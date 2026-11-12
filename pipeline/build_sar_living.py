"""Add Hong Kong and Macau residents to the Guangdong demand grid.

Hong Kong comes from the 2021 census zones in HKPOP. That download is already
longitude and latitude, so it matches the rest of the map. People are placed
on OpenStreetMap buildings inside each zone, which keeps country parks empty.
Zones with no buildings fall back to one point.

Macau comes from the 30 m raster in MCPOP. The tile is also longitude and
latitude, but it covers Zhuhai as well, so it is clipped to the Macau outline
before use. The raster has no ages. Each cell is split with the 2021 census
shares: under 15, 15-64, and 65 and over.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import osmium
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely import STRtree
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid

from build_city_bounds import OUTPUTS, BoundaryHandler, contains, vertex_count
from build_province_mask import main as build_province_mask

ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = ROOT / "data" / "raw" / "guangdong-latest.osm.pbf"
ZONES_PATH = ROOT / "HKPOP" / "ltpug" / "LTPUG_21C_LTPUG_21C_converted.geojson"
MACAU_RASTER = ROOT / "MCPOP" / "mac_general_2020.tif"
DEMAND_PATH = ROOT / "viewer" / "demand.json"
ALBERS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs"
)
CELL = 100
MIN_PEOPLE = 50
# 2021 census, end-2021: 682,070 people. Under 15: 98,981. Ages 15-64: 500,277.
# Age 65 and over: 82,812. The raster is only a total, so every cell uses these shares.
MACAU_CENSUS = (98981, 500277, 82812)
HK_RELATIONS = (913110, 20044132)
MACAU_RELATION = 1867188
# Buildings that stand in for where people live. building=yes is included because
# most Hong Kong towers are tagged that way.
RESIDENTIAL_BUILDINGS = {
    "yes",
    "residential",
    "apartments",
    "house",
    "detached",
    "terrace",
    "semidetached_house",
    "dormitory",
    "bungalow",
    "static_caravan",
}


def polygonal(geom):
    geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        parts = [part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")]
        geom = unary_union(parts) if parts else geom
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise RuntimeError(f"boundary is {geom.geom_type}")
    return geom


def hamilton(weights: np.ndarray, total: int) -> np.ndarray:
    """Split an integer total across weights without changing the sum."""
    allocated = np.zeros(len(weights), dtype=np.int64)
    if total <= 0 or len(weights) == 0:
        return allocated
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return allocated
    raw = total * weights / total_weight
    allocated[:] = np.floor(raw).astype(np.int64)
    left = int(total - int(allocated.sum()))
    if left > 0:
        order = np.argsort(-(raw - allocated), kind="stable")
        allocated[order[:left]] += 1
    return allocated


def ring_area(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 4:
        return 0.0
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) * 0.5)


def extract_boundaries() -> tuple[dict, dict]:
    wanted = {
        913110: ("Hong Kong", "香港"),
        20044132: ("Hong Kong", "香港"),
        MACAU_RELATION: ("Macau", "澳门"),
    }
    handler = BoundaryHandler(wanted)
    handler.apply_file(str(PBF_PATH), locations=True, idx="flex_mem")
    if MACAU_RELATION not in handler.found or not any(relation_id in handler.found for relation_id in HK_RELATIONS):
        found = ", ".join(str(relation_id) for relation_id in handler.found) or "none"
        raise SystemExit(f"missing Hong Kong or Macau boundary, found {found}")

    def accept(feature: dict, checks: list[tuple[str, float, float, bool]]) -> bool:
        geometry = feature["geometry"]
        ok = True
        for label, lon, lat, expect in checks:
            hit = contains(geometry, lon, lat)
            print(f"  {feature['properties']['name']} {label}: {hit} expect {expect}", flush=True)
            ok = ok and hit == expect
        return ok

    hk_checks = [
        ("Central", 114.1583, 22.2818, True),
        ("Kowloon", 114.175, 22.320, True),
        ("Tung Chung", 113.948, 22.289, True),
        ("Shenzhen", 114.10, 22.55, False),
        ("Macau", 113.543, 22.198, False),
    ]
    hk_choices = []
    for relation_id in HK_RELATIONS:
        if relation_id not in handler.found:
            print(f"Hong Kong relation {relation_id} has no area", flush=True)
            continue
        feature = handler.found[relation_id]
        feature["properties"]["osm_relation"] = relation_id
        print(
            f"Hong Kong relation {relation_id} parts={len(feature['geometry']['coordinates'])} "
            f"vertices={vertex_count(feature['geometry'])}",
            flush=True,
        )
        if accept(feature, hk_checks):
            hk_choices.append(feature)
    if not hk_choices:
        raise SystemExit("no Hong Kong boundary passed the coastline check")
    hong_kong = max(hk_choices, key=lambda feature: vertex_count(feature["geometry"]))

    macau = handler.found[MACAU_RELATION]
    print(
        f"Macau relation {MACAU_RELATION} parts={len(macau['geometry']['coordinates'])} "
        f"vertices={vertex_count(macau['geometry'])}",
        flush=True,
    )
    mo_checks = [
        ("peninsula", 113.543, 22.198, True),
        ("Taipa", 113.559, 22.158, True),
        ("Coloane", 113.565, 22.123, True),
        ("Zhuhai", 113.55, 22.25, False),
    ]
    if not accept(macau, mo_checks):
        raise SystemExit("Macau boundary failed the coastline check")
    return hong_kong, macau


def overlap_km2(left, right) -> float:
    shared = left.intersection(right)
    return shared.area * (111_320 ** 2) * math.cos(math.radians(22.3)) / 1_000_000


def publish_cities(hong_kong: dict, macau: dict, hong_kong_geom, macau_geom) -> None:
    cities = json.loads((ROOT / "viewer" / "cities.json").read_text(encoding="utf-8"))
    by_name = {feature["properties"]["name"]: feature for feature in cities["features"]}
    for other_name in ("Shenzhen", "Zhuhai"):
        other = polygonal(shape(by_name[other_name]["geometry"]))
        shared_hk = overlap_km2(hong_kong_geom, other)
        shared_mo = overlap_km2(macau_geom, other)
        print(f"overlap {other_name}: Hong Kong {shared_hk:.2f} km2, Macau {shared_mo:.2f} km2", flush=True)
        if shared_hk > 5 or shared_mo > 5:
            raise SystemExit(f"{other_name} overlaps a new boundary by more than 5 km2")
    cities["features"] = [
        feature for feature in cities["features"] if feature["properties"]["name"] not in ("Hong Kong", "Macau")
    ]
    cities["features"].extend([hong_kong, macau])
    payload = json.dumps(cities, ensure_ascii=False, separators=(",", ":"))
    for path in OUTPUTS:
        path.write_text(payload, encoding="utf-8")
        print(path, flush=True)


def load_zones(hong_kong_geom):
    data = json.loads(ZONES_PATH.read_text(encoding="utf-8"))
    zones = []
    for feature in data["features"]:
        props = feature["properties"]
        ages = [int(props[key]) for key in ("age_1", "age_2", "age_3", "age_4", "age_5")]
        youth, labor, old = ages[0], ages[1] + ages[2] + ages[3], ages[4]
        if youth + labor + old != int(props["t_pop"]):
            raise SystemExit(f"zone {props['ltpug']} ages do not add up to its population")
        geom = polygonal(shape(feature["geometry"]).intersection(hong_kong_geom))
        if geom.is_empty:
            continue
        zones.append({"id": str(props["ltpug"]), "youth": youth, "labor": labor, "old": old, "geom": geom})
    print(f"Hong Kong zones={len(zones)}", flush=True)
    return zones


class HomeFinder:
    """Building and residential-land weights inside each census zone."""

    def __init__(self, zones, hong_kong_geom):
        self.zones = zones
        self.geoms = [zone["geom"] for zone in zones]
        self.tree = STRtree(self.geoms)
        self.hong_kong = prep(hong_kong_geom)
        self.to_proj = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
        self.to_wgs = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
        self.buildings: list[dict[tuple[int, int], list[float]]] = [dict() for _ in zones]
        self.landuse: list[dict[tuple[int, int], list[float]]] = [dict() for _ in zones]
        self.seen = 0

    def add(self, store: dict[tuple[int, int], list[float]], ix: int, iy: int, area: float, x: float, y: float) -> None:
        bucket = store.get((ix, iy))
        if bucket is None:
            store[(ix, iy)] = [area, area * x, area * y]
            return
        bucket[0] += area
        bucket[1] += area * x
        bucket[2] += area * y

    def consume(self, obj, landuse: bool) -> None:
        parts = []
        try:
            outers = list(obj.outer_rings())
        except osmium.InvalidLocationError:
            return
        for outer in outers:
            try:
                coords = [(node.lon, node.lat) for node in outer if node.location.valid()]
            except osmium.InvalidLocationError:
                continue
            if len(coords) < 4:
                continue
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]
            if max(lons) < 113.80 or min(lons) > 114.46 or max(lats) < 22.14 or min(lats) > 22.58:
                continue
            xs, ys = self.to_proj.transform(lons, lats)
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            area = ring_area(xs, ys)
            for inner in obj.inner_rings(outer):
                try:
                    inner_coords = [(node.lon, node.lat) for node in inner if node.location.valid()]
                except osmium.InvalidLocationError:
                    continue
                if len(inner_coords) < 4:
                    continue
                ixs, iys = self.to_proj.transform(
                    [coord[0] for coord in inner_coords], [coord[1] for coord in inner_coords]
                )
                area -= ring_area(np.asarray(ixs, dtype=np.float64), np.asarray(iys, dtype=np.float64))
            if area <= 1:
                continue
            parts.append((float(xs.mean()), float(ys.mean()), area))
        if not parts:
            return
        total = sum(part[2] for part in parts)
        x = sum(part[0] * part[2] for part in parts) / total
        y = sum(part[1] * part[2] for part in parts) / total
        lon, lat = self.to_wgs.transform(x, y)
        point = Point(float(lon), float(lat))
        if not self.hong_kong.covers(point):
            return
        hits = self.tree.query(point, predicate="covered_by")
        if len(hits) == 0:
            return
        zone_index = int(hits[0])
        ix = int(math.floor(x / CELL))
        iy = int(math.floor(y / CELL))
        store = self.landuse[zone_index] if landuse else self.buildings[zone_index]
        self.add(store, ix, iy, total, x, y)


def scan_homes(zones, hong_kong_geom) -> HomeFinder:
    finder = HomeFinder(zones, hong_kong_geom)
    processor = (
        osmium.FileProcessor(str(PBF_PATH))
        .with_filter(osmium.filter.KeyFilter("building", "landuse"))
        .with_locations()
        .with_areas()
    )
    for obj in processor:
        finder.seen += 1
        if finder.seen % 400_000 == 0:
            print(f"scanned {finder.seen:,} map objects", flush=True)
        if not obj.is_area():
            continue
        building = obj.tags.get("building")
        landuse = obj.tags.get("landuse")
        if building in RESIDENTIAL_BUILDINGS:
            finder.consume(obj, landuse=False)
        elif not building and landuse == "residential":
            finder.consume(obj, landuse=True)
    filled = sum(1 for zone in finder.buildings if zone)
    print(f"scanned {finder.seen:,} map objects, zones with buildings={filled}", flush=True)
    return finder


def weights_from_buckets(buckets: dict[tuple[int, int], list[float]]):
    keys = list(buckets)
    if not keys:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
    weight = np.array([buckets[key][0] for key in keys], dtype=np.float64)
    xs = np.array([buckets[key][1] / buckets[key][0] for key in keys], dtype=np.float64)
    ys = np.array([buckets[key][2] / buckets[key][0] for key in keys], dtype=np.float64)
    cells = np.array(keys, dtype=np.int64)
    return cells, weight, xs, ys


def fallback_weights(geom, to_proj: Transformer):
    minx, miny, maxx, maxy = geom.bounds
    xs, ys = to_proj.transform([minx, maxx], [miny, maxy])
    ix0 = int(math.floor(min(xs) / CELL))
    ix1 = int(math.floor(max(xs) / CELL))
    iy0 = int(math.floor(min(ys) / CELL))
    iy1 = int(math.floor(max(ys) / CELL))
    if (ix1 - ix0 + 1) * (iy1 - iy0 + 1) > 4000:
        point = geom.representative_point()
        x, y = to_proj.transform(point.x, point.y)
        return np.array([[int(math.floor(x / CELL)), int(math.floor(y / CELL))]]), np.array([1.0]), np.array([x]), np.array([y])
    prepared = prep(geom)
    to_wgs = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    cells = []
    xs_out = []
    ys_out = []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            x = (ix + 0.5) * CELL
            y = (iy + 0.5) * CELL
            lon, lat = to_wgs.transform(x, y)
            if prepared.covers(Point(float(lon), float(lat))):
                cells.append((ix, iy))
                xs_out.append(x)
                ys_out.append(y)
    if not cells:
        point = geom.representative_point()
        x, y = to_proj.transform(point.x, point.y)
        return np.array([[int(math.floor(x / CELL)), int(math.floor(y / CELL))]]), np.array([1.0]), np.array([x]), np.array([y])
    return np.array(cells, dtype=np.int64), np.ones(len(cells)), np.array(xs_out), np.array(ys_out)


def accumulate(cells, weight, xs, ys, youth: int, labor: int, old: int, into: dict) -> None:
    if youth + labor + old <= 0 or len(weight) == 0:
        return
    parts = (hamilton(weight, labor), hamilton(weight, youth), hamilton(weight, old))
    for index in range(len(weight)):
        key = (int(cells[index, 0]), int(cells[index, 1]))
        bucket = into.get(key)
        people = (int(parts[0][index]), int(parts[1][index]), int(parts[2][index]))
        if bucket is None:
            into[key] = [people[0], people[1], people[2], people[0] * xs[index], people[1] * xs[index], people[2] * xs[index], people[0] * ys[index], people[1] * ys[index], people[2] * ys[index]]
            continue
        for offset, value in enumerate(people):
            bucket[offset] += value
            bucket[3 + offset] += value * xs[index]
            bucket[6 + offset] += value * ys[index]


def emit_points(cells: dict) -> list[list]:
    to_wgs = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    kept = []
    groups: dict[tuple[int, int], list[int]] = {}
    for key, bucket in cells.items():
        residents = bucket[0] + bucket[1] + bucket[2]
        if residents >= MIN_PEOPLE:
            kept.append(point_from_bucket(bucket, to_wgs))
        elif residents > 0:
            group = (key[0] // 5, key[1] // 5)
            groups.setdefault(group, [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            dest = groups[group]
            for index, value in enumerate(bucket):
                dest[index] += value
    villages = 0
    for bucket in groups.values():
        if bucket[0] + bucket[1] + bucket[2] >= MIN_PEOPLE:
            kept.append(point_from_bucket(bucket, to_wgs))
            villages += 1
    print(f"cells={len(kept) - villages:,} village groups={villages:,}", flush=True)
    return kept


def point_from_bucket(bucket, to_wgs: Transformer) -> list:
    labor, youth, old = int(bucket[0]), int(bucket[1]), int(bucket[2])
    residents = labor + youth + old
    weight = residents or 1
    x = (bucket[3] + bucket[4] + bucket[5]) / weight
    y = (bucket[6] + bucket[7] + bucket[8]) / weight
    lon, lat = to_wgs.transform(x, y)
    return [round(float(lon), 5), round(float(lat), 5), residents, labor, youth, old, 0]


def hong_kong_points(hong_kong_geom) -> list[list]:
    zones = load_zones(hong_kong_geom)
    finder = scan_homes(zones, hong_kong_geom)
    to_proj = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    cells: dict[tuple[int, int], list] = {}
    fallbacks = 0
    for index, zone in enumerate(zones):
        buckets = finder.buildings[index] or finder.landuse[index]
        if buckets:
            parsed = weights_from_buckets(buckets)
        else:
            parsed = fallback_weights(zone["geom"], to_proj)
            fallbacks += 1
            print(f"no buildings in zone {zone['id']}: {zone['youth'] + zone['labor'] + zone['old']:,} people", flush=True)
        accumulate(*parsed, zone["youth"], zone["labor"], zone["old"], cells)
    print(f"Hong Kong fallback zones={fallbacks}", flush=True)
    return emit_points(cells)


def macau_points(macau_geom) -> list[list]:
    with rasterio.open(MACAU_RASTER) as src:
        values = src.read(1).astype(np.float64)
        transform = src.transform
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise SystemExit(f"Macau raster CRS is {src.crs}, expected longitude/latitude")
    values[~np.isfinite(values)] = 0
    values[values < 0] = 0
    inside = geometry_mask([mapping(macau_geom)], out_shape=values.shape, transform=transform, invert=True)
    values[~inside] = 0
    clipped = float(values.sum())
    print(f"Macau raster inside the outline={clipped:,.0f}", flush=True)
    rows, cols = np.nonzero(values > 0)
    if len(rows) == 0:
        raise SystemExit("Macau outline missed the population raster")
    lons = transform.c + transform.a * (cols.astype(np.float64) + 0.5)
    lats = transform.f + transform.e * (rows.astype(np.float64) + 0.5)
    to_proj = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    xs, ys = to_proj.transform(lons, lats)
    ix = np.floor(np.asarray(xs) / CELL).astype(np.int64)
    iy = np.floor(np.asarray(ys) / CELL).astype(np.int64)
    pairs = np.stack([ix, iy], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    weight = np.bincount(inverse, weights=values[rows, cols])
    pop = values[rows, cols]
    xw = np.bincount(inverse, weights=np.asarray(xs) * pop) / weight
    yw = np.bincount(inverse, weights=np.asarray(ys) * pop) / weight
    youth_share, labor_share, old_share = MACAU_CENSUS
    census = youth_share + labor_share + old_share
    total = int(round(clipped))
    youth = int(round(clipped * youth_share / census))
    labor = int(round(clipped * labor_share / census))
    old = int(round(clipped * old_share / census))
    labor += total - (youth + labor + old)
    cells: dict[tuple[int, int], list] = {}
    accumulate(unique, weight, xw, yw, youth, labor, old, cells)
    return emit_points(cells)


def merge_demand(points: list[list], hong_kong_geom, macau_geom) -> None:
    demand = json.loads(DEMAND_PATH.read_text(encoding="utf-8"))
    regions = prep(unary_union([hong_kong_geom, macau_geom]))
    kept = []
    removed = 0
    for point in demand["points"]:
        if regions.covers(Point(float(point[0]), float(point[1]))):
            removed += point[2]
            continue
        kept.append(point)
    if removed > 20_000:
        raise SystemExit(f"refusing to drop {removed:,} existing residents inside Hong Kong or Macau")
    print(f"existing residents inside the new outlines={removed:,}", flush=True)
    kept.extend(points)
    totals = {
        "residents": sum(point[2] for point in kept),
        "labor": sum(point[3] for point in kept),
        "youth": sum(point[4] for point in kept),
        "old": sum(point[5] for point in kept),
    }
    demand["totals"] = totals
    demand["points"] = kept
    DEMAND_PATH.write_text(json.dumps(demand, separators=(",", ":")), encoding="utf-8")
    added = sum(point[2] for point in points)
    print(
        f"added={added:,} blobs={len(points):,}  residents={totals['residents']:,}  "
        f"labor={totals['labor']:,}  youth={totals['youth']:,}  old={totals['old']:,}",
        flush=True,
    )


def main() -> None:
    if not PBF_PATH.exists():
        raise SystemExit(f"missing OpenStreetMap extract: {PBF_PATH}")
    if not ZONES_PATH.exists():
        raise SystemExit(f"missing Hong Kong census zones: {ZONES_PATH}")
    if not MACAU_RASTER.exists():
        raise SystemExit(f"missing Macau raster: {MACAU_RASTER}")
    hong_kong, macau = extract_boundaries()
    hong_kong_geom = polygonal(shape(hong_kong["geometry"]))
    macau_geom = polygonal(shape(macau["geometry"]))
    publish_cities(hong_kong, macau, hong_kong_geom, macau_geom)
    build_province_mask()
    points = hong_kong_points(hong_kong_geom)
    points.extend(macau_points(macau_geom))
    merge_demand(points, hong_kong_geom, macau_geom)


if __name__ == "__main__":
    sys.exit(main())

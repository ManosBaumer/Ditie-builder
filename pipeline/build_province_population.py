"""Extend living demand from the Guangzhou box to the rest of Guangdong.

The Guangzhou box keeps its 100 m age grids. Everywhere else, each city's
2020 census population (minus anyone already counted in that box) is placed
on OpenStreetMap housing: residential buildings where they are mapped, and
residential land-use only in the gaps. The placement is a weight. The count
is the census. Age bands outside the box are that city's own census shares
for 0-14, 15-64 and 65+.

Metro cities are kept in 100 m bins. Other prefectures are summed to 500 m.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = ROOT / "data" / "raw" / "guangdong-latest.osm.pbf"
CITIES_PATH = ROOT / "viewer" / "cities.json"
DEMAND_PATH = ROOT / "viewer" / "demand.json"

GZ_WEST, GZ_SOUTH, GZ_EAST, GZ_NORTH = 112.920295, 22.730316, 113.74153, 23.431411
MIN_PEOPLE = 50
LAT_M = 110_746.0
LON_M = 111_320.0 * math.cos(math.radians(23.15))

# 2020 seventh-census permanent population, and age shares in percent:
# 0-14, 15-59, 60+, 65+. Labor on the map is 15-64, which is 15-59 plus 60-64.
# Guangdong Bureau of Statistics, seventh census communiques no. 2 and no. 4.
CENSUS = {
    "Guangzhou": (18_676_605, 13.87, 74.72, 11.41, 7.82),
    "Shenzhen": (17_560_061, 15.11, 79.53, 5.36, 3.22),
    "Zhuhai": (2_439_585, 15.88, 74.12, 10.00, 6.64),
    "Shantou": (5_502_031, 22.28, 62.20, 15.52, 10.67),
    "Foshan": (9_498_863, 15.11, 74.37, 10.52, 7.35),
    "Shaoguan": (2_855_131, 21.60, 60.02, 18.38, 13.12),
    "Heyuan": (2_837_686, 25.28, 58.25, 16.47, 11.74),
    "Meizhou": (3_873_239, 23.50, 56.46, 20.04, 14.41),
    "Huizhou": (6_042_852, 20.76, 69.19, 10.05, 6.83),
    "Shanwei": (2_672_819, 25.88, 59.20, 14.92, 10.60),
    "Dongguan": (10_466_625, 13.12, 81.41, 5.47, 3.54),
    "Zhongshan": (4_418_060, 15.69, 75.44, 8.87, 5.98),
    "Jiangmen": (4_798_090, 16.02, 65.72, 18.26, 13.01),
    "Yangjiang": (2_602_959, 22.48, 60.53, 16.99, 12.48),
    "Zhanjiang": (6_981_236, 26.08, 57.13, 16.79, 11.93),
    "Maoming": (6_174_050, 27.07, 56.13, 16.80, 12.09),
    "Zhaoqing": (4_113_594, 22.16, 61.43, 16.41, 11.81),
    "Qingyuan": (3_969_473, 23.33, 60.34, 16.33, 11.62),
    "Chaozhou": (2_568_387, 19.30, 61.53, 19.17, 13.27),
    "Jieyang": (5_577_814, 25.53, 58.36, 16.11, 11.30),
    "Yunfu": (2_383_350, 25.12, 57.44, 17.44, 12.50),
}
METRO = {"Guangzhou", "Shenzhen", "Foshan", "Dongguan", "Zhuhai"}
HOUSING = {"apartments", "residential", "house", "detached", "terrace", "semidetached_house", "dormitory", "yes"}
FLOORS = {
    "apartments": 18,
    "residential": 12,
    "dormitory": 8,
    "house": 2,
    "detached": 2,
    "terrace": 3,
    "semidetached_house": 2,
    "yes": 6,
}


def age_shares(youth: float, age15: float, age60: float, age65: float) -> tuple[float, float, float]:
    labor = age15 + age60 - age65
    total = youth + labor + age65
    return youth / total, labor / total, age65 / total


def split_ages(total: float, shares: tuple[float, float, float]) -> tuple[int, int, int, int]:
    youth_s, labor_s, old_s = shares
    people = int(round(total))
    youth = int(round(people * youth_s))
    labor = int(round(people * labor_s))
    old = people - youth - labor
    if old < 0:
        labor += old
        old = 0
    if labor < 0:
        youth += labor
        labor = 0
    return youth + labor + old, labor, youth, old


def ring_contains(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        yi, yj = ring[i][1], ring[j][1]
        if (yi > lat) != (yj > lat):
            xi, xj = ring[i][0], ring[j][0]
            if lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def ring_metrics(coords: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [lon * LON_M for lon, _ in coords]
    ys = [lat * LAT_M for _, lat in coords]
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(xs) - 1):
        cross = xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
        area2 += cross
        cx += (xs[i] + xs[i + 1]) * cross
        cy += (ys[i] + ys[i + 1]) * cross
    if abs(area2) < 1e-6:
        return 0.0, coords[0][0], coords[0][1]
    cx /= 3 * area2
    cy /= 3 * area2
    return abs(area2) / 2, cx / LON_M, cy / LAT_M


def floors_of(tags, kind: str) -> float:
    raw = tags.get("building:levels")
    if raw:
        try:
            value = float(str(raw).split(";")[0])
        except ValueError:
            value = 0
        if 1 <= value <= 80:
            return value
    return FLOORS.get(kind, 6)


def main() -> None:
    if not PBF_PATH.exists():
        raise SystemExit(f"missing OpenStreetMap extract: {PBF_PATH}")

    cities = json.loads(CITIES_PATH.read_text(encoding="utf-8"))
    names = [feature["properties"]["name"] for feature in cities["features"]]
    missing = [name for name in names if name not in CENSUS]
    if missing or len(names) != len(CENSUS):
        raise SystemExit(f"census table does not match the city file: {missing}")
    shares = {name: age_shares(*row[1:]) for name, row in CENSUS.items()}

    prepared = []
    west = south = 180.0
    east = north = -180.0
    for feature in cities["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        rings = [polygon[0] for polygon in polygons]
        xs = [point[0] for ring in rings for point in ring]
        ys = [point[1] for ring in rings for point in ring]
        prepared.append((feature["properties"]["name"], (min(xs), min(ys), max(xs), max(ys)), rings))
        west, south = min(west, min(xs)), min(south, min(ys))
        east, north = max(east, max(xs)), max(north, max(ys))

    def city_of(lon: float, lat: float) -> str | None:
        for name, (cwest, csouth, ceast, cnorth), rings in prepared:
            if lon < cwest or lon > ceast or lat < csouth or lat > cnorth:
                continue
            if any(ring_contains(lon, lat, ring) for ring in rings):
                return name
        return None

    previous = json.loads(DEMAND_PATH.read_text(encoding="utf-8"))
    kept = []
    already = {name: 0 for name in names}
    outside = 0
    for point in previous["points"]:
        if not any(point[i] for i in (2, 3, 4, 5)):
            continue
        lon, lat = float(point[0]), float(point[1])
        if not (GZ_WEST <= lon <= GZ_EAST and GZ_SOUTH <= lat <= GZ_NORTH):
            continue
        kept.append([round(lon, 5), round(lat, 5), int(point[2]), int(point[3]), int(point[4]), int(point[5]), 0])
        name = city_of(lon, lat)
        if name is None:
            outside += 1
        else:
            already[name] += int(point[2])
    print(f"kept {len(kept):,} age-grid cells from the Guangzhou box, {outside:,} not inside a city", flush=True)

    housing: dict[tuple, float] = {}
    landuse: dict[tuple, float] = {}

    def cell(name: str, lon: float, lat: float) -> tuple:
        size = 100 if name in METRO else 500
        return name, round(lon * LON_M / size), round(lat * LAT_M / size)

    def add(store: dict, name: str, lon: float, lat: float, weight: float) -> None:
        if weight <= 0 or (GZ_WEST <= lon <= GZ_EAST and GZ_SOUTH <= lat <= GZ_NORTH):
            return
        key = cell(name, lon, lat)
        store[key] = store.get(key, 0.0) + weight

    def shape_of(area) -> tuple[float, float, float, list] | None:
        total = 0.0
        cx = 0.0
        cy = 0.0
        outer_ring = None
        try:
            outers = list(area.outer_rings())
        except osmium.InvalidLocationError:
            return None
        for outer in outers:
            try:
                coords = [(node.lon, node.lat) for node in outer]
            except osmium.InvalidLocationError:
                continue
            if len(coords) < 4:
                continue
            footprint, lon, lat = ring_metrics(coords)
            hole = 0.0
            try:
                inners = list(area.inner_rings(outer))
            except osmium.InvalidLocationError:
                inners = []
            for inner in inners:
                try:
                    inner_coords = [(node.lon, node.lat) for node in inner]
                except osmium.InvalidLocationError:
                    continue
                if len(inner_coords) >= 4:
                    hole += ring_metrics(inner_coords)[0]
            footprint = max(0.0, footprint - hole)
            if footprint <= 0:
                continue
            total += footprint
            cx += lon * footprint
            cy += lat * footprint
            outer_ring = coords
        if total <= 0 or outer_ring is None:
            return None
        return total, cx / total, cy / total, outer_ring

    def spread(name: str, ring: list, weight: float, store: dict) -> None:
        step = 150 if name in METRO else 400
        lons = [point[0] for point in ring]
        lats = [point[1] for point in ring]
        dlon = step / LON_M
        dlat = step / LAT_M
        width = max(1, int((max(lons) - min(lons)) / dlon) + 1)
        height = max(1, int((max(lats) - min(lats)) / dlat) + 1)
        if width * height > 4000:
            scale = math.sqrt(width * height / 4000)
            dlon *= scale
            dlat *= scale
        hits = []
        lat = min(lats) + dlat / 2
        while lat < max(lats):
            lon = min(lons) + dlon / 2
            while lon < max(lons):
                if ring_contains(lon, lat, ring):
                    hits.append((lon, lat))
                lon += dlon
            lat += dlat
        if not hits:
            hits = [(sum(lons) / len(lons), sum(lats) / len(lats))]
        share = weight / len(hits)
        for lon, lat in hits:
            add(store, name, lon, lat, share)

    class Handler(osmium.SimpleHandler):
        def area(self, area) -> None:
            building = area.tags.get("building") or ""
            land = area.tags.get("landuse") or ""
            housing_kind = building if building in HOUSING else ""
            if housing_kind == "yes" and (area.tags.get("shop") or area.tags.get("office") or area.tags.get("amenity")):
                housing_kind = ""
            if not housing_kind and land != "residential":
                return
            shaped = shape_of(area)
            if shaped is None:
                return
            footprint, lon, lat, ring = shaped
            if footprint < 25:
                return
            name = city_of(lon, lat)
            if name is None:
                return
            if housing_kind:
                weight = footprint * floors_of(area.tags, housing_kind)
                if footprint > 15_000:
                    spread(name, ring, weight, housing)
                else:
                    add(housing, name, lon, lat, weight)
            else:
                # A residential area stands in only where no housing was drawn.
                spread(name, ring, footprint * 0.35 * 6, landuse)

    Handler().apply_file(str(PBF_PATH), locations=True)
    for key, weight in landuse.items():
        if key not in housing:
            housing[key] = weight
    print(f"housing bins={len(housing):,}", flush=True)

    by_city: dict[str, list[tuple[int, int, float]]] = {name: [] for name in names}
    for (name, ix, iy), weight in housing.items():
        by_city[name].append((ix, iy, weight))

    added: list[list[float]] = []
    for name, items in by_city.items():
        target = CENSUS[name][0] - already[name]
        weight = sum(item[2] for item in items)
        if target <= 0 or weight <= 0:
            print(f"  {name:<12} census={CENSUS[name][0]:>10,} already={already[name]:>10,} added=0", flush=True)
            continue
        scale = target / weight
        size = 100 if name in METRO else 500
        floor = 50
        for candidate in (50, 25, 10):
            shown = sum(item[2] * scale for item in items if item[2] * scale >= candidate)
            floor = candidate
            if shown >= target * 0.85:
                break
        city_people = 0
        city_bins = 0
        for ix, iy, item_weight in items:
            count = item_weight * scale
            if count < floor:
                continue
            residents, labor, youth, old = split_ages(count, shares[name])
            if residents < floor:
                continue
            added.append([
                round(ix * size / LON_M, 5),
                round(iy * size / LAT_M, 5),
                residents, labor, youth, old, 0,
            ])
            city_people += residents
            city_bins += 1
        print(
            f"  {name:<12} census={CENSUS[name][0]:>10,} already={already[name]:>10,} "
            f"added={city_people:>10,} bins={city_bins:,} min={floor}",
            flush=True,
        )

    points = kept + added
    points.sort(key=lambda point: point[2])
    totals = {
        "residents": sum(point[2] for point in points),
        "labor": sum(point[3] for point in points),
        "youth": sum(point[4] for point in points),
        "old": sum(point[5] for point in points),
    }
    payload = {
        "bounds": [round(west, 4), round(south, 4), round(east, 4), round(north, 4)],
        "cell_m": 100,
        "min_people": MIN_PEOPLE,
        "fields": ["lon", "lat", "residents", "labor", "youth", "old", "jobs"],
        "totals": totals,
        "population": {
            "guangzhou_box": "100 m age grids kept as they were",
            "elsewhere": "2020 census population of each city, placed on OpenStreetMap housing",
            "age_elsewhere": "each city's seventh-census shares for 0-14, 15-64 and 65+",
            "metro_100m": sorted(METRO),
            "other_cell_m": 500,
        },
        "points": points,
    }
    DEMAND_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"{len(points):,} cells ({len(kept):,} kept, {len(added):,} added)  "
        f"residents={totals['residents']:,} labor={totals['labor']:,} "
        f"youth={totals['youth']:,} old={totals['old']:,}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())

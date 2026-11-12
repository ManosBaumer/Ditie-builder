"""Estimate workplace demand across Guangdong from OpenStreetMap.

OpenStreetMap does not count workers. Each workplace polygon contributes
floor area times a workers-per-square-metre rate, and each workplace point
contributes a small fixed number. The weights are then scaled so total jobs
equal the resident labor force already on the map.

Small buildings and shops keep their real coordinates, so the bubbles follow
streets instead of a grid. A footprint large enough to cover several 100 m
cells is spread across the cells inside it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import osmium
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = ROOT / "data" / "raw" / "guangdong-latest.osm.pbf"
DEMAND_PATH = ROOT / "viewer" / "demand.json"
CITIES_PATH = ROOT / "viewer" / "cities.json"
# Same Krasovsky Albers grid the Guangzhou age rasters use, so a large
# workplace still spreads across 100 m cells.
ALBERS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs"
)
CELL = 100
# Footprints smaller than this stay on their true centroid. Larger sites are
# spread across the 100 m cells they actually cover.
SMALL_FOOTPRINT_M2 = 15_000

# Default floors, workers per m2 of floor area, and how much of a land-use
# polygon is treated as actual floorplate.
SPEC = {
    "office": (10, 1 / 12, 0.45),
    "commercial": (6, 1 / 20, 0.40),
    "retail": (2, 1 / 25, 0.45),
    "industrial": (2, 1 / 45, 0.40),
    "warehouse": (1, 1 / 80, 0.50),
    "education": (4, 1 / 35, 0.25),
    "hospital": (6, 1 / 30, 0.30),
    "public": (5, 1 / 30, 0.30),
}

BUILDING_KIND = {
    "office": "office",
    "commercial": "commercial",
    "retail": "retail",
    "supermarket": "retail",
    "kiosk": "retail",
    "industrial": "industrial",
    "warehouse": "warehouse",
    "factory": "industrial",
    "manufacture": "industrial",
    "school": "education",
    "university": "education",
    "college": "education",
    "hospital": "hospital",
    "clinic": "hospital",
    "public": "public",
    "government": "public",
    "civic": "public",
    "hotel": "commercial",
    "train_station": "public",
    "transportation": "public",
}

LANDUSE_KIND = {
    "commercial": "commercial",
    "retail": "retail",
    "industrial": "industrial",
}

AMENITY_KIND = {
    "school": "education",
    "kindergarten": "education",
    "college": "education",
    "university": "education",
    "hospital": "hospital",
    "clinic": "hospital",
    "government": "public",
    "townhall": "public",
    "courthouse": "public",
}

# Used when a workplace is only a point, not a footprint.
POI_JOBS = {
    "education": 80,
    "hospital": 400,
    "public": 120,
    "office": 40,
    "retail": 4,
    "commercial": 25,
    "industrial": 30,
}

SKIP_BUILDINGS = {
    "yes",
    "residential",
    "apartments",
    "house",
    "detached",
    "terrace",
    "semidetached_house",
    "dormitory",
    "bungalow",
    "farm",
    "garage",
    "garages",
    "shed",
    "roof",
    "ruins",
    "construction",
    "hut",
    "cabin",
    "static_caravan",
}


def tag(tags, key: str) -> str:
    value = tags.get(key)
    return "" if value is None else str(value)


def classify_area(tags) -> tuple[str, str] | None:
    building = tag(tags, "building")
    amenity = tag(tags, "amenity")
    office = tag(tags, "office")
    landuse = tag(tags, "landuse")
    if building and building not in SKIP_BUILDINGS:
        kind = BUILDING_KIND.get(building)
        if kind is None and office:
            kind = "office"
        if kind is None and amenity in AMENITY_KIND:
            kind = AMENITY_KIND[amenity]
        if kind:
            return kind, "building"
    if office and building:
        return "office", "building"
    if landuse in LANDUSE_KIND:
        return LANDUSE_KIND[landuse], "landuse"
    return None


def classify_node(tags) -> str | None:
    amenity = tag(tags, "amenity")
    if amenity in AMENITY_KIND:
        return AMENITY_KIND[amenity]
    if tag(tags, "office"):
        return "office"
    landuse = tag(tags, "landuse")
    if landuse in LANDUSE_KIND:
        return LANDUSE_KIND[landuse]
    if tag(tags, "shop"):
        return "retail"
    return None


def parse_floors(tags, default: float) -> float:
    raw = tag(tags, "building:levels")
    if not raw:
        return default
    try:
        value = float(raw.split(";")[0])
    except ValueError:
        return default
    if 1 <= value <= 120:
        return value
    return default


def ring_area(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 4:
        return 0.0
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) * 0.5)


def ring_mask(px: np.ndarray, py: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    inside = np.zeros(px.shape, dtype=bool)
    x1 = xs[:-1]
    y1 = ys[:-1]
    x2 = xs[1:]
    y2 = ys[1:]
    for i in range(len(x1)):
        yi = y1[i]
        yj = y2[i]
        crosses = (yi > py) != (yj > py)
        if not np.any(crosses):
            continue
        xi = x1[i]
        xj = x2[i]
        cross_x = (xj - xi) * (py[crosses] - yi) / (yj - yi) + xi
        inside[crosses] ^= px[crosses] < cross_x
    return inside


def point_in_ring(x: float, y: float, xs: np.ndarray, ys: np.ndarray) -> bool:
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        yi, yj = float(ys[i]), float(ys[j])
        if (yi > y) != (yj > y):
            xi, xj = float(xs[i]), float(xs[j])
            cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def intersects(lons: list[float], lats: list[float], bounds: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bounds
    return max(lons) >= west and min(lons) <= east and max(lats) >= south and min(lats) <= north


def main() -> None:
    if not PBF_PATH.exists() or PBF_PATH.stat().st_size < 1_000_000:
        raise SystemExit(f"missing OpenStreetMap extract: {PBF_PATH}")
    demand = json.loads(DEMAND_PATH.read_text(encoding="utf-8"))
    west, south, east, north = demand["bounds"]
    bounds = (west, south, east, north)
    labor_target = float(demand["totals"]["labor"])

    to_proj = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    to_wgs = Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)
    edge_lon = np.concatenate([
        np.linspace(west, east, 40), np.linspace(west, east, 40),
        np.full(40, west), np.full(40, east),
    ])
    edge_lat = np.concatenate([
        np.full(40, south), np.full(40, north),
        np.linspace(south, north, 40), np.linspace(south, north, 40),
    ])
    edge_x, edge_y = to_proj.transform(edge_lon, edge_lat)
    left = math.floor(float(np.min(edge_x)) / CELL) * CELL
    top = math.ceil(float(np.max(edge_y)) / CELL) * CELL
    width = int(math.ceil((float(np.max(edge_x)) - left) / CELL))
    height = int(math.ceil((top - float(np.min(edge_y))) / CELL))
    if width <= 0 or height <= 0 or width * height > 80_000_000:
        raise SystemExit(f"job grid is an unexpected size: {width}x{height}")
    transform = Affine(CELL, 0, left, 0, -CELL, top)

    cities = json.loads(CITIES_PATH.read_text(encoding="utf-8"))

    def project_geometry(geometry) -> dict:
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        projected = []
        for polygon in polygons:
            rings = []
            for ring in polygon:
                xs, ys = to_proj.transform([point[0] for point in ring], [point[1] for point in ring])
                rings.append([[float(x), float(y)] for x, y in zip(xs, ys)])
            projected.append(rings)
        return {"type": "MultiPolygon", "coordinates": projected}

    city_id = rasterize(
        [(project_geometry(feature["geometry"]), 1) for feature in cities["features"]],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    print(f"job grid {width}x{height}, city cells={int(city_id.sum()):,}", flush=True)

    raw_points: list[tuple[float, float, float]] = []
    counts = {"building": 0, "landuse": 0, "poi": 0, "spread_cells": 0}

    def add_weight(lon: float, lat: float, weight: float) -> None:
        if weight <= 0 or not (west <= lon <= east and south <= lat <= north):
            return
        raw_points.append((float(lon), float(lat), float(weight)))

    def cells_inside(
        xs: np.ndarray,
        ys: np.ndarray,
        holes: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[float, float]]:
        def row_col(x: float, y: float) -> tuple[float, float]:
            col = (x - transform.c) / transform.a - 0.5
            row = (y - transform.f) / transform.e - 0.5
            return row, col

        r_a, c_a = row_col(float(xs.min()), float(ys.min()))
        r_b, c_b = row_col(float(xs.max()), float(ys.max()))
        r0 = max(0, int(np.floor(min(r_a, r_b))))
        r1 = min(height - 1, int(np.floor(max(r_a, r_b))))
        c0 = max(0, int(np.floor(min(c_a, c_b))))
        c1 = min(width - 1, int(np.floor(max(c_a, c_b))))
        if r1 < r0 or c1 < c0:
            return []
        rows = np.arange(r0, r1 + 1)
        cols = np.arange(c0, c1 + 1)
        center_x = transform.c + transform.a * (cols + 0.5)
        center_y = transform.f + transform.e * (rows + 0.5)
        grid_x, grid_y = np.meshgrid(center_x, center_y)
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        inside = ring_mask(flat_x, flat_y, xs, ys)
        for hole_x, hole_y in holes:
            inside &= ~ring_mask(flat_x, flat_y, hole_x, hole_y)
        if not np.any(inside):
            return []
        chosen_x = flat_x[inside]
        chosen_y = flat_y[inside]
        lon, lat = to_wgs.transform(chosen_x, chosen_y)
        return list(zip(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)))

    def area_weight(obj, kind: str, source: str) -> None:
        floors, rate, coverage = SPEC[kind]
        if source == "building":
            floors = parse_floors(obj.tags, floors)
            coverage = 1.0
        parts: list[tuple[np.ndarray, np.ndarray, float, list[tuple[np.ndarray, np.ndarray]]]] = []
        for outer in obj.outer_rings():
            coords = [(node.lon, node.lat) for node in outer]
            if len(coords) < 4 or not intersects([c[0] for c in coords], [c[1] for c in coords], bounds):
                continue
            xs, ys = to_proj.transform([c[0] for c in coords], [c[1] for c in coords])
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            outer_area = ring_area(xs, ys)
            holes: list[tuple[np.ndarray, np.ndarray]] = []
            for inner in obj.inner_rings(outer):
                inner_coords = [(node.lon, node.lat) for node in inner]
                ixs, iys = to_proj.transform(
                    [c[0] for c in inner_coords], [c[1] for c in inner_coords]
                )
                ixs = np.asarray(ixs, dtype=np.float64)
                iys = np.asarray(iys, dtype=np.float64)
                outer_area -= ring_area(ixs, iys)
                holes.append((ixs, iys))
            if outer_area <= 0:
                continue
            parts.append((xs, ys, outer_area, holes))
        if not parts:
            return
        total = sum(part[2] for part in parts)
        weight = total * coverage * floors * rate
        centroid_x = sum(float(xs.mean()) * area for xs, _, area, _ in parts) / total
        centroid_y = sum(float(ys.mean()) * area for _, ys, area, _ in parts) / total
        if total < SMALL_FOOTPRINT_M2:
            lon, lat = to_wgs.transform(centroid_x, centroid_y)
            add_weight(lon, lat, weight)
        else:
            samples: list[tuple[float, float]] = []
            for xs, ys, _, holes in parts:
                samples.extend(cells_inside(xs, ys, holes))
            if not samples:
                lon, lat = to_wgs.transform(centroid_x, centroid_y)
                add_weight(lon, lat, weight)
            else:
                share = weight / len(samples)
                for lon, lat in samples:
                    add_weight(lon, lat, share)
                counts["spread_cells"] += len(samples)
        counts[source] += 1

    processor = (
        osmium.FileProcessor(str(PBF_PATH))
        .with_filter(osmium.filter.KeyFilter("building", "landuse", "amenity", "office", "shop"))
        .with_locations()
        .with_areas()
    )
    for obj in processor:
        if obj.is_node():
            kind = classify_node(obj.tags)
            if kind is None or not obj.location.valid():
                continue
            add_weight(obj.location.lon, obj.location.lat, POI_JOBS[kind])
            counts["poi"] += 1
            continue
        if not obj.is_area():
            continue
        classified = classify_area(obj.tags)
        if classified is None:
            continue
        try:
            area_weight(obj, *classified)
        except osmium.InvalidLocationError:
            continue

    if raw_points:
        located = np.asarray(raw_points, dtype=np.float64)
        xs, ys = to_proj.transform(located[:, 0], located[:, 1])
        cols = np.floor((xs - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((ys - transform.f) / transform.e).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        inside[inside] = city_id[rows[inside], cols[inside]] > 0
        raw_points = [tuple(point) for point in located[inside]]
        print(f"workplaces inside a city: {len(raw_points):,}", flush=True)

    raw = sum(weight for _, _, weight in raw_points)
    if raw <= 0:
        raise SystemExit("no workplaces found inside Guangdong")
    scale = labor_target / raw

    grouped: dict[tuple[float, float], float] = {}
    for lon, lat, weight in raw_points:
        key = (round(lon, 6), round(lat, 6))
        grouped[key] = grouped.get(key, 0.0) + weight * scale

    homes = []
    for point in demand["points"]:
        if not any(point[i] for i in (2, 3, 4, 5)):
            continue
        homes.append(
            [round(float(point[0]), 6), round(float(point[1]), 6), int(point[2]), int(point[3]), int(point[4]), int(point[5]), 0]
        )

    job_points = []
    for (lon, lat), jobs_value in grouped.items():
        jobs = int(round(jobs_value))
        if jobs < 1 or not (west <= lon <= east and south <= lat <= north):
            continue
        job_points.append([lon, lat, 0, 0, 0, 0, jobs])
    shown_jobs = sum(point[6] for point in job_points)
    if shown_jobs > 0:
        factor = labor_target / shown_jobs
        for point in job_points:
            point[6] = int(round(point[6] * factor))
        job_points = [point for point in job_points if point[6] >= 1]

    points = homes + job_points
    points.sort(key=lambda point: point[2] + point[6])
    job_total = sum(point[6] for point in points)
    demand["fields"] = ["lon", "lat", "residents", "labor", "youth", "old", "jobs"]
    demand["jobs"] = {
        "source": "OpenStreetMap workplaces at their real coordinates, scaled to the resident labor force",
        "placement": "small buildings and shops keep their coordinate; large footprints are spread across the 100 m cells they cover",
        "workplace_counts": counts,
        "total": job_total,
    }
    demand["points"] = points
    DEMAND_PATH.write_text(json.dumps(demand, separators=(",", ":")), encoding="utf-8")
    print(
        f"buildings={counts['building']} landuse={counts['landuse']} pois={counts['poi']}  "
        f"spread_cells={counts['spread_cells']}  workplaces={len(job_points)}  jobs={job_total:,}",
        flush=True,
    )
    from build_bubbles import main as build_bubbles
    build_bubbles()


if __name__ == "__main__":
    sys.exit(main())

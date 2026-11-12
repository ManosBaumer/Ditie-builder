"""Turn the Guangdong age grids into residential demand blobs.

Each blob of 50 or more people is one native 100 m cell, placed on that cell's
center. Cells with fewer people are summed across 500 m so a village is kept
without drawing a speck for every house. The 60-64 band is already inside the
labor-force grid, so it is not added again. Cells outside a Guangdong city are
dropped, because the clip itself is the rectangular envelope and that rectangle
also covers sea and neighboring provinces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "POPgridChina" / "guangdong"
CITIES_PATH = ROOT / "viewer" / "cities.json"
BOUNDS_PATH = ROOT / "bounds.geojson"
OUTPUT_PATH = ROOT / "viewer" / "demand.json"
BLOCK = 1  # native 100 m cell
MIN_PEOPLE = 50

LAYERS = {
    "labor": "towndasymetric劳动力.tif",
    "youth": "towndasymetric青少年.tif",
    "old": "towndasymetric老65.tif",
}


def read_layer(path: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float64)
        values[~np.isfinite(values)] = 0
        values[values < 0] = 0
        return values, src.transform, src.crs


def wgs84_bounds(path: Path) -> tuple[float, float, float, float]:
    """Envelope of every city polygon in the bounds file."""
    collection = json.loads(path.read_text(encoding="utf-8"))
    xs: list[float] = []
    ys: list[float] = []
    for feature in collection["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        for polygon in polygons:
            xs.extend(point[0] for point in polygon[0])
            ys.extend(point[1] for point in polygon[0])
    if not xs:
        raise SystemExit(f"no city bounds in {path}")
    return min(xs), min(ys), max(xs), max(ys)


def city_mask(transform: rasterio.Affine, shape: tuple[int, int], crs: object) -> np.ndarray:
    """Cells whose center falls inside a Guangdong city."""
    cities = json.loads(CITIES_PATH.read_text(encoding="utf-8"))
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    shapes = []
    for feature in cities["features"]:
        geometry = feature["geometry"]
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
        shapes.append(({"type": "MultiPolygon", "coordinates": projected}, 1))
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8")


def main() -> None:
    layers: dict[str, np.ndarray] = {}
    transform = None
    crs = None
    for key, name in LAYERS.items():
        values, layer_transform, layer_crs = read_layer(SOURCE_DIR / name)
        if transform is None:
            transform = layer_transform
            crs = layer_crs
            shape = values.shape
        elif values.shape != shape or layer_crs.to_proj4() != crs.to_proj4():
            raise SystemExit(f"{name} does not match the other age grids")
        elif abs(layer_transform.c - transform.c) > 1 or abs(layer_transform.f - transform.f) > 1:
            raise SystemExit(f"{name} is shifted relative to the other grids")
        layers[key] = values

    bounds = wgs84_bounds(BOUNDS_PATH)
    height, width = shape
    # BLOCK is 1, so each kept cell is emitted at its own center.
    inside = city_mask(transform, (height, width), crs) > 0
    total = layers["labor"] + layers["youth"] + layers["old"]
    rows, cols = np.nonzero(inside & (total >= MIN_PEOPLE))
    print(f"kept cells={len(rows):,} of {height}x{width}", flush=True)
    labor_v = np.rint(layers["labor"][rows, cols]).astype(np.int64)
    youth_v = np.rint(layers["youth"][rows, cols]).astype(np.int64)
    old_v = np.rint(layers["old"][rows, cols]).astype(np.int64)
    residents_v = labor_v + youth_v + old_v
    xs = transform.c + transform.a * (cols.astype(np.float64) + 0.5)
    ys = transform.f + transform.e * (rows.astype(np.float64) + 0.5)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    order = np.argsort(residents_v, kind="stable")
    payload_points = []
    for index in order:
        payload_points.append(
            [
                round(float(lon[index]), 5),
                round(float(lat[index]), 5),
                int(residents_v[index]),
                int(labor_v[index]),
                int(youth_v[index]),
                int(old_v[index]),
                0,
            ]
        )

    # The same 50-person rule on its own drops villages, where a 100 m cell
    # rarely reaches 50 people. Group those cells to 500 m and keep the group.
    small_rows, small_cols = np.nonzero(inside & (total > 0) & (total < MIN_PEOPLE))
    if len(small_rows):
        group = (small_rows // 5).astype(np.int64) * 1_000_000 + (small_cols // 5).astype(np.int64)
        _, inverse = np.unique(group, return_inverse=True)
        cell_total = total[small_rows, small_cols]
        sums = np.bincount(inverse, weights=cell_total)
        labor_s = np.rint(np.bincount(inverse, weights=layers["labor"][small_rows, small_cols])).astype(np.int64)
        youth_s = np.rint(np.bincount(inverse, weights=layers["youth"][small_rows, small_cols])).astype(np.int64)
        old_s = np.rint(np.bincount(inverse, weights=layers["old"][small_rows, small_cols])).astype(np.int64)
        xs_s = transform.c + transform.a * (small_cols.astype(np.float64) + 0.5)
        ys_s = transform.f + transform.e * (small_rows.astype(np.float64) + 0.5)
        xw = np.bincount(inverse, weights=xs_s * cell_total)
        yw = np.bincount(inverse, weights=ys_s * cell_total)
        labor_k = labor_s[sums >= MIN_PEOPLE]
        youth_k = youth_s[sums >= MIN_PEOPLE]
        old_k = old_s[sums >= MIN_PEOPLE]
        residents_k = labor_k + youth_k + old_k
        kept = residents_k >= MIN_PEOPLE
        labor_k, youth_k, old_k, residents_k = labor_k[kept], youth_k[kept], old_k[kept], residents_k[kept]
        group_x = (xw / sums)[sums >= MIN_PEOPLE][kept]
        group_y = (yw / sums)[sums >= MIN_PEOPLE][kept]
        group_lon, group_lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(group_x, group_y)
        payload_points.extend(
            [
                round(float(group_lon[index]), 5),
                round(float(group_lat[index]), 5),
                int(residents_k[index]),
                int(labor_k[index]),
                int(youth_k[index]),
                int(old_k[index]),
                0,
            ]
            for index in range(len(residents_k))
        )
        print(f"village groups={len(residents_k):,}", flush=True)
    payload_points.sort(key=lambda point: point[2])

    totals = {
        "residents": sum(p[2] for p in payload_points),
        "labor": sum(p[3] for p in payload_points),
        "youth": sum(p[4] for p in payload_points),
        "old": sum(p[5] for p in payload_points),
    }
    payload = {
        "bounds": [bounds[0], bounds[1], bounds[2], bounds[3]],
        "cell_m": BLOCK * 100,
        "min_people": MIN_PEOPLE,
        "fields": ["lon", "lat", "residents", "labor", "youth", "old", "jobs"],
        "totals": totals,
        "points": payload_points,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"{len(payload_points)} blobs  residents={totals['residents']:,}  "
        f"labor={totals['labor']:,}  youth={totals['youth']:,}  old={totals['old']:,}",
        flush=True,
    )
    print(OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    sys.exit(main())

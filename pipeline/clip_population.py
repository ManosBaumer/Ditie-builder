"""Clip the national population rasters to the Guangdong bounds.

Reads only the overlapping tiles, writes compressed float32 GeoTIFFs in the
source grid (Krasovsky 1940 Albers, 100 m), and does not resample.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
BOUNDS_PATH = ROOT / "bounds.geojson"
OUTPUT_DIR = ROOT / "POPgridChina" / "guangdong"


def guangzhou_bounds_wgs84(path: Path) -> tuple[float, float, float, float]:
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


def clip_raster(src_path: Path, dest_path: Path, wgs84_bounds: tuple[float, float, float, float]) -> None:
    west, south, east, north = wgs84_bounds
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise SystemExit(f"{src_path.name} has no CRS")
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, west, south, east, north, densify_pts=21
        )
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            raise SystemExit(f"{src_path.name} does not overlap the Guangdong bounds")

        raw = src.read(1, window=window, out_dtype="float64")
        # National grids mark the outside with -inf or the float nodata sentinel,
        # not with NaN. Tiny negatives are subtraction noise in the 60-64 band.
        nodata = ~np.isfinite(raw) | (raw < -1.0e20)
        cleaned = np.where(raw < 0, 0, raw)
        data = np.full(raw.shape, np.nan, dtype=np.float32)
        data[~nodata] = cleaned[~nodata].astype(np.float32)
        finite = ~nodata
        total = float(data[finite].sum()) if finite.any() else 0.0
        profile = {
            "driver": "GTiff",
            "height": int(window.height),
            "width": int(window.width),
            "count": 1,
            "dtype": "float32",
            "transform": src.window_transform(window),
            "crs": src.crs,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_NEEDED",
        }

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dest_path, "w", **profile) as dst:
            dst.write(data, 1)

    print(
        f"{src_path.name} -> {dest_path.name}  "
        f"{int(window.width)}x{int(window.height)}  "
        f"finite={int(finite.sum())}  sum={total:,.0f}",
        flush=True,
    )


def national_sources() -> list[Path]:
    """The China-wide age grids, not the smaller clips already on disk."""
    found = [
        path for path in (ROOT / "GDPOP").rglob("*.tif")
        if "POPgridChina" not in path.parts
    ]
    if not found:
        found = [path for path in (ROOT / "POPgridChina").glob("*.tif")]
    if not found:
        raise SystemExit("no national tifs in GDPOP or POPgridChina")
    return sorted(found, key=lambda path: path.name)


def main() -> None:
    bounds = guangzhou_bounds_wgs84(BOUNDS_PATH)
    print(
        "WGS84 box "
        f"west={bounds[0]:.6f} south={bounds[1]:.6f} "
        f"east={bounds[2]:.6f} north={bounds[3]:.6f}",
        flush=True,
    )
    for src_path in national_sources():
        clip_raster(src_path, OUTPUT_DIR / src_path.name, bounds)


if __name__ == "__main__":
    sys.exit(main())

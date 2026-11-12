"""Turn the GBA-UBF Guangzhou sample shapefiles into colored building footprints.

The published repo includes one study patch, split into five function classes.
Each footprint has a height, which the map extrudes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "gba-ubf-demo"
OUTPUT_PATH = ROOT / "viewer" / "buildings.geojson"

FOLDERS = {
    "commercial": "commercial",
    "education": "education",
    "industry": "industry",
    "public": "public",
    "residential": "residential",
}


def ring_area_m2(ring: list[list[float]]) -> float:
    if len(ring) < 4:
        return 0.0
    lat = math.radians(ring[0][1])
    mx = 111_320 * math.cos(lat)
    my = 110_540
    twice = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0] * mx, ring[i][1] * my
        x2, y2 = ring[i + 1][0] * mx, ring[i + 1][1] * my
        twice += x1 * y2 - x2 * y1
    return abs(twice) * 0.5


def rings_of(shape) -> list[list[list[float]]]:
    points = shape.points
    starts = list(shape.parts) + [len(points)]
    rings = []
    for start, end in zip(starts, starts[1:]):
        ring = [[round(x, 6), round(y, 6)] for x, y in points[start:end]]
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        rings.append(ring)
    return rings


def main() -> None:
    features = []
    counts = {sector: 0 for sector in FOLDERS}
    malls = 0
    west = south = 180.0
    east = north = -180.0
    for sector, folder in FOLDERS.items():
        path = next((SOURCE_DIR / folder).glob("*.shp"))
        reader = shapefile.Reader(str(path), encoding="gbk")
        fields = [field[0] for field in reader.fields[1:]]
        height_i = fields.index("Height")
        mall_i = fields.index("商场")
        id_i = fields.index("id")
        for record, shape in zip(reader.iterRecords(), reader.iterShapes()):
            rings = rings_of(shape)
            if not rings:
                continue
            height = float(record[height_i])
            if not math.isfinite(height) or height < 0:
                height = 0.0
            mall = 1 if int(record[mall_i] or 0) else 0
            malls += mall
            area = sum(ring_area_m2(ring) for ring in rings)
            for ring in rings:
                for lon, lat in ring:
                    west, east = min(west, lon), max(east, lon)
                    south, north = min(south, lat), max(north, lat)
            geometry = (
                {"type": "Polygon", "coordinates": rings}
                if len(rings) == 1
                else {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}
            )
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "sector": sector,
                        "height": round(height, 1),
                        "mall": mall,
                        "area": int(round(area)),
                        "id": int(record[id_i]),
                    },
                    "geometry": geometry,
                }
            )
            counts[sector] += 1

    payload = {
        "type": "FeatureCollection",
        "source": "GBA-UBF Guangzhou sample (chenchs0629/GBA-UBF demo)",
        "counts": counts,
        "malls": malls,
        "bounds": [west, south, east, north],
        "features": features,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"{len(features)} footprints  malls={malls}  "
        + "  ".join(f"{key}={value}" for key, value in counts.items()),
        flush=True,
    )
    print(f"{OUTPUT_PATH.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""Union the city outlines into one province shape for the map mask.

Separate city holes share borders, and the map's fill then paints over the
Pearl River Delta. One outline does not have that seam.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parents[1]
CITIES_PATH = ROOT / "viewer" / "cities.json"
OUTPUT_PATH = ROOT / "viewer" / "province.json"


def main() -> None:
    cities = json.loads(CITIES_PATH.read_text(encoding="utf-8"))
    parts = [make_valid(shape(feature["geometry"])) for feature in cities["features"]]
    province = unary_union(parts)
    # About 40 m. Keeps Shekou and drops the points that make panning stutter.
    province = province.simplify(0.0004, preserve_topology=True)
    if province.geom_type == "Polygon":
        geometry = {"type": "MultiPolygon", "coordinates": [mapping(province)["coordinates"]]}
    else:
        geometry = mapping(province)
    payload = {
        "type": "Feature",
        "properties": {"name": "Guangdong"},
        "geometry": geometry,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rings = geometry["coordinates"]
    vertices = sum(len(ring) for polygon in rings for ring in polygon)
    print(f"parts={len(rings)} vertices={vertices} -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

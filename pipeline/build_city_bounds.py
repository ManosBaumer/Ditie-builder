"""Replace the coarse city outlines with OpenStreetMap boundaries.

The previous polygons were a simplified extract: Shenzhen had a few hundred
corners, so the mask sliced off real shoreline. The Guangdong OpenStreetMap
file already contains the prefecture relations. Exterior rings are stored
clockwise, which is what the map mask expects when it uses them as holes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = ROOT / "data" / "raw" / "guangdong-latest.osm.pbf"
# Relation id, English name used in the viewer, Chinese name.
CITIES = [
    (3244431, "Chaozhou", "潮州市"),
    (3464319, "Dongguan", "东莞市"),
    (3464719, "Foshan", "佛山市"),
    (3287346, "Guangzhou", "广州市"),
    (3241970, "Heyuan", "河源市"),
    (3209912, "Huizhou", "惠州市"),
    (3463901, "Jiangmen", "江门市"),
    (3282749, "Jieyang", "揭阳市"),
    (2689018, "Maoming", "茂名市"),
    (3233743, "Meizhou", "梅州市"),
    (3124223, "Qingyuan", "清远市"),
    (3282750, "Shantou", "汕头市"),
    (3282610, "Shanwei", "汕尾市"),
    (3203994, "Shaoguan", "韶关市"),
    (3464353, "Shenzhen", "深圳市"),
    (3032037, "Yangjiang", "阳江市"),
    (3032080, "Yunfu", "云浮市"),
    (3484419, "Zhanjiang", "湛江市"),
    (3205802, "Zhaoqing", "肇庆市"),
    (3464878, "Zhongshan", "中山市"),
    (3464829, "Zhuhai", "珠海市"),
]
OUTPUTS = [
    ROOT / "viewer" / "cities.json",
    ROOT / "bounds.geojson",
    ROOT / "guangdong.geojson",
]


def signed_area(ring: list[list[float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        area += x1 * y2 - x2 * y1
    return area / 2


def clean_ring(ring: list[list[float]], want_clockwise: bool) -> list[list[float]]:
    points = [[round(lon, 6), round(lat, 6)] for lon, lat in ring]
    deduped = [points[0]]
    for point in points[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    if len(deduped) < 4:
        return []
    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])
    clockwise = signed_area(deduped) < 0
    if clockwise != want_clockwise:
        deduped.reverse()
    return deduped


def vertex_count(geometry: dict) -> int:
    polygons = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [polygons]
    return sum(len(ring) for polygon in polygons for ring in polygon)


class BoundaryHandler(osmium.SimpleHandler):
    def __init__(self, wanted: dict[int, tuple[str, str]]):
        super().__init__()
        self.wanted = wanted
        self.factory = osmium.geom.GeoJSONFactory()
        self.found: dict[int, dict] = {}

    def area(self, area: osmium.osm.Area) -> None:
        if area.from_way():
            return
        relation_id = area.orig_id()
        if relation_id not in self.wanted or relation_id in self.found:
            return
        try:
            geometry = json.loads(self.factory.create_multipolygon(area))
        except RuntimeError:
            return
        name, zh = self.wanted[relation_id]
        polygons = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        cleaned = []
        for polygon in polygons:
            rings = []
            for index, ring in enumerate(polygon):
                # Exterior clockwise so the viewer can use it as a mask hole.
                cleaned_ring = clean_ring(ring, want_clockwise=(index == 0))
                if len(cleaned_ring) >= 4:
                    rings.append(cleaned_ring)
            if rings:
                cleaned.append(rings)
        if not cleaned:
            return
        self.found[relation_id] = {
            "type": "Feature",
            "properties": {"name": name, "zh": zh, "osm_relation": relation_id},
            "geometry": {"type": "MultiPolygon", "coordinates": cleaned},
        }


def contains(geometry: dict, lon: float, lat: float) -> bool:
    polygons = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [polygons]

    def in_ring(ring: list[list[float]]) -> bool:
        inside = False
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            if (y1 > lat) != (y2 > lat):
                cross = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
                if lon < cross:
                    inside = not inside
        return inside

    for polygon in polygons:
        if not polygon or not in_ring(polygon[0]):
            continue
        if any(in_ring(hole) for hole in polygon[1:]):
            continue
        return True
    return False


def feature_named(collection: dict, name: str) -> dict:
    for feature in collection["features"]:
        if feature["properties"]["name"] == name:
            return feature
    raise SystemExit(f"missing {name}")


def main() -> None:
    if not PBF_PATH.exists():
        raise SystemExit(f"missing OpenStreetMap extract: {PBF_PATH}")
    wanted = {relation_id: (name, zh) for relation_id, name, zh in CITIES}
    handler = BoundaryHandler(wanted)
    handler.apply_file(str(PBF_PATH), locations=True, idx="flex_mem")
    missing = [name for relation_id, name, _zh in CITIES if relation_id not in handler.found]
    if missing:
        raise SystemExit(f"missing city boundaries: {', '.join(missing)}")

    features = [handler.found[relation_id] for relation_id, _name, _zh in CITIES]
    collection = {"type": "FeatureCollection", "features": features}
    for feature in features:
        print(
            f"{feature['properties']['name']:12} parts={len(feature['geometry']['coordinates']):4} "
            f"vertices={vertex_count(feature['geometry']):7}",
            flush=True,
        )
    old = json.loads((ROOT / "viewer" / "cities.json").read_text(encoding="utf-8"))
    checks = [
        ("Shenzhen", "Shekou", 113.912, 22.479, True),
        ("Shenzhen", "Chiwan tip", 113.886, 22.468, True),
        ("Shenzhen", "Qianhai", 113.887, 22.522, True),
        ("Shenzhen", "Sheung Shui", 114.129, 22.502, False),
        ("Shenzhen", "Hong Kong Island", 114.174, 22.278, False),
        ("Zhuhai", "Macau", 113.543, 22.198, False),
        ("Guangzhou", "city center", 113.264, 23.129, True),
    ]
    failed = False
    for city, label, lon, lat, expect in checks:
        hit = contains(feature_named(collection, city)["geometry"], lon, lat)
        old_hit = contains(feature_named(old, city)["geometry"], lon, lat)
        print(f"{city} {label}: osm={hit} old={old_hit} expect={expect}", flush=True)
        if hit != expect:
            failed = True
    shenzhen = feature_named(collection, "Shenzhen")["geometry"]
    old_shenzhen = feature_named(old, "Shenzhen")["geometry"]
    print(
        f"Shenzhen vertices osm={vertex_count(shenzhen):,} old={vertex_count(old_shenzhen):,}",
        flush=True,
    )
    if failed or vertex_count(shenzhen) < vertex_count(old_shenzhen) * 2:
        raise SystemExit("OpenStreetMap boundaries failed the coastline check")

    payload = json.dumps(collection, ensure_ascii=False, separators=(",", ":"))
    for path in OUTPUTS:
        path.write_text(payload, encoding="utf-8")
        print(path, flush=True)


if __name__ == "__main__":
    sys.exit(main())

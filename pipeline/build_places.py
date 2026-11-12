"""Build the destination places layer inside the Guangzhou box.

A place is somewhere people travel to for a reason other than their shift:
catching a train, flying, class, a hospital visit, shopping, a day out.

Visit numbers come only from published figures in place_counts.csv, each with
its source. Yearly figures are spread evenly over 365 days. Everything else
OpenStreetMap knows about is kept as an unrated place: it is on the map, but
it carries no visit number because nobody publishes one.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parents[1]
PBF_PATH = ROOT / "data" / "raw" / "guangdong-latest.osm.pbf"
COUNTS_PATH = ROOT / "pipeline" / "place_counts.csv"
DEMAND_PATH = ROOT / "viewer" / "demand.json"
PLACES_PATH = ROOT / "viewer" / "places.json"

LAT_M = 110_746.0
LON_M = 111_320.0 * math.cos(math.radians(23.15))

# Visits per day at or above which a place counts as a city or district destination.
CITY_VISITS = 15_000
DISTRICT_VISITS = 3_000
# Handpicked counts below this stay off the map. Unrated OpenStreetMap places are unaffected.
MIN_DAILY = 20_000
# A published name match only absorbs mapped places this close.
ABSORB_NAME_M = 5_000

KINDS = {
    "airport": {"label": "Airport", "peak": "all day"},
    "rail": {"label": "Railway station", "peak": "all day"},
    "university": {"label": "University", "peak": "morning and evening"},
    "school": {"label": "School", "peak": "morning and mid-afternoon"},
    "hospital": {"label": "Hospital", "peak": "morning"},
    "mall": {"label": "Mall", "peak": "evening and weekend"},
    "shopping": {"label": "Shopping street", "peak": "evening and weekend"},
    "theme_park": {"label": "Theme park", "peak": "weekend"},
    "zoo": {"label": "Zoo or aquarium", "peak": "weekend"},
    "museum": {"label": "Museum or library", "peak": "weekend"},
    "attraction": {"label": "Landmark", "peak": "weekend and evening"},
    "park": {"label": "Park", "peak": "weekend and early morning"},
    "stadium": {"label": "Stadium", "peak": "event evenings"},
    "exhibition": {"label": "Exhibition centre", "peak": "fair days"},
    "border": {"label": "Border crossing", "peak": "all day"},
}
KIND_ORDER = list(KINDS)
TIERS = ["city", "district", "local", "unrated"]
# Kinds a published place may absorb by name; a hospital called 白云山医院 is not part of Baiyun Mountain.
NAME_ABSORBS = {"park", "attraction", "theme_park", "zoo", "museum", "shopping", "mall", "exhibition", "university"}


def tag(tags, key: str) -> str:
    value = tags.get(key)
    return "" if value is None else str(value)


def ring_metrics(coords: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Area in m2 and area centroid of a closed lon/lat ring."""
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


def point_in_ring(lon: float, lat: float, coords: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(coords) - 1
    for i in range(len(coords)):
        xi, yi = coords[i]
        xj, yj = coords[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def area_shape(obj) -> tuple[float, float, float] | None:
    """Total area and a point inside the largest ring."""
    best = None
    total = 0.0
    for outer in obj.outer_rings():
        coords = [(node.lon, node.lat) for node in outer]
        if len(coords) < 4:
            continue
        area, lon, lat = ring_metrics(coords)
        total += area
        if best is None or area > best[0]:
            best = (area, lon, lat, coords)
    if best is None or total <= 0:
        return None
    _, lon, lat, coords = best
    if not point_in_ring(lon, lat, coords):
        lon, lat = min(coords, key=lambda c: (c[0] - lon) ** 2 + (c[1] - lat) ** 2)
    return total, lon, lat


DIRECTIONS = {"南": ("nan", "South"), "北": ("bei", "North"), "东": ("dong", "East"), "西": ("xi", "West")}


def english_name(tags) -> str:
    en = tag(tags, "name:en")
    zh = tag(tags, "name:zh") or tag(tags, "name")
    if not en:
        return ""
    if zh and zh[-1] in DIRECTIONS:
        pinyin, word = DIRECTIONS[zh[-1]]
        if en.lower().endswith(pinyin) and len(en) > len(pinyin):
            en = en[: -len(pinyin)] + " " + word
    return en


def classify(tags, area: float | None) -> str | None:
    """Kind of destination from OpenStreetMap tags, or None if it is not one worth showing."""
    amenity = tag(tags, "amenity")
    tourism = tag(tags, "tourism")
    leisure = tag(tags, "leisure")
    shop = tag(tags, "shop")
    named = bool(tag(tags, "name"))
    famous = bool(tag(tags, "wikidata"))

    if amenity in ("university", "college"):
        return "university"
    if amenity in ("school", "kindergarten"):
        return "school"
    if amenity == "hospital":
        return "hospital"
    if amenity == "library" and named and (famous or (area or 0) >= 5_000):
        return "museum"
    if amenity == "exhibition_centre" and area:
        return "exhibition"
    if shop in ("mall", "department_store") and named:
        return "mall"
    if tourism == "theme_park":
        return "theme_park"
    if tourism in ("zoo", "aquarium"):
        return "zoo"
    if tourism == "museum":
        return "museum"
    if tourism == "attraction" and named and (famous or (area or 0) >= 20_000):
        return "attraction"
    if leisure in ("park", "nature_reserve") and named and (area or 0) >= 10_000:
        return "park"
    if leisure == "stadium" and named and (area or 0) >= 10_000:
        return "stadium"
    if leisure == "sports_centre" and famous and area:
        return "stadium"
    return None


def load_counts() -> list[dict]:
    rows = []
    with COUNTS_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] not in KINDS:
                raise SystemExit(f"unknown kind {row['kind']!r} for {row['name']}")
            figure = float(row["figure"])
            daily = figure / 365 if row["per"] == "year" else figure
            daily *= float(row["factor"] or 1)
            kept = (row.get("keep") or "").strip().lower() == "yes"
            if daily < MIN_DAILY and not kept:
                print(f"skip {row['name']}: {daily:,.0f} visits/day", flush=True)
                continue
            rows.append({
                "kind": row["kind"],
                "name": row["name"],
                "zh": row["zh"],
                "lon": float(row["lon"]),
                "lat": float(row["lat"]),
                "visits": daily,
                "basis": row["basis"],
                "source": row["source"],
                "url": row["url"],
                "absorb_m": float(row["absorb_m"] or 0),
                "absorb_zh": re.compile(row["absorb_zh"]) if row["absorb_zh"] else None,
            })
    return rows


def metres(a: dict, b: dict) -> float:
    return math.hypot((a["lon"] - b["lon"]) * LON_M, (a["lat"] - b["lat"]) * LAT_M)


def main() -> None:
    if not PBF_PATH.exists() or PBF_PATH.stat().st_size < 1_000_000:
        raise SystemExit(f"missing OpenStreetMap extract: {PBF_PATH}")
    demand = json.loads(DEMAND_PATH.read_text(encoding="utf-8"))
    west, south, east, north = demand["bounds"]

    def inside(lon: float, lat: float) -> bool:
        return west <= lon <= east and south <= lat <= north

    counted = [row for row in load_counts() if inside(row["lon"], row["lat"])]
    mapped: list[dict] = []

    def add(kind: str, lon: float, lat: float, tags, name: str = "", area: float = 0.0) -> None:
        zh = tag(tags, "name")
        en = name or english_name(tags) or zh
        if en:
            mapped.append({"kind": kind, "lon": lon, "lat": lat, "name": en, "zh": zh if zh != en else "", "area": area})

    processor = (
        osmium.FileProcessor(str(PBF_PATH))
        .with_filter(osmium.filter.KeyFilter("aeroway", "railway", "amenity", "shop", "tourism", "leisure"))
        .with_locations()
        .with_areas()
    )
    for obj in processor:
        tags = obj.tags
        if obj.is_node():
            if not obj.location.valid():
                continue
            lon, lat, area = obj.location.lon, obj.location.lat, None
        elif obj.is_area():
            try:
                shape = area_shape(obj)
            except osmium.InvalidLocationError:
                continue
            if shape is None:
                continue
            area, lon, lat = shape
        else:
            continue
        if not inside(lon, lat):
            continue

        if tag(tags, "aeroway") == "aerodrome" and tag(tags, "iata"):
            add("airport", lon, lat, tags, area=area or 0.0)
            continue
        if tag(tags, "railway") == "station":
            station = tag(tags, "station")
            if station in ("subway", "light_rail", "tram", "monorail") or tag(tags, "subway") == "yes":
                continue
            if tag(tags, "train") != "yes" and station != "train":
                continue
            zh = tag(tags, "name")
            add("rail", lon, lat, tags, (english_name(tags) or zh) + " Station")
            continue
        kind = classify(tags, area)
        if kind:
            add(kind, lon, lat, tags, area=area or 0.0)

    def absorbed(place: dict) -> bool:
        for row in counted:
            distance = metres(place, row)
            if place["kind"] == row["kind"] and distance <= row["absorb_m"]:
                return True
            if (row["absorb_zh"] and place["kind"] in NAME_ABSORBS and distance <= ABSORB_NAME_M
                    and row["absorb_zh"].search(place["zh"] or place["name"])):
                return True
        return False

    # The same place is often mapped twice (a station node and its building,
    # a park and a node inside it). Keep the largest.
    mapped.sort(key=lambda p: -p["area"])
    unrated: list[dict] = []
    seen: dict[tuple[str, str], list[dict]] = {}
    for place in mapped:
        if absorbed(place):
            continue
        twins = seen.setdefault((place["kind"], place["name"].lower()), [])
        if any(metres(t, place) < 1_500 for t in twins):
            continue
        twins.append(place)
        unrated.append(place)

    rows = []
    counts = {kind: 0 for kind in KIND_ORDER}
    tier_counts = {tier: 0 for tier in TIERS}
    for place in counted:
        visits = int(round(place["visits"] / 100.0) * 100) or 100
        tier = "city" if visits >= CITY_VISITS else "district" if visits >= DISTRICT_VISITS else "local"
        counts[place["kind"]] += 1
        tier_counts[tier] += 1
        rows.append([
            round(place["lon"], 6), round(place["lat"], 6), visits,
            KIND_ORDER.index(place["kind"]), TIERS.index(tier),
            place["name"], place["zh"], place["basis"], place["source"], place["url"],
        ])
    for place in unrated:
        counts[place["kind"]] += 1
        tier_counts["unrated"] += 1
        rows.append([
            round(place["lon"], 6), round(place["lat"], 6), 0,
            KIND_ORDER.index(place["kind"]), TIERS.index("unrated"),
            place["name"], place["zh"], "", "", "",
        ])
    rows.sort(key=lambda row: row[2])

    out = {
        "bounds": demand["bounds"],
        "source": "Published visitor and passenger figures (place_counts.csv); other destinations from OpenStreetMap without a count",
        "fields": ["lon", "lat", "visits", "kind", "tier", "name", "zh", "basis", "source", "url"],
        "kinds": [{"id": kind, **KINDS[kind]} for kind in KIND_ORDER],
        "tiers": TIERS,
        "tier_visits": {"city": CITY_VISITS, "district": DISTRICT_VISITS},
        "counts": counts,
        "tier_counts": tier_counts,
        "places": rows,
    }
    PLACES_PATH.write_text(json.dumps(out, ensure_ascii=False, separators=(",\n", ":")), encoding="utf-8")
    print(f"places={len(rows)} counted={len(counted)} tiers={tier_counts}", flush=True)
    print("kinds=" + ", ".join(f"{k}:{v}" for k, v in counts.items()), flush=True)
    for row in sorted((r for r in rows if r[2]), key=lambda r: -r[2]):
        print(f"  {row[2]:>8,}  {KIND_ORDER[row[3]]:<11} {TIERS[row[4]]:<8} {row[5]}  {row[0]:.4f},{row[1]:.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

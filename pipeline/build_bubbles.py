"""Bake the bubble groups so the map does not freeze on load.

The viewer used to walk every living cell and workplace in the browser.
That is the same grouping as before: crowded cells fold together, then
nearby leftovers fold in, and the circle sits on their weighted center.
The province view gets a second, wider pass. Zooming in swaps to the
finer bubbles, which are loaded after the first paint.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMAND_PATH = ROOT / "viewer" / "demand.json"
PLACES_PATH = ROOT / "viewer" / "places.json"
OVERVIEW_PATH = ROOT / "viewer" / "bubbles.json"
DETAIL_PATH = ROOT / "viewer" / "bubbles-detail.json"

LON_M = 111320 * math.cos(math.radians(23.15))
LAT_M = 110746
LON0 = 109.0
LAT0 = 20.0


def bins_for(x: np.ndarray, y: np.ndarray, bin_m: float):
    ix = np.floor(x / bin_m).astype(np.int32)
    iy = np.floor(y / bin_m).astype(np.int32)
    key = (ix.astype(np.int64) << 20) + iy.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    ordered = key[order]
    change = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(order)]))
    lookup = {int(ordered[start]): (int(start), int(end)) for start, end in zip(starts, ends)}
    return ix, iy, order, lookup


def within(i: int, x, y, ix, iy, order, lookup, radius_m: float, bin_m: float) -> list[int]:
    reach = int(radius_m // bin_m) + 1
    limit = radius_m * radius_m
    cx = x[i]
    cy = y[i]
    bx = int(ix[i])
    by = int(iy[i])
    found = []
    for row in range(by - reach, by + reach + 1):
        for col in range(bx - reach, bx + reach + 1):
            span = lookup.get((col << 20) + row)
            if span is None:
                continue
            start, end = span
            for pos in range(start, end):
                other = int(order[pos])
                dx = x[other] - cx
                dy = y[other] - cy
                if dx * dx + dy * dy <= limit:
                    found.append(other)
    return found


def pack(points: np.ndarray, indexes: list[int], weight_index: int) -> list[float]:
    chosen = points[np.asarray(indexes, dtype=np.int64)]
    weight = chosen[:, weight_index]
    weight = np.where(weight > 0, weight, 1.0)
    total = float(weight.sum())
    summed = chosen[:, 2:].sum(axis=0)
    return [
        round(float(np.dot(chosen[:, 0], weight) / total), 5),
        round(float(np.dot(chosen[:, 1], weight) / total), 5),
        *[int(round(float(value))) for value in summed],
    ]


def cluster(points: np.ndarray, weight_index: int, scale: float) -> list[list[float]]:
    if len(points) == 0:
        return []
    x = (points[:, 0] - LON0) * LON_M
    y = (points[:, 1] - LAT0) * LAT_M
    bin_m = 160.0 * scale
    ix, iy, order, lookup = bins_for(x, y, bin_m)

    # One distance check per occupied cell, not per person. Cells whose
    # whole neighborhood holds fewer than six points cannot be crowded.
    reach = int((180.0 * scale) // bin_m) + 1
    limit = (180.0 * scale) * (180.0 * scale)
    crowded = np.zeros(len(points), dtype=bool)
    for key, (start, end) in lookup.items():
        col = key >> 20
        row = key - (col << 20)
        chunks = []
        for drow in range(-reach, reach + 1):
            for dcol in range(-reach, reach + 1):
                span = lookup.get(((col + dcol) << 20) + (row + drow))
                if span is not None:
                    chunks.append(order[span[0] : span[1]])
        if not chunks:
            continue
        neigh = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if len(neigh) < 6:
            continue
        center = order[start:end]
        dx = x[center, None] - x[neigh]
        dy = y[center, None] - y[neigh]
        crowded[center] = np.count_nonzero((dx * dx) + (dy * dy) <= limit, axis=1) >= 6

    weight = points[:, weight_index]
    rank = np.argsort(-np.where(weight > 0, weight, 1.0), kind="mergesort")
    used = np.zeros(len(points), dtype=bool)
    grouped_rows = []
    for index in rank:
        index = int(index)
        if not crowded[index] or used[index]:
            continue
        group = [
            other
            for other in within(index, x, y, ix, iy, order, lookup, 210.0 * scale, bin_m)
            if crowded[other] and not used[other]
        ]
        if len(group) < 4:
            continue
        for other in group:
            used[other] = True
        grouped_rows.append(pack(points, group, weight_index))

    singles = points[~used]
    grouped = np.asarray(grouped_rows, dtype=np.float64) if grouped_rows else np.zeros((0, points.shape[1]))
    bubbles = combine(singles, 260.0 * scale, 2, weight_index) + combine(grouped, 440.0 * scale, 3, weight_index)
    bubbles.sort(key=lambda row: row[weight_index])
    return bubbles


def combine(points: np.ndarray, radius_m: float, min_count: int, weight_index: int) -> list[list[float]]:
    if len(points) == 0:
        return []
    x = (points[:, 0] - LON0) * LON_M
    y = (points[:, 1] - LAT0) * LAT_M
    bin_m = 180.0
    ix, iy, order, lookup = bins_for(x, y, bin_m)
    weight = points[:, weight_index]
    rank = np.argsort(-np.where(weight > 0, weight, 1.0), kind="mergesort")
    used = np.zeros(len(points), dtype=bool)
    out = []
    for index in rank:
        index = int(index)
        if used[index]:
            continue
        group = [other for other in within(index, x, y, ix, iy, order, lookup, radius_m, bin_m) if not used[other]]
        if len(group) < min_count:
            continue
        for other in group:
            used[other] = True
        out.append(pack(points, group, weight_index))
    for index in range(len(points)):
        if not used[index]:
            row = points[index]
            out.append([round(float(row[0]), 5), round(float(row[1]), 5), *[int(round(float(value))) for value in row[2:]]])
    return out


def as_rows(raw_points: list, width: int) -> np.ndarray:
    rows = np.zeros((len(raw_points), width), dtype=np.float64)
    for index, point in enumerate(raw_points):
        rows[index, : len(point)] = point
    return rows


def split_demand(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homes = points[np.any(points[:, 2:6] > 0, axis=1)]
    jobs = points[(points[:, 6] > 0) & ~np.any(points[:, 2:6] > 0, axis=1)]
    return homes, jobs


def cluster_demand(points: np.ndarray, scale: float) -> list[list[float]]:
    homes, jobs = split_demand(points)
    return cluster(homes, 2, scale) + cluster(jobs, 6, scale)


def cluster_place_blobs(points: np.ndarray) -> list[list[float]]:
    """Soft general-draw circles from unrated places.

    The living/job packer looks for crowded homes and workplaces. Unrated
    places are thinner on the map, so that pass either skips a busy street
    or parks the circle hundreds of metres away. Bin the points at about a
    city-block size, grow each peak from the densest remaining bin, and put
    the circle on the places it ate so commercial corridors stay on the
    corridor.
    """
    if len(points) == 0:
        return []
    x = (points[:, 0] - LON0) * LON_M
    y = (points[:, 1] - LAT0) * LAT_M
    bin_m = 180.0
    peak_radius = 380.0
    min_peak = 4
    leftover_radius = 220.0
    ix = np.floor(x / bin_m).astype(np.int32)
    iy = np.floor(y / bin_m).astype(np.int32)
    bins: dict[tuple[int, int], list[int]] = {}
    for index in range(len(points)):
        bins.setdefault((int(ix[index]), int(iy[index])), []).append(index)

    weight = np.where(points[:, 7] > 0, points[:, 7], 1.0)
    scores: dict[tuple[int, int], tuple[float, float]] = {}
    for key, members in bins.items():
        bx, by = key
        score = 0.0
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                neigh = bins.get((bx + dcol, by + drow), ())
                if neigh:
                    score += float(weight[neigh].sum())
        scores[key] = (score, float(weight[members].sum()))

    used = np.zeros(len(points), dtype=bool)
    out: list[list[float]] = []
    ranked = sorted(scores, key=lambda key: (-scores[key][0], -scores[key][1], key[0], key[1]))
    reach = int(peak_radius // bin_m) + 1
    for key in ranked:
        seed = [index for index in bins[key] if not used[index]]
        if not seed:
            continue
        seed_w = weight[seed]
        seed_total = float(seed_w.sum())
        cx = float(np.dot(x[seed], seed_w) / seed_total)
        cy = float(np.dot(y[seed], seed_w) / seed_total)
        group: list[int] = []
        bx, by = key
        limit = peak_radius * peak_radius
        for drow in range(-reach, reach + 1):
            for dcol in range(-reach, reach + 1):
                for index in bins.get((bx + dcol, by + drow), ()):
                    if used[index]:
                        continue
                    dx = x[index] - cx
                    dy = y[index] - cy
                    if dx * dx + dy * dy <= limit:
                        group.append(index)
        if len(group) < min_peak:
            continue
        for index in group:
            used[index] = True
        out.append(pack(points, group, 7))

    leftovers = np.flatnonzero(~used)
    if len(leftovers):
        leftover_points = points[leftovers]
        for row in combine(leftover_points, leftover_radius, 2, 7):
            if row[7] >= 2:
                out.append(row)

    out.sort(key=lambda row: row[7])
    return out


def main() -> None:
    started = time.perf_counter()
    demand = json.loads(DEMAND_PATH.read_text(encoding="utf-8"))
    raw = as_rows(demand["points"], 8)
    print(f"raw points={len(raw):,}", flush=True)
    detail = cluster_demand(raw, 1)
    print(f"detail bubbles={len(detail):,}  {time.perf_counter() - started:.1f}s", flush=True)

    detail_rows = as_rows(detail, 8)
    overview = cluster_demand(detail_rows, 12)
    print(f"overview bubbles={len(overview):,}  {time.perf_counter() - started:.1f}s", flush=True)
    if len(overview) > 12000:
        overview = cluster_demand(detail_rows, 20)
        print(f"overview widened={len(overview):,}", flush=True)

    if PLACES_PATH.exists():
        places = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
        kind_ids = [row["id"] for row in places["kinds"]]
        # Malls and landmarks pull foot traffic harder than schools, so they
        # win density peaks on mixed streets like Beijing Road.
        kind_weight = {
            "mall": 3,
            "shopping": 3,
            "attraction": 3,
            "exhibition": 3,
            "theme_park": 3,
            "museum": 2,
            "park": 2,
            "stadium": 2,
            "university": 2,
            "zoo": 2,
        }
        unrated = [point for point in places["places"] if point[4] == 3]
        place_rows = np.zeros((len(unrated), 8), dtype=np.float64)
        for index, point in enumerate(unrated):
            place_rows[index, 0] = point[0]
            place_rows[index, 1] = point[1]
            kind = kind_ids[point[3]] if 0 <= point[3] < len(kind_ids) else ""
            place_rows[index, 7] = kind_weight.get(kind, 1)
        place_detail = cluster_place_blobs(place_rows)
        print(f"place detail blobs={len(place_detail):,}", flush=True)
        # Second, wider pass so the province view matches living/jobs:
        # fewer, larger soft circles instead of every street peak.
        place_rows_detail = as_rows(place_detail, 8)
        place_overview = combine(place_rows_detail, 2800.0, 2, 7)
        place_overview.sort(key=lambda row: row[7])
        print(f"place overview blobs={len(place_overview):,}", flush=True)
    else:
        place_detail = []
        place_overview = []

    OVERVIEW_PATH.write_text(
        json.dumps({"points": overview, "placeBlobs": place_overview}, separators=(",\n", ":")),
        encoding="utf-8",
    )
    DETAIL_PATH.write_text(
        json.dumps({"points": detail, "placeBlobs": place_detail}, separators=(",\n", ":")),
        encoding="utf-8",
    )
    print(OVERVIEW_PATH, flush=True)
    print(DETAIL_PATH, flush=True)
    print(f"done in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())

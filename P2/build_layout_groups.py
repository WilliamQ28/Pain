#!/usr/bin/env python3
"""
Derive higher-level flex grouping hints from the existing manifest + children JSON.

The script walks each band, reuses the same grid-normalization logic from render_html,
and groups children that occupy the exact same column span into vertical stacks.
The output JSON can be consumed by downstream renderers to mimic the trimmed PNG
layout while still using flexbox (stacked columns remain a single flex item).

Usage:
  python3 P2/build_layout_groups.py \
      --manifest P2/p2_tiles/manifest_seed7.json \
      --children P2/p4_populate/children_l1_seed7.json \
      --out P2/p4_populate/layout_groups_seed7.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from render_html import (
    load_json,
    build_row_map,
    build_col_map,
    place_children,
    global_col_extents,
)


def serialize_group(key, items: List[dict], max_cols: int) -> Dict[str, Any]:
    col_start, col_span = key
    order = min(info["row_start"] * max_cols + info["col_start"] for info in items)
    raw_pct = (col_span / max_cols) * 100.0 if max_cols else 100.0
    payload = {
        "type": "stack" if len(items) > 1 else "single",
        "col_start": col_start,
        "col_span": col_span,
        "order": order,
        "basis_pct": raw_pct,
        "items": [],
    }
    for info in sorted(items, key=lambda x: x["row_start"]):
        child = info["child"]
        payload["items"].append(
            {
                "id": child.get("id"),
                "class": child.get("class"),
                "pattern": child.get("pattern"),
                "row_start": info["row_start"],
                "row_span": info["row_span"],
                "row_orig": info["row_orig"],
                "col_start": info["col_start"],
                "col_span": info["col_span"],
            }
        )
    return payload


def build_groups(manifest: Dict[str, Any], children: List[Dict[str, Any]], preserve_extent: bool = True) -> Dict[str, Any]:
    tiles_x = int(manifest.get("tiles_x", 40))
    col_start, col_end = (0, tiles_x) if preserve_extent else global_col_extents(children, fallback_cols=tiles_x)
    canvas_cols = max(1, col_end - col_start)
    result = {"bands": []}

    children_by_band: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for child in children:
        children_by_band[child.get("band", "Unknown")].append(child)

    for band in manifest.get("bands", []):
        name = band.get("name")
        band_children = children_by_band.get(name, [])
        band_cols = int(band.get("tiles_w", tiles_x))
        band_rows = int(band.get("rows0") or band.get("tiles_h") or 0)

        row_map, row_vals = build_row_map(band_children, total_rows=band_rows if preserve_extent else None)
        col_map, col_count = build_col_map(
            band_children,
            col_offset=col_start,
            total_cols=band_cols if preserve_extent else None,
        )
        if not row_vals or not col_count:
            continue
        placements, _ = place_children(
            band_children,
            col_map=col_map,
            row_map=row_map,
            col_offset=col_start,
            col_count=col_count,
            nav_rows=row_vals[:2],
        )
        groups = defaultdict(list)
        for info in placements:
            key = (info["col_start"], info["col_span"])
            groups[key].append(info)
        serialized = [
            serialize_group(key, infos, col_count) for key, infos in sorted(groups.items(), key=lambda kv: kv[0])
        ]
        result["bands"].append(
            {
                "name": name,
                "cols": col_count,
                "canvas_cols": canvas_cols,
                "groups": serialized,
            }
        )
    return result


def parse_args():
    ap = argparse.ArgumentParser(description="Build flex grouping metadata from manifest + children.")
    ap.add_argument("--manifest", required=True, help="Path to manifest_seed*.json")
    ap.add_argument("--children", required=True, help="Path to children_l1_seed*.json")
    ap.add_argument("--out", required=True, help="Where to write the grouping JSON")
    ap.add_argument(
        "--compact_extent",
        action="store_true",
        help="Compact grids to used rows/cols (legacy). Default preserves full band extent.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = load_json(Path(args.manifest))
    children = load_json(Path(args.children))

    groups = build_groups(manifest, children, preserve_extent=not args.compact_extent)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print(f"[OK] wrote grouping hints → {out_path}")


if __name__ == "__main__":
    main()

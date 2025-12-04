#!/usr/bin/env python3
"""
Derive richer flex-friendly grouping metadata (column stacks + row clusters)
from the existing manifest + children JSON. This script does not alter the core
generation pipeline; it simply emits an auxiliary JSON payload that downstream
renderers can consume to recreate the trimmed PNG layout with flex.

Usage:
  python3 P2/build_flex_layout.py \
      --manifest P2/p2_tiles/manifest_seed7.json \
      --children P2/p4_populate/children_l1_seed7.json \
      --out P2/p4_populate/flex_layout_seed7.json
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


def serialize_columns(placements: List[dict], max_cols: int) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for info in placements:
        groups[(info["col_start"], info["col_span"])].append(info)
    serialized: List[Dict[str, Any]] = []
    for (col_start, col_span), members in groups.items():
        order_val = min(m["row_start"] for m in members)
        basis_pct = (col_span / max(max_cols, 1)) * 100.0
        serialized.append(
            {
                "type": "stack" if len(members) > 1 else "single",
                "col_start": col_start,
                "col_span": col_span,
                "order": order_val,
                "basis_pct": basis_pct,
                "items": [
                    {
                        "id": m["child"].get("id"),
                        "class": m["child"].get("class"),
                        "pattern": m["child"].get("pattern"),
                        "row_start": m["row_start"],
                        "row_span": m["row_span"],
                        "col_start": m["col_start"],
                        "col_span": m["col_span"],
                    }
                    for m in sorted(members, key=lambda x: x["row_start"])
                ],
            }
        )
    return sorted(serialized, key=lambda g: (g["order"], g["col_start"]))


def serialize_rows(placements: List[dict]) -> List[Dict[str, Any]]:
    row_groups: Dict[int, List[dict]] = defaultdict(list)
    for info in placements:
        row_groups[info["row_start"]].append(info)
    serialized = []
    for row_start, members in row_groups.items():
        serialized.append(
            {
                "row_start": row_start,
                "row_span": max(m["row_span"] for m in members),
                "items": [
                    {
                        "id": m["child"].get("id"),
                        "class": m["child"].get("class"),
                        "pattern": m["child"].get("pattern"),
                        "col_start": m["col_start"],
                        "col_span": m["col_span"],
                        "row_span": m["row_span"],
                    }
                    for m in sorted(members, key=lambda x: x["col_start"])
                ],
            }
        )
    return sorted(serialized, key=lambda g: g["row_start"])


def build_flex_layout(manifest: Dict[str, Any], children: List[Dict[str, Any]], preserve_extent: bool = True) -> Dict[str, Any]:
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
        columns = serialize_columns(placements, max_cols=col_count)
        rows = serialize_rows(placements)
        result["bands"].append(
            {
                "name": name,
                "cols": col_count,
                "canvas_cols": canvas_cols,
                "columns": columns,
                "rows": rows,
            }
        )
    return result


def parse_args():
    ap = argparse.ArgumentParser(description="Build flex layout metadata from manifest + children.")
    ap.add_argument("--manifest", required=True, help="Path to manifest_seed*.json")
    ap.add_argument("--children", required=True, help="Path to children_l1_seed*.json")
    ap.add_argument("--out", required=True, help="Where to write the layout JSON")
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
    payload = build_flex_layout(manifest, children, preserve_extent=not args.compact_extent)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote flex layout metadata → {out_path}")


if __name__ == "__main__":
    main()

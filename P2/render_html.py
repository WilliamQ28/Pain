#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render an HTML mock of a generated layout using WFC manifest + children JSON.
Media blocks use a solid-blue PNG placeholder, and text-style blocks show Lorem Ipsum.

Usage:
  python P2/render_html.py \
      --manifest P2/p2_tiles/manifest_seed7.json \
      --children P2/p4_populate/children_l1_seed7.json \
      --outdir P2/html
"""

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 1x1 PNG with RGB(0, 87, 255)
BLUE_PIXEL_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4XmP4"
    "//8/AwAI/AL+eN6V5QAAAABJRU5ErkJggg=="
)

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam."
)
LOREM_WORDS = LOREM.split()
BUTTON_WORDS = ["Action", "Primary", "Secondary", "Confirm", "Submit"]
NAV_WORDS = ["Home", "About", "Services", "Work", "Blog", "Contact", "Shop", "Docs", "Pricing"]
CTA_WORDS = ["Sign Up", "Get Started", "Join Now", "Learn More", "Book Demo"]
NAV_AREA_CAP = 4
LEGAL_WORDS = ["Privacy", "Terms", "Accessibility", "Cookies", "Do Not Sell", "Careers", "© 2025 Example Co."]
BAND_TAGS = {
    "header": "header",
    "nav": "nav",
    "main": "main",
    "supplement": "aside",
    "footer": "footer",
}
BAND_COLORS = {
    "Header": "#182235",
    "Nav": "#111827",
    "Main": "#0e111b",
    "Supplement": "#151921",
    "Footer": "#0b0d12",
}

AVG_CHAR_REM = 0.8
LINE_HEIGHT_REM = 1.5

def lorem_for_chars(char_capacity: Optional[int]) -> str:
    if not char_capacity or char_capacity <= 0:
        return LOREM
    words = []
    total = 0
    idx = 0
    while total < char_capacity:
        word = LOREM_WORDS[idx % len(LOREM_WORDS)]
        words.append(word)
        total += len(word) + 1
        idx += 1
    return " ".join(words)

def estimate_char_capacity(col_span: int, row_span: int, tile_rem: float, scale: float) -> int:
    width_rem = max(1.0, col_span * tile_rem * scale)
    height_rem = max(tile_rem, row_span * tile_rem)
    chars_per_line = max(6.0, width_rem / AVG_CHAR_REM)
    lines = max(1.0, height_rem / LINE_HEIGHT_REM)
    capacity = int(chars_per_line * lines)
    return max(20, min(800, capacity))

def lorem_for_span(area: int) -> str:
    if area <= 0:
        return " ".join(LOREM_WORDS[:5])
    words = max(5, min(len(LOREM_WORDS), area * 2))
    return " ".join(LOREM_WORDS[:words])

def label_for_class(cls: str, pattern: str, idx: int, area: int, row_orig: int, top_rows: List[int], bottom_row: Optional[int], char_capacity: Optional[int] = None) -> Tuple[str, bool]:
    if row_orig in top_rows:
        word = NAV_WORDS[idx % len(NAV_WORDS)]
        return word, True
    if bottom_row is not None and row_orig >= bottom_row:
        word = LEGAL_WORDS[idx % len(LEGAL_WORDS)]
        return word, False
    if pattern == "menu_row":
        word = NAV_WORDS[idx % len(NAV_WORDS)]
        return word, True
    if cls in {"button", "role_button"}:
        if area <= NAV_AREA_CAP:
            word = CTA_WORDS[idx % len(CTA_WORDS)]
            return word, True
        return lorem_for_span(area), False
    if cls in {"input", "select"}:
        return " ".join(LOREM_WORDS[:min(4, len(LOREM_WORDS))]), True
    if char_capacity:
        return lorem_for_chars(char_capacity), False
    return lorem_for_span(area), False


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def child_grid_position(
    shadow: Dict[str, int],
    col_map: Dict[int, int],
    row_map: Dict[int, int],
    col_offset: int = 0,
) -> Tuple[int, int, int, int, int]:
    col = int(shadow.get("col", 0)) - col_offset
    row = int(shadow.get("row", 0))
    col_span = max(1, int(shadow.get("col_span", 1)))
    row_span = max(1, int(shadow.get("row_span", 1)))

    if col_map:
        c_start = col_map.get(col, 0)
        c_end = col_map.get(col + col_span - 1, c_start)
        col_span = c_end - c_start + 1
    else:
        c_start = col

    if row_map:
        r_start = row_map.get(row, 0)
        r_end = row_map.get(row + row_span - 1, r_start)
        row_span = r_end - r_start + 1
    else:
        r_start = row

    return c_start, r_start, col_span, row_span, row


def build_css(tile_rem: float, canvas_cols: int, wireframe: bool = False) -> str:
    if wireframe:
        return f"""
:root {{
  --tile-rem: {tile_rem:.4f}rem;
  --canvas-cols: {canvas_cols};
}}
* {{
  box-sizing: border-box;
  font-family: "Segoe UI", "Inter", sans-serif;
}}
body {{
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: #0f172a;
}}
.layout-shell {{
  width: 100%;
  margin: 0 auto;
  padding: 0;
  display: flex;
  justify-content: center;
  overflow-x: auto;
}}
.layout {{
  width: calc(var(--tile-rem) * var(--canvas-cols));
  background: #ffffff;
  position: relative;
  flex-shrink: 0;
  padding: 0;
}}
.band {{
  position: relative;
  margin: 0 0 0.75rem 0;
  padding: 0;
  border: 0;
  background: transparent;
}}
.band-body {{
  box-sizing: border-box;
}}
.band-body.grid-mode {{
  display: grid;
  grid-template-columns: repeat(var(--band-cols, var(--canvas-cols)), var(--band-tile-rem, var(--tile-rem)));
  grid-auto-rows: var(--band-tile-rem, var(--tile-rem));
  gap: 0;
  padding: 0;
  margin: 0;
  width: calc(var(--band-tile-rem, var(--tile-rem)) * var(--band-cols, var(--canvas-cols)));
  justify-content: start;
  align-content: start;
}}
.band-body.flex-mode {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: flex-start;
  align-content: flex-start;
  padding: 0;
  margin: 0;
  width: calc(var(--band-tile-rem, var(--tile-rem)) * var(--band-cols, var(--canvas-cols)));
}}
.child {{
  border: 1px solid #94a3b8;
  background: #f8fafc;
  color: #0f172a;
  border-radius: 0;
  padding: 0;
  font-size: 0.75rem;
  line-height: 1.3;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  min-height: 100%;
  min-width: 100%;
}}
.child.media img {{
  display: block;
  height: 100%;
  width: 100%;
  object-fit: cover;
  background: #e2e8f0;
}}
.content {{
  width: 100%;
}}
@media (max-width: 50rem) {{
  .layout {{
    width: min(100%, calc(var(--tile-rem) * var(--canvas-cols)));
  }}
  .band-body.grid-mode {{
    width: calc(var(--tile-rem) * var(--band-cols, var(--canvas-cols)));
  }}
}}
""".strip()

    return f"""
:root {{
  --tile-rem: {tile_rem:.4f}rem;
  --canvas-cols: {canvas_cols};
  --card: #f7f9fc;
  --border: #e2e8f0;
  --text: #111827;
  --muted: #64748b;
  --accent: #2b6cb0;
}}
* {{
  box-sizing: border-box;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
}}
body {{
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: var(--text);
}}
.layout-shell {{
  width: 100%;
  margin: 0 auto;
  padding: 0;
  display: flex;
  justify-content: center;
  overflow-x: auto;
  scrollbar-color: rgba(0,0,0,0.25) transparent;
}}
.layout {{
  width: calc(var(--tile-rem) * var(--canvas-cols));
  background: #ffffff;
  position: relative;
  flex-shrink: 0;
  padding: 0;
}}
.band {{
  position: relative;
  margin: 0 0 0.75rem 0;
  padding: 0;
  background: #ffffff;
  border: 0;
  box-shadow: none;
}}
.band-body {{
  box-sizing: border-box;
}}
.band-body.grid-mode {{
  display: grid;
  grid-template-columns: repeat(var(--band-cols, var(--canvas-cols)), var(--band-tile-rem, var(--tile-rem)));
  grid-auto-rows: var(--band-tile-rem, var(--tile-rem));
  gap: 0;
  padding: 0;
  margin: 0;
  width: calc(var(--band-tile-rem, var(--tile-rem)) * var(--band-cols, var(--canvas-cols)));
  justify-content: start;
  align-content: start;
}}
.band-body.flex-mode {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: flex-start;
  align-content: flex-start;
  padding: 0;
  margin: 0;
  width: calc(var(--band-tile-rem, var(--tile-rem)) * var(--band-cols, var(--canvas-cols)));
}}
.column-stack {{
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  flex: 1 1 100%;
  max-width: 100%;
}}
.column-stack > .child {{
  width: 100%;
  max-width: 100%;
}}
.child {{
  border-radius: 0.35rem;
  padding: 0.35rem;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  color: var(--text);
  border: 1px solid var(--border);
  background: var(--card);
  overflow: hidden;
}}
.child.button,
.child.role_button,
.child.input,
.child.select {{
  background: #eef2ff;
  border-color: #e0e7ff;
  font-weight: 600;
  color: #1e3a8a;
}}
.child.text {{
  background: var(--card);
  border-color: var(--border);
}}
.child.media {{
  padding: 0;
  background: #e2e8f0;
}}
.child.media img {{
  display: block;
  height: 100%;
  width: 100%;
  object-fit: cover;
  background: #cbd5e1;
}}
.child .label {{
  display: none;
}}
.child .content {{
  font-size: 0.9rem;
  line-height: 1.45;
  flex: 1;
  overflow: hidden;
  overflow-wrap: anywhere;
  hyphens: auto;
  display: flex;
  align-items: flex-start;
}}
.content.heading {{
  font-weight: 700;
  font-size: 1.05rem;
  margin-bottom: 0.25rem;
}}
.content.small {{
  font-size: 0.78rem;
  color: var(--muted);
}}
.sm-text .content {{
  font-size: 0.75rem;
}}
.legal-text .content {{
  font-size: 0.8rem;
  color: var(--muted);
}}
@media (max-width: 60rem) {{
  .layout {{
    width: min(100%, calc(var(--tile-rem) * var(--canvas-cols)));
  }}
}}
@media (max-width: 50rem) {{
  .band-body.flex-mode {{
    gap: 0.5rem;
  }}
  .band-body .child {{
    flex-basis: 100% !important;
    max-width: 100% !important;
    min-height: auto !important;
  }}
  .child.media img {{
    height: auto;
  }}
}}
""".strip()


def build_row_map(children: List[Dict], total_rows: Optional[int] = None) -> Tuple[Dict[int, int], List[int]]:
    """
    Build a dense mapping from L0 rows to compact rows for rendering.
    If total_rows is provided, preserve the full band height (keeps whitespace).
    """
    if total_rows is not None and total_rows > 0:
        sorted_rows = list(range(total_rows))
        return {r: r for r in sorted_rows}, sorted_rows

    used_rows: set[int] = set()
    for child in children:
        shadow = child.get("grid_l0_shadow") or child.get("grid_l0") or {}
        row = int(shadow.get("row", 0))
        row_span = max(1, int(shadow.get("row_span", 1)))
        for r in range(row, row + row_span):
            used_rows.add(r)
    if not used_rows:
        return {}, []
    sorted_rows = sorted(used_rows)
    row_map = {r: idx for idx, r in enumerate(sorted_rows)}
    return row_map, sorted_rows


def build_col_map(children: List[Dict], col_offset: int, total_cols: Optional[int] = None) -> Tuple[Dict[int, int], int]:
    """
    Build a dense mapping from L0 cols to compact cols for rendering.
    If total_cols is provided, preserve the full band width starting at col_offset.
    """
    if total_cols is not None and total_cols > 0:
        sorted_cols = list(range(col_offset, col_offset + total_cols))
        return {c: idx for idx, c in enumerate(sorted_cols)}, total_cols

    used_cols: set[int] = set()
    for child in children:
        shadow = child.get("grid_l0_shadow") or child.get("grid_l0") or {}
        col = int(shadow.get("col", 0)) - col_offset
        col_span = max(1, int(shadow.get("col_span", 1)))
        for c in range(max(0, col), max(0, col) + col_span):
            used_cols.add(c)
    if not used_cols:
        return {}, 0
    sorted_cols = sorted(used_cols)
    col_map = {c: idx for idx, c in enumerate(sorted_cols)}
    return col_map, len(sorted_cols)


def place_children(
    children: List[Dict],
    col_map: Dict[int, int],
    row_map: Dict[int, int],
    col_offset: int,
    col_count: int,
    nav_rows: List[int],
) -> Tuple[List[dict], int]:
    placements: List[dict] = []
    occupancy: List[List[bool]] = []
    band_cols = max(1, col_count)
    nav_area_limit = NAV_AREA_CAP

    def ensure_rows(n: int):
        while len(occupancy) < n:
            occupancy.append([False] * band_cols)

    def has_collision(r: int, c: int, r_span: int, c_span: int) -> bool:
        for rr in range(r, r + r_span):
            if rr >= len(occupancy):
                continue
            row_cells = occupancy[rr]
            for cc in range(c, c + c_span):
                if cc < 0 or cc >= len(row_cells):
                    return True
                if row_cells[cc]:
                    return True
        return False

    def mark_cells(r: int, c: int, r_span: int, c_span: int):
        ensure_rows(r + r_span)
        for rr in range(r, r + r_span):
            row_cells = occupancy[rr]
            for cc in range(c, c + c_span):
                if 0 <= cc < len(row_cells):
                    row_cells[cc] = True

    sorted_children = sorted(
        children,
        key=lambda ch: (
            row_map.get((ch.get("grid_l0_shadow") or ch.get("grid_l0") or {}).get("row", 0), 0),
            col_map.get((ch.get("grid_l0_shadow") or ch.get("grid_l0") or {}).get("col", 0) - col_offset, 0),
        ),
    )

    for child in sorted_children:
        c_idx, r_idx, c_span, r_span, row_orig = child_grid_position(
            child.get("grid_l0_shadow") or child.get("grid_l0") or {},
            col_map=col_map,
            row_map=row_map,
            col_offset=col_offset,
        )
        ensure_rows(r_idx + r_span)
        rr = r_idx
        cc = c_idx

        # If overlapping within nav (top rows) move horizontally
        if row_orig in nav_rows:
            while cc + c_span <= band_cols and has_collision(rr, cc, r_span, c_span):
                cc += 1
        while has_collision(rr, cc, r_span, c_span):
            cc += 1
            if cc + c_span > band_cols:
                cc = 0
                rr += 1
            ensure_rows(rr + r_span)

        mark_cells(rr, cc, r_span, c_span)
        placements.append(
            {
                "child": child,
                "col_start": cc,
                "row_start": rr,
                "col_span": c_span,
                "row_span": r_span,
                "row_orig": row_orig,
            }
        )
    return placements, len(occupancy)


def global_col_extents(children: List[Dict], fallback_cols: int) -> Tuple[int, int]:
    if not children:
        return 0, fallback_cols
    cols = [child.get("grid_l0_shadow") or child.get("grid_l0") or {} for child in children]
    col_start = min(int(c.get("col", 0)) for c in cols)
    col_end = max(int(c.get("col", 0)) + int(c.get("col_span", 1)) for c in cols)
    return col_start, col_end


def render_band(
    band: Dict,
    children: List[Dict],
    tile_rem: float,
    col_offset: int,
    row_map: Dict[int, int],
    row_values: List[int],
    col_map: Dict[int, int],
    col_count: int,
    canvas_cols: int,
    band_hint: Optional[Dict[str, Any]] = None,
    wireframe: bool = False,
    copy_map: Optional[Dict[str, Any]] = None,
) -> str:
    if band_hint and band_hint.get("columns"):
        return render_band_flex(
            band,
            children,
            tile_rem,
            row_values=row_values,
            col_count=col_count,
            canvas_cols=canvas_cols,
            band_hint=band_hint,
            copy_map=copy_map,
        )
    return render_band_grid(
        band,
        children,
        tile_rem,
        col_offset,
        row_map,
        row_values,
        col_map,
        col_count,
        canvas_cols,
        wireframe=wireframe,
        copy_map=copy_map,
    )


def render_band_grid(
    band: Dict,
    children: List[Dict],
    tile_rem: float,
    col_offset: int,
    row_map: Dict[int, int],
    row_values: List[int],
    col_map: Dict[int, int],
    col_count: int,
    canvas_cols: int,
    wireframe: bool = False,
    copy_map: Optional[Dict[str, Any]] = None,
) -> str:
    nav_rows = row_values[:2]
    placements, _ = place_children(children, col_map, row_map, col_offset, col_count, nav_rows)
    if not placements:
        return ""

    top_rows_orig = row_values[:2]
    bottom_row_orig = row_values[-1] if row_values else None
    label_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    band_name = band.get("name", "Section")
    tag = BAND_TAGS.get(band_name.strip().lower(), "section")
    band_color = BAND_COLORS.get(band_name, "#101522")
    max_cols = max(1, col_count)
    target_cols = min(canvas_cols, col_count)
    scale = target_cols / float(max_cols)

    parts = [f'<{tag} class="band" data-band="{band_name}" style="--band-bg:{band_color};">']
    grid_style = (
        f"--band-cols:{col_count};"
        f"--band-tile-rem:calc(var(--tile-rem) * {scale:.4f});"
    )
    parts.append(f'<div class="band-body grid-mode" style="{grid_style}">')

    for info in sorted(placements, key=lambda inf: (inf["row_start"], inf["col_start"])):
        child = info["child"]
        cls = child.get("class", "unknown")
        pattern = child.get("pattern", "")
        row_orig = info["row_orig"]
        area = info["col_span"] * info["row_span"]

        display_cls = cls

        key = (display_cls, pattern)
        idx = label_counts[key]
        label_counts[key] += 1

        char_cap = estimate_char_capacity(info["col_span"], info["row_span"], tile_rem, scale)
        snippet_override = None
        heading_override = None
        if copy_map and child.get("id") in copy_map:
            val = copy_map.get(child["id"])
            if isinstance(val, dict):
                heading_override = val.get("heading")
                snippet_override = val.get("body") or val.get("text") or val.get("content")
            else:
                snippet_override = str(val)
        if snippet_override is not None:
            snippet = snippet_override
            small = False
        else:
            snippet, small = label_for_class(
                display_cls,
                pattern,
                idx,
                area,
                row_orig=row_orig,
                top_rows=top_rows_orig,
                bottom_row=bottom_row_orig,
                char_capacity=char_cap,
            )
        size_class = ""
        if char_cap < 80:
            size_class = "small"
        heading = heading_override
        if heading is None and char_cap > 250 and cls in {"text", "button", "role_button"}:
            base = "Heading"
            heading = f"{base} {idx+1}"
        extra = []
        if bottom_row_orig is not None and row_orig >= bottom_row_orig:
            extra.append("legal-text")
        class_tokens = ["child", display_cls]
        if small:
            class_tokens.append("sm-text")
        class_tokens.extend(extra)
        classes = " ".join(class_tokens)
        style = (
            f"grid-column:{info['col_start'] + 1} / span {info['col_span']};"
            f"grid-row:{info['row_start'] + 1} / span {info['row_span']};"
            f"min-height:calc(var(--tile-rem) * {info['row_span']});"
        )
        parts.append(f'<div class="{classes}" style="{style}" data-pattern="{pattern}">')
        if cls == "media":
            parts.append(
                f'<img src="data:image/png;base64,{BLUE_PIXEL_BASE64}" '
                f'alt="{cls} placeholder" />'
            )
        else:
            if heading:
                parts.append(f'<div class="content heading">{heading}</div>')
            body_cls = "content"
            if size_class == "small":
                body_cls += " small"
            parts.append(f'<div class="{body_cls}">{snippet}</div>')
        parts.append("</div>")

    parts.append("</div>")
    parts.append(f"</{tag}>")
    return "\n".join(parts)


def render_band_flex(
    band: Dict,
    children: List[Dict],
    tile_rem: float,
    row_values: List[int],
    col_count: int,
    canvas_cols: int,
    band_hint: Dict[str, Any],
    copy_map: Optional[Dict[str, Any]] = None,
) -> str:
    child_lookup = {child.get("id"): child for child in children if child.get("id")}
    top_rows_orig = row_values[:2]
    bottom_row_orig = row_values[-1] if row_values else None
    label_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    band_name = band.get("name", "Section")
    tag = BAND_TAGS.get(band_name.strip().lower(), "section")
    band_color = BAND_COLORS.get(band_name, "#101522")
    max_cols = max(1, int(band_hint.get("cols") or col_count))
    target_cols = min(canvas_cols, max_cols)
    scale = target_cols / float(max_cols)

    parts = [f'<{tag} class="band" data-band="{band_name}" style="--band-bg:{band_color};">']
    parts.append(f'<div class="band-body flex-mode" style="--band-cols:{col_count};--band-tile-rem:calc(var(--tile-rem) * {scale:.4f});">')

    def render_member(item: Dict[str, Any], flex_mode: bool, basis_override: Optional[float] = None, order_override: Optional[int] = None):
        child = child_lookup.get(item.get("id"))
        if not child:
            return
        cls = child.get("class", "unknown")
        pattern = child.get("pattern", "")
        row_idx = int(item.get("row_start", 0))
        if 0 <= row_idx < len(row_values):
            row_orig = row_values[row_idx]
        else:
            row_orig = row_idx
        col_span = int(item.get("col_span", 1))
        row_span = int(item.get("row_span", 1))
        area = col_span * row_span

        display_cls = cls

        key = (display_cls, pattern)
        idx = label_counts[key]
        label_counts[key] += 1

        char_cap = estimate_char_capacity(col_span, row_span, tile_rem, scale)
        snippet_override = None
        heading_override = None
        if copy_map and child.get("id") in copy_map:
            val = copy_map.get(child["id"])
            if isinstance(val, dict):
                heading_override = val.get("heading")
                snippet_override = val.get("body") or val.get("text") or val.get("content")
            else:
                snippet_override = str(val)
        if snippet_override is not None:
            snippet = snippet_override
            small = False
        else:
            snippet, small = label_for_class(
                display_cls,
                pattern,
                idx,
                area,
                row_orig=row_orig,
                top_rows=top_rows_orig,
                bottom_row=bottom_row_orig,
                char_capacity=char_cap,
            )
        size_class = ""
        if char_cap < 80:
            size_class = "small"
        heading = heading_override
        if heading is None and char_cap > 250 and cls in {"text", "button", "role_button"}:
            base = "Heading"
            heading = f"{base} {idx+1}"
        extra = []
        if bottom_row_orig is not None and row_orig >= bottom_row_orig:
            extra.append("legal-text")
        class_tokens = ["child", display_cls]
        if small:
            class_tokens.append("sm-text")
        class_tokens.extend(extra)
        classes = " ".join(class_tokens)
        if flex_mode:
            raw_pct = (col_span / max_cols) * 100.0
            basis_pct = basis_override if basis_override is not None else max(5.0, raw_pct * 0.97)
            order_val = order_override if order_override is not None else row_idx * max_cols + int(item.get("col_start", 0))
            style = (
                f"flex:1 1 {basis_pct:.4f}%;"
                f"max-width:{basis_pct:.4f}%;"
                f"order:{order_val};"
                f"min-height:calc(var(--tile-rem) * {row_span});"
            )
        else:
            style = f"width:100%;max-width:100%;min-height:calc(var(--tile-rem) * {row_span});"
        parts.append(f'<div class="{classes}" style="{style}" data-pattern="{pattern}">')
        if cls == "media":
            parts.append(
                f'<img src="data:image/png;base64,{BLUE_PIXEL_BASE64}" '
                f'alt="{cls} placeholder" />'
            )
        else:
            if heading:
                parts.append(f'<div class="content heading">{heading}</div>')
            body_cls = "content"
            if size_class == "small":
                body_cls += " small"
            parts.append(f'<div class="{body_cls}">{snippet}</div>')
        parts.append("</div>")

    columns = sorted(
        band_hint.get("columns", []),
        key=lambda col: (int(col.get("order", 0)), int(col.get("col_start", 0))),
    )
    for column in columns:
        items = [item for item in column.get("items", []) if item.get("id") in child_lookup]
        if not items:
            continue
        basis_pct = max(5.0, float(column.get("basis_pct", 100.0)) * 0.97)
        order_val = int(column.get("order", 0))
        if column.get("type") == "stack" and len(items) > 1:
            stack_style = (
                f"flex:1 1 {basis_pct:.4f}%;"
                f"max-width:{basis_pct:.4f}%;"
                f"order:{order_val};"
            )
            parts.append(f'<div class="column-stack" style="{stack_style}">')
            for item in sorted(items, key=lambda x: int(x.get("row_start", 0))):
                render_member(item, flex_mode=False)
            parts.append("</div>")
        else:
            render_member(items[0], flex_mode=True, basis_override=basis_pct, order_override=order_val)

    row_items = band_hint.get("rows", [])
    for row in row_items:
        members = [m for m in row.get("items", []) if m.get("id") in child_lookup]
        if len(members) <= 1:
            continue
        row_span = int(row.get("row_span", 1))
        width_pct = sum((int(m.get("col_span", 1)) / max_cols) * 100.0 for m in members)
        width_pct = max(10.0, min(100.0, width_pct * 0.97))
        order_val = int(row.get("row_start", 0))
        row_style = (
            f"flex:1 1 {width_pct:.4f}%;"
            f"max-width:{width_pct:.4f}%;"
            f"order:{order_val};"
            "display:flex;flex-wrap:wrap;gap:0.5rem;"
        )
        parts.append(f'<div class="row-stack" style="{row_style}">')
        for item in sorted(members, key=lambda x: int(x.get("col_start", 0))):
            render_member(item, flex_mode=True)
        parts.append("</div>")

    parts.append("</div>")
    parts.append(f"</{tag}>")
    return "\n".join(parts)

def render_html(
    manifest: Dict,
    children: List[Dict],
    title: str,
    tile_rem: float,
    flex_layout: Optional[Dict[str, Any]] = None,
    preserve_extent: bool = True,
    wireframe: bool = False,
    grid_only: bool = False,
    copy_map: Optional[Dict[str, Any]] = None,
) -> str:
    tiles_x = int(manifest.get("tiles_x", 40))
    col_start, col_end = (0, tiles_x) if preserve_extent else global_col_extents(children, fallback_cols=tiles_x)
    canvas_cols = max(1, col_end - col_start)

    max_cols = 1
    children_by_band: Dict[str, List[Dict]] = defaultdict(list)
    for child in children:
        band = child.get("band", "Unknown")
        children_by_band[band].append(child)

    bands_html = []
    flex_map = {}
    if flex_layout and not grid_only:
        flex_map = {band.get("name"): band for band in flex_layout.get("bands", [])}
    for band in manifest.get("bands", []):
        name = band.get("name")
        band_children = children_by_band.get(name, [])
        band_cols = int(band.get("tiles_w", tiles_x))
        band_rows = int(band.get("rows0") or band.get("tiles_h") or 0)

        row_map, row_rows = build_row_map(band_children, total_rows=band_rows if preserve_extent else None)
        row_count = len(row_rows)
        col_map, col_count = build_col_map(
            band_children,
            col_offset=col_start,
            total_cols=band_cols if preserve_extent else None,
        )
        if row_count == 0 or col_count == 0:
            continue
        max_cols = max(max_cols, col_count)
        band_hint = flex_map.get(name)
        band_html = render_band(
            band,
            band_children,
            tile_rem=tile_rem,
            col_offset=col_start,
            row_map=row_map,
            row_values=row_rows,
            col_map=col_map,
            col_count=col_count,
            canvas_cols=canvas_cols,
            band_hint=band_hint,
            wireframe=wireframe,
            copy_map=copy_map,
        )
        bands_html.append(band_html)

    css = build_css(tile_rem=tile_rem, canvas_cols=max(canvas_cols, max_cols), wireframe=wireframe)

    body = "\n".join(bands_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
{css}
  </style>
</head>
<body>
  <div class="layout-shell">
  <main class="layout">
{body}
  </main>
  </div>
</body>
</html>
""".strip()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Render procedural layout HTML from manifest + children JSON."
    )
    ap.add_argument("--manifest", required=True, help="Path to manifest_seed*.json")
    ap.add_argument("--children", required=True, help="Path to children_l1_seed*.json")
    ap.add_argument("--outdir", default="P2/html", help="Where to write the HTML file")
    ap.add_argument("--seed", type=int, default=None, help="Override seed for filename")
    ap.add_argument(
        "--tile_px",
        type=float,
        default=None,
        help="Pixels per tile column/row (defaults to manifest png_scale or 36)",
    )
    ap.add_argument(
        "--tile_rem",
        type=float,
        default=None,
        help="Override tile size directly in rem (skips px-to-rem conversion)",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional document title (defaults to 'Seed {seed} Layout')",
    )
    ap.add_argument(
        "--flex_layout",
        default=None,
        help="Optional flex layout metadata JSON (from build_flex_layout.py)",
    )
    ap.add_argument(
        "--compact_extent",
        action="store_true",
        help="Compact grids to used rows/cols (legacy). Default preserves full band extent.",
    )
    ap.add_argument(
        "--wireframe",
        action="store_true",
        help="Render outline-only wireframe styling (no dark theme).",
    )
    ap.add_argument(
        "--grid_only",
        action="store_true",
        help="Force pure grid rendering even if flex metadata is provided.",
    )
    ap.add_argument(
        "--copy_map",
        default=None,
        help="Optional JSON file mapping child_id -> text or {heading,body}.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    children_path = Path(args.children)

    manifest = load_json(manifest_path)
    children = load_json(children_path)

    seed = args.seed if args.seed is not None else manifest.get("seed", 0)
    if args.tile_rem:
        tile_size = float(args.tile_rem)
    else:
        px_scale = args.tile_px or float(manifest.get("png_scale", 36) or 36)
        tile_size = px_scale / 16.0
    title = args.title or f"Seed {seed} Layout"

    flex_layout = None
    if args.flex_layout:
        flex_layout = load_json(Path(args.flex_layout))
    copy_map = None
    if args.copy_map:
        try:
            copy_map = load_json(Path(args.copy_map))
        except Exception:
            copy_map = None

    html = render_html(
        manifest=manifest,
        children=children,
        title=title,
        tile_rem=tile_size,
        flex_layout=flex_layout,
        preserve_extent=not args.compact_extent,
        wireframe=args.wireframe,
        grid_only=args.grid_only,
        copy_map=copy_map,
    )

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    out_path = outdir / f"seed{seed}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] wrote HTML mock → {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pass B — Populate zones with L1 children (square-grid refinement)

Inputs
  --manifest   p2_tiles/manifest_seedX.json
  --zones      p3_orchestrate/zones_l0_seedX.json
  --patterns   wfc_stats/patterns.json

Outputs
  p4_populate/children_l1_seedX.json
  debug/children_l1_overlay_seedX.png

Notes
- Works in L1 grid units inside each zone: L1 = L0 * R, where R comes from the zone (or pattern).
- Also records a convenience L0 'shadow' span (integer-div rounding) for easy rendering/QC.
- Keeps everything in integer grid units; no pixel math here.
"""

import argparse, csv, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from PIL import Image, ImageDraw

# ---------------- CLI ----------------

GEOM_GUIDE = None
PX_PER_TILE = 36.0
CANVAS_WIDTH = 1440.0
QA_WARNINGS: List[str] = []


def parse_args():
    ap = argparse.ArgumentParser("Populate Zones L1")
    ap.add_argument("--manifest", required=True, help="p2_tiles/manifest_seedX.json")
    ap.add_argument("--zones",     required=True, help="p3_orchestrate/zones_l0_seedX.json")
    ap.add_argument("--patterns",  required=True, help="wfc_stats/patterns.json")
    ap.add_argument("--geometry",  default=None, help="optional element_geometry.json for sizing guidance")
    ap.add_argument("--outdir",    default="p4_populate", help="output folder for children json")
    ap.add_argument("--debugdir",  default="debug", help="debug overlay folder")
    ap.add_argument("--seed",      type=int, default=None, help="override seed in filenames, else from manifest")
    ap.add_argument("--png_scale", type=int, default=8, help="scale for overlay")
    ap.add_argument("--logfile",   default=None, help="optional CSV file to log generated children")
    ap.add_argument("--main_slice_px", type=int, default=None, help="optional Main window height in px (e.g., 960)")
    ap.add_argument("--main_slice_rows", type=int, default=None, help="optional Main window height in rows (overrides px)")
    ap.add_argument("--gutter_cols", type=int, default=None, help="optional side gutter (cols per side) to clamp children")
    ap.add_argument("--macro_grid", default=None, help="optional macro_grid_config.json (for gutter defaults)")
    ap.add_argument("--viewport_px", type=int, default=None, help="target viewport height in px for windowed denoise")
    ap.add_argument("--viewport_rows", type=int, default=None, help="target viewport height in rows (overrides px)")
    ap.add_argument("--window_stride_frac", type=float, default=0.5, help="stride fraction of viewport rows when windowing")
    ap.add_argument("--window_attempts", type=int, default=2, help="attempts per window with slight randomness")
    ap.add_argument("--min_window_coverage", type=float, default=0.7, help="coverage threshold to inject filler in a window")
    return ap.parse_args()

# -------------- IO helpers -------------

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -------------- Grid helpers -------------

def clamp(v, lo, hi): return max(lo, min(hi, v))

def l1_dims(z_grid: dict, R: int) -> Tuple[int,int]:
    """Return (rows_l1, cols_l1) inside the zone."""
    return int(z_grid["row_span"] * R), int(z_grid["col_span"] * R)

def l1_to_l0_span(r: int, c: int, rs: int, cs: int, R: int, parent_l0: dict) -> Tuple[int,int,int,int]:
    """
    Convert an L1 child span to an L0 'shadow' (rows/cols in integer tiles),
    then clamp to the parent zone's L0 box.
    parent_l0 is the zone's L0 grid dict: {"row","col","row_span","col_span"}.
    """
    # raw L0 (ceil on end so we fully cover the child)
    r0 = r // R
    c0 = c // R
    r1 = (r + rs + R - 1) // R
    c1 = (c + cs + R - 1) // R
    rs0 = max(1, r1 - r0)
    cs0 = max(1, c1 - c0)

    # clamp inside the zone
    z_r0 = int(parent_l0["row"])
    z_c0 = int(parent_l0["col"])
    z_rs = int(parent_l0["row_span"])
    z_cs = int(parent_l0["col_span"])

    # local to zone, then clamp
    lr0 = max(0, min(r0, z_rs - 1))
    lc0 = max(0, min(c0, z_cs - 1))
    lr1 = max(lr0 + 1, min(r0 + rs0, z_rs))
    lc1 = max(lc0 + 1, min(c0 + cs0, z_cs))

    return lr0, lc0, max(1, lr1 - lr0), max(1, lc1 - lc0)


def within(l1_rows: int, l1_cols: int, r: int, c: int, rs: int, cs: int) -> bool:
    return 0 <= r and 0 <= c and (r + rs) <= l1_rows and (c + cs) <= l1_cols

def validate_and_clamp_child(child: dict, zone_l0: dict, tiles_x: int, band_rows: int) -> bool:
    """Ensure child's L0 shadow sits inside its band and parent zone."""
    g = child["grid_l0_shadow"]
    zr, zc, zrs, zcs = zone_l0["row"], zone_l0["col"], zone_l0["row_span"], zone_l0["col_span"]

    # Must be inside parent zone in L0
    if not (zr <= g["row"] < zr + zrs): return False
    if not (zc <= g["col"] < zc + zcs): return False
    if g["row"] + g["row_span"] > zr + zrs: return False
    if g["col"] + g["col_span"] > zc + zcs: return False

    # Must be inside band lattice
    if g["col"] < 0 or g["col"] + g["col_span"] > tiles_x: return False
    if g["row"] < 0 or g["row"] + g["row_span"] > band_rows: return False
    return True


# -------------- Geometry guidance -------------

class GeometryGuide:
    def __init__(self, data: Optional[dict], canvas_width: float, tiles_x: int):
        self.data = data or {}
        self.canvas_width = float(canvas_width)
        self.tiles_x = max(1, int(tiles_x))
        self.px_per_tile = self.canvas_width / float(self.tiles_x)
        self.by_band = self.data.get("by_band", {})
        self.by_band_type = self.data.get("by_band_and_type", {})

    def _stat_dict(self, band: str, cls: str, field: str) -> Optional[dict]:
        if not band:
            return None
        band_dict = self.by_band_type.get(band, {}).get(cls, {})
        if field in band_dict and isinstance(band_dict[field], dict):
            return band_dict[field]
        broader = self.by_band.get(band, {}).get(field, {})
        if isinstance(broader, dict):
            return broader
        return None

    def _stat(self, band: str, cls: str, field: str) -> Optional[float]:
        d = self._stat_dict(band, cls, field)
        if d:
            val = d.get("p50")
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def _relative(self, band: str, cls: str, field: str) -> Optional[float]:
        if not band:
            return None
        band_dict = self.by_band_type.get(band, {}).get(cls, {})
        rel = band_dict.get(field)
        if isinstance(rel, dict):
            val = rel.get("p50")
            if isinstance(val, (int, float)):
                return float(val)
        broader = self.by_band.get(band, {}).get(field, {})
        if isinstance(broader, dict):
            val = broader.get("p50")
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def width_px(self, band: str, cls: str) -> Optional[float]:
        val = self._stat(band, cls, "width_px")
        if val is not None:
            return val
        rel = self._relative(band, cls, "width_relative")
        if rel is not None:
            return rel * self.canvas_width
        return None

    def height_px(self, band: str, cls: str) -> Optional[float]:
        val = self._stat(band, cls, "height_px")
        if val is not None:
            return val
        rel = self._relative(band, cls, "height_relative")
        if rel is not None:
            return rel * self.canvas_width
        return None

    def width_cols_l1(self, band: str, cls: str, zone_cols_l0: int, R: int, l1_cols: int) -> Optional[int]:
        px = self.width_px(band, cls)
        if px is None or self.px_per_tile <= 0:
            return None
        zone_width_px = zone_cols_l0 * self.px_per_tile
        px = min(max(px, self.px_per_tile), zone_width_px)
        cols_l0 = px / self.px_per_tile
        return max(1, min(int(round(cols_l0 * R)), l1_cols))

    def width_cols_stats_l1(self, band: str, cls: str, zone_cols_l0: int, R: int, l1_cols: int) -> Optional[Tuple[int, int, int]]:
        stats = self._stat_dict(band, cls, "width_px")
        if not stats:
            return None
        vals: List[int] = []
        for key in ("p25", "p50", "p75"):
            v = stats.get(key)
            if not isinstance(v, (int, float)):
                return None
            zone_width_px = zone_cols_l0 * self.px_per_tile
            v = min(max(v, self.px_per_tile), zone_width_px)
            cols_l0 = v / self.px_per_tile
            vals.append(max(1, min(int(round(cols_l0 * R)), l1_cols)))
        return tuple(vals)

    def height_rows_l1(self, band: str, cls: str, zone_rows_l0: int, R: int, l1_rows: int) -> Optional[int]:
        px = self.height_px(band, cls)
        if px is None or self.px_per_tile <= 0:
            return None
        zone_height_px = zone_rows_l0 * self.px_per_tile
        px = min(max(px, self.px_per_tile), zone_height_px)
        rows_l0 = px / self.px_per_tile
        return max(1, min(int(round(rows_l0 * R)), l1_rows))

    def height_rows_stats_l1(self, band: str, cls: str, zone_rows_l0: int, R: int, l1_rows: int) -> Optional[Tuple[int, int, int]]:
        stats = self._stat_dict(band, cls, "height_px")
        if not stats:
            return None
        vals: List[int] = []
        zone_height_px = zone_rows_l0 * self.px_per_tile
        for key in ("p25", "p50", "p75"):
            v = stats.get(key)
            if not isinstance(v, (int, float)):
                return None
            v = min(max(v, self.px_per_tile), zone_height_px)
            rows_l0 = v / self.px_per_tile
            vals.append(max(1, min(int(round(rows_l0 * R)), l1_rows)))
        return tuple(vals)

    def expected_count(self, band: str, cls: str, zone_cols_l0: int) -> Optional[int]:
        px = self.width_px(band, cls)
        if px is None or px <= 0 or self.px_per_tile <= 0:
            return None
        zone_width_px = zone_cols_l0 * self.px_per_tile
        count = max(1, int(round(zone_width_px / px)))
        return count


def qa_band_coverage(children: List[dict], zones: List[dict]) -> List[str]:
    warnings: List[str] = []
    band_spans: Dict[str, Tuple[int, int]] = {}
    for z in zones:
        band = z.get("band")
        if not band:
            continue
        g = z.get("grid", {})
        c0 = int(g.get("col", 0))
        span = int(g.get("col_span", 0))
        left, right = band_spans.get(band, (c0, c0))
        left = min(left, c0)
        right = max(right, c0 + span)
        band_spans[band] = (left, right)
    for band in ("Nav", "Footer"):
        band_children = [c for c in children if c.get("band") == band]
        if not band_children:
            warnings.append(f"{band}: no children generated")
            continue
        span_info = band_spans.get(band)
        if not span_info:
            continue
        denominator = max(1, span_info[1] - span_info[0])
        covered_cols = set()
        for c in band_children:
            shadow = c.get("grid_l0_shadow", {})
            start = int(shadow.get("col", 0))
            span = int(shadow.get("col_span", 0))
            for idx in range(start, start + span):
                covered_cols.add(idx)
        coverage = len(covered_cols) / float(denominator)
        threshold = 0.85 if band == "Nav" else 0.7
        if coverage < threshold:
            warnings.append(f"{band}: horizontal coverage low ({coverage:.2f})")
        if band == "Nav":
            count = sum(1 for c in band_children if c.get("class") == "text")
            if not (4 <= count <= 8):
                warnings.append(f"{band}: unexpected link count ({count})")
    card_count = sum(1 for c in children if c.get("class") == "card")
    if card_count > 12:
        warnings.append(f"Main: too many cards ({card_count})")
    return warnings


# -------------- Main band balancing -------------

def rebalance_main(children: List[dict], logs: List[dict], target_ratio: float = 0.35) -> None:
    main_indices = [i for i, ch in enumerate(children) if ch.get("band") == "Main"]
    if not main_indices:
        return
    areas = [(i, children[i]["grid_l0_shadow"]["row_span"] * children[i]["grid_l0_shadow"]["col_span"])
             for i in main_indices]
    total_area = sum(area for _, area in areas)
    if total_area <= 0:
        return
    media_entries = [(area, i) for i, area in areas if children[i]["class"] == "media"]
    if not media_entries:
        return
    media_entries.sort(key=lambda t: t[0])
    media_area = sum(area for area, _ in media_entries)
    target_area = total_area * target_ratio
    min_area = total_area * max(0.0, target_ratio - 0.05)
    kept = len(media_entries)
    for area, idx in media_entries:
        if media_area <= target_area or kept <= 1:
            break
        if media_area - area < min_area:
            break
        children[idx]["class"] = "text"
        logs[idx]["class"] = "text"
        media_area -= area
        kept -= 1


# -------------- Window helpers -------------

def slice_zone_windows(zone: dict, viewport_rows: int, stride_frac: float) -> List[dict]:
    """Legacy; unused in band-wide windowing."""
    return [zone]


def score_children_l1(kids: List[dict], l1_rows: int, l1_cols: int, target_media_ratio: float = 0.35) -> Tuple[float, float]:
    covered = set()
    media_area = 0
    text_area = 0
    for ch in kids:
        rs = int(ch.get("rs", ch.get("row_span", 1)))
        cs = int(ch.get("cs", ch.get("col_span", 1)))
        r0 = int(ch.get("r", ch.get("row", 0)))
        c0 = int(ch.get("c", ch.get("col", 0)))
        for r in range(r0, r0 + rs):
            if 0 <= r < l1_rows:
                for c in range(c0, c0 + cs):
                    if 0 <= c < l1_cols:
                        covered.add((r, c))
        area = rs * cs
        if ch.get("class") == "media":
            media_area += area
        elif ch.get("class") == "text":
            text_area += area
    coverage = len(covered) / float(max(1, l1_rows * l1_cols))
    media_ratio = media_area / float(media_area + text_area + 1e-6)
    balance = -abs(media_ratio - target_media_ratio)
    return coverage, balance


def compress_vertical(children: List[dict], band_rows: int) -> None:
    """
    Pack children upward within their band while respecting lock_y.
    Uses L0 shadows; mutates in place.
    """
    # sort by current row, then col
    movable = [ch for ch in children if "lock_y" not in ch.get("locks", [])]
    fixed = [ch for ch in children if "lock_y" in ch.get("locks", [])]
    movable.sort(key=lambda ch: (ch["grid_l0_shadow"]["row"], ch["grid_l0_shadow"]["col"]))
    occupied: List[Tuple[int,int,int,int]] = []
    for ch in fixed:
        g = ch["grid_l0_shadow"]
        occupied.append((g["row"], g["col"], g["row_span"], g["col_span"]))
    def collides(r, c, rs, cs) -> bool:
        for ro, co, rso, cso in occupied:
            if not (r + rs <= ro or ro + rso <= r or c + cs <= co or co + cso <= c):
                return True
        return False
    for ch in movable:
        g = ch["grid_l0_shadow"]
        r, c, rs, cs = g["row"], g["col"], g["row_span"], g["col_span"]
        target = r
        while target > 0 and not collides(target - 1, c, rs, cs):
            target -= 1
        g["row"] = max(0, target)
        occupied.append((g["row"], c, rs, cs))
    # clamp inside band
    for ch in children:
        g = ch["grid_l0_shadow"]
        if g["row"] + g["row_span"] > band_rows:
            g["row"] = max(0, band_rows - g["row_span"])


# -------------- Pattern fillers -------------

def fill_pill_row(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    cfg = pat.get("children", [{}])[0]
    row_h = int(cfg.get("rows_l1", 2))
    min_w = int(cfg.get("min_cols_l1", 3))
    gap_c = int(cfg.get("gap_cols_l1", 2))
    max_items = int(cfg.get("max_items", 8))
    desired = max(3, min(7, l1_cols // max(2, min_w)))
    desired += rng.randint(-1, 1)
    desired = clamp(desired, 3, max_items)
    available = max(1, l1_cols - max(0, desired - 1) * gap_c)
    base_w = max(min_w, available // desired)
    remainder = available - base_w * desired
    r = max(0, (l1_rows - row_h) // 2)
    items: List[dict] = []
    c = 0
    for idx in range(desired):
        if c >= l1_cols:
            break
        w = base_w
        if remainder > 0:
            w += 1
            remainder -= 1
        w = clamp(w, min_w, max(1, l1_cols - c))
        items.append({"class": "role_button", "r": r, "c": c, "rs": row_h, "cs": w})
        c += w + gap_c
    if not items:
        items.append({"class": "role_button", "r": r, "c": 0, "rs": row_h, "cs": max(min_w, l1_cols)})
    return items

def fill_utility_row(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    # one input and one button aligned right
    inp_w = int(pat["children"][0].get("cols_l1", 8))
    inp_h = int(pat["children"][0].get("rows_l1", 2))
    btn_w = int(pat["children"][1].get("cols_l1", 4))
    btn_h = int(pat["children"][1].get("rows_l1", 2))
    pad   = 1
    r = 0
    items = []
    # input left
    items.append({"class":"input","r":r,"c":pad,"rs":inp_h,"cs":min(inp_w, l1_cols//2)})
    # button right
    rb_c = max(pad, l1_cols - btn_w - pad)
    items.append({"class":"role_button","r":r,"c":rb_c,"rs":btn_h,"cs":btn_w})
    return items

def fill_menu_row(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    txt_h = int(pat["children"][0].get("rows_l1", 1))
    min_w = int(pat["children"][0].get("min_cols_l1", 2))
    gap   = int(pat["children"][0].get("gap_cols_l1", 1))
    max_items = int(pat["children"][0].get("max_items", 8))
    desired = clamp((l1_cols // max(3, min_w)) + rng.randint(-1, 1), 4, max_items)
    total_gap = max(0, desired - 1) * gap
    available = max(1, l1_cols - total_gap)
    base_w = max(min_w, available // desired)
    remainder = available - base_w * desired

    r = max(0, (l1_rows - txt_h) // 2)
    items: List[dict] = []
    c = 0
    for idx in range(desired):
        if c >= l1_cols:
            break
        w = base_w
        if remainder > 0:
            w += 1
            remainder -= 1
        w = clamp(w, min_w, max(1, l1_cols - c))
        items.append({"class": "text", "r": r, "c": c, "rs": txt_h, "cs": w})
        c += w + gap
    # optional right CTA
    if len(pat["children"]) > 1 and pat["children"][1].get("class") == "role_button":
        bw = int(pat["children"][1].get("cols_l1", 4))
        bh = int(pat["children"][1].get("rows_l1", 2))
        c = max(0, l1_cols - bw)
        items.append({"class":"role_button","r":0,"c":c,"rs":bh,"cs":bw})
    return items

def fill_media_single(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    min_w = int(pat["children"][0].get("min_cols_l1", 6))
    min_h = int(pat["children"][0].get("min_rows_l1", 6))
    w = clamp(min_w, 1, l1_cols)
    h = clamp(min_h, 1, l1_rows)
    # center-ish
    r = max(0, (l1_rows - h)//2); c = max(0, (l1_cols - w)//2)
    return [{"class":"media","r":r,"c":c,"rs":h,"cs":w}]

def fill_hero(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    # Harmonize hero toward feature-card proportions: fixed media left, generous plate
    media_w = clamp(int(l1_cols * 0.45), 12, max(14, l1_cols - 10))
    media_h = clamp(int(l1_rows * 0.60), 8, max(10, l1_rows - 6))
    m_r = clamp((l1_rows - media_h) // 2, 0, max(0, l1_rows - media_h))
    m_c = 0
    items = [{"class": "media", "r": m_r, "c": m_c, "rs": media_h, "cs": media_w}]
    # text lines
    line_h = int(pat["children"][1].get("line_rows_l1", 2))
    min_lines = int(pat["children"][1].get("min_lines", 4))
    max_lines = int(pat["children"][1].get("max_lines", 6))
    gap_cols = 2
    text_c0 = media_w + gap_cols
    text_width = clamp(l1_cols - media_w - gap_cols, 10, l1_cols - gap_cols)
    lines = clamp(rng.randint(min_lines, max_lines), min_lines, max_lines)
    total_lines_height = lines * line_h + (lines - 1)
    r = max(0, (l1_rows - total_lines_height - 3) // 2)
    for _ in range(lines):
        if r + line_h > l1_rows: break
        items.append({"class": "text", "r": r, "c": text_c0, "rs": line_h, "cs": text_width})
        r += line_h + 1
    cta_h = min(3, max(2, line_h))
    cta_width = clamp(text_width // 2, 10, text_width)
    cta_row = clamp(r + 1, 0, max(0, l1_rows - cta_h))
    cta_col = text_c0 + max(0, (text_width - cta_width) // 2)
    items.append({"class": "role_button", "r": cta_row, "c": cta_col, "rs": cta_h, "cs": cta_width})
    return items

def ensure_media_bounds(media_height: int, txt_height: int, btn_height: int, media_w: int, text_w: int,
                        card_h: int, min_h: int, max_h: int, min_share: float = 0.4, max_share: float = 0.6):
    """
    Keep media/text within a reasonable share of the card and clamp heights.
    """
    media_height = clamp(media_height, min_h, max_h)
    text_height = max(1, txt_height)
    btn_local = btn_height
    media_area = media_height * media_w
    text_area = text_height * text_w + btn_local * text_w
    total = max(1, media_area + text_area)
    share = media_area / total
    if share > max_share:
        target_area = max_share * total
        desired_h = max(1, int(target_area / max(1, media_w)))
        media_height = clamp(desired_h, min_h, max_h)
    elif share < min_share:
        target_area = min_share * total
        desired_h = max(1, int(math.ceil(target_area / max(1, media_w))))
        media_height = clamp(desired_h, min_h, max_h)
    remaining = max(1, card_h - media_height - 1)
    text_height = min(text_height, remaining - (1 if btn_local else 0))
    if text_height < 1:
        text_height = max(1, remaining)
        btn_local = 0
    return media_height, text_height, btn_local

def fill_card_grid(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    p = pat["children"][0]
    gap_c = int(p.get("gap_cols_l1", 2))
    gap_r = int(p.get("gap_rows_l1", 2))

    target_cov = 0.75
    # Card size bounds for cohesive real-world proportions
    min_card_w = max(24, int(l1_cols * 0.35))
    max_card_w = max(min_card_w, int(l1_cols * 0.65))
    min_card_h = max(30, int(l1_rows * 0.30))
    max_card_h = max(min_card_h, int(l1_rows * 0.60))
    # Gutters: keep a soft inset so cards don’t hug edges
    inner_gutter = max(1, gap_c)
    layouts = [(2, 2), (1, 2), (1, 3), (2, 1)]
    candidates = []
    for rows, cols in layouts:
        slots = rows * cols
        if slots > 4:
            continue
        avail_c = l1_cols - max(0, cols - 1) * gap_c
        avail_r = l1_rows - max(0, rows - 1) * gap_r
        if avail_c <= 0 or avail_r <= 0:
            continue
        base_w = clamp(avail_c // cols, min_card_w, min(max_card_w, avail_c))
        base_h = clamp(avail_r // rows, min_card_h, min(max_card_h, avail_r))
        if base_w < min_card_w or base_h < min_card_h:
            continue
        card_w = clamp(base_w, min_card_w, avail_c)
        # choose height to hit target coverage with this slot count
        target_h = int(round(target_cov * l1_rows * l1_cols / float(max(1, slots * card_w))))
        card_h = clamp(target_h, min_card_h, min(base_h, max_card_h))
        cov = (slots * card_w * card_h) / float(max(1, l1_rows * l1_cols))
        candidates.append(((abs(target_cov - cov), -cov), rows, cols, card_w, card_h, avail_c, avail_r))

    if not candidates:
        # Fallback: single large card centered
        card_w = clamp(l1_cols, min_card_w, max_card_w)
        card_h = clamp(int(l1_rows * 0.6), min_card_h, max_card_h)
        rows = cols = 1
        avail_c, avail_r = l1_cols, l1_rows
    else:
        candidates.sort(key=lambda t: (t[0][0], t[0][1]))
        pick = min(len(candidates) - 1, rng.randint(0, min(2, len(candidates) - 1)))
        _, rows, cols, card_w, card_h, avail_c, avail_r = candidates[pick]

    total_gap_c = max(0, cols - 1) * gap_c
    total_gap_r = max(0, rows - 1) * gap_r
    avail_c = max(1, min(avail_c, l1_cols - total_gap_c - inner_gutter * 2))
    avail_r = max(1, min(avail_r, l1_rows - total_gap_r))
    base_w = clamp(card_w, min_card_w, min(max_card_w, avail_c // cols if cols else avail_c))
    base_h = clamp(card_h, min_card_h, min(max_card_h, avail_r // rows if rows else avail_r))
    leftover_c = max(0, avail_c - base_w * cols)
    leftover_r = max(0, avail_r - base_h * rows)

    # Randomize how leftover width/height is distributed to avoid identical cards.
    col_widths = [base_w] * cols
    if leftover_c > 0 and cols > 0:
        take = min(leftover_c, cols)
        for idx in rng.sample(range(cols), take):
            col_widths[idx] += 1
    jitter_w = max(1, base_w // 8)
    for _ in range(cols):
        a, b = rng.sample(range(cols), 2) if cols > 1 else (0, 0)
        delta = rng.randint(-jitter_w, jitter_w)
        if delta > 0 and col_widths[a] - delta >= min_card_w and col_widths[b] + delta <= min(max_card_w, avail_c):
            col_widths[a] -= delta
            col_widths[b] += delta
        elif delta < 0 and col_widths[a] - delta <= min(max_card_w, avail_c) and col_widths[b] + delta >= min_card_w:
            col_widths[a] -= delta
            col_widths[b] += delta
    col_widths = [clamp(w, min_card_w, min(max_card_w, avail_c)) for w in col_widths]
    total_w = sum(col_widths) + max(0, cols - 1) * gap_c
    if total_w > avail_c and cols > 0:
        over = total_w - avail_c
        for idx in sorted(range(cols), key=lambda i: col_widths[i], reverse=True):
            if over <= 0:
                break
            reducible = max(0, col_widths[idx] - min_card_w)
            reduce_by = min(reducible, over)
            col_widths[idx] -= reduce_by
            over -= reduce_by
    total_w = sum(col_widths) + max(0, cols - 1) * gap_c
    if total_w < avail_c and cols > 0:
        col_widths[-1] = clamp(col_widths[-1] + (avail_c - total_w), min_card_w, min(max_card_w, avail_c))

    row_heights = [base_h] * rows
    if leftover_r > 0 and rows > 0:
        take = min(leftover_r, rows)
        for idx in rng.sample(range(rows), take):
            row_heights[idx] += 1
    jitter_h = max(1, base_h // 8)
    for _ in range(rows):
        if rows < 2:
            break
        a, b = rng.sample(range(rows), 2)
        delta = rng.randint(-jitter_h, jitter_h)
        if delta > 0 and row_heights[a] - delta >= min_card_h and row_heights[b] + delta <= min(max_card_h, avail_r):
            row_heights[a] -= delta
            row_heights[b] += delta
        elif delta < 0 and row_heights[a] - delta <= min(max_card_h, avail_r) and row_heights[b] + delta >= min_card_h:
            row_heights[a] -= delta
            row_heights[b] += delta
    row_heights = [clamp(h, min_card_h, min(max_card_h, avail_r)) for h in row_heights]
    total_h = sum(row_heights) + max(0, rows - 1) * gap_r
    if total_h > avail_r and rows > 0:
        over = total_h - avail_r
        for idx in sorted(range(rows), key=lambda i: row_heights[i], reverse=True):
            if over <= 0:
                break
            reducible = max(0, row_heights[idx] - min_card_h)
            reduce_by = min(reducible, over)
            row_heights[idx] -= reduce_by
            over -= reduce_by
    total_h = sum(row_heights) + max(0, rows - 1) * gap_r
    if total_h < avail_r and rows > 0:
        row_heights[-1] = clamp(row_heights[-1] + (avail_r - total_h), min_card_h, min(max_card_h, avail_r))

    items: List[dict] = []
    y = 0
    slots = min(rows * cols, 4)
    buttons_budget = rng.randint(1, 2)
    slot_idx = 0
    for r_idx in range(rows):
        if slot_idx >= slots:
            break
        this_h = row_heights[r_idx]
        x = 0
        for c_idx in range(cols):
            if slot_idx >= slots:
                break
            this_w = col_widths[c_idx]
            if x + this_w > l1_cols or y + this_h > l1_rows:
                break
            style = rng.random()
            media_first = style < 0.5
            side_by_side = 0.2 < style < 0.45
            media_h = clamp(int(this_h * (0.6 if media_first else 0.55)), max(12, int(this_h * 0.5)), max(10, this_h - 3))
            text_h = max(2, this_h - media_h - 1)
            btn_h = 0
            if this_h - media_h - text_h >= 2 and buttons_budget > 0 and rng.random() < 0.5:
                btn_h = 1
                buttons_budget -= 1

            if side_by_side and this_w >= min_card_w + 4:
                min_text_w = max(8, int(this_w * 0.35))
                media_w = clamp(int(this_w * 0.55), min_card_w, max(min_card_w, this_w - min_text_w - max(1, gap_c // 2)))
                text_w = max(min_text_w, this_w - media_w - max(1, gap_c // 2))
                if text_w < min_text_w:
                    # fallback to stacked if we can’t keep a healthy text width
                    side_by_side = False
                else:
                    media_h = clamp(
                        media_h,
                        int(media_w * 0.56),  # ~16:9 min
                        min(int(media_w * 0.9), this_h - 2),
                    )
                    media_h, text_h, btn_h = ensure_media_bounds(
                        media_h, text_h, btn_h, media_w, text_w, this_h,
                        min_h=int(media_w * 0.56),
                        max_h=min(int(media_w * 0.9), this_h - 2),
                        min_share=0.4,
                        max_share=0.6,
                    )
                    text_h = clamp(text_h, 2, min(6, this_h - media_h - (1 if btn_h else 0)))
                    media_c = x if rng.random() < 0.5 else x + (this_w - media_w)
                    text_c = x + (media_w + max(1, gap_c // 2) if media_c == x else 0)
                    items.append({"class": "media", "r": y, "c": media_c, "rs": media_h, "cs": media_w})
                    items.append({"class": "text", "r": y, "c": text_c, "rs": text_h, "cs": text_w})
                    if btn_h:
                        btn_row = y + text_h + 1
                        btn_w = clamp(int(text_w * 0.6), 6, text_w)
                        btn_c = text_c + max(0, (text_w - btn_w) // 2)
                        if btn_row + btn_h <= y + this_h:
                            items.append({"class": "role_button", "r": btn_row, "c": btn_c, "rs": btn_h, "cs": btn_w})
            else:
                media_h = clamp(media_h + rng.randint(-2, 2), max(12, int(this_h * 0.5)), max(10, this_h - 3))
                media_h = clamp(
                    media_h,
                    int(this_w * 0.56),
                    min(int(this_w * 0.9), this_h - 2),
                )
                text_h = max(2, this_h - media_h - 1)
                media_h, text_h, btn_h = ensure_media_bounds(
                    media_h, text_h, btn_h, this_w, this_w, this_h,
                    min_h=int(this_w * 0.56),
                    max_h=min(int(this_w * 0.9), this_h - 2),
                    min_share=0.4,
                    max_share=0.6,
                )
                text_h = clamp(text_h, 2, min(6, this_h - media_h - (1 if btn_h else 0)))
                items.append({"class": "media", "r": y, "c": x, "rs": media_h, "cs": this_w})
                text_r = y + media_h + 1
                items.append({"class": "text", "r": text_r, "c": x, "rs": text_h, "cs": this_w})
                if btn_h:
                    btn_row = text_r + text_h + 1
                    btn_w = clamp(int(this_w * 0.55), 6, this_w)
                    btn_c = x + max(0, (this_w - btn_w) // 2)
                    if btn_row + btn_h <= y + this_h:
                        items.append({"class": "role_button", "r": btn_row, "c": btn_c, "rs": btn_h, "cs": btn_w})
            x += this_w + gap_c
            slot_idx += 1
        y += this_h + gap_r

    return items

def fill_feature_rail(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    p_media = pat["children"][0]
    p_text  = pat["children"][1]
    mw = int(p_media.get("cols_l1", 22))
    mh = int(p_media.get("rows_l1", 16))
    gap_c = int(p_media.get("gap_cols_l1", 3))
    th = int(p_text.get("rows_l1", 3))
    gap_r = int(p_text.get("gap_rows_l1", 1))
    btn_h = min(3, max(2, int(th * 0.75)))
    max_tiles_cfg = int(p_media.get("max_tiles", 3))
    # inset to avoid hugging edges
    inner_gutter = max(1, gap_c)
    max_tiles = clamp((l1_cols - inner_gutter * 2 + gap_c) // max(1, mw + gap_c), 1, max_tiles_cfg)
    max_tiles = min(max_tiles, 3)
    total_gap = max(0, max_tiles - 1) * gap_c
    available = max(1, l1_cols - total_gap - inner_gutter * 2)
    mw = clamp(mw, max(8, int(l1_cols * 0.18)), max(18, int(l1_cols * 0.5)))
    mw = min(mw, max(6, available // max_tiles))
    remainder = available - mw * max_tiles
    items = []
    c = inner_gutter
    r_media = 0
    tiles_placed = 0
    cta_budget = rng.randint(1, 2)
    while tiles_placed < max_tiles and c < l1_cols - inner_gutter:
        this_w = mw
        if remainder > 0:
            this_w += 1
            remainder -= 1
        if c + this_w > l1_cols - inner_gutter:
            break
        media_h = clamp(mh, max(8, int(l1_rows * 0.4)), max(14, int(l1_rows * 0.65)))
        items.append({"class":"media","r":r_media,"c":c,"rs":media_h,"cs":this_w})
        rt = r_media + media_h + gap_r
        txt_h = clamp(th, 2, 5)
        if rt + txt_h <= l1_rows:
            items.append({"class":"text","r":rt,"c":c,"rs":txt_h,"cs":this_w})
            r_btn = rt + txt_h + 1
            if r_btn + btn_h <= l1_rows and cta_budget > 0 and rng.random() < 0.6:
                items.append({"class":"role_button","r":r_btn,"c":c,"rs":btn_h,"cs":min(this_w, max(8, this_w // 2))})
                cta_budget -= 1
        c += this_w + gap_c
        tiles_placed += 1
    return items

def fill_promo_rail(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    p = pat["children"][0]
    w = int(p.get("cols_l1", 8))
    h = int(p.get("rows_l1", 8))
    gap_c = int(p.get("gap_cols_l1", 2))
    items = []
    max_tiles_cfg = int(p.get("max_tiles", 4))
    max_tiles = clamp((l1_cols + gap_c) // max(1, w + gap_c), 2, max_tiles_cfg)
    total_gap = max(0, max_tiles - 1) * gap_c
    available = max(1, l1_cols - total_gap)
    tile_w = max(w, available // max_tiles)
    tile_w = min(tile_w, available)
    remainder = available - tile_w * max_tiles
    c = 0
    for idx in range(max_tiles):
        if c >= l1_cols:
            break
        this_w = tile_w
        if remainder > 0:
            this_w += 1
            remainder -= 1
        cls = "media" if idx % 2 == 0 else "text"
        items.append({"class": cls, "r": 0, "c": c, "rs": h, "cs": this_w})
        c += this_w + gap_c
    return items

def fill_columns(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    p = pat["children"][0]
    col_w = int(p.get("cols_per_col_l1", 10))
    gap_c = int(p.get("gap_cols_l1", 2))
    item_h = int(p.get("item_rows_l1", 2))
    title_h = int(p.get("title_rows_l1", 3))
    min_items = int(p.get("min_items_per_col", 3))
    max_items_cfg = int(p.get("max_items_per_col", max(3, min_items)))
    max_cols = clamp((l1_cols + gap_c) // max(1, col_w + gap_c), 3, 5)
    total_gap = max(0, max_cols - 1) * gap_c
    available = max(1, l1_cols - total_gap)
    col_width = max(col_w, available // max_cols)
    col_width = min(col_width, available)
    remainder = available - col_width * max_cols

    items: List[dict] = []
    c = 0
    for col_idx in range(max_cols):
        if c >= l1_cols:
            break
        this_w = col_width
        if remainder > 0:
            this_w += 1
            remainder -= 1
        r = 0
        items.append({"class": "text", "r": r, "c": c, "rs": title_h, "cs": this_w})
        r += title_h + 1
        max_items = clamp(min_items + rng.randint(-1, 2), min_items, max_items_cfg)
        for _ in range(max_items):
            if r + item_h > l1_rows:
                break
            items.append({"class": "text", "r": r, "c": c, "rs": item_h, "cs": this_w})
            r += item_h + 1
        c += this_w + gap_c
    return items

def fill_legal(zone, pat, rng: random.Random) -> List[dict]:
    z = zone["grid"]; R = int(zone["R"])
    l1_rows, l1_cols = l1_dims(z, R)
    # a baseline row of text with tiny gaps, maybe a small media on far right
    txt = []
    min_w = int(pat["children"][0].get("min_cols_l1", 2))
    gap = int(pat["children"][0].get("gap_cols_l1", 1))
    max_items_cfg = int(pat["children"][0].get("max_items", 6))
    max_items = clamp((l1_cols // (min_w + gap)), 3, max_items_cfg)
    total_gap = max(0, max_items - 1) * gap
    available = max(1, l1_cols - total_gap)
    base_w = max(min_w, available // max_items)
    remainder = available - base_w * max_items
    c = 0
    for idx in range(max_items):
        this_w = base_w
        if remainder > 0:
            this_w += 1
            remainder -= 1
        txt.append({"class": "text", "r": 0, "c": c, "rs": 1, "cs": this_w})
        c += this_w + gap
    items = txt
    if len(pat.get("children",[]))>1 and pat["children"][1].get("class")=="media":
        w = int(pat["children"][1].get("cols_l1", 2))
        h = int(pat["children"][1].get("rows_l1", 2))
        items.append({"class":"media","r":0,"c":max(0, l1_cols - w),"rs":h,"cs":w})
    return items

FILLERS = {
    "pill_row":      fill_pill_row,
    "utility_row":   fill_utility_row,
    "menu_row":      fill_menu_row,
    "media_single":  fill_media_single,
    "hero":          fill_hero,
    "card_grid":     fill_card_grid,
    "feature_rail":  fill_feature_rail,
    "promo_rail":    fill_promo_rail,
    "columns":       fill_columns,
    "legal":         fill_legal,
}

# -------------- Overlay drawing -------------

def draw_overlay(manifest: dict, zones: List[dict], children: List[dict], scale: int, macro_grid: Optional[dict] = None, main_viewports: Optional[List[Tuple[int,int]]] = None, main_viewport_labels: Optional[List[str]] = None) -> Image.Image:
    bands = [b["name"] for b in manifest.get("bands",[])]
    rows_map = {b["name"]: b["rows0"] for b in manifest.get("bands",[])}
    tiles_x  = int(manifest.get("tiles_x", 40))
    total_rows = sum(rows_map.values())
    img_w = tiles_x*scale
    img_h = total_rows*scale
    img = Image.new("RGBA", (img_w, img_h), (0,0,0,255))
    drw = ImageDraw.Draw(img)

    # Precompute Main zone spans for viewport labeling
    main_zone_spans: List[Tuple[int,int,dict]] = []
    for z in zones:
        if z.get("band") != "Main":
            continue
        g = z.get("grid", {})
        r0 = int(g.get("row", 0))
        r1 = r0 + int(g.get("row_span", 0))
        main_zone_spans.append((r0, r1, z))
    main_zone_spans.sort(key=lambda t: (t[0], t[1]))

    # helper to get band top in px
    def band_top(bname: str) -> int:
        y = 0
        for b in bands:
            if b == bname:
                return y
            y += rows_map[b]*scale
        return 0

    # band separators
    y0 = 0
    for b in bands:
        r = rows_map[b]
        drw.rectangle([0, y0, tiles_x*scale-1, y0 + r*scale - 1], outline=(120,0,0,150))
        y0 += r*scale

    # design grid overlay (macro grid + gutters) for snapping visualization
    if macro_grid:
        gutter_tiles = int(macro_grid.get("tiles_gutter_each_side", 0) or 0)
        macro_cols = int(macro_grid.get("macro_cols", 0) or 0)
        tiles_per_macro_col = int(macro_grid.get("tiles_per_macro_col", 0) or 0)
        tiles_per_macro_row = int(macro_grid.get("tiles_per_macro_row", 0) or 0)
        macro_band_rows = (macro_grid.get("bands") or {})
        # vertical guides across canvas (12-col style if configured)
        if macro_cols > 0 and tiles_per_macro_col > 0:
            x_start = gutter_tiles * scale
            guide_col = (90, 90, 90, 110)
            for i in range(macro_cols + 1):
                x = x_start + i * tiles_per_macro_col * scale
                drw.line([(x, 0), (x, img_h - 1)], fill=guide_col)
            if gutter_tiles > 0:
                gutter_col = (110, 110, 110, 140)
                gx0 = gutter_tiles * scale
                gx1 = (tiles_x - gutter_tiles) * scale
                drw.line([(gx0, 0), (gx0, img_h - 1)], fill=gutter_col)
                drw.line([(gx1, 0), (gx1, img_h - 1)], fill=gutter_col)
        # horizontal guides per band using macro row step across full band height
        if tiles_per_macro_row > 0:
            y_cursor = 0
            guide_row = (80, 80, 80, 110)
            for b in bands:
                band_rows = rows_map[b]
                mr = tiles_per_macro_row
                y = y_cursor + mr * scale
                while y < y_cursor + band_rows * scale:
                    drw.line([(0, y), (tiles_x*scale - 1, y)], fill=guide_row)
                    y += mr * scale
                y_cursor += band_rows * scale

    window_outlines: List[Tuple[int,int,int,int,str]] = []
    # draw zones
    for z in zones:
        b = z["band"]; g = z["grid"]
        x0 = g["col"]*scale
        y0 = band_top(b) + g["row"]*scale
        x1 = x0 + g["col_span"]*scale - 1
        y1 = y0 + g["row_span"]*scale - 1
        # Red outline for bands; yellow outline for viewport windows in Main with specific labels
        zone_outline = (200,40,40,180)
        zone_fill = (200,40,40,30)
        label = z.get("pattern","zone")
        if z.get("band") == "Main" and ("_win" in z.get("id","") or "_vp" in z.get("id","")):
            zone_outline = (250,220,40,220)
            zone_fill = (250,220,40,40)
            # label by pattern: hero/content/callouts
            pat = z.get("pattern","").lower()
            base = "Main"
            if "hero" in pat:
                base = "Hero"
            elif "content" in pat or "card" in pat:
                base = "Content"
            elif "callout" in pat or "feature" in pat:
                base = "Callouts"
            try:
                if "_win" in z.get("id",""):
                    suffix = z.get("id","").split("_win")[-1]
                else:
                    suffix = z.get("id","").split("_vp")[-1]
                label = f"{base}{int(suffix)+1}"
            except Exception:
                label = f"{base} {z.get('id','')}"
            window_outlines.append((x0,y0,x1,y1,label))
        drw.rectangle([x0,y0,x1,y1], outline=zone_outline, fill=zone_fill)
        if not (z.get("band") == "Main" and ("_win" in z.get("id","") or "_vp" in z.get("id",""))):
            drw.text((x0+3,y0+3), label, fill=(220,40,40,220))

    # draw children (using L0 shadow spans for clarity)
    for ch in children:
        b = ch["band"]; g0 = ch["grid_l0_shadow"]
        x0 = g0["col"]*scale
        y0 = band_top(b) + g0["row"]*scale
        x1 = x0 + g0["col_span"]*scale - 1
        y1 = y0 + g0["row_span"]*scale - 1
        if ch["class"] == "media":
            col = (60,120,220,220); fill = (60,120,220,140)
        elif ch["class"] == "text":
            col = (230,140,60,220); fill = (230,140,60,140)
        else:
            col = (200,200,200,200); fill = (200,200,200,100)
        drw.rectangle([x0,y0,x1,y1], outline=col, fill=fill)

    # redraw window outlines on top for visibility, adjust labels to avoid overlaps where possible
    for x0,y0,x1,y1,label in window_outlines:
        drw.rectangle([x0,y0,x1,y1], outline=(250,220,40,255), width=2)
        drw.text((x0+3,y0+3), label, fill=(250,220,40,255))

    # Draw viewport markers for Main on a separate top layer so they always stay above zones/children.
    if main_viewports:
        vp_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        vp_drw = ImageDraw.Draw(vp_layer)
        y_base = band_top("Main")
        x0 = 0
        x1 = tiles_x * scale - 1
        for idx, (r0, r1) in enumerate(main_viewports):
            y0 = y_base + r0 * scale
            y1 = y_base + r1 * scale - 1
            # pick label by dominant overlapping Main zone (fallback to viewportN) unless provided
            if main_viewport_labels and idx < len(main_viewport_labels):
                best_label = main_viewport_labels[idx]
            else:
                best_label = f"viewport{idx+1}"
                best_overlap = 0
                best_row0 = None
                for zr0, zr1, z in main_zone_spans:
                    overlap = max(0, min(r1, zr1) - max(r0, zr0))
                    if overlap > best_overlap or (overlap == best_overlap and overlap > 0 and (best_row0 is None or zr0 < best_row0)):
                        best_overlap = overlap
                        best_row0 = zr0
                        lbl = z.get("pattern") or z.get("id") or best_label
                        best_label = str(lbl)
            vp_drw.rectangle([x0, y0, x1, y1], outline=(250,220,40,255), width=2)
            vp_drw.text((x0 + 3, y0 + 3), best_label, fill=(250,220,40,255))
        img = Image.alpha_composite(img, vp_layer)
    return img

# -------------- Main pipeline -------------

def main():
    global GEOM_GUIDE, PX_PER_TILE, CANVAS_WIDTH, QA_WARNINGS
    QA_WARNINGS = []
    args = parse_args()
    manifest_path = Path(args.manifest)
    zones_path = Path(args.zones)
    manifest = load_json(manifest_path)
    zones     = load_json(zones_path)
    patterns  = load_json(Path(args.patterns))
    macro_grid = load_json(Path(args.macro_grid)) if args.macro_grid else {}
    # optional per-pattern denoise targets
    denoise_targets = {}
    denoise_path = Path("wfc_stats/denoise_targets.json")
    if denoise_path.exists():
        try:
            denoise_targets = load_json(denoise_path)
        except Exception:
            denoise_targets = {}
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    debugdir = Path(args.debugdir); debugdir.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(manifest.get("seed", 0))
    rng = random.Random(seed)

    tiles_x = int(manifest.get("tiles_x", 40))
    png_scale = int(manifest.get("png_scale", getattr(args, "png_scale", 8)))
    canvas_width = tiles_x * png_scale
    CANVAS_WIDTH = canvas_width
    PX_PER_TILE = canvas_width / float(tiles_x) if tiles_x else float(png_scale)
    gutter_cols = args.gutter_cols
    if gutter_cols is None and macro_grid:
        gutter_cols = int(macro_grid.get("tiles_gutter_each_side", 0) or 0)
    gutter_cols = max(0, int(gutter_cols or 0))

    geometry_data = None
    if args.geometry:
        geom_path = Path(args.geometry)
        if geom_path.exists():
            geometry_data = load_json(geom_path)
        else:
            print(f"[WARN] geometry file not found: {geom_path}")
    GEOM_GUIDE = GeometryGuide(geometry_data, CANVAS_WIDTH, tiles_x)
    rows_map = {b["name"]: int(b.get("rows0", 0)) for b in manifest.get("bands", [])}

    # Build quick index of zone -> pattern payload
    patt_map: Dict[str, dict] = {}
    for band, cfg in patterns.items():
        # top-level also has "globals"
        pass
    # Just keep the per-pattern dicts directly for lookup
    patt_map = {k: v for k, v in patterns.items() if k not in ("globals",)}

    # Optional slicing of Main band zones into windowed chunks (rows remain band-relative/global).
    DEFAULT_VIEWPORT_PX = 900  # approximate Chrome desktop viewport height
    viewport_rows = None
    if args.viewport_rows and args.viewport_rows > 0:
        viewport_rows = int(args.viewport_rows)
    elif args.viewport_px and args.viewport_px > 0 and PX_PER_TILE > 0:
        viewport_rows = max(1, int(round(float(args.viewport_px) / PX_PER_TILE)))
    elif args.main_slice_rows and args.main_slice_rows > 0:
        viewport_rows = int(args.main_slice_rows)
    elif args.main_slice_px and args.main_slice_px > 0 and PX_PER_TILE > 0:
        viewport_rows = max(1, int(round(float(args.main_slice_px) / PX_PER_TILE)))
    # Default viewport size if none provided: approximate Chrome desktop viewport height.
    if viewport_rows is None and PX_PER_TILE > 0:
        viewport_rows = max(1, int(round(DEFAULT_VIEWPORT_PX / PX_PER_TILE)))

    # Optional hard trim of Main to a multiple of viewport_rows; adjust manifest/zones in-memory.
    trimmed_main = False
    if viewport_rows and "Main" in rows_map and rows_map["Main"] > 0:
        main_rows_orig = rows_map["Main"]
        trimmed_rows = max(viewport_rows, (main_rows_orig // viewport_rows) * viewport_rows)
        if trimmed_rows != main_rows_orig:
            rows_map["Main"] = trimmed_rows
            for b in manifest.get("bands", []):
                if b.get("name") == "Main":
                    b["rows0"] = trimmed_rows
                    b["tiles_h"] = trimmed_rows
                    break
            new_zones = []
            main_zones = [z for z in zones if z.get("band") == "Main"]
            other_zones = [z for z in zones if z.get("band") != "Main"]
            # trim main zones that exceed trimmed height
            trimmed_main = []
            for z in sorted(main_zones, key=lambda zz: int(zz.get("grid", {}).get("row", 0))):
                g = z.get("grid", {})
                r0 = int(g.get("row", 0)); rs = int(g.get("row_span", 0))
                r1 = r0 + rs
                if r0 >= trimmed_rows:
                    continue
                if r1 > trimmed_rows:
                    rs = max(1, trimmed_rows - r0)
                    g["row_span"] = rs
                trimmed_main.append(z)
            # snap main zones to multiples of viewport_rows and make them contiguous
            required_windows = max(1, trimmed_rows // viewport_rows)
            windows = []
            for z in trimmed_main:
                rs = int(z.get("grid", {}).get("row_span", 0))
                w = max(1, int(round(rs / float(viewport_rows))))
                windows.append(w)
            total_w = sum(windows) if windows else 0
            if windows:
                diff = required_windows - total_w
                if diff != 0:
                    # adjust last zone to absorb diff
                    windows[-1] = max(1, windows[-1] + diff)
                # ensure we don't exceed required_windows
                if sum(windows) > required_windows:
                    excess = sum(windows) - required_windows
                    for i in range(len(windows)-1, -1, -1):
                        can_cut = windows[i] - 1
                        cut = min(can_cut, excess)
                        windows[i] -= cut
                        excess -= cut
                        if excess <= 0:
                            break
                # rebuild zones contiguously
                r_cursor = 0
                adjusted_main = []
                for w, z in zip(windows, trimmed_main):
                    span = w * viewport_rows
                    g = z.get("grid", {})
                    g["row"] = r_cursor
                    g["row_span"] = span
                    r_cursor += span
                    adjusted_main.append(z)
                # if we undershot, extend last zone
                if r_cursor < trimmed_rows and adjusted_main:
                    extra = trimmed_rows - r_cursor
                    adjusted_main[-1]["grid"]["row_span"] += extra
                new_zones = other_zones + adjusted_main
                trimmed_main = True
            else:
                new_zones = zones
            zones = new_zones
    # Trim all bands to their occupied height (remove trailing blank space)
    band_extents: Dict[str, int] = defaultdict(int)
    for z in zones:
        band = z.get("band")
        if not band:
            continue
        g = z.get("grid", {})
        r0 = int(g.get("row", 0))
        rs = int(g.get("row_span", 0))
        band_extents[band] = max(band_extents[band], r0 + rs)
    for b in manifest.get("bands", []):
        name = b.get("name")
        if not name:
            continue
        occupied = band_extents.get(name, 0)
        orig_rows = int(b.get("rows0", 0))
        if occupied > 0 and occupied < orig_rows:
            b["rows0"] = occupied
            b["tiles_h"] = occupied
            rows_map[name] = occupied
            # clamp zones for this band
            for z in zones:
                if z.get("band") != name:
                    continue
                g = z.get("grid", {})
                r0 = int(g.get("row", 0))
                rs = int(g.get("row_span", 0))
                r1 = r0 + rs
                if r1 > occupied:
                    g["row_span"] = max(1, occupied - r0)
    # recompute band extents after trims (used for overlay sizing)
    # If we trimmed, write out copies of the adjusted manifest/zones for downstream use.
    if trimmed_main or band_extents:
        manifest_trim = manifest_path.with_name(manifest_path.stem + "_trimmed" + manifest_path.suffix)
        zones_trim = zones_path.with_name(zones_path.stem + "_trimmed" + zones_path.suffix)
        save_json(manifest_trim, manifest)
        save_json(zones_trim, zones)

    stride_frac = max(0.1, float(getattr(args, "window_stride_frac", 0.5)))
    # Build viewport markers for Main; keep zones unchanged for placement.
    main_viewports: List[Tuple[int,int]] = []
    main_viewport_labels: List[str] = []
    main_zone_spans_labels: List[Tuple[int,int,dict]] = []
    if viewport_rows:
        main_rows_manifest = 0
        for b in manifest.get("bands", []):
            if b.get("name") == "Main":
                main_rows_manifest = int(b.get("rows0", 0))
                break
        main_rows_zones = 0
        main_rows_non_filler = 0
        for z in zones:
            if z.get("band") == "Main":
                g = z.get("grid", {})
                r0 = int(g.get("row", 0))
                rs = int(g.get("row_span", 0))
                main_rows_zones = max(main_rows_zones, r0 + rs)
                if "_filler" not in z.get("id", "") and z.get("pattern") != "filler":
                    main_rows_non_filler = max(main_rows_non_filler, r0 + rs)
                main_zone_spans_labels.append((r0, r0 + rs, z))
        # Use the larger of manifest Main height or zone extent so we always have space to draw viewports.
        main_rows = max(main_rows_manifest, main_rows_zones, main_rows_non_filler)
        if main_rows > 0:
            # zone-aware slicing: card_grid can tile into multiple windows, others single window capped at viewport_rows
            for r0, r1, z in sorted(main_zone_spans_labels, key=lambda t: (t[0], t[1])):
                span = max(0, r1 - r0)
                if span <= 0:
                    continue
                pat = (z.get("pattern") or "").lower()
                if pat == "card_grid":
                    win_count = max(1, int(math.ceil(span / max(1, viewport_rows))))
                    base_h = span // win_count
                    remainder = span % win_count
                    v0 = r0
                    for i in range(win_count):
                        h = base_h + (1 if i < remainder else 0)
                        v1 = v0 + h
                        main_viewports.append((v0, v1))
                        main_viewport_labels.append(pat or f"viewport{len(main_viewports)}")
                        v0 = v1
                elif pat == "feature_rail":
                    win_count = max(1, int(math.ceil(span / max(1, viewport_rows))))
                    base_h = span // win_count
                    remainder = span % win_count
                    v0 = r0
                    for i in range(win_count):
                        h = base_h + (1 if i < remainder else 0)
                        v1 = v0 + h
                        main_viewports.append((v0, v1))
                        main_viewport_labels.append(pat or f"viewport{len(main_viewports)}")
                        v0 = v1
                else:
                    win_count = max(1, int(math.ceil(span / max(1, viewport_rows))))
                    base_h = span // win_count
                    remainder = span % win_count
                    v0 = r0
                    for i in range(win_count):
                        h = base_h + (1 if i < remainder else 0)
                        v1 = v0 + h
                        main_viewports.append((v0, v1))
                        main_viewport_labels.append(pat or f"viewport{len(main_viewports)}")
                        v0 = v1
            # fallback to tiling if no zones or labels computed
            if not main_viewports:
                viewport_rows = max(1, min(viewport_rows, main_rows))
                v0 = 0
                if viewport_rows >= main_rows:
                    main_viewports.append((0, main_rows))
                    main_viewport_labels.append("viewport1")
                else:
                    while v0 + viewport_rows <= main_rows:
                        v1 = v0 + viewport_rows
                        main_viewports.append((v0, v1))
                        main_viewport_labels.append(f"viewport{len(main_viewports)}")
                        v0 = v1

    children: List[dict] = []
    log_rows: List[Dict[str, Any]] = []
    max_elements = {"card_grid": 10, "card_grid_dense": 10, "card_grid_sparse": 10, "hero": 8, "feature_rail": 10}
    occupied_by_band: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)

    def snap_to_grid(col_global: int, col_span: int, row_global: int, row_span: int, band_rows: int,
                     macro_col_w: int, macro_row_h: int, gutter_cols: int, tiles_x: int) -> Tuple[int,int,int,int]:
        # Clamp to gutters horizontally
        col_global = max(gutter_cols, min(col_global, tiles_x - gutter_cols - col_span))
        # Snap columns to macro grid without expanding spans (avoid overlap)
        if macro_col_w > 0:
            col_global = gutter_cols + max(0, ((col_global - gutter_cols) // macro_col_w) * macro_col_w)
            col_span = min(col_span, max(1, tiles_x - gutter_cols - col_global))
        # Vertical: snap to macro rows or midlines, but never stretch span
        row_global = max(0, min(row_global, max(0, band_rows - row_span)))
        if macro_row_h > 0:
            candidates = []
            base = (row_global // macro_row_h) * macro_row_h
            for off in (0, macro_row_h, -macro_row_h, macro_row_h // 2, -macro_row_h // 2):
                cand = max(0, min(band_rows - row_span, base + off))
                candidates.append(cand)
            candidates.append(0)
            candidates.append(max(0, band_rows - row_span))
            row_global = min(candidates, key=lambda v: abs(v - row_global))
        if row_global + row_span > band_rows:
            row_span = max(1, band_rows - row_global)
        return col_global, col_span, row_global, row_span

    def cap_items(kids: List[dict], cap: int, ensure_text: bool = False) -> List[dict]:
        if cap <= 0 or len(kids) <= cap:
            return kids
        texts = [k for k in kids if k.get("class") == "text"]
        medias = [k for k in kids if k.get("class") == "media"]
        others = [k for k in kids if k.get("class") not in ("text", "media")]
        out: List[dict] = []
        if texts:
            out.append(texts[0]); texts = texts[1:]
        if medias:
            out.append(medias[0]); medias = medias[1:]
        priority = {"text": 0, "media": 1, "role_button": 2, "button": 2, "input": 2, "select": 2}
        remaining = texts + medias + others
        remaining = sorted(remaining, key=lambda k: (priority.get(k.get("class"), 3), -(int(k.get("rs",0))*int(k.get("cs",0)))))
        for k in remaining:
            if len(out) >= cap:
                break
            out.append(k)
        # ensure at least one text if requested
        if ensure_text and not any(k.get("class") == "text" for k in out):
            if texts:
                out[-1] = texts[0]
            else:
                out[-1] = {"class": "text", "r": 0, "c": 0, "rs": 2, "cs": max(1, kids[0].get("cs",1))}
        return out

    def enforce_card_balance(kids: List[dict], win_rows: int, win_cols: int, min_media: int, max_media: int,
                             target_media_ratio: Optional[float] = None, max_text_ratio: float = 0.95) -> List[dict]:
        # Ensure media/text counts and reasonable media:text ratio; add text if missing; cap media count.
        media = [k for k in kids if k.get("class") == "media"]
        text  = [k for k in kids if k.get("class") == "text"]
        # Cap media count
        if max_media and len(media) > max_media:
            media = sorted(media, key=lambda k: -(int(k.get("rs",0))*int(k.get("cs",0))))[:max_media]
        # Ensure minimum media
        if min_media and len(media) < min_media:
            media_needed = min_media - len(media)
            for i in range(media_needed):
                media.append({"class": "media", "r": 0, "c": 0, "rs": max(4, win_rows//2), "cs": max(6, win_cols//2)})
        # Ensure at least one text
        if not text:
            text.append({"class": "text", "r": 0, "c": 0, "rs": max(2, win_rows//5), "cs": win_cols})
        # Trim text so that text area stays below media area (text:media < 1:1)
        media_area = sum(int(k.get("rs",0)) * int(k.get("cs",0)) for k in media)
        text_area = sum(int(k.get("rs",0)) * int(k.get("cs",0)) for k in text)
        if media_area > 0:
            max_text_area = int(media_area * max_text_ratio)
            excess = max(0, text_area - max_text_area)
            if excess > 0:
                text_sorted = sorted(text, key=lambda k: int(k.get("rs",1))*int(k.get("cs",1)), reverse=True)
                for t in text_sorted:
                    if excess <= 0:
                        break
                    rs = int(t.get("rs", 1)); cs = int(t.get("cs", 1))
                    reducible = max(0, rs - 1)
                    if reducible <= 0:
                        continue
                    reduce_rows = min(reducible, math.ceil(excess / max(1, cs)))
                    t["rs"] = max(1, rs - reduce_rows)
                    excess -= (rs - t["rs"]) * cs
                # If still too text-heavy, drop smallest text blocks until under budget (keep one)
                text_sorted = sorted(text, key=lambda k: int(k.get("rs",1))*int(k.get("cs",1)))
                while excess > 0 and len(text_sorted) > 1:
                    t = text_sorted.pop(0)
                    area = int(t.get("rs",1))*int(t.get("cs",1))
                    excess -= area
                    kids.remove(t)
                    text.remove(t)
        others = [k for k in kids if k.get("class") not in ("media","text")]
        kids = media + text + others
        return kids

    # Helper to inject a filler child into a window if coverage is low
    def maybe_add_window_filler(band: str, window_row0: int, window_rows: int, target_cov: float = 0.70, min_children: int = 8):
        # compute coverage within this window from current children
        window = [[False] * tiles_x for _ in range(window_rows)]
        child_count = 0
        for ch in children:
            if ch.get("band") != band:
                continue
            sh = ch.get("grid_l0_shadow", {})
            r0 = int(sh.get("row", 0))
            rs = int(sh.get("row_span", 1))
            c0 = int(sh.get("col", 0))
            cs = int(sh.get("col_span", 1))
            for r in range(r0, r0 + rs):
                rr = r - window_row0
                if 0 <= rr < window_rows:
                    for c in range(c0, c0 + cs):
                        if 0 <= c < tiles_x:
                            window[rr][c] = True
                    child_count += 1
        covered = sum(sum(1 for c in row if c) for row in window)
        total = window_rows * tiles_x
        cov = covered / float(total) if total else 0.0
        if cov >= target_cov:
            # if coverage is OK but item count is too low, still add small fillers
            pass
        if cov < target_cov:
            # add structured fillers instead of a single slab
            filler_rs = max(4, min(window_rows, int(window_rows * 0.4)))
            filler_r0 = window_row0 + max(0, (window_rows - filler_rs) // 2)
            cols_available = tiles_x - gutter_cols * 2
            col0 = gutter_cols
            cols_per = max(6, cols_available // 2)
            for idx in range(2):
                if col0 >= tiles_x - gutter_cols:
                    break
                col_span = min(cols_per, tiles_x - gutter_cols - col0)
                media_id = f"filler_media_{band}_{window_row0}_{idx}"
                child_media = {
                    "id": f"child_{media_id}",
                    "parent_id": media_id,
                    "band": band,
                    "level": 1,
                    "class": "media",
                    "grid_l1": {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "R": 1},
                    "grid_l0_shadow": {"row": filler_r0, "col": col0, "row_span": filler_rs, "col_span": col_span},
                    "locks": ["lock_y"],
                    "pattern": "filler",
                }
                children.append(child_media)
                log_rows.append({
                    "seed": seed,
                    "band": band,
                    "zone_id": media_id,
                    "pattern": "filler",
                    "class": child_media["class"],
                    "l1_row": 0,
                    "l1_col": 0,
                    "l1_rows": 1,
                    "l1_cols": 1,
                    "l0_row": filler_r0,
                    "l0_col": col0,
                    "l0_rows": filler_rs,
                    "l0_cols": col_span,
                })
                # text beneath media
                txt_rs = max(2, int(filler_rs * 0.35))
                txt_r0 = min(window_row0 + window_rows - txt_rs, filler_r0 + filler_rs + 1)
                text_id = f"filler_text_{band}_{window_row0}_{idx}"
                child_text = {
                    "id": f"child_{text_id}",
                    "parent_id": text_id,
                    "band": band,
                    "level": 1,
                    "class": "text",
                    "grid_l1": {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "R": 1},
                    "grid_l0_shadow": {"row": txt_r0, "col": col0, "row_span": txt_rs, "col_span": col_span},
                    "locks": ["lock_y"],
                    "pattern": "filler",
                }
                children.append(child_text)
                log_rows.append({
                    "seed": seed,
                    "band": band,
                    "zone_id": text_id,
                    "pattern": "filler",
                    "class": child_text["class"],
                    "l1_row": 0,
                    "l1_col": 0,
                    "l1_rows": 1,
                    "l1_cols": 1,
                    "l0_row": txt_r0,
                    "l0_col": col0,
                    "l0_rows": txt_rs,
                    "l0_cols": col_span,
                })
                child_count += 2
                col0 += col_span + 1
        # enforce a minimum item count by sprinkling small text chips
        needed = max(0, min_children - child_count)
        if needed > 0:
            chip_h = max(2, window_rows // 5)
            chip_r0 = window_row0 + max(0, (window_rows - chip_h) // 2)
            cols_per_chip = max(3, tiles_x // max(needed, 1))
            c = 0
            for i in range(needed):
                chip_w = min(cols_per_chip, tiles_x - c)
                if chip_w <= 0:
                    break
                filler_id = f"chip_{band}_{window_row0}_{i}"
                child = {
                    "id": f"child_{filler_id}",
                    "parent_id": filler_id,
                    "band": band,
                    "level": 1,
                    "class": "text",
                    "grid_l1": {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "R": 1},
                    "grid_l0_shadow": {"row": chip_r0, "col": c, "row_span": chip_h, "col_span": chip_w},
                    "locks": ["lock_y"],
                    "pattern": "filler",
                }
                children.append(child)
                log_rows.append({
                    "seed": seed,
                    "band": band,
                    "zone_id": filler_id,
                    "pattern": "filler",
                    "class": child["class"],
                    "l1_row": 0,
                    "l1_col": 0,
                    "l1_rows": 1,
                    "l1_cols": 1,
                    "l0_row": chip_r0,
                    "l0_col": c,
                    "l0_rows": chip_h,
                    "l0_cols": chip_w,
                })
                c += chip_w + 1

    window_attempts = max(1, int(getattr(args, "window_attempts", 1)))
    min_window_cov = max(0.0, min(1.0, float(getattr(args, "min_window_coverage", 0.7))))

    for z in zones:
        pat_name = z.get("pattern", "")
        # randomly swap variants for Main patterns
        if z.get("band") == "Main" and pat_name in ("card_grid", "feature_rail"):
            if pat_name == "card_grid" and rng.random() < 0.5:
                pat_name = "card_grid_dense"
            if pat_name == "feature_rail" and rng.random() < 0.5:
                pat_name = "feature_rail_alt"
        # optional sparse variant for card_grid to add variation
        if z.get("band") == "Main" and pat_name == "card_grid" and rng.random() < 0.4:
            pat_name = "card_grid_sparse"
        filler = FILLERS.get(pat_name)
        if filler is None:
            # default: single text block filling zone
            filler = lambda zone, pat, r: [{"class":"text","r":0,"c":0,"rs":int(zone["grid"]["row_span"]*zone["R"]), "cs":int(zone["grid"]["col_span"]*zone["R"])}]
            pat_def = {}
        else:
            pat_def = patt_map.get(pat_name, {})
        # pattern-specific coverage/constraints if provided
        target_cov = min_window_cov
        min_media = None
        max_media = None
        target_media_ratio = None
        if pat_name in denoise_targets:
            tgt = denoise_targets.get(pat_name, {})
            if isinstance(tgt, dict):
                target_cov = float(tgt.get("target_cov", target_cov))
                min_media = tgt.get("min_media", None)
                max_media = tgt.get("max_media", None)
                target_media_ratio = tgt.get("media_ratio", None)
        if pat_name.startswith("card_grid"):
            target_cov = max(target_cov, 0.70)
            target_media_ratio = target_media_ratio if target_media_ratio is not None else 0.55
            min_media = max(int(min_media) if min_media is not None else 2, 2)
            max_media = max(int(max_media) if max_media is not None else 6, 4)
        # Children in L1 inside the zone; card_grid is generated per viewport slice to vary content.
        zone_grid = z["grid"]
        R = int(z["R"])
        zone_rows_l0 = int(zone_grid.get("row_span", 0))
        window_splits: List[Tuple[int, int]] = []
        if pat_name == "card_grid" and viewport_rows:
            win_count = max(1, int(math.ceil(zone_rows_l0 / float(viewport_rows))))
            base_h = zone_rows_l0 // win_count
            remainder = zone_rows_l0 % win_count
            offset = 0
            for idx in range(win_count):
                h = base_h + (1 if idx < remainder else 0)
                window_splits.append((offset, h))
                offset += h
        else:
            window_splits.append((0, zone_rows_l0))

        win_global_idx = 0
        for win_offset_l0, win_rows_l0 in window_splits:
            sub_grid = dict(zone_grid)
            sub_grid["row"] = zone_grid["row"] + win_offset_l0
            sub_grid["row_span"] = win_rows_l0
            l1_rspan, l1_cspan = l1_dims(sub_grid, R)
            best_kids: List[dict] = []
            best_score = (-1.0, -1.0)
            # per-window RNG to diversify content; seed by global + window index
            rng_window = random.Random(seed * 10**6 + win_global_idx + 1)
            for _ in range(window_attempts):
                trial = filler({"grid": sub_grid, "R": R}, pat_def, rng_window)
                cov, bal = score_children_l1(trial, l1_rspan, l1_cspan, target_media_ratio=target_media_ratio or 0.35)
                if (cov, bal) > best_score:
                    best_score = (cov, bal)
                    best_kids = trial
            kids = best_kids
            if best_score[0] < target_cov and not pat_name.startswith("card_grid"):
                need = max(1, int((target_cov - best_score[0]) * l1_rspan))
                filler_rs = clamp(need, 2, l1_rspan)
                filler_r0 = max(0, l1_rspan - filler_rs)
                # split filler into two stacked paragraphs to avoid giant text slabs
                half = max(2, filler_rs // 2)
                kids.append({"class": "text", "r": filler_r0, "c": 0, "rs": half, "cs": l1_cspan})
                kids.append({"class": "text", "r": min(l1_rspan - half, filler_r0 + half + 1), "c": 0, "rs": filler_rs - half, "cs": l1_cspan})

            # cap elements to avoid overly busy slices
            cap = max_elements.get(pat_name, 0)
            kids = cap_items(kids, cap)
            # expand bottom-most items to fill unused height if any
            if kids and not pat_name.startswith("card_grid"):
                used_height = max(int(k.get("r", 0)) + int(k.get("rs", 0)) for k in kids)
                extra = max(0, l1_rspan - used_height)
                if extra > 0:
                    # give extra rows to items that end at used_height
                    tail_items = [k for k in kids if int(k.get("r", 0)) + int(k.get("rs", 0)) == used_height]
                    if tail_items:
                        inc = extra // len(tail_items)
                        rem = extra % len(tail_items)
                        for idx, k in enumerate(tail_items):
                            add = inc + (1 if idx < rem else 0)
                            k["rs"] = int(k.get("rs", 0)) + add

            # Enforce media/text balance and caps for card_grid
            if pat_name.startswith("card_grid"):
                kids = enforce_card_balance(
                    kids,
                    l1_rspan,
                    l1_cspan,
                    int(min_media) if min_media else 1,
                    int(max_media) if max_media else 3,
                    target_media_ratio=target_media_ratio,
                    max_text_ratio=0.95,
                )
            # cap elements to avoid overly busy slices and keep text
            cap = max_elements.get(pat_name, 0)
            kids = cap_items(kids, cap, ensure_text=pat_name.startswith("card_grid"))
            # expand existing items to fill unused height if any (favor text)
            if kids and not pat_name.startswith("card_grid"):
                used_height = max(int(k.get("r", 0)) + int(k.get("rs", 0)) for k in kids)
                extra = max(0, l1_rspan - used_height)
                if extra > 0:
                    text_items = [k for k in kids if k.get("class") == "text"]
                    targets = text_items or kids
                    inc = extra // len(targets)
                    rem = extra % len(targets)
                    for idx, k in enumerate(targets):
                        add = inc + (1 if idx < rem else 0)
                        k["rs"] = int(k.get("rs", 0)) + add

            for i, k in enumerate(kids):
                l1_r, l1_c, l1_rs, l1_cs = k["r"], k["c"], k["rs"], k["cs"]
                s_row, s_col, s_rs, s_cs = l1_to_l0_span(
                    l1_r, l1_c, l1_rs, l1_cs, R, sub_grid
                )
                z0 = sub_grid["row"]; zc = sub_grid["col"]
                col_global = zc + s_col
                row_global = z0 + s_row
                macro_col_w = int(macro_grid.get("tiles_per_macro_col", 0) or 0)
                macro_row_h = int(macro_grid.get("tiles_per_macro_row", 0) or 0)
                band_rows = rows_map.get(z["band"], 0)
                col_global, s_cs, row_global, s_rs = snap_to_grid(
                    col_global, s_cs, row_global, s_rs, band_rows, macro_col_w, macro_row_h, gutter_cols, tiles_x
                )
                # prevent overlaps by shifting horizontally if needed
                def rects_overlap(r1, c1, rs1, cs1, r2, c2, rs2, cs2) -> bool:
                    return not (r1 + rs1 <= r2 or r2 + rs2 <= r1 or c1 + cs1 <= c2 or c2 + cs2 <= c1)

                def find_non_overlapping(band: str, r: int, c: int, rs: int, cs: int, step: int) -> Tuple[Optional[int], Optional[int]]:
                    occupied = occupied_by_band.get(band, [])
                    if all(not rects_overlap(r, c, rs, cs, ro, co, rso, cso) for ro, co, rso, cso in occupied):
                        return r, c
                    # try shifting horizontally within gutters
                    max_right = tiles_x - gutter_cols - cs
                    candidates = []
                    for delta in range(step, tiles_x, step):
                        if c + delta <= max_right:
                            candidates.append(c + delta)
                        if c - delta >= gutter_cols:
                            candidates.append(c - delta)
                        if len(candidates) > 20:
                            break
                    for cand_c in candidates:
                        if all(not rects_overlap(r, cand_c, rs, cs, ro, co, rso, cso) for ro, co, rso, cso in occupied):
                            return r, cand_c
                    return None, None

                step = max(1, macro_grid.get("tiles_per_macro_col", 1) or 1)
                new_r, new_c = find_non_overlapping(z["band"], row_global, col_global, s_rs, s_cs, step)
                if new_r is None or new_c is None:
                    continue
                row_global, col_global = new_r, new_c

                child = {
                    "id": f"child_{z['id']}_{win_global_idx}_{i}",
                    "parent_id": z["id"],
                    "band": z["band"],
                    "level": 1,
                    "class": k["class"],
                    "grid_l1": {"row": l1_r, "col": l1_c, "row_span": l1_rs, "col_span": l1_cs, "R": R},
                    "grid_l0_shadow": {"row": row_global, "col": col_global, "row_span": s_rs, "col_span": s_cs},
                    "locks": z.get("locks", ["lock_y"]),
                    "pattern": pat_name
                }
                occupied_by_band[z["band"]].append((row_global, col_global, s_rs, s_cs))
                children.append(child)
                log_rows.append({
                    "seed": seed,
                    "band": child["band"],
                    "zone_id": z["id"],
                    "pattern": pat_name,
                    "class": child["class"],
                    "l1_row": l1_r,
                    "l1_col": l1_c,
                    "l1_rows": l1_rs,
                    "l1_cols": l1_cs,
                    "l0_row": child["grid_l0_shadow"]["row"],
                    "l0_col": child["grid_l0_shadow"]["col"],
                    "l0_rows": child["grid_l0_shadow"]["row_span"],
                    "l0_cols": child["grid_l0_shadow"]["col_span"],
                })
            win_global_idx += 1
    # Compress vertical gaps per band (after snapping)
    by_band: Dict[str, List[dict]] = defaultdict(list)
    for ch in children:
        by_band[ch.get("band","")].append(ch)
    for b, lst in by_band.items():
        compress_vertical(lst, rows_map.get(b, 0))

    rebalance_main(children, log_rows, target_ratio=0.5)
    QA_WARNINGS.extend(qa_band_coverage(children, zones))

    # Save children JSON
    out_path = outdir / f"children_l1_seed{seed}.json"
    save_json(out_path, children)

    # Overlay
    overlay = draw_overlay(
        manifest,
        zones,
        children,
        scale=int(args.png_scale),
        macro_grid=macro_grid,
        main_viewports=main_viewports if viewport_rows else None,
        main_viewport_labels=main_viewport_labels if main_viewport_labels else None,
    )
    png_path = debugdir / f"children_l1_overlay_seed{seed}.png"
    overlay.save(png_path)

    log_path = Path(args.logfile) if args.logfile else (debugdir / f"children_l1_seed{seed}_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_rows:
        fieldnames = ["seed", "band", "zone_id", "pattern", "class",
                      "l1_row", "l1_col", "l1_rows", "l1_cols",
                      "l0_row", "l0_col", "l0_rows", "l0_cols"]
        with open(log_path, "w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"[OK] Log:      {log_path}")

    print(f"[OK] Children: {out_path}")
    print(f"[OK] Overlay:  {png_path}")
    print(f"[SUMMARY] total_children={len(children)} zones={len(zones)}")
    for warn in QA_WARNINGS:
        print(f"[WARN] {warn}")
    # stdout trimmed manifest/zones for inspection
    try:
        print(json.dumps({"manifest": manifest, "zones": zones}, indent=2))
    except Exception:
        pass

if __name__ == "__main__":
    main()

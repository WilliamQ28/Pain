#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pass A — Coarse band orchestration (Level 0 zones)

Inputs
  --manifest  C:\wfc\P2\p2_tiles\manifest_seed7.json
  --rules     C:\wfc\P2\wfc_stats\zone_rules.json

Outputs
  p3_orchestrate\zones_l0_seed{seed}.json     (list of zone records)
  debug\zones_l0_overlay_seed{seed}.png       (stacked bands + zone boxes)

Notes
- Works in square grid units derived from manifest (tiles_x and rows0 per band).
- Uses zone_rules.json 'layout' to determine vertical/horizontal composition.
- Obeys min/max rows (and max_rows_frac) and min/max cols where applicable.
- For Header horizontal split (logo|primary_nav|utilities), it estimates
  col spans using per-column class densities (media on the left, utilities right).
- All other bands default to full-width zones stacked vertically unless rules
  specify horizontal groupings.

This is intentionally lightweight so you can iterate heuristics easily.
"""

import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------ CLI ------------------

def parse_args():
    ap = argparse.ArgumentParser("Orchestrate bands L0 (zones)")
    ap.add_argument("--manifest", required=True, help="path to manifest_seedX.json")
    ap.add_argument("--rules", required=True, help="path to zone_rules.json")
    ap.add_argument("--macro_grid", default=None, help="optional macro_grid_config.json to snap zones to macro cols/rows")
    ap.add_argument("--outdir", default="p3_orchestrate", help="where to write zones JSON")
    ap.add_argument("--debugdir", default="debug", help="where to write debug overlays")
    ap.add_argument("--seed", type=int, default=None, help="override seed in manifest (for filenames only)")
    ap.add_argument("--tiles_root", default=None, help="optional override root for *_tiles_seedX.npy (else use manifest.files.tiles)")
    ap.add_argument("--png_scale", type=int, default=8, help="scale for debug overlay")
    ap.add_argument("--gutter_cols", type=int, default=0, help="optional side gutter (cols per side) for all bands")
    return ap.parse_args()

# ------------------ IO ------------------

def load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------------ Helpers ------------------

DEFAULT_BANDS = ["Header", "Nav", "Main", "Supplement", "Footer"]

def band_order(manifest: dict) -> List[str]:
    names = [b.get("name","") for b in manifest.get("bands",[])]
    return names if names else DEFAULT_BANDS

def band_info(manifest: dict, name: str) -> Dict[str,int]:
    for b in manifest.get("bands",[]):
        if b.get("name")==name:
            return b
    return {}

def load_tiles_for_band(band: str, manifest: dict, tiles_root: Path=None) -> np.ndarray:
    files = manifest.get("files",{}).get("tiles",{})
    if band in files and files[band]:
        p = Path(files[band])
    else:
        # fallback to conventional name under tiles_root
        if tiles_root is None:
            raise FileNotFoundError(f"No tiles path for {band} in manifest.files and no --tiles_root provided")
        p = tiles_root / f"{band}_tiles_seed{manifest.get('seed',0)}.npy"
    return np.load(p)


def load_macro_grid(p: Path) -> dict:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def moving_avg(v: np.ndarray, k: int) -> np.ndarray:
    if k<=1: return v.copy()
    pad = k//2
    vp = np.pad(v, (pad,pad), mode="edge")
    w = np.ones(k, dtype=np.float32)/float(k)
    return np.convolve(vp, w, mode="valid")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ------------------ Zone sizing ------------------

def compute_zone_row_span(rule: dict, band_rows: int, remaining_rows: int, is_last: bool) -> int:
    min_rows = int(rule.get("min_rows", 1))
    max_rows = int(rule.get("max_rows", remaining_rows))
    max_frac = rule.get("max_rows_frac", None)
    if isinstance(max_frac, (int,float)):
        max_rows = min(max_rows, int(band_rows * float(max_frac)))
    # if last zone in vertical stack, give it the remainder (bounded)
    if is_last:
        return clamp(remaining_rows, min_rows, max_rows)
    # otherwise choose the tightest feasible value leaning to mid of min/max
    target = clamp((min_rows + max_rows)//2, min_rows, max_rows)
    target = clamp(target, 1, remaining_rows)
    return target

def estimate_header_cols(grid: np.ndarray, class_map: Dict[str,int], tiles_w: int, rule_seq: List[str], rules: dict) -> Dict[str, Tuple[int,int]]:
    """
    Estimate column allocations for Header horizontal zones: e.g. ["logo","primary_nav","utilities"].
    Uses simple per-column densities to bias: media on left, utilities right. Falls back to min/max cols.
    Returns dict zone_id -> (col_start, col_span)
    """
    H, W = grid.shape
    # Per-column class densities
    col_counts = {name: np.zeros(W, dtype=np.float32) for name in class_map.keys()}
    for c_name, c_id in class_map.items():
        col_counts[c_name] = (grid == c_id).sum(axis=0).astype(np.float32) / float(max(1,H))

    # defaults from rules
    spans = {}
    remaining = tiles_w
    c0 = 0
    for i, z in enumerate(rule_seq):
        zr = None
        for cand in rules.get("zones",[]):
            if cand.get("id")==z:
                zr = cand; break
        min_c = int((zr or {}).get("min_cols", 1))
        max_c = int((zr or {}).get("max_cols", tiles_w))
        # heuristic: logo ~ media heavy; utilities ~ input/select/role_button; nav = rest
        score = np.zeros(W, dtype=np.float32)
        if z == "logo":
            score = col_counts.get("media", np.zeros(W))
        elif z in ("utilities","utility_row"):
            score = (col_counts.get("input",0)+col_counts.get("select",0)+col_counts.get("role_button",0))
        elif z in ("primary_nav","menu_row"):
            score = (col_counts.get("text",0)+col_counts.get("role_button",0))
        # normalize score into a soft window width
        # basic split: proportional to (sum score) but clamped by min/max
        want = int(round( (score.sum()/max(1.0, sum(v.sum() for v in col_counts.values()))) * tiles_w ))
        if want <= 0:
            want = (min_c + max_c)//2
        want = clamp(want, min_c, max_c)
        # ensure we leave room for remaining zones
        zones_left = len(rule_seq) - (i+1)
        min_left = 0
        for j in range(i+1, len(rule_seq)):
            # sum future mins
            z2 = rule_seq[j]
            zr2 = None
            for cand in rules.get("zones",[]):
                if cand.get("id")==z2:
                    zr2 = cand; break
            min_left += int((zr2 or {}).get("min_cols", 1))
        max_here = remaining - min_left
        want = clamp(want, min_c, max_here)
        spans[z] = (c0, want)
        c0 += want
        remaining -= want
    return spans

# ------------------ Orchestration core ------------------

def orchestrate(manifest: dict, rules: dict, tiles_root: Path=None, gutter_cols: int = 0, macro_cfg: dict | None = None) -> Tuple[List[dict], Image.Image]:
    seed = manifest.get("seed", 0)
    tiles_x = int(manifest.get("tiles_x", 40))
    scale = 8  # will be overridden by caller for PNG
    # Prepare stacked debug canvas size based on rows per band
    bands = band_order(manifest)
    rows_by_band = {b: band_info(manifest,b).get("rows0",1) for b in bands}
    total_rows = sum(rows_by_band.values())
    img_w = tiles_x * scale
    img_h = total_rows * scale
    img = Image.new("RGB", (img_w, img_h), (245,245,245))
    draw = ImageDraw.Draw(img)

    class_map: Dict[str,int] = manifest.get("class_map", {})
    inv_class = {v:k for k,v in class_map.items()}

    y_cursor = 0
    zones_out: List[dict] = []

    macro_cols = int(macro_cfg.get("macro_cols", 0) or 0) if macro_cfg else 0
    tiles_per_macro_col = int(macro_cfg.get("tiles_per_macro_col", 0) or 0) if macro_cfg else 0
    tiles_per_macro_row = int(macro_cfg.get("tiles_per_macro_row", 0) or 0) if macro_cfg else 0
    macro_band_rows = (macro_cfg or {}).get("bands", {}) if macro_cfg else {}
    # Allow overriding target ratios for Main; defaults used if not provided.
    main_target_ratios = {"hero": 0.2, "content": 0.6, "callouts": 0.2}
    try:
        cfg_ratios = (rules.get("Main", {}).get("target_ratios") or {})
        merged = {}
        merged.update(main_target_ratios)
        merged.update({k: float(v) for k, v in cfg_ratios.items() if isinstance(v, (int, float, str))})
        # normalize
        total = sum(merged.values())
        if total > 0:
            merged = {k: v / total for k, v in merged.items()}
        main_target_ratios = merged
    except Exception:
        pass

    for b in bands:
        binfo = band_info(manifest, b)
        rows = int(binfo.get("rows0", 1))
        tiles_w_full = int(binfo.get("tiles_w", tiles_x))
        gutter = max(0, int(gutter_cols))
        c_offset = min(gutter, tiles_w_full//2)
        tiles_w = max(1, tiles_w_full - c_offset*2)
        # Target proportional split for Main to avoid oversized filler; scale to band rows.
        forced_rows: Dict[str, int] = {}
        # load grid to compute simple per-row/col densities for heuristics
        grid = load_tiles_for_band(b, manifest, tiles_root)
        H, W = grid.shape
        assert H == rows and W == tiles_w_full, f"{b}: grid shape {grid.shape} != ({rows},{tiles_w_full})"

        # Draw band boundary
        draw.rectangle([0, y_cursor, img_w-1, y_cursor + rows*scale - 1], outline=(0,0,0))
        draw.text((4, y_cursor+2), b, fill=(0,0,0))

        # Fetch rules for this band
        br = rules.get(b, {})
        vert_layout = br.get("layout", {}).get("vertical", [])
        horiz_layout = br.get("layout", {}).get("horizontal", [])

        # If horizontal-only (like Header), we still need to place a single row-span area
        # Determine vertical zones
        vertical_groups: List[List[str]] = []
        if vert_layout:
            for group in vert_layout:
                vertical_groups.append([z for z in group])
        else:
            # No vertical grouping: treat whole band as one group (horiz handled later)
            if horiz_layout:
                # one vertical slice with horizontal split inside
                vertical_groups.append([f"__H__:{'+'.join(horiz_layout[0])}"])
            else:
                # No layout specified; whole band is a generic single zone
                vertical_groups.append(["content"])

        if tiles_per_macro_row > 0 and macro_band_rows and b in macro_band_rows:
            rows = max(rows, int(macro_band_rows.get(b, {}).get("macro_rows", 1) * tiles_per_macro_row))
        remaining_rows = rows
        row0 = 0
        # For Main, precompute desired row spans to fill the band proportionally (avoid giant filler).
        if b == "Main" and vertical_groups:
            # allow overriding via rules.Main.target_ratios but default to 20/60/20
            target_ratios = main_target_ratios
            base_rows = rows
            for group in vertical_groups:
                if len(group) == 1 and group[0] in target_ratios:
                    z = group[0]
                    forced_rows[z] = max(1, int(round(base_rows * target_ratios[z])))
            if forced_rows:
                total_forced = sum(forced_rows.values())
                if total_forced > 0:
                    scale = base_rows / float(total_forced)
                    forced_rows = {k: max(1, int(round(v * scale))) for k, v in forced_rows.items()}
                    diff = base_rows - sum(forced_rows.values())
                    order = ["content", "callouts", "hero"]
                    while diff != 0:
                        for z in order:
                            if diff == 0: break
                            if diff > 0:
                                forced_rows[z] = forced_rows.get(z, 0) + 1
                                diff -= 1
                            elif forced_rows.get(z, 0) > 1:
                                forced_rows[z] -= 1
                                diff += 1
                        else:
                            break

        for gi, group in enumerate(vertical_groups):
            # If the group encodes a horizontal split marker
            if len(group)==1 and group[0].startswith("__H__:"):
                toks = group[0].split(":",1)[1].split("+")
                # Height for this horizontal group: use min of all child rules, or clamp 1..rows
                # Try to find a reasonable band-row allocation (use max min_rows among children)
                child_min = max(int(next((z.get("min_rows",1) for z in br.get("zones",[]) if z.get("id")==t), {"min_rows":1}) ) for t in toks)
                # Prefer small headers/navs; use min(child_min, 2) but ensure <= remaining_rows
                this_rows = clamp(min(remaining_rows, max(child_min, 1)), 1, remaining_rows)
                # Horizontal allocation inside this vertical slice
                spans = estimate_header_cols(grid[row0:row0+this_rows, c_offset:c_offset+tiles_w], class_map, tiles_w, toks, br)
                for z in toks:
                    c0, cspan = spans[z]
                    c0 += c_offset
                    zones_out.append({
                        "id": f"zone_{b}_{z}_{row0}_{c0}",
                        "band": b,
                        "level": 0,
                        "grid": {"row": row0, "col": int(c0), "row_span": int(this_rows), "col_span": int(cspan)},
                        "pattern": next((zdef.get("pattern") for zdef in br.get("zones",[]) if zdef.get("id")==z), z),
                        "locks": list(set(["lock_y"] + next((zdef.get("locks",[]) for zdef in br.get("zones",[]) if zdef.get("id")==z), []))),
                        "R": int(next((zdef.get("R", rules.get("global",{}).get("R_default",3)) for zdef in br.get("zones",[]) if zdef.get("id")==z), rules.get("global",{}).get("R_default",3)))
                    })
                    # draw box
                    x0 = c0*scale
                    y0p = y_cursor + row0*scale
                    draw.rectangle([x0, y0p, x0 + cspan*scale -1, y0p + this_rows*scale -1], outline=(0,120,240))
                    draw.text((x0+3, y0p+3), z, fill=(0,0,0))
                row0 += this_rows
                remaining_rows -= this_rows
                continue

            # Otherwise, this vertical group is a single zone id in a stacked column
            for zi, z in enumerate(group):
                # Find rule record
                zr = None
                for cand in br.get("zones",[]):
                    if cand.get("id")==z:
                        zr = cand; break
                # rows for this zone
                is_last = (gi == len(vertical_groups)-1) and (zi == len(group)-1)
                if b == "Main" and z in forced_rows:
                    z_rows = forced_rows[z]
                else:
                    z_rows = compute_zone_row_span(zr or {}, rows, remaining_rows, is_last)
                # snap rows to macro grid if available
                if tiles_per_macro_row > 0:
                    z_rows = max(1, int(round(z_rows / tiles_per_macro_row)) * tiles_per_macro_row)
                    z_rows = min(z_rows, remaining_rows)
                if b == "Main" and is_last:
                    # ensure we soak up any remainder to avoid giant filler zones
                    z_rows = max(1, rows - row0)
                # full-width by default
                zones_out.append({
                    "id": f"zone_{b}_{z}_{row0}_0",
                    "band": b,
                    "level": 0,
                    "grid": {"row": row0, "col": c_offset, "col_span": int(tiles_w), "row_span": int(z_rows)},
                    "pattern": (zr or {}).get("pattern", z),
                    "locks": list(set(["lock_y"] + (zr or {}).get("locks", []))),
                    "R": int((zr or {}).get("R", rules.get("global",{}).get("R_default",3)))
                })
                # draw
                x0 = 0
                y0p = y_cursor + row0*scale
                draw.rectangle([x0, y0p, x0 + tiles_w*scale -1, y0p + z_rows*scale -1], outline=(240,120,0))
                draw.text((x0+3, y0p+3), z, fill=(0,0,0))
                row0 += z_rows
                remaining_rows -= z_rows

        # After placing all vertical groups, drop any leftover rows (no filler zones) for Main.
        if b == "Main":
            remaining_rows = 0

        # advance cursor for next band
        y_cursor += rows*scale

    return zones_out, img

# ------------------ Main ------------------

def main():
    args = parse_args()
    manifest = load_json(Path(args.manifest))
    rules = load_json(Path(args.rules))
    outdir = Path(args.outdir); ensure_dir(outdir)
    debugdir = Path(args.debugdir); ensure_dir(debugdir)

    # let filenames reflect either CLI-override or manifest seed
    seed = args.seed if args.seed is not None else int(manifest.get("seed", 0))
    tiles_root = Path(args.tiles_root) if args.tiles_root else None

    macro_cfg = load_macro_grid(Path(args.macro_grid)) if args.macro_grid else None
    zones, overlay = orchestrate(manifest, rules, tiles_root=tiles_root, gutter_cols=int(args.gutter_cols), macro_cfg=macro_cfg)
    zpath = outdir / f"zones_l0_seed{seed}.json"
    with open(zpath, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2)

    # Debug overlay
    png = debugdir / f"zones_l0_overlay_seed{seed}.png"
    overlay.save(png)

    print(f"[OK] Zones: {zpath}")
    print(f"[OK] Overlay: {png}")
    print(f"[SUMMARY] total_zones={len(zones)}")

if __name__ == "__main__":
    main()

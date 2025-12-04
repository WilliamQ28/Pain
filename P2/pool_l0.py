#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- IO ----------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------- pooling ----------
def mode_pool(grid: np.ndarray, k: int) -> np.ndarray:
    H, W = grid.shape
    Hp, Wp = math.ceil(H / k), math.ceil(W / k)
    pad_h, pad_w = Hp * k - H, Wp * k - W
    if pad_h or pad_w:
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode="edge")
    g = grid.reshape(Hp, k, Wp, k).swapaxes(1, 2).reshape(Hp, Wp, k * k)
    C = int(grid.max()) + 1
    out = np.empty((Hp, Wp), dtype=np.int32)
    for y in range(Hp):
        for x in range(Wp):
            bc = np.bincount(g[y, x], minlength=C)
            out[y, x] = int(np.argmax(bc))  # ties -> smallest id
    return out

# ---------- palette helpers ----------
def _stable_random_palette(vals: np.ndarray, seed: int = 1234) -> Dict[int, tuple]:
    rng = np.random.default_rng(seed)
    return {int(i): tuple(int(c) for c in rng.integers(32, 224, size=3)) for i in vals}

def build_palette_from_manifest(manifest: dict, uniq_vals: np.ndarray) -> Dict[int, tuple]:
    # 1) numeric->rgb if present
    for key in ("class_colors", "class_palette"):
        if key in manifest and isinstance(manifest[key], dict):
            pal = {int(k): tuple(v) for k, v in manifest[key].items()}
            # ensure all classes have a color
            rnd = _stable_random_palette(uniq_vals)
            for i in uniq_vals:
                pal.setdefault(int(i), rnd[int(i)])
            return pal
    # 2) fallback random
    return _stable_random_palette(uniq_vals)

def guess_png_scale(manifest: dict, default_scale: int = 8) -> int:
    # Common places we store scale; otherwise assume 8
    for key in ("png_scale", "debug_png_scale", "scale"):
        if key in manifest and isinstance(manifest[key], int) and manifest[key] > 0:
            return int(manifest[key])
    return default_scale

def find_debug_png(mani_path: Path, mani: dict, seed: int) -> Optional[Path]:
    # Prefer explicit path in manifest
    files = mani.get("files", {})
    if isinstance(files, dict):
        p = files.get("debug_png")
        if p:
            pth = Path(p)
            if not pth.is_absolute():
                pth = mani_path.parent / p
            if pth.exists():
                return pth
    # Fallback: sibling of manifest, common name
    candidate = mani_path.parent / f"debug_seed{seed}.png"
    return candidate if candidate.exists() else None

def detect_gutter_px(img: Image.Image, tiles_w: int, scale: int) -> int:
    """
    If the stitched debug has a left label gutter, its width is usually a small
    multiple of 'scale'. Try to infer by comparing image width to W*scale.
    """
    expected = tiles_w * scale
    extra = img.width - expected
    if extra <= 0:
        return 0
    # Choose nearest multiple of scale up to 12 tiles as gutter
    m = (extra // scale) * scale
    if 0 < m <= 12 * scale:
        return int(m)
    return 0

def extract_palette_from_debug(debug_png: Path, mani: dict) -> Optional[Dict[int, tuple]]:
    """
    Build {class_id: (r,g,b)} by sampling colors from the original stitched debug.
    Assumes each band is stacked vertically, optional left gutter and 1px rules are tolerated.
    """
    try:
        img = Image.open(debug_png).convert("RGB")
    except Exception:
        return None

    bands = mani.get("bands", [])
    tiles = mani.get("files", {}).get("tiles", {})
    if not bands or not tiles:
        return None

    # Need tile width (same for all bands)
    # Pick first band grid to get width
    first = np.load(tiles[bands[0]["name"]])
    tiles_w = first.shape[1]
    scale = guess_png_scale(mani, default_scale=8)
    gutter_px = detect_gutter_px(img, tiles_w, scale)

    y = 0
    palette: Dict[int, Tuple[int, int, int]] = {}
    for b in bands:
        g = np.load(tiles[b["name"]])
        h, w = g.shape
        # crop band area in tile space from stitched debug
        x0 = gutter_px
        y0 = y
        x1 = gutter_px + w * scale
        y1 = y + h * scale
        # guard
        x1 = min(x1, img.width)
        y1 = min(y1, img.height)
        band_img = np.asarray(img.crop((x0, y0, x1, y1)).resize((w, h), Image.NEAREST))
        # for each class id present in grid, compute mean color
        for cid in np.unique(g):
            mask = (g == cid)
            if mask.any():
                rgb = band_img[mask].mean(axis=0)
                palette[int(cid)] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        # advance y (allow 1px rule lines between bands without caring)
        y += h * scale
        if y < img.height and img.height - y < 2 * scale:
            # skip possible 1px rule
            y += 1
    return palette if palette else None

# ---------- rendering ----------
def grid_to_img(grid: np.ndarray, palette: Dict[int, tuple]) -> Image.Image:
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Map by palette (unseen ids remain black)
    for cid, rgb in palette.items():
        img[grid == cid] = rgb
    return Image.fromarray(img, "RGB")

def render_band_png(grid: np.ndarray, palette: Dict[int, tuple], out_path: Path, scale: int = 8):
    grid_to_img(grid, palette).resize((grid.shape[1]*scale, grid.shape[0]*scale), Image.NEAREST).save(out_path)

def render_stitched_debug(bands: List[dict], palette: Dict[int, tuple],
                          out_path: Path, scale: int = 8, label: bool = True, gutter_tiles: int = 6):
    W = bands[0]["grid"].shape[1]
    H = sum(b["grid"].shape[0] for b in bands)
    label_gutter_px = (gutter_tiles if label else 0) * scale
    canvas = Image.new("RGB", (W*scale + label_gutter_px, H*scale), (24,24,24))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.load_default()
    except Exception: font = None
    y_off = 0
    for b in bands:
        g = b["grid"]; h, w = g.shape
        im = grid_to_img(g, palette).resize((w*scale, h*scale), Image.NEAREST)
        canvas.paste(im, (label_gutter_px, y_off))
        if label:
            draw.text((4, y_off + 4), b["name"], fill=(230,230,230), font=font)
            draw.line([(0, y_off), (W*scale + label_gutter_px, y_off)], fill=(60,60,60), width=1)
        y_off += h*scale
    draw.line([(0, y_off-1), (W*scale + label_gutter_px, y_off-1)], fill=(60,60,60), width=1)
    canvas.save(out_path)

def render_compare(original_bands: List[dict], pooled_bands: List[dict],
                   palette: Dict[int, tuple], out_path: Path, scale: int = 8, label: bool = True):
    W1 = original_bands[0]["grid"].shape[1]
    W2 = pooled_bands[0]["grid"].shape[1]
    H1 = sum(b["grid"].shape[0] for b in original_bands)
    H2 = sum(b["grid"].shape[0] for b in pooled_bands)
    H = max(H1, H2)
    gutter_px = (6 if label else 0) * scale
    sep_px = 8

    left = Image.new("RGB", (W1*scale + gutter_px, H*scale), (24,24,24))
    right = Image.new("RGB", (W2*scale + gutter_px, H*scale), (24,24,24))
    drawL, drawR = ImageDraw.Draw(left), ImageDraw.Draw(right)
    try: font = ImageFont.load_default()
    except Exception: font = None

    def paste_col(img, draw, bands):
        y = 0
        for b in bands:
            g = b["grid"]; h, w = g.shape
            im = grid_to_img(g, palette).resize((w*scale, h*scale), Image.NEAREST)
            img.paste(im, (gutter_px, y))
            if label:
                draw.text((4, y+4), b["name"], fill=(230,230,230), font=font)
                draw.line([(0,y), (img.width, y)], fill=(60,60,60), width=1)
            y += h*scale
        draw.line([(0, y-1), (img.width, y-1)], fill=(60,60,60), width=1)
        return img

    left  = paste_col(left,  drawL, original_bands)
    right = paste_col(right, drawR, pooled_bands)

    canvas = Image.new("RGB", (left.width + sep_px + right.width, H*scale), (16,16,16))
    canvas.paste(left, (0,0))
    canvas.paste(right, (left.width + sep_px, 0))
    canvas.save(out_path)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Pool L0 tiles with palette extracted from original debug")
    ap.add_argument("--manifest", required=True, help="p2_tiles/manifest_seedX.json")
    ap.add_argument("--outdir", default="p2_tiles_pooled", help="output dir")
    ap.add_argument("--k", type=int, default=2, help="pooling factor")
    ap.add_argument("--scale", type=int, default=8, help="scale for debug PNGs")
    ap.add_argument("--compare", action="store_true", help="write side-by-side original vs pooled")
    ap.add_argument("--debug_src", default="", help="optional path to original stitched debug png")
    args = ap.parse_args()

    mani_path = Path(args.manifest)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mani = load_json(mani_path)

    seed = int(mani.get("seed", 0))
    tiles_x_in = int(mani.get("tiles_x", 40))
    k = max(1, int(args.k))
    tiles_x_out = math.ceil(tiles_x_in / k)
    bands_in = mani["bands"]
    files_in = mani["files"]["tiles"]

    # Load original + pooled grids
    pooled_grids, original_grids = [], []
    files_out, rows_out, all_classes = {}, {}, set()

    for b in bands_in:
        name = b["name"]
        grid = np.load(Path(files_in[name]))
        pooled = mode_pool(grid, k)

        npy_out = outdir / f"{name}_tiles_seed{seed}_pooled{int(k)}.npy"
        np.save(npy_out, pooled)

        original_grids.append({"name": name, "grid": grid})
        pooled_grids.append({"name": name, "grid": pooled})
        files_out[name] = str(npy_out.resolve())
        rows_out[name] = int(pooled.shape[0])
        all_classes.update(int(v) for v in np.unique(grid))
        all_classes.update(int(v) for v in np.unique(pooled))

    uniq_vals = np.array(sorted(all_classes), dtype=int)

    # ----- Palette: prefer extracting from original stitched debug -----
    debug_src = Path(args.debug_src) if args.debug_src else find_debug_png(mani_path, mani, seed)
    palette = None
    if debug_src and debug_src.exists():
        palette = extract_palette_from_debug(debug_src, mani)
        if palette:
            print(f"[INFO] Palette extracted from original debug: {debug_src.name}")
    if not palette:
        palette = build_palette_from_manifest(mani, uniq_vals)
        print("[WARN] Using palette from manifest / fallback (no debug PNG palette found)")

    # Per-band pooled debug PNGs (canonical/sampled palette)
    for b in pooled_grids:
        out_png = outdir / f"{b['name']}_tiles_seed{seed}_pooled{int(k)}.png"
        render_band_png(b["grid"], palette, out_png, scale=args.scale)

    # Stitched pooled debug
    stitched_pooled = outdir / f"debug_seed{seed}_pooled{int(k)}.png"
    render_stitched_debug(pooled_grids, palette, stitched_pooled, scale=args.scale, label=True)
    print(f"[OK] stitched pooled debug: {stitched_pooled}")

    # Optional side-by-side
    if args.compare:
        comp = outdir / f"debug_seed{seed}_pooled{int(k)}_compare.png"
        render_compare(original_grids, pooled_grids, palette, comp, scale=args.scale, label=True)
        print(f"[OK] side-by-side compare: {comp}")

    # Pooled manifest
    bands_out = []
    for b in bands_in:
        name = b["name"]
        bands_out.append({
            **{kk: vv for kk, vv in b.items() if kk not in ("rows0","tiles_w","tiles_h")},
            "rows0": rows_out[name],
            "tiles_w": tiles_x_out,
            "tiles_h": rows_out[name]
        })
    mani_out = {
        **mani,
        "tiles_x": tiles_x_out,
        "bands": bands_out,
        "files": { **mani.get("files", {}), "tiles": files_out },
        "notes": (mani.get("notes","") + f" | Pooled {k}×{k} with palette-from-debug + stitched debug").strip()
    }
    save_json(outdir / f"manifest_seed{seed}_pooled{int(k)}.json", mani_out)
    print(f"[OK] pooled manifest written (tiles_x {tiles_x_in} → {tiles_x_out})")

if __name__ == "__main__":
    main()

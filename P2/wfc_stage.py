#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFC Stage (L0 seed) with coarse-to-fine and graceful adjacency relaxation.

- Solve per-band integer grids with WFC.
- NEW: --coarse_factor lets you run on a smaller lattice and upscale.
- NEW: --relax_on_fail / --relax_pct / --restarts provide automatic retries
       with progressively widened adjacency if the strict solve fails.

Outputs:
  per-band *.npy, stitched debug PNG, manifest JSON
"""

import argparse, hashlib, json, math, os, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------- CLI --------------------------
def parse_args():
    ap = argparse.ArgumentParser("wfc_stage")
    ap.add_argument("--prob", required=True, help="probabilities.json")
    ap.add_argument("--adj", required=True, help="adjacency.json")
    ap.add_argument("--outdir", default="p2_tiles", help="output directory")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--canvas_width", type=int, default=1440)
    ap.add_argument("--tiles_x", type=int, default=40, help="target lattice width (tile columns)")
    ap.add_argument("--png_scale", type=int, default=8, help="debug PNG scale per tile")

    # WFC params
    ap.add_argument("--wfc_temp", type=float, default=0.90, help="temperature for randomness (1.0 = none)")
    ap.add_argument("--wfc_eps", type=float, default=0.03, help="min entropy epsilon")
    ap.add_argument("--wfc_backtrack", type=int, default=300, help="max backtracks")

    # Coarse-to-fine
    ap.add_argument("--coarse_factor", type=int, default=3,
                    help="solve on tiles/coarse_factor and nearest-upscale back")

    # Band-wise tweak (optional)
    ap.add_argument("--media_add", default="", help="e.g., 'Main=+0.12,Header=-0.05'")

    # NEW: graceful fallback when strict adjacency fails
    ap.add_argument("--relax_on_fail", action="store_true",
                    help="if strict solve fails, retry with relaxed adjacency")
    ap.add_argument("--relax_pct", type=float, default=1.0,
                    help="0..1 fraction of remaining classes to add into each neighbor set on retry")
    ap.add_argument("--restarts", type=int, default=3,
                    help="max attempts per band (progressively more relaxed)")
    ap.add_argument("--main_slice_px", type=int, default=None,
                    help="optional target slice height in px for Main; converts to rows via canvas_width/tiles_x")
    ap.add_argument("--main_slice_rows", type=int, default=None,
                    help="optional target slice height in rows (overrides main_slice_px)")

    return ap.parse_args()

# -------------------------- utils / IO --------------------------
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_media_add(s: str) -> Dict[str, float]:
    if not s: return {}
    out = {}
    for part in [x.strip() for x in s.split(",") if x.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            try: out[k.strip()] = float(v.strip())
            except: pass
    return out

# -------------------------- Palette / Debug --------------------------
DEFAULT_CLASS_COLORS = {
    "text":        (226, 91, 58),
    "media":       (99, 180, 246),
    "button":      (56, 142, 60),
    "input":       (142, 36, 170),
    "select":      (233, 178, 76),
    "role_button": (66, 66, 66),
}

def stable_color_for_label(label: str, cid: int) -> Tuple[int, int, int]:
    key = (label or str(cid)).strip().lower()
    if key in DEFAULT_CLASS_COLORS:
        return DEFAULT_CLASS_COLORS[key]
    digest = hashlib.sha1((label or str(cid)).encode("utf-8")).digest()
    return tuple(int(40 + (digest[i] % 181)) for i in range(3))

def build_palette(prob_cfg: dict,
                  uniq_ids: np.ndarray,
                  id_to_label: Dict[int, str]) -> Dict[int, Tuple[int,int,int]]:
    for key in ("class_colors", "class_palette"):
        pal = prob_cfg.get(key)
        if isinstance(pal, dict):
            out = {int(k): tuple(v) for k, v in pal.items()}
            for cid in uniq_ids:
                lab = id_to_label.get(int(cid), f"class_{int(cid)}")
                out.setdefault(int(cid), stable_color_for_label(lab, int(cid)))
            return out
    cmap = prob_cfg.get("class_map", {})
    if isinstance(cmap, dict) and cmap:
        lower = {k.lower(): int(v) for k, v in cmap.items()}
        out = {}
        for nm, rgb in DEFAULT_CLASS_COLORS.items():
            if nm in lower: out[lower[nm]] = rgb
        for cid in uniq_ids:
            lab = id_to_label.get(int(cid), f"class_{int(cid)}")
            out.setdefault(int(cid), stable_color_for_label(lab, int(cid)))
        return out
    out = {}
    for cid in uniq_ids:
        lab = id_to_label.get(int(cid), f"class_{int(cid)}")
        out[int(cid)] = stable_color_for_label(lab, int(cid))
    return out

def grid_to_img(grid: np.ndarray, palette: Dict[int, Tuple[int,int,int]]) -> Image.Image:
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in palette.items():
        img[grid == cid] = rgb
    return Image.fromarray(img, "RGB")

def render_stitched_debug(bands: List[Tuple[str, np.ndarray]],
                          palette: Dict[int, Tuple[int,int,int]],
                          out_path: Path,
                          scale: int = 8,
                          legend: Optional[List[Tuple[str, Tuple[int,int,int]]]] = None,
                          target_width_px: Optional[int] = None) -> int:
    if not bands:
        return max(1, int(scale))

    legend = legend or []
    requested_scale = max(1, int(scale))
    W = bands[0][1].shape[1]
    H_tiles = sum(g.shape[0] for _, g in bands)

    scale_px = requested_scale
    left_margin = 0
    right_margin = 0
    target = int(target_width_px) if target_width_px else None

    if target and target >= W:
        scale_px = max(1, target // W)
        grid_width = W * scale_px
        if grid_width > target:
            left_margin = right_margin = 0
        else:
            extra = target - grid_width
            left_margin = extra // 2
            right_margin = extra - left_margin
    else:
        grid_width = W * scale_px
        left_margin = requested_scale // 2
        right_margin = left_margin
    grid_width = W * scale_px

    legend_pad_x = 16 if legend else 0
    legend_width = 0
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if legend:
        legend_width = max(200, scale_px * 5)
    canvas_width = legend_width + legend_pad_x + left_margin + grid_width + right_margin
    canvas_height = H_tiles * scale_px
    canvas = Image.new("RGB", (canvas_width, canvas_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    grid_x0 = legend_width + legend_pad_x

    y = 0
    for name, g in bands:
        im = grid_to_img(g, palette).resize((g.shape[1]*scale_px, g.shape[0]*scale_px), Image.NEAREST)
        canvas.paste(im, (grid_x0 + left_margin, y))
        draw.line([(0, y), (canvas_width, y)], fill=(60, 60, 60), width=1)
        label_x = grid_x0 + left_margin + 4
        label_y = y + 4
        text_w = text_h = 0
        if font:
            try:
                bbox = font.getbbox(name)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = font.getsize(name)
        if text_w and text_h:
            pad = 4
            draw.rectangle([label_x - pad,
                            label_y - pad,
                            label_x + text_w + pad,
                            label_y + text_h + pad],
                           fill=(16, 16, 16))
        draw.text((label_x, label_y), name, fill=(230, 230, 230), font=font)
        y += g.shape[0] * scale_px
    draw.line([(0, y - 1), (canvas_width, y - 1)], fill=(60, 60, 60), width=1)

    if legend and canvas_height > 0:
        swatch = max(18, scale_px)
        legend_bg = [0, 0, legend_width + legend_pad_x, canvas_height]
        draw.rectangle(legend_bg, fill=(20, 20, 20))
        total_needed = len(legend) * (swatch + 10) + 20
        line_height = swatch + 10
        if total_needed > canvas_height and len(legend) > 0:
            line_height = max(swatch + 6, math.floor((canvas_height - 20) / max(1, len(legend))))
        base_y = 12
        for label, rgb in legend:
            swatch_box = [12, base_y, 12 + swatch, base_y + swatch]
            draw.rectangle(swatch_box, fill=rgb, outline=(10, 10, 10))
            text_x = swatch_box[2] + 8
            if font:
                try:
                    bbox = font.getbbox(label)
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_h = font.getsize(label)[1]
            else:
                text_h = 0
            text_y = base_y + max(0, (swatch - text_h) // 2)
            draw.text((text_x, text_y), label, fill=(230, 230, 230), font=font)
            base_y += line_height

    canvas.save(out_path)
    return scale_px

# -------------------------- Geometry --------------------------
def upscale_nearest(grid: np.ndarray, target_h: int, target_w: int, factor: int) -> np.ndarray:
    if factor <= 1: return grid[:target_h, :target_w]
    expanded = np.kron(grid, np.ones((factor, factor), dtype=np.int32))
    return expanded[:target_h, :target_w]

# -------------------------- Probabilities / Bands --------------------------
@dataclass
class BandSpec:
    name: str
    y0_px: int
    y1_px: int
    rows0: int
    tiles_w: int

def normalize(p: np.ndarray) -> np.ndarray:
    s = p.sum()
    if s <= 0: return np.ones_like(p) / len(p)
    return p / s

def adjust_media_weights(band_name: str, weights: Dict[str, float], class_map: Dict[str,int], p: np.ndarray) -> np.ndarray:
    if not weights: return p
    delta = weights.get(band_name, 0.0)
    if abs(delta) < 1e-9: return p
    if "media" in class_map:
        idx = class_map["media"]
        p = p.copy()
        p[idx] = max(0.0, p[idx] + delta)
        p = normalize(p)
    return p

def compute_band_specs(canvas_width: int, tiles_x: int, prob_cfg: dict) -> List[BandSpec]:
    default_order = ["Header", "Nav", "Main", "Supplement", "Footer"]
    height_pct = prob_cfg.get("band_height_pct", {
        "Header": 8, "Nav": 5, "Main": 70, "Supplement": 10, "Footer": 7
    })
    rows_total = tiles_x * 10
    pct_total = sum(height_pct.get(b, 0) for b in default_order) or 100
    bands: List[BandSpec] = []
    y0 = 0
    for name in default_order:
        rows = max(1, int(round(rows_total * (height_pct.get(name, 0) / pct_total))))
        y1 = y0 + rows
        bands.append(BandSpec(name=name, y0_px=y0, y1_px=y1, rows0=rows, tiles_w=tiles_x))
        y0 = y1
    if bands:
        diff = rows_total - bands[-1].y1_px
        if diff:
            last = bands[-1]
            bands[-1] = BandSpec(last.name, last.y0_px, last.y1_px + diff, last.rows0 + diff, tiles_x)
    return bands

# -------------------------- WFC core --------------------------
class WFCGrid:
    def __init__(self, H, W, class_count, prior, neighbors, rng, temp, eps, max_backtrack):
        self.H, self.W, self.C = H, W, class_count
        self.prior = normalize(prior.astype(np.float64))
        self.NN = neighbors
        self.rng = rng
        self.temp = temp
        self.eps = eps
        self.max_backtrack = max_backtrack
        self.poss = np.ones((H, W, self.C), dtype=bool)
        self.collapsed = np.full((H, W), -1, dtype=np.int32)

    def entropy(self, y, x):
        ps = self.prior[self.poss[y, x]]
        if ps.size == 0: return -1.0
        ps = ps / ps.sum()
        return float(-(ps * np.log(ps + 1e-12)).sum())

    def select_cell(self):
        min_e, cand = 1e9, None
        for y in range(self.H):
            for x in range(self.W):
                if self.collapsed[y, x] >= 0: continue
                e = self.entropy(y, x)
                if e < 0: return (y, x)
                if e < min_e - 1e-12:
                    min_e, cand = e, (y, x)
        return cand

    def sample_class(self, y, x):
        mask = self.poss[y, x]
        idx = np.nonzero(mask)[0]
        if idx.size == 0: return -1
        probs = self.prior[idx].astype(np.float64)
        probs = probs / probs.sum()
        if self.temp != 1.0:
            probs = probs ** (1.0 / max(1e-6, self.temp))
            probs = probs / probs.sum()
        return int(self.rng.choice(idx, p=probs))

    def propagate(self):
        Q = []
        for y in range(self.H):
            for x in range(self.W):
                if self.collapsed[y, x] >= 0: Q.append((y, x))

        while Q:
            y, x = Q.pop()
            cls = self.collapsed[y, x]
            for dy, dx, d_self, d_nbr in [(-1,0,'N','S'), (1,0,'S','N'), (0,-1,'W','E'), (0,1,'E','W')]:
                yy, xx = y+dy, x+dx
                if not (0 <= yy < self.H and 0 <= xx < self.W): continue
                if self.collapsed[yy, xx] >= 0:
                    nbr_cls = self.collapsed[yy, xx]
                    if nbr_cls not in self.NN.get(cls, {}).get(d_self, []):
                        return False
                    continue
                poss_idx = np.nonzero(self.poss[yy, xx])[0]
                ok = []
                for c2 in poss_idx:
                    if cls in self.NN.get(c2, {}).get(d_nbr, []):
                        ok.append(c2)
                new_mask = np.zeros(self.C, dtype=bool)
                new_mask[ok] = True
                if not new_mask.any(): return False
                if not np.array_equal(new_mask, self.poss[yy, xx]):
                    self.poss[yy, xx] = new_mask
                    single = np.nonzero(new_mask)[0]
                    if single.size == 1:
                        self.collapsed[yy, xx] = int(single[0])
                        Q.append((yy, xx))
        return True

    def solve(self):
        bt, stack = 0, []
        while True:
            c = self.select_cell()
            if c is None: return self.collapsed.copy()
            y, x = c
            k = self.sample_class(y, x)
            if k < 0:
                if not stack or bt >= self.max_backtrack: return None
                bt += 1
                (state, yb, xb, forb) = stack.pop()
                self.poss = state["poss"].copy()
                self.collapsed = state["collapsed"].copy()
                self.poss[yb, xb][forb] = False
                continue
            state = {"poss": self.poss.copy(), "collapsed": self.collapsed.copy()}
            stack.append((state, y, x, k))
            self.poss[y, x] = False
            self.poss[y, x][k] = True
            self.collapsed[y, x] = k
            if not self.propagate():
                if not stack or bt >= self.max_backtrack: return None
                bt += 1
                (state, yb, xb, forb) = stack.pop()
                self.poss = state["poss"].copy()
                self.collapsed = state["collapsed"].copy()
                self.poss[yb, xb][forb] = False

# -------------------------- Adjacency helpers --------------------------
def build_neighbors(adj_cfg: dict, class_count: int) -> Dict[int, Dict[str, List[int]]]:
    all_ids = list(range(class_count))
    out: Dict[int, Dict[str, List[int]]] = {}
    for k in range(class_count):
        s = adj_cfg.get(str(k), {}) if isinstance(adj_cfg, dict) else {}
        out[k] = {
            "N": s.get("N", all_ids[:]),
            "S": s.get("S", all_ids[:]),
            "E": s.get("E", all_ids[:]),
            "W": s.get("W", all_ids[:]),
        }
    return out

def relax_neighbors(orig: Dict[int, Dict[str, List[int]]],
                    class_count: int,
                    pct: float,
                    rng: np.random.Generator) -> Dict[int, Dict[str, List[int]]]:
    """Add a fraction of missing classes into each direction's allow-list."""
    pct = max(0.0, min(1.0, float(pct)))
    all_ids = np.arange(class_count, dtype=int)
    relaxed: Dict[int, Dict[str, List[int]]] = {}
    for c, dirs in orig.items():
        relaxed[c] = {}
        for d, lst in dirs.items():
            s = set(lst)
            if pct >= 1.0:
                relaxed[c][d] = list(all_ids)
                continue
            missing = [i for i in all_ids if i not in s]
            k = int(math.ceil(pct * len(missing)))
            if k > 0:
                add = rng.choice(missing, size=k, replace=False).tolist()
                s.update(add)
            relaxed[c][d] = sorted(list(s))
    return relaxed

# -------------------------- Main --------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    slice_rows_main = None
    if args.main_slice_rows and args.main_slice_rows > 0:
        slice_rows_main = int(args.main_slice_rows)
    elif args.main_slice_px and args.main_slice_px > 0:
        # derive rows from viewport px using horizontal px_per_tile = canvas_width / tiles_x
        px_per_tile = max(1e-6, float(args.canvas_width) / float(args.tiles_x))
        slice_rows_main = max(1, int(round(float(args.main_slice_px) / px_per_tile)))

    outdir = Path(args.outdir); ensure_dir(outdir)
    prob_cfg = load_json(Path(args.prob))
    adj_cfg  = load_json(Path(args.adj))

    class_map_cfg = prob_cfg.get("class_map", {})
    class_map: Dict[str, int] = {}
    if isinstance(class_map_cfg, dict):
        for name, idx in class_map_cfg.items():
            try:
                class_map[str(name)] = int(idx)
            except Exception:
                continue
    if class_map:
        C = max(class_map.values()) + 1
    else:
        C = int(prob_cfg.get("class_count", 6))

    id_to_label: Dict[int, str] = {}
    if class_map:
        for name, idx in class_map.items():
            try:
                id_to_label[int(idx)] = str(name)
            except Exception:
                continue
    if not id_to_label:
        fallback_order = ["button", "input", "media", "role_button", "select", "text", "textarea"]
        for cid in range(C):
            if cid < len(fallback_order):
                id_to_label[cid] = fallback_order[cid]
            else:
                id_to_label[cid] = f"class_{cid}"
    if not class_map:
        class_map = {label: cid for cid, label in id_to_label.items()}
    else:
        for cid, label in id_to_label.items():
            class_map.setdefault(label, cid)

    bands = compute_band_specs(args.canvas_width, args.tiles_x, prob_cfg)

    media_tweak = parse_media_add(args.media_add)
    per_band_prior: Dict[str, np.ndarray] = {}
    for b in bands:
        pw = prob_cfg.get("type_weights", {}).get(b.name, {})
        if pw and class_map:
            vec = np.zeros(C, dtype=np.float64)
            for nm, w in pw.items():
                if nm in class_map: vec[class_map[nm]] = float(w)
            vec = adjust_media_weights(b.name, media_tweak, class_map, vec)
            per_band_prior[b.name] = normalize(vec); continue
        pv = prob_cfg.get("priors", {}).get(b.name)
        if isinstance(pv, list) and len(pv) == C:
            per_band_prior[b.name] = normalize(np.array(pv, dtype=np.float64)); continue
        per_band_prior[b.name] = np.ones(C, dtype=np.float64) / C

    neighbors_strict = build_neighbors(adj_cfg, C)

    files_tiles: Dict[str, str] = {}
    tiles_by_band: Dict[str, np.ndarray] = {}
    uniq_ids = set()

    for b in bands:
        tiles_w_tgt, tiles_h_tgt = b.tiles_w, b.rows0
        cf = max(1, int(args.coarse_factor))
        prior = per_band_prior[b.name]

        def solve_slice(h_target_rows: int, fixed_top: Optional[np.ndarray]):
            tiles_w = max(1, int(math.ceil(tiles_w_tgt / cf)))
            tiles_h = max(1, int(math.ceil(h_target_rows / cf)))
            temp0 = args.wfc_temp
            solved_local = None
            for attempt in range(1, max(1, args.restarts) + 1):
                if attempt == 1 or not args.relax_on_fail:
                    NN = neighbors_strict
                    temp = temp0
                    relax_note = ""
                else:
                    pct = min(1.0, args.relax_pct * (attempt - 1) / max(1, args.restarts - 1))
                    NN = relax_neighbors(neighbors_strict, C, pct, rng)
                    temp = min(1.25, temp0 + 0.05 * (attempt - 1))
                    relax_note = f" (relaxed adjacency pct={pct:.2f}, temp={temp:.2f})"

                wfc = WFCGrid(H=tiles_h, W=tiles_w, class_count=C, prior=prior,
                              neighbors=NN, rng=rng, temp=temp,
                              eps=args.wfc_eps, max_backtrack=args.wfc_backtrack)
                if fixed_top is not None:
                    for x in range(min(tiles_w, fixed_top.shape[0])):
                        cls = int(fixed_top[x])
                        wfc.poss[0, x] = False
                        wfc.poss[0, x][cls] = True
                        wfc.collapsed[0, x] = cls
                    if not wfc.propagate():
                        continue
                grid_coarse = wfc.solve()
                if grid_coarse is not None:
                    solved_local = (grid_coarse, relax_note, tiles_h, tiles_w)
                    break
            return solved_local

        slice_rows = slice_rows_main if b.name.lower() == "main" else None
        if slice_rows and tiles_h_tgt > slice_rows:
            grid_full = np.zeros((tiles_h_tgt, tiles_w_tgt), dtype=np.int32)
            remaining = tiles_h_tgt
            y_cursor = 0
            fixed_top = None
            slice_idx = 0
            while remaining > 0:
                h_slice = min(slice_rows, remaining)
                solved = solve_slice(h_slice, fixed_top)
                if solved is None:
                    print(f"[ERROR] WFC failed for band {b.name} slice {slice_idx} (rows {h_slice}).", file=sys.stderr)
                    sys.exit(2)
                grid_coarse, relax_note, tiles_h, tiles_w = solved
                grid_up = upscale_nearest(grid_coarse, h_slice, tiles_w_tgt, cf)
                grid_full[y_cursor:y_cursor + h_slice, :] = grid_up
                fixed_top = grid_coarse[-1, :].copy()
                y_cursor += h_slice
                remaining -= h_slice
                slice_idx += 1
            grid = grid_full
            note = f" (sliced {slice_idx}x ~{slice_rows} rows)"
        else:
            solved = solve_slice(tiles_h_tgt, None)
            if solved is None:
                print(f"[ERROR] WFC failed for band {b.name} after retries. "
                      f"Try higher --restarts or --relax_pct, or review adjacency/probabilities.", file=sys.stderr)
                sys.exit(2)
            grid_coarse, relax_note, tiles_h, tiles_w = solved
            grid = upscale_nearest(grid_coarse, tiles_h_tgt, tiles_w_tgt, cf)
            note = relax_note

        npy_path = outdir / f"{b.name}_tiles_seed{args.seed}.npy"
        np.save(npy_path, grid.astype(np.int32))
        files_tiles[b.name] = str(npy_path.resolve())
        tiles_by_band[b.name] = grid
        for v in np.unique(grid): uniq_ids.add(int(v))

        print(f"[OK] {b.name:<10} -> upscaled {tiles_h_tgt}x{tiles_w_tgt} -> {npy_path.name}{note}")

    uniq_ids_arr = np.array(sorted(list(uniq_ids)), dtype=int)
    palette = build_palette(prob_cfg, uniq_ids_arr, id_to_label)
    legend_info: List[Tuple[str, Tuple[int, int, int]]] = []
    seen_labels: Set[str] = set()
    for cid in uniq_ids_arr:
        label = id_to_label.get(int(cid), f"class_{int(cid)}")
        if label in seen_labels:
            continue
        seen_labels.add(label)
        legend_info.append((label, palette[int(cid)]))
    legend_info.sort(key=lambda item: item[0].lower())
    legend_entries = legend_info

    stitched_png = outdir / f"debug_seed{args.seed}.png"
    stitch_list = [(b.name, tiles_by_band[b.name]) for b in bands]
    png_scale_used = render_stitched_debug(stitch_list, palette, stitched_png,
                                           scale=args.png_scale,
                                           legend=legend_entries,
                                           target_width_px=args.canvas_width)
    print(f"[OK] stitched debug: {stitched_png} (scale={png_scale_used})")

    mani = {
        "seed": args.seed,
        "tiles_x": args.tiles_x,
        "png_scale": int(png_scale_used),
        "class_map": class_map,
        "bands": [
            {
                "name": b.name,
                "y0_px": b.y0_px, "y1_px": b.y1_px,
                "rows0": b.rows0, "tiles_w": b.tiles_w, "tiles_h": b.rows0
            } for b in bands
        ],
        "files": {"tiles": files_tiles, "debug_png": str(stitched_png.resolve())},
        "notes": f"WFC stage (coarse_factor={int(args.coarse_factor)})"
    }
    mani_path = outdir / f"manifest_seed{args.seed}.json"
    save_json(mani_path, mani)
    print(f"[OK] manifest: {mani_path}\nWFC stage complete")

if __name__ == "__main__":
    main()


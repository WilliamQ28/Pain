#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate per-band, per-type spatial densities into block/tile heatmaps + summaries.
Adds --debug logging and periodic heartbeats with --every.

Usage (Windows):
  python aggregate_element_layout.py ^
    --labels "C:\wfc\project-2-at-2025-10-20-22-45-dff17110.json" ^
    --rgdir "C:\wfc\capture\rg" ^
    --outdir "C:\wfc\out\agg_blocks" ^
    --tiles_x 64 --tiles_y 16 ^
    --quickviz --debug --every 20
"""

import argparse, json, csv, re, urllib.parse, time, os, sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw

# ---------- Element type encoding (G-channel anchors) ----------
TYPE_BY_G = {
    0:   "bg",
    16:  "text",
    32:  "media",
    48:  "button",
    64:  "input",
    80:  "select",
    96:  "textarea",
    112: "role_button",
}
G_KEYS = sorted(TYPE_BY_G.keys())

def nearest_anchor(g: int) -> int:
    gq = int(round(g / 8.0)) * 8
    gq = max(0, min(248, gq))
    return min(G_KEYS, key=lambda a: abs(a - gq))

# ---------- I/O helpers ----------
def load_rg(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGBA"))

def write_csv_matrix(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for r in arr:
            w.writerow([f"{x:.6f}" for x in r])

# ---------- Label Studio path normalization ----------
def _basename_from_ls(value: str) -> str:
    """
    Accepts:
      - /data/local-files/?d=C:\\wfc\\capture\\raw\\python-org.png
      - file:///C:/wfc/capture/raw/python-org.png
      - /data/upload/2/0d1bb443-nodered-org.png
    Returns basename only, e.g., 'python-org.png'.
    """
    if not value:
        return ""
    s = str(value)

    if "/data/local-files/" in s:
        parts = urllib.parse.urlsplit(s)
        q = urllib.parse.parse_qs(parts.query)
        if "d" in q and q["d"]:
            s = q["d"][0]

    if s.startswith("file://"):
        s = urllib.parse.urlsplit(s).path
        if re.match(r"^/[A-Za-z]:/", s):
            s = s[1:]

    s = urllib.parse.unquote(s).replace("\\", "/")
    return s.split("/")[-1]

def _resolve_rg_path(rgdir: Path, any_name: str) -> Path | None:
    """
    Try to find the RG image by basename (case-insensitive), allowing .png/.jpg/.jpeg.
    Only searches inside rgdir.
    """
    if not any_name:
        return None
    base = Path(any_name).name
    stem = Path(base).stem.lower()
    for ext in (".png", ".jpg", ".jpeg"):
        p = rgdir / f"{stem}{ext}"
        if p.exists():
            return p
    # last resort scan (avoid if huge)
    for p in rgdir.glob("*"):
        if p.is_file() and p.stem.lower() == stem:
            return p
    return None

# ---------- Label Studio band iterators (support both formats) ----------
def iter_bands_flat(task: dict):
    """
    Flat export shape:
      task["image"] -> path
      task["band"]  -> list of {x,y,width,height,rectanglelabels,original_width,original_height}
    y/height in %.
    """
    rects = task.get("band") or []
    if not isinstance(rects, list):
        return
    for v in rects:
        labels = v.get("rectanglelabels") or []
        if not labels:
            continue
        label = labels[0]
        y = v.get("y"); h = v.get("height")
        if y is None or h is None:
            continue
        try:
            y = float(y); h = float(h)
        except:
            continue
        yield label, y, h, "%"

def iter_bands_annot(task: dict):
    """
    Standard LS shape:
      task["annotations"][].result[] with type == 'rectanglelabels'
      value.y/value.height in % by default (unit present sometimes).
    """
    anns = task.get("annotations") or []
    for ann in anns:
        for r in (ann.get("result") or []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r.get("value") or {}
            labels = v.get("rectanglelabels") or []
            if not labels:
                continue
            label = labels[0]
            y = v.get("y"); h = v.get("height")
            unit = v.get("unit") or v.get("original_units") or "%"
            if y is None or h is None:
                continue
            try:
                y = float(y); h = float(h)
            except:
                continue
            yield label, y, h, unit

def main():
    ap = argparse.ArgumentParser(description="Per-band element density aggregation into tiles.")
    ap.add_argument("--labels", nargs="+", required=True, help="Label Studio JSON export(s)")
    ap.add_argument("--rgdir", required=True, help="Folder with RG-encoded screenshots")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tiles_x", type=int, default=64)
    ap.add_argument("--tiles_y", type=int, default=16)
    ap.add_argument("--quickviz", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Verbose progress logging")
    ap.add_argument("--every", type=int, default=25, help="Heartbeat frequency in tasks")
    args = ap.parse_args()

    t0 = time.time()
    rgdir = Path(args.rgdir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    blocks_dir = outdir / "blocks"; blocks_dir.mkdir(exist_ok=True)
    summaries_dir = outdir / "summaries"; summaries_dir.mkdir(exist_ok=True)
    global_dir = outdir / "global"; global_dir.mkdir(exist_ok=True)

    if args.debug:
        print(f"[dbg] rgdir={rgdir}", flush=True)
        print(f"[dbg] outdir={outdir}", flush=True)

    # accumulators
    accum = defaultdict(lambda: defaultdict(lambda: np.zeros((args.tiles_y, args.tiles_x), dtype=np.float64)))
    band_contrib = Counter()
    area_records = []
    height_records = []

    matched_images = 0
    matched_bands = 0
    total_tasks = 0
    total_with_bands = 0

    # Load tasks
    all_tasks = []
    for lp in args.labels:
        if args.debug:
            print(f"[dbg] loading labels: {lp}", flush=True)
        txt = Path(lp).read_text(encoding="utf-8")
        try:
            j = json.loads(txt)
        except Exception as e:
            print(f"[err] failed to parse {lp}: {e}", flush=True)
            continue
        if isinstance(j, list):
            all_tasks.extend(j)
        else:
            all_tasks.extend(j.get("tasks") or [])

    if args.debug:
        print(f"[dbg] total tasks loaded: {len(all_tasks)}", flush=True)

    # Process tasks
    for idx, t in enumerate(all_tasks, 1):
        total_tasks += 1

        img_field = t.get("image") or (t.get("data", {}) or {}).get("image") or t.get("file_upload")
        basename = _basename_from_ls(img_field)
        rgp = _resolve_rg_path(rgdir, basename)

        if not rgp:
            if args.debug:
                print(f"[dbg] ({idx}) no RG found for image='{img_field}' basename='{basename}'", flush=True)
            continue

        try:
            arr = load_rg(rgp)
        except Exception as e:
            print(f"[err] ({idx}) failed to open RG image {rgp}: {e}", flush=True)
            continue

        H, W, _ = arr.shape
        G = arr[:, :, 1]
        matched_images += 1

        # Gather bands
        rects = list(iter_bands_flat(t))
        if not rects:
            rects = list(iter_bands_annot(t))

        if not rects:
            if args.debug:
                print(f"[dbg] ({idx}) image '{basename}': 0 bands", flush=True)
            continue

        total_with_bands += 1
        if args.debug:
            print(f"[dbg] ({idx}) image '{basename}': {len(rects)} bands; H={H} W={W}", flush=True)

        # Process bands
        for b_i, (label, y_val, h_val, unit) in enumerate(rects, 1):
            if str(unit).lower().startswith("px"):
                y1 = max(0, min(H, int(round(y_val))))
                y2 = max(0, min(H, int(round(y_val + h_val))))
            else:
                y1 = int(round((y_val/100.0)*H))
                y2 = int(round(((y_val+h_val)/100.0)*H))
                y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
            if y2 <= y1:
                if args.debug:
                    print(f"[dbg]   band#{b_i} '{label}' skipped (y2<=y1) y1={y1} y2={y2}", flush=True)
                continue

            matched_bands += 1
            band_h = y2 - y1
            band_px = band_h * W
            if args.debug and b_i <= 3:  # only show first few per task to avoid spam
                print(f"[dbg]   band#{b_i} '{label}' y1={y1} y2={y2} h={band_h}px area={band_px}", flush=True)

            # (1) per-band element area summary
            gh = G[y1:y2, :].reshape(-1)
            cnt = Counter(int(nearest_anchor(int(v))) for v in gh)

            rec = {"file_upload": basename, "band_label": label, "px_total": int(band_px)}
            for a in G_KEYS:
                tname = TYPE_BY_G[a]
                rec[f"px_{tname}"] = int(cnt.get(a, 0))
            denom = float(band_px) if band_px > 0 else 1.0
            for a in G_KEYS:
                tname = TYPE_BY_G[a]
                rec[f"frac_{tname}"] = rec[f"px_{tname}"] / denom
            area_records.append(rec)

            height_records.append({
                "file_upload": basename, "band_label": label,
                "band_px": int(band_px), "band_h": int(band_h), "W": int(W)
            })

            # (2) tile/block accumulation
            tiles_x, tiles_y = args.tiles_x, args.tiles_y
            tile_w = W / tiles_x
            tile_h = band_h / tiles_y
            view = G[y1:y2, :]

            for ty in range(tiles_y):
                r0 = int(round(ty * tile_h))
                r1 = int(round((ty + 1) * tile_h))
                if r1 <= r0:
                    continue
                row_slice = view[r0:r1, :]
                if row_slice.size == 0:
                    continue
                for tx in range(tiles_x):
                    c0 = int(round(tx * tile_w))
                    c1 = int(round((tx + 1) * tile_w))
                    if c1 <= c0:
                        continue
                    tile = row_slice[:, c0:c1]
                    if tile.size == 0:
                        continue
                    flat = tile.reshape(-1)
                    local = Counter()
                    q = np.round(flat / 8.0).astype(int) * 8
                    for val in q:
                        if val < 0: val = 0
                        if val > 248: val = 248
                        a = min(G_KEYS, key=lambda A: abs(A - int(val)))
                        local[a] += 1
                    for a, n in local.items():
                        tname = TYPE_BY_G[a]
                        accum[label][tname][ty, tx] += n
            band_contrib[label] += 1

        # heartbeat
        if args.every > 0 and (idx % args.every == 0):
            elapsed = time.time() - t0
            print(f"[hb] tasks={idx}/{len(all_tasks)} imgs={matched_images} bands={matched_bands} "
                  f"with_bands={total_with_bands} elapsed={elapsed:.1f}s", flush=True)

    # ---------- finalize ----------
    print(f"[info] matched_images={matched_images}, matched_bands={matched_bands}, "
          f"tasks_total={total_tasks}, tasks_with_bands={total_with_bands}", flush=True)

    # Write blocks (normalized 0..1 by max per (band,type))
    for label, type_maps in accum.items():
        band_dir = (outdir / "blocks" / label)
        band_dir.mkdir(parents=True, exist_ok=True)
        for tname, M in type_maps.items():
            D = (M / M.max()).astype(np.float64) if M.max() > 0 else M.astype(np.float64)
            np.save(band_dir / f"{tname}.npy", D)
            write_csv_matrix(band_dir / f"{tname}.csv", D)
            if args.debug:
                print(f"[dbg] wrote {label}/{tname} -> {band_dir}", flush=True)

            if args.quickviz:
                scale = 8
                Ht, Wt = D.shape
                img = (D * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img, mode="L").resize((Wt*scale, Ht*scale), resample=Image.NEAREST).convert("RGB")
                draw = ImageDraw.Draw(img)
                draw.text((4, 4), f"{label}/{tname}", fill=(255, 255, 0))
                img.save((outdir / "global" / f"{label}__{tname}__avg.png"))

    # Write summaries
    if area_records:
        cols = ["file_upload","band_label","px_total"] + \
               [f"px_{TYPE_BY_G[a]}" for a in G_KEYS] + \
               [f"frac_{TYPE_BY_G[a]}" for a in G_KEYS]
        with (outdir / "summaries" / "band_area_fractions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(area_records)
        if args.debug:
            print(f"[dbg] wrote summaries/band_area_fractions.csv ({len(area_records)} rows)", flush=True)

    if height_records:
        cols = ["file_upload","band_label","band_px","band_h","W"]
        with (outdir / "summaries" / "band_height_stats.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(height_records)
        if args.debug:
            print(f"[dbg] wrote summaries/band_height_stats.csv ({len(height_records)} rows)", flush=True)

    elapsed = time.time() - t0
    print("[OK] Wrote block density maps to:", str(outdir / "blocks"), flush=True)
    print("[OK] Summaries in:", str(outdir / "summaries"), flush=True)
    print(f"[done] total_elapsed={elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()

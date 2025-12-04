#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create color-agnostic variants to prevent color memorization:
  1) Grayscale replicated to 3 channels: name__gray.ext
  2) Keep luminance, randomize chroma slightly: name__Lrand.ext

Usage:
  python color_decolorize.py --dataset C:\wfc\datasets --split train --ab_amp 20 --p_shuffle 0.0 --b_gain 0.95 1.05 --c_gain 0.95 1.05
"""

import argparse, random, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

def to_gray3(im: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(im)
    return Image.merge("RGB", (g, g, g))

def lab_like_L_constant_chroma_rand(im: Image.Image, ab_amp: int) -> Image.Image:
    # Use YCbCr as a LAB-like proxy (no extra deps): keep Y (luminance), jitter Cb/Cr in [-ab_amp, ab_amp]
    im = im.convert("RGB")
    ycbcr = im.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    cb_arr = np.array(cb, dtype=np.int16)
    cr_arr = np.array(cr, dtype=np.int16)
    off_cb = random.randint(-ab_amp, ab_amp)
    off_cr = random.randint(-ab_amp, ab_amp)
    cb_arr = np.clip(cb_arr + off_cb, 0, 255).astype(np.uint8)
    cr_arr = np.clip(cr_arr + off_cr, 0, 255).astype(np.uint8)
    out = Image.merge("YCbCr", (y, Image.fromarray(cb_arr), Image.fromarray(cr_arr))).convert("RGB")
    return out

def maybe_channel_shuffle(im: Image.Image, p: float) -> Image.Image:
    if random.random() >= p:
        return im
    r, g, b = im.split()
    orders = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    o = random.choice(orders)
    chans = [r, g, b]
    return Image.merge("RGB", (chans[o[0]], chans[o[1]], chans[o[2]]))

def apply_bc(im: Image.Image, b_rng: tuple, c_rng: tuple) -> Image.Image:
    if b_rng:
        im = ImageEnhance.Brightness(im).enhance(random.uniform(*b_rng))
    if c_rng:
        im = ImageEnhance.Contrast(im).enhance(random.uniform(*c_rng))
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"C:\wfc\datasets")
    ap.add_argument("--split", default="train", choices=["train","val"])
    ap.add_argument("--ab_amp", type=int, default=20, help="± range for chroma jitter")
    ap.add_argument("--p_shuffle", type=float, default=0.0, help="probability of RGB channel shuffle")
    ap.add_argument("--b_gain", nargs=2, type=float, default=[0.95, 1.05], help="brightness gain range")
    ap.add_argument("--c_gain", nargs=2, type=float, default=[0.95, 1.05], help="contrast gain range")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    img_dir = Path(args.dataset) / "images" / args.split
    lbl_dir = Path(args.dataset) / "labels" / args.split
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}

    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"[ERR] Missing split dirs: {img_dir} / {lbl_dir}")

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    made, skipped = 0, 0

    for ip in imgs:
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists():
            skipped += 1
            continue

        base = Image.open(ip).convert("RGB")

        # Variant 1: grayscale (3ch) + light B/C (keeps structure, kills hue)
        v1 = to_gray3(base)
        v1 = apply_bc(v1, tuple(args.b_gain), tuple(args.c_gain))
        v1 = maybe_channel_shuffle(v1, args.p_shuffle)
        out1 = img_dir / f"{ip.stem}__gray{ip.suffix}"
        v1.save(out1, **({"quality":95} if out1.suffix.lower() in {".jpg",".jpeg"} else {}))
        shutil.copyfile(lp, lbl_dir / f"{ip.stem}__gray.txt")
        made += 1

        # Variant 2: luminance preserved, chroma randomized + light B/C
        v2 = lab_like_L_constant_chroma_rand(base, args.ab_amp)
        v2 = apply_bc(v2, tuple(args.b_gain), tuple(args.c_gain))
        v2 = maybe_channel_shuffle(v2, args.p_shuffle)
        out2 = img_dir / f"{ip.stem}__Lrand{ip.suffix}"
        v2.save(out2, **({"quality":95} if out2.suffix.lower() in {".jpg",".jpeg"} else {}))
        shutil.copyfile(lp, lbl_dir / f"{ip.stem}__Lrand.txt")
        made += 1

    print(f"[OK] Created {made} variants in '{args.split}'. Skipped (no label): {skipped}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# color_augment.py
import argparse, random, shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

def hue_shift(im: Image.Image, deg: float) -> Image.Image:
    hsv = im.convert("HSV"); h,s,v = hsv.split()
    h = h.point(lambda p: (p + int(deg/360.0*255)) % 255)
    return Image.merge("HSV", (h,s,v)).convert("RGB")

def clamp01(x): return max(0.0, min(2.0, x))

def make_variants(im: Image.Image, seed: int, k: int):
    random.seed(seed)
    pool = []
    pool.append(("hue", hue_shift(im, random.choice([30,60,120,180]))))
    sat = ImageEnhance.Color(im).enhance(clamp01(0.8 + random.random()*0.6))
    bri = ImageEnhance.Brightness(sat).enhance(clamp01(0.9 + random.random()*0.4))
    con = ImageEnhance.Contrast(bri).enhance(clamp01(0.9 + random.random()*0.4))
    pool.append(("jitter", con))
    pool.append(("gray", ImageOps.grayscale(im).convert("RGB")))
    tint = Image.new("RGB", im.size, random.choice([(230,240,255),(255,240,230),(235,255,240)]))
    pool.append(("tint", Image.blend(im, tint, alpha=0.15)))
    random.seed(seed ^ 0xA5A5A5A5)
    idx = list(range(len(pool))); random.shuffle(idx)
    return [pool[i] for i in idx[:k]]

def main():
    ap = argparse.ArgumentParser(description="Color-only YOLO augmentation (labels unchanged).")
    ap.add_argument("--dataset", default=r"C:\wfc\datasets")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per_image", type=int, default=2)
    ap.add_argument("--exts", default=".jpg,.jpeg,.png")
    args = ap.parse_args()

    root = Path(args.dataset)
    img_dir = root/"images"/args.split
    lab_dir = root/"labels"/args.split
    assert img_dir.exists() and lab_dir.exists(), "Run split_dataset.py first."

    exts = tuple(e.strip().lower() for e in args.exts.split(","))
    total_src = total_new = 0

    for ip in sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts]):
        lp = lab_dir/(ip.stem + ".txt")
        if not lp.exists(): 
            continue
        total_src += 1
        im = Image.open(ip).convert("RGB")
        variants = make_variants(im, seed=hash(ip.stem) & 0xffffffff, k=args.per_image)
        for tag, aug in variants:
            out_img = img_dir/(ip.stem + f"__aug_{tag}" + ip.suffix.lower())
            out_lab = lab_dir/(ip.stem + f"__aug_{tag}.txt")
            aug.save(out_img, quality=95)
            shutil.copyfile(lp, out_lab)
            total_new += 1

    print(f"[OK] Augmented {total_src} originals → {total_new} new images in '{args.split}'")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# top_up_to_total.py
import argparse, random, shutil
from pathlib import Path
from PIL import Image
from color_augment import make_variants

def count_images(root: Path):
    def n(p): return len([x for x in p.iterdir() if x.suffix.lower() in (".jpg",".jpeg",".png")])
    ti = n(root/"images"/"train"); vi = n(root/"images"/"val")
    return ti + vi, ti, vi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"C:\wfc\datasets")
    ap.add_argument("--target_total", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.dataset)
    img_tr = root/"images"/"train"
    lab_tr = root/"labels"/"train"
    assert img_tr.exists() and lab_tr.exists(), "Run split + base augmentation first."

    total, train_ct, val_ct = count_images(root)
    need = args.target_total - total
    print(f"[INFO] Current total={total} (train={train_ct}, val={val_ct}); target={args.target_total}; need={need}")
    if need <= 0:
        print("[OK] Already at or above target."); return

    originals = [p for p in img_tr.iterdir()
                 if p.suffix.lower() in (".jpg",".jpeg",".png") and "__aug_" not in p.stem]
    if not originals:
        print("No originals found in train to top-up."); return

    random.seed(args.seed)
    picks = [random.choice(originals) for _ in range(need)]
    added = 0
    for ip in picks:
        lp = lab_tr/(ip.stem + ".txt")
        if not lp.exists(): continue
        im = Image.open(ip).convert("RGB")
        tag, aug = make_variants(im, seed=random.randint(0, 2**31-1), k=1)[0]
        out_img = img_tr/(ip.stem + f"__aug_extra_{tag}" + ip.suffix.lower())
        out_lab = lab_tr/(ip.stem + f"__aug_extra_{tag}.txt")
        aug.save(out_img, quality=95)
        shutil.copyfile(lp, out_lab)
        added += 1

    total2, train2, val2 = count_images(root)
    print(f"[OK] Added {added} images. New totals: total={total2} (train={train2}, val={val2})")

if __name__ == "__main__":
    main()

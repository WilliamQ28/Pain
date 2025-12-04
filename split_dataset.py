#!/usr/bin/env python
# split_dataset.py
from pathlib import Path
import random, shutil, sys

ROOT = Path(r"C:\wfc\datasets")
IMG = ROOT / "images"
LAB = ROOT / "labels"
SPLITS = ["train", "val"]

def main(seed=42, train_ratio=0.8):
    assert IMG.exists() and LAB.exists(), "Expected C:\\wfc\\datasets\\images and \\labels folders"
    imgs = sorted([p for p in IMG.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
    if not imgs:
        print("No images found in datasets\\images"); sys.exit(1)

    random.seed(seed); random.shuffle(imgs)
    k = int(train_ratio * len(imgs))
    plan = {"train": imgs[:k], "val": imgs[k:]}

    for sp in SPLITS:
        (IMG/sp).mkdir(parents=True, exist_ok=True)
        (LAB/sp).mkdir(parents=True, exist_ok=True)

    moved = 0
    for sp, files in plan.items():
        for ip in files:
            lp = LAB / (ip.stem + ".txt")
            if not lp.exists(): 
                print(f"[WARN] label missing for {ip.name}")
                continue
            shutil.move(str(ip), str((IMG/sp)/ip.name))
            shutil.move(str(lp), str((LAB/sp)/(lp.name)))
            moved += 1

    print(f"[OK] Split complete. Train={len(plan['train'])}, Val={len(plan['val'])}, moved={moved}")

if __name__ == "__main__":
    main()

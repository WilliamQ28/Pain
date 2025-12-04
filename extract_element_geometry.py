#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract element geometry statistics from YOLO bounding boxes.

Analyzes aspect ratios, size distributions, and spatial properties
from YOLO-labeled screenshots to inform WFC generation constraints.

Usage:
    python extract_element_geometry.py \
        --labels_dir datasets/labels/train \
        --images_dir datasets/images/train \
        --classes datasets/classes.txt \
        --output P2/wfc_stats/element_geometry.json \
        --rg_dir capture/rg
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


# Band class mapping (YOLO class IDs)
BAND_CLASSES = {
    0: "Banner",
    1: "Footer",
    2: "Header",
    3: "Main",
    4: "Nav",
    5: "Supplement"
}

# Element type mapping from RG G-channel
G_TO_TYPE = {
    16:  "text",
    32:  "media",
    48:  "button",
    64:  "input",
    80:  "select",
    96:  "textarea",
    112: "role_button"
}


def parse_args():
    ap = argparse.ArgumentParser(description="Extract element geometry from YOLO labels")
    ap.add_argument("--labels_dir", required=True, help="Path to YOLO labels directory")
    ap.add_argument("--images_dir", required=True, help="Path to images directory")
    ap.add_argument("--classes", required=True, help="Path to classes.txt")
    ap.add_argument("--output", default="P2/wfc_stats/element_geometry.json", help="Output JSON path")
    ap.add_argument("--rg_dir", default="capture/rg", help="RG-encoded screenshots directory")
    ap.add_argument("--min_samples", type=int, default=5, help="Minimum samples for statistics")
    return ap.parse_args()


def load_class_names(classes_path: Path) -> List[str]:
    """Load class names from classes.txt"""
    with open(classes_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def parse_yolo_label(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """Parse YOLO format: class_id x_center y_center width height (normalized 0-1)"""
    try:
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        return class_id, x_center, y_center, width, height
    except (ValueError, IndexError):
        return None


def denormalize_bbox(x_center: float, y_center: float, width: float, height: float,
                     img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO bbox to absolute pixel coordinates"""
    x_px = int((x_center - width / 2) * img_width)
    y_px = int((y_center - height / 2) * img_height)
    w_px = int(width * img_width)
    h_px = int(height * img_height)
    return x_px, y_px, w_px, h_px


def sample_element_types_from_rg(rg_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Dict[str, int]:
    """Sample G-channel from RG image to determine element types within bbox"""
    x, y, w, h = bbox
    rg_arr = np.array(rg_img.convert("RGB"))

    # Crop to bbox (with bounds checking)
    img_h, img_w = rg_arr.shape[:2]
    x1 = max(0, min(x, img_w))
    y1 = max(0, min(y, img_h))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))

    if x2 <= x1 or y2 <= y1:
        return {}

    region = rg_arr[y1:y2, x1:x2, 1]  # G-channel

    # Count element types
    type_counts = defaultdict(int)
    for g_val in np.unique(region):
        # Find nearest G anchor
        g_rounded = int(round(g_val / 8.0)) * 8
        if g_rounded in G_TO_TYPE:
            elem_type = G_TO_TYPE[g_rounded]
            count = int(np.sum(region == g_val))
            type_counts[elem_type] += count

    return dict(type_counts)


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute common percentiles"""
    if not values:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95))
    }


def classify_aspect_ratio(aspect: float) -> str:
    """Classify aspect ratio into common categories"""
    if aspect < 0.9:
        return "portrait"
    elif aspect < 1.1:
        return "1:1"
    elif 1.5 < aspect < 1.9:
        return "16:9"
    elif 1.2 < aspect < 1.4:
        return "4:3"
    elif aspect > 4.0:
        return "wide"
    else:
        return "other"


def analyze_geometry(labels_dir: Path, images_dir: Path, rg_dir: Path,
                     class_names: List[str], min_samples: int) -> Dict:
    """Main analysis function"""

    # Data collectors
    by_band = defaultdict(lambda: {
        "aspect_ratios": [],
        "widths_px": [],
        "heights_px": [],
        "widths_rel": [],
        "heights_rel": [],
        "areas_rel": [],
        "x_positions": [],
        "y_positions": []
    })

    by_band_and_type = defaultdict(lambda: defaultdict(lambda: {
        "aspect_ratios": [],
        "widths_px": [],
        "heights_px": []
    }))

    total_processed = 0
    total_boxes = 0

    # Process all label files
    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files")

    for label_file in label_files:
        # Skip augmented versions for cleaner stats
        if "__gray" in label_file.stem or "__Lrand" in label_file.stem:
            continue

        # Find corresponding image
        img_file = images_dir / f"{label_file.stem}.png"
        if not img_file.exists():
            img_file = images_dir / f"{label_file.stem}.jpg"
        if not img_file.exists():
            continue

        # Load image dimensions
        try:
            img = Image.open(img_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error loading {img_file}: {e}", file=sys.stderr)
            continue

        # Try to load RG version
        rg_file = rg_dir / f"{label_file.stem}.png"
        rg_img = None
        if rg_file.exists():
            try:
                rg_img = Image.open(rg_file)
            except Exception:
                pass

        # Parse labels
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = parse_yolo_label(line)
                if not parsed:
                    continue

                class_id, x_c, y_c, w_norm, h_norm = parsed

                # Get band name
                band = BAND_CLASSES.get(class_id, f"class_{class_id}")

                # Convert to pixels
                x_px, y_px, w_px, h_px = denormalize_bbox(x_c, y_c, w_norm, h_norm, img_width, img_height)

                # Skip invalid boxes
                if w_px <= 0 or h_px <= 0:
                    continue

                # Calculate metrics
                aspect = w_px / h_px
                area_rel = w_norm * h_norm

                # Store band-level stats
                by_band[band]["aspect_ratios"].append(aspect)
                by_band[band]["widths_px"].append(w_px)
                by_band[band]["heights_px"].append(h_px)
                by_band[band]["widths_rel"].append(w_norm)
                by_band[band]["heights_rel"].append(h_norm)
                by_band[band]["areas_rel"].append(area_rel)
                by_band[band]["x_positions"].append(x_c)
                by_band[band]["y_positions"].append(y_c)

                # Sample element types from RG if available
                if rg_img:
                    type_counts = sample_element_types_from_rg(rg_img, (x_px, y_px, w_px, h_px))
                    if type_counts:
                        # Use dominant type
                        dominant_type = max(type_counts, key=type_counts.get)
                        by_band_and_type[band][dominant_type]["aspect_ratios"].append(aspect)
                        by_band_and_type[band][dominant_type]["widths_px"].append(w_px)
                        by_band_and_type[band][dominant_type]["heights_px"].append(h_px)

                total_boxes += 1

        total_processed += 1
        if total_processed % 50 == 0:
            print(f"Processed {total_processed}/{len(label_files)} files, {total_boxes} boxes")

    print(f"\nTotal: {total_processed} files, {total_boxes} bounding boxes")

    # Compile results
    results = {
        "metadata": {
            "files_processed": total_processed,
            "total_boxes": total_boxes,
            "min_samples_threshold": min_samples
        },
        "by_band": {},
        "by_band_and_type": {}
    }

    # Aggregate band statistics
    for band, data in by_band.items():
        if len(data["aspect_ratios"]) < min_samples:
            continue

        # Aspect ratio distribution
        aspects = data["aspect_ratios"]
        aspect_categories = defaultdict(int)
        for a in aspects:
            cat = classify_aspect_ratio(a)
            aspect_categories[cat] += 1

        total_aspects = len(aspects)
        aspect_histogram = {k: v / total_aspects for k, v in aspect_categories.items()}

        # Detect if dimensions are fixed or variable
        height_std = np.std(data["heights_px"])
        width_std = np.std(data["widths_px"])

        results["by_band"][band] = {
            "count": len(aspects),
            "aspect_ratios": {
                "histogram": aspect_histogram,
                **compute_percentiles(aspects)
            },
            "width_px": compute_percentiles(data["widths_px"]),
            "height_px": compute_percentiles(data["heights_px"]),
            "width_relative": compute_percentiles(data["widths_rel"]),
            "height_relative": compute_percentiles(data["heights_rel"]),
            "area_relative": compute_percentiles(data["areas_rel"]),
            "position_x": compute_percentiles(data["x_positions"]),
            "position_y": compute_percentiles(data["y_positions"]),
            "sizing": {
                "height_fixed": bool(height_std < np.mean(data["heights_px"]) * 0.15),  # <15% variation
                "width_fixed": bool(width_std < np.mean(data["widths_px"]) * 0.15)
            }
        }

    # Aggregate band+type statistics
    for band, types in by_band_and_type.items():
        if band not in results["by_band_and_type"]:
            results["by_band_and_type"][band] = {}

        for elem_type, data in types.items():
            if len(data["aspect_ratios"]) < min_samples:
                continue

            aspects = data["aspect_ratios"]
            aspect_categories = defaultdict(int)
            for a in aspects:
                cat = classify_aspect_ratio(a)
                aspect_categories[cat] += 1

            total = len(aspects)

            results["by_band_and_type"][band][elem_type] = {
                "count": total,
                "aspect_ratios": {
                    "histogram": {k: v / total for k, v in aspect_categories.items()},
                    **compute_percentiles(aspects)
                },
                "width_px": compute_percentiles(data["widths_px"]),
                "height_px": compute_percentiles(data["heights_px"])
            }

    return results


def main():
    args = parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    classes_file = Path(args.classes)
    output_file = Path(args.output)
    rg_dir = Path(args.rg_dir)

    # Validate inputs
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not classes_file.exists():
        print(f"Error: Classes file not found: {classes_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting geometry from YOLO labels...")
    print(f"  Labels: {labels_dir}")
    print(f"  Images: {images_dir}")
    print(f"  RG screenshots: {rg_dir}")
    print()

    # Load class names
    class_names = load_class_names(classes_file)
    print(f"Loaded {len(class_names)} classes: {', '.join(class_names)}")
    print()

    # Analyze
    results = analyze_geometry(labels_dir, images_dir, rg_dir, class_names, args.min_samples)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Geometry statistics saved to: {output_file}")
    print(f"     Analyzed {results['metadata']['files_processed']} files")
    print(f"     Extracted {results['metadata']['total_boxes']} bounding boxes")
    print(f"     Generated stats for {len(results['by_band'])} bands")

    # Print summary
    print("\nSummary by band:")
    for band, stats in results["by_band"].items():
        aspect = stats["aspect_ratios"]["mean"]
        w = stats["width_px"]["mean"]
        h = stats["height_px"]["mean"]
        print(f"  {band:12} n={stats['count']:4d}  aspect={aspect:.2f}  size={w:.0f}x{h:.0f}px")


if __name__ == "__main__":
    main()

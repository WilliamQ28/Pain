#!/usr/bin/env python3
# qa_validate_and_preview.py
# Validates Label Studio band JSON ↔ RG images; emits a report and small preview artifacts.

import argparse, json, re, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # bar charts optional

def pct_to_rows(y_pct: float, h_pct: float, H: int):
    y1 = int(round((y_pct/100.0)*H))
    y2 = int(round(((y_pct+h_pct)/100.0)*H))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    if y2 < y1: y1, y2 = y2, y1
    return y1, y2

def load_rg(p: Path):
    return np.array(Image.open(p).convert("RGBA"))

def nice(s: str, n=80):
    return (s[:n] + "…") if len(s) > n else s

def main():
    ap = argparse.ArgumentParser(description="Validate band JSON against RG images; emit previews and a QA report.")
    ap.add_argument("--labels", nargs="+", required=True, help="Label Studio JSON export(s)")
    ap.add_argument("--rgdir", required=True, help="Folder containing RG images named as file_upload")
    ap.add_argument("--outdir", required=True, help="Where to write qa artifacts")
    ap.add_argument("--max_preview_sites", type=int, default=12, help="Limit overlays to avoid huge output")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    overlay_dir = outdir / "overlays"; overlay_dir.mkdir(exist_ok=True)
    charts_dir = outdir / "charts"; charts_dir.mkdir(exist_ok=True)

    rgdir = Path(args.rgdir)
    issues = []
    site_stats = []
    site_count = 0
    overlays_done = 0

    def push_issue(kind, site, detail):
        issues.append({"kind": kind, "site": site, "detail": detail})

    for lbl in args.labels:
        data = json.loads(Path(lbl).read_text(encoding="utf-8"))
        tasks = data if isinstance(data, list) else (data.get("tasks") or [data])

        for t in tasks:
            fu = t.get("file_upload") or (t.get("data", {}) or {}).get("image")
            if not fu:
                push_issue("missing_file_upload", "<unknown>", f"in {lbl}")
                continue
            fname = Path(fu).name
            rgp = rgdir / fname
            if not rgp.exists():
                push_issue("rg_missing", fname, f"expected {rgp}")
                continue

            arr = load_rg(rgp)
            H, W, _ = arr.shape
            G = arr[:,:,1]

            bands = []
            for ann in (t.get("annotations") or []):
                for r in (ann.get("result") or []):
                    if r.get("type") != "rectanglelabels": continue
                    v = r.get("value", {})
                    if "y" not in v or "height" not in v:
                        push_issue("missing_coords", fname, f"result without y/height: {nice(json.dumps(r))}")
                        continue
                    labels = v.get("rectanglelabels") or []
                    label = labels[0] if labels else "Unknown"
                    y_pct, h_pct = float(v["y"]), float(v["height"])
                    # schema checks
                    if not (0.0 <= y_pct <= 100.0): push_issue("y_out_of_range", fname, f"y={y_pct}")
                    if not (0.0 <= h_pct <= 100.0): push_issue("h_out_of_range", fname, f"h={h_pct}")
                    if y_pct + h_pct > 100.0001: push_issue("y_plus_h_gt_100", fname, f"y+h={y_pct+h_pct}")
                    y1, y2 = pct_to_rows(y_pct, h_pct, H)
                    if y2 <= y1:
                        push_issue("empty_band_pixels", fname, f"{label} → rows {y1}-{y2}")
                        continue
                    bands.append((label, y1, y2))

            if not bands:
                push_issue("no_bands", fname, "no rectanglelabels bands found")
                continue

            # sort by vertical position for order sanity
            bands.sort(key=lambda x: x[1])

            # Arithmetic invariants + quick density
            per_band = []
            for (label, y1, y2) in bands:
                cnt = Counter(G[y1:y2,:].flatten().tolist())
                total = int((y2-y1) * W)
                if total != sum(cnt.values()):
                    push_issue("hist_sum_mismatch", fname, f"{label}: total={total} sum(hist)={sum(cnt.values())}")
                # store top-3 bins for report
                top = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:3]
                per_band.append({
                    "label": label, "y1": y1, "y2": y2, "px": total,
                    "g_top": [{"g": int(k), "n": int(v), "pct": (v/total if total else 0.0)} for k, v in top]
                })

                # minimal bar chart (optional)
                if plt is not None:
                    if len(cnt) > 0:
                        # small chart per band
                        xs = [int(k) for k,_ in top]
                        ys = [int(v) for _,v in top]
                        fig = plt.figure(figsize=(2.6, 1.6), dpi=130)
                        ax = plt.gca()
                        ax.bar([str(x) for x in xs], ys)
                        ax.set_title(f"{label} top-G\n{fname}", fontsize=8)
                        ax.tick_params(axis='both', which='major', labelsize=7)
                        fig.tight_layout()
                        chart_p = charts_dir / f"{fname}-{label}-{y1}-{y2}.png"
                        fig.savefig(chart_p, bbox_inches="tight")
                        plt.close(fig)

            site_stats.append({
                "file_upload": fname, "H": H, "W": W,
                "bands": per_band
            })
            site_count += 1

            # Overlay preview (limit count)
            if overlays_done < args.max_preview_sites:
                base = Image.open(rgp).convert("RGB")
                draw = ImageDraw.Draw(base)
                # optional: font = ImageFont.load_default()  # keep default for portability
                for b in per_band:
                    y1, y2 = b["y1"], b["y2"]
                    draw.line([(0, y1), (base.width, y1)], fill=(0,255,0), width=2)
                    draw.line([(0, y2), (base.width, y2)], fill=(255,0,0), width=2)
                    label = f"{b['label']} ({y2-y1}px)"
                    draw.text((8, max(0, y1+3)), label, fill=(255,255,0))
                base.save(overlay_dir / fname)
                overlays_done += 1

    report = {
        "sites_checked": site_count,
        "issues": issues,
        "stats_sample": site_stats[:30],  # cap to keep report readable
        "overlay_previews": str(overlay_dir),
        "charts_dir": str(charts_dir),
    }
    (outdir / "qa_report.json").write_text(json.dumps(report, indent=2))
    print(f"[OK] QA done. Report → {outdir/'qa_report.json'}")
    print(f"[OK] Overlays in   → {overlay_dir}")
    if plt is not None:
        print(f"[OK] Charts in     → {charts_dir} (top-3 G bins per band)")
    if issues:
        print(f"[WARN] Found {len(issues)} issues; see qa_report.json")
    else:
        print("[OK] No issues found.")

if __name__ == "__main__":
    main()

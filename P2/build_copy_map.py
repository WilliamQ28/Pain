#!/usr/bin/env python3
"""
Build a simple copy map (child_id -> text/heading) from manifest + children.
By default it generates deterministic placeholder text sized to each child so
you can preview layouts without Lorem clones. Optionally, if --use_hf is set
and HUGGINGFACE_TOKEN is available, it will call the gen_text helper to get
model-generated copy per child.

Usage:
  python3 build_copy_map.py \
    --manifest p2_tiles/manifest_seed0_trimmed.json \
    --children p4_populate/children_l1_seed0.json \
    --out p4_populate/copy_seed0.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from render_html import load_json, estimate_char_capacity, LOREM

try:
    from NLP import gen_text
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


TARGET_CLASSES = {"text", "button", "role_button", "input", "select"}


def lorem_fill(char_cap: int) -> str:
    words = LOREM.split()
    out: List[str] = []
    total = 0
    idx = 0
    while total < max(20, char_cap):
        w = words[idx % len(words)]
        out.append(w)
        total += len(w) + 1
        idx += 1
    return " ".join(out)


def build_prompts(
    children: List[Dict[str, Any]],
    px_per_tile: float,
    content_type: str,
    tone: str,
    audience: str,
    notes: Optional[str],
    use_hf: bool,
    allowed_bands: Optional[set[str]] = None,
) -> Dict[str, Dict[str, str]]:
    copy_map: Dict[str, Dict[str, str]] = {}
    for ch in children:
        cid = ch.get("id")
        cls = ch.get("class")
        band = ch.get("band")
        if allowed_bands and band not in allowed_bands:
            continue
        if not cid or cls not in TARGET_CLASSES:
            continue
        shadow = ch.get("grid_l0_shadow") or ch.get("grid_l0") or {}
        cols = int(shadow.get("col_span", 1))
        rows = int(shadow.get("row_span", 1))
        # estimate capacity in rem space (render_html uses tile_rem = px/16)
        tile_rem = px_per_tile / 16.0
        char_cap = estimate_char_capacity(cols, rows, tile_rem, 1.0)
        heading = None
        body = None
        if use_hf and HF_AVAILABLE and os.getenv("HUGGINGFACE_TOKEN"):
            prompt = gen_text.build_prompt(
                content_type=content_type,
                char_budget=char_cap,
                tone=tone,
                audience=audience,
                context=[],
                notes=notes,
            )
            try:
                max_tokens = gen_text.budget_to_max_tokens(char_cap)
                body = gen_text.call_hf_chat(
                    prompt=prompt,
                    model_id=os.getenv("MODEL_ID", gen_text.DEFAULT_MODEL),
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
            except Exception:
                body = lorem_fill(char_cap)
        else:
            body = lorem_fill(char_cap)
        if not body:
            body = lorem_fill(char_cap)
        # simple heading if roomy and class is text/button-ish
        if char_cap > 250 and cls in {"text", "button", "role_button"}:
            heading = f"Heading for {cid}"
        payload: Dict[str, str] = {}
        if heading:
            payload["heading"] = heading
        if body:
            payload["body"] = body
        copy_map[cid] = payload or body or ""
    return copy_map


def parse_args():
    ap = argparse.ArgumentParser(description="Build copy map sized per child.")
    ap.add_argument("--manifest", required=True, help="Path to manifest_seed*.json")
    ap.add_argument("--children", required=True, help="Path to children_l1_seed*.json")
    ap.add_argument("--out", required=True, help="Where to write copy_map JSON")
    ap.add_argument("--content_type", default="tech hardware retailer", help="Content type for prompts")
    ap.add_argument("--tone", default="confident and clear", help="Tone for prompts")
    ap.add_argument("--audience", default="IT buyers", help="Audience for prompts")
    ap.add_argument("--notes", default="Highlight performance, reliability, and quick shipping.", help="Optional extra notes for the model")
    ap.add_argument("--bands", default="Main", help="Comma-separated bands to populate (default: Main).")
    ap.add_argument("--use_hf", action="store_true", help="Use HF router via gen_text if token is available")
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = load_json(Path(args.manifest))
    children = load_json(Path(args.children))
    px_per_tile = float(manifest.get("png_scale", 36) or 36)
    allowed_bands = None
    if args.bands:
        allowed_bands = {b.strip() for b in args.bands.split(",") if b.strip()}
    copy_map = build_prompts(
        children=children,
        px_per_tile=px_per_tile,
        content_type=args.content_type,
        tone=args.tone,
        audience=args.audience,
        notes=args.notes,
        use_hf=bool(args.use_hf),
        allowed_bands=allowed_bands,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(copy_map, indent=2), encoding="utf-8")
    print(f"[OK] wrote copy map → {out_path}")


if __name__ == "__main__":
    main()

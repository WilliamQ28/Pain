#!/usr/bin/env python3
"""
Generate web copy with optional RAG context using the HF router chat API.
Supports staged mode (site name -> hero -> sections with light linkage) or
legacy single-body mode sized by width/height.
"""

import argparse
import json
import math
import os
import random
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct:together"  # chat-completions via HF router
DEFAULT_CHAR_WIDTH_FACTOR = 0.52  # avg em width vs font size
DEFAULT_LINE_HEIGHT = 1.4
DEFAULT_FILL_RATIO = 0.9
MAX_CONTEXT_CHUNKS = 3
DEFAULT_SECTION_CHAR_MIN = 250
DEFAULT_SECTION_CHAR_MAX = 400
DEFAULT_HERO_TITLE_CHARS = 80
DEFAULT_HERO_SUBTITLE_CHARS = 200
DEFAULT_SECTION_PARA_COUNT = 3


# ---------------- helpers ----------------

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def estimate_char_budget(
    width_px: float,
    height_px: float,
    font_size_px: float = 16.0,
    line_height: float = DEFAULT_LINE_HEIGHT,
    char_width_factor: float = DEFAULT_CHAR_WIDTH_FACTOR,
    fill_ratio: float = DEFAULT_FILL_RATIO,
    min_chars: int = 120,
    max_chars: int = 1200,
) -> int:
    """Estimate characters that fit inside a text box."""
    width_px = max(16.0, width_px)
    height_px = max(16.0, height_px)
    font_size_px = max(8.0, font_size_px)
    line_height_px = font_size_px * max(1.05, line_height)

    chars_per_line = width_px / (font_size_px * max(0.3, char_width_factor))
    line_count = height_px / line_height_px
    rough_budget = int(chars_per_line * line_count * clamp(fill_ratio, 0.5, 1.0))
    return int(clamp(rough_budget, float(min_chars), float(max_chars)))


def budget_to_max_tokens(char_budget: int, safety: float = 1.3) -> int:
    """Convert character budget to token limit for LLM."""
    return int(math.ceil((char_budget / 4.0) * safety))


def load_rag(path: Optional[Path], max_items: Optional[int] = None) -> List[dict]:
    if not path:
        return []
    if not path.exists():
        return []
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                items.append(rec)
            except Exception:
                continue
            if max_items and len(items) >= max_items:
                break
    return items


def pick_context(
    rag: Sequence[dict],
    strategy: str,
    max_chunks: int,
    content_type: str,
) -> List[dict]:
    if not rag or max_chunks <= 0:
        return []
    if strategy == "topic" and content_type:
        filtered = [r for r in rag if content_type.lower() in str(r.get("topic", "")).lower()]
        if filtered:
            rag = filtered
    return random.sample(rag, k=min(max_chunks, len(rag)))


def parse_section_lengths(raw: Optional[str], sections: int) -> List[int]:
    """Parse a JSON list of section lengths; fill missing with defaults."""
    if not raw:
        return [DEFAULT_SECTION_CHAR_MAX] * max(0, sections)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            vals = []
            for v in arr:
                if isinstance(v, (int, float)):
                    vals.append(int(v))
            if len(vals) < sections:
                vals.extend([DEFAULT_SECTION_CHAR_MAX] * (sections - len(vals)))
            return vals[:sections]
    except Exception:
        pass
    return [DEFAULT_SECTION_CHAR_MAX] * max(0, sections)


def parse_section_para_counts(raw: Optional[str], sections: int, default_count: int) -> List[int]:
    """Parse a JSON list of paragraph counts; fill missing with default."""
    if not raw:
        return [default_count] * max(0, sections)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            vals = []
            for v in arr:
                if isinstance(v, int) and v > 0:
                    vals.append(v)
            if len(vals) < sections:
                vals.extend([default_count] * (sections - len(vals)))
            return vals[:sections]
    except Exception:
        pass
    return [default_count] * max(0, sections)


def build_prompt(
    content_type: str,
    char_budget: int,
    tone: str,
    audience: str,
    context: List[dict],
    notes: Optional[str] = None,
) -> str:
    ctx_lines = []
    for i, rec in enumerate(context, start=1):
        txt = str(rec.get("text", "")).strip()
        if not txt:
            continue
        slot = rec.get("slot_type", "") or rec.get("band", "")
        ctx_lines.append(f"{i}) [{slot}] {txt}")
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "None provided."
    note_line = f"- Context notes: {notes}" if notes else ""
    prompt = textwrap.dedent(
        f"""
        You are writing the main body copy for a {content_type} website.
        - Audience: {audience}
        - Tone: {tone}
        - Length: about {char_budget} characters. Keep it concise; avoid filler.
        - Style: cohesive paragraph(s), no headings or lists, avoid placeholders, plain text.
        - Context snippets (use as inspiration, do not quote verbatim):
        {ctx_block}
        {note_line}
        If anything is ambiguous, pick reasonable details that fit the category. Return only the body text.
        """
    ).strip()
    return prompt


def call_hf_chat(
    prompt: str,
    model_id: str,
    token: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Call HF router using OpenAI-compatible chat/completions API."""
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN (or HF_TOKEN) is required to call the API")

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected HF chat response: {data}") from exc


def generate_site_name(
    token: str,
    model_id: str,
    content_type: str,
    audience: str,
    tone: str,
    notes: Optional[str],
    context: List[dict],
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """Return (site_name, prompt_used)."""
    ctx_lines = []
    for i, rec in enumerate(context[:2], start=1):
        txt = str(rec.get("text", "")).strip()
        if txt:
            ctx_lines.append(f"{i}) {txt}")
    ctx_block = "\n".join(ctx_lines) if ctx_lines else "None provided."
    note_line = f"- Notes: {notes}" if notes else ""
    prompt = textwrap.dedent(
        f"""
        Propose a short brand-style site name for a {content_type} website.
        - Audience: {audience}
        - Tone: {tone}
        - Context snippets (inspiration only): {ctx_block}
        {note_line}
        Return only the name, 1-3 words, no quotes or punctuation.
        """
    ).strip()
    name = call_hf_chat(
        prompt=prompt,
        model_id=model_id,
        token=token,
        max_new_tokens=48,
        temperature=temperature,
        top_p=top_p,
    )
    return name.strip(), prompt


def generate_hero(
    token: str,
    model_id: str,
    site_name: str,
    content_type: str,
    audience: str,
    tone: str,
    notes: Optional[str],
    target_title_chars: int,
    target_subtitle_chars: int,
    temperature: float,
    top_p: float,
) -> Tuple[Dict[str, str], str]:
    """Return (hero_dict, prompt_used)."""
    prompt = textwrap.dedent(
        f"""
        You are writing the hero copy for a {content_type} site named "{site_name}".
        - Audience: {audience}
        - Tone: {tone}
        - Title length: about {target_title_chars} characters.
        - Subtitle length: about {target_subtitle_chars} characters.
        {f"- Notes: {notes}" if notes else ""}
        Return ONLY valid JSON with keys hero_title and hero_subtitle. No backticks, no markdown.
        """
    ).strip()
    raw = call_hf_chat(
        prompt=prompt,
        model_id=model_id,
        token=token,
        max_new_tokens=budget_to_max_tokens(target_title_chars + target_subtitle_chars),
        temperature=temperature,
        top_p=top_p,
    )
    hero = {"hero_title": "", "hero_subtitle": ""}
    # Clean possible fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            hero["hero_title"] = str(parsed.get("hero_title", "")).strip()
            hero["hero_subtitle"] = str(parsed.get("hero_subtitle", "")).strip()
    except Exception:
        # fallback: naive split on newline/period
        parts = cleaned.splitlines()
        if parts:
            hero["hero_title"] = parts[0].strip()
            if len(parts) > 1:
                hero["hero_subtitle"] = " ".join(p.strip() for p in parts[1:] if p.strip())
    return hero, prompt


def plan_sections(
    token: str,
    model_id: str,
    site_name: str,
    content_type: str,
    audience: str,
    tone: str,
    notes: Optional[str],
    sections: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[dict], str]:
    """Return (sections_plan, prompt_used), where plan is list of {heading,intent}."""
    prompt = textwrap.dedent(
        f"""
        Propose exactly {sections} section headings for a {content_type} site named "{site_name}".
        - Audience: {audience}
        - Tone: {tone}
        {f"- Notes: {notes}" if notes else ""}
        Return JSON array of objects: [{{"heading": "...", "intent": "one-line purpose"}}].
        """
    ).strip()
    raw = call_hf_chat(
        prompt=prompt,
        model_id=model_id,
        token=token,
        max_new_tokens=budget_to_max_tokens(sections * 80),
        temperature=temperature,
        top_p=top_p,
    )
    plan: List[dict] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                heading = str(item.get("heading", "")).strip()
                intent = str(item.get("intent", "")).strip()
                if heading:
                    plan.append({"heading": heading, "intent": intent})
    except Exception:
        # fallback: split lines "Heading: intent"
        for line in raw.splitlines():
            if not line.strip():
                continue
            if ":" in line:
                h, i = line.split(":", 1)
                plan.append({"heading": h.strip(), "intent": i.strip()})
            else:
                plan.append({"heading": line.strip(), "intent": ""})
    return plan[:sections], prompt


def sanitize_heading(text: str, fallback: str) -> str:
    """Avoid feeding malformed headings downstream."""
    if not text:
        return fallback
    if text.startswith("[") or text.startswith("{"):
        return fallback
    return text


def generate_section_body(
    token: str,
    model_id: str,
    site_name: str,
    content_type: str,
    audience: str,
    tone: str,
    heading: str,
    intent: str,
    target_chars: int,
    previous_body: Optional[str],
    para_count: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """Return (body_text, prompts_used joined). Generate paragraphs separately to reduce repetition."""
    prior_takeaway = ""
    if previous_body:
        prior_takeaway = previous_body.split(".")[0][:120]
    paragraphs: List[str] = []
    prompts_used: List[str] = []
    per_para_chars = max(80, int(target_chars / max(1, para_count)))
    for idx in range(para_count):
        prior_section_line = f"- Prior section takeaway (use in one clause only): {prior_takeaway}" if prior_takeaway else "- This is the first section; no prior reference."
        prior_in_section = ""
        if paragraphs:
            prior_in_section = "Prior paragraphs started with: " + "; ".join(p.split(".")[0][:60] for p in paragraphs if p)
        banned_starts = [ " ".join(p.split()[:5]) for p in paragraphs if p.split() ]
        banned_line = f"- Do not start with: {', '.join(banned_starts)}" if banned_starts else ""
        prompt = textwrap.dedent(
            f"""
            Write paragraph {idx+1} for a section on a {content_type} site named "{site_name}".
            - Section heading: {heading}
            - Intent: {intent or "n/a"}
            - Audience: {audience}
            - Tone: {tone}
            - Length: about {per_para_chars} characters.
            - Style: cohesive, no headings/lists, avoid placeholders, plain text.
            - Include at least one concrete feature (e.g., budget tracking, savings goals, bill reminders, small investments). In community/learning contexts, mention tips/webinars/guides.
            {prior_section_line}
            - Start differently from prior paragraphs; add a new point, do not restate earlier sentences. {prior_in_section}
            - Avoid repeating phrases already used (e.g., "Once you sign up") or re-explaining account linking.
            {banned_line}
            - Ensure this paragraph is a complete thought; do not end mid-sentence.
            Return only this paragraph text.
            """
        ).strip()
        attempts = 0
        para_text = ""
        while attempts < 2:
            para = call_hf_chat(
                prompt=prompt if attempts == 0 else (prompt + "\nThe previous attempt was incomplete or repetitive. Rewrite with a different opening and a full sentence ending."),
                model_id=model_id,
                token=token,
                max_new_tokens=budget_to_max_tokens(per_para_chars),
                temperature=temperature,
                top_p=top_p,
            )
            para_text = para.strip()
            too_short = len(para_text) < per_para_chars * 0.5
            no_end = para_text and para_text[-1] not in ".!?"
            start_words = " ".join(para_text.split()[:5])
            repeats_start = start_words in banned_starts if start_words else False
            if not (too_short or no_end or repeats_start):
                break
            attempts += 1
        paragraphs.append(para_text)
        prompts_used.append(prompt)
    body_text = "\n\n".join(paragraphs)
    return body_text, "\n\n".join(prompts_used)


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Generate web body copy sized to a wireframe box")
    ap.add_argument("--content_type", default="product", help="e.g., fintech, SaaS, nonprofit")
    ap.add_argument("--audience", default="a general audience evaluating the offering")
    ap.add_argument("--tone", default="clear and confident")
    ap.add_argument("--notes", default=None, help="optional brand/context notes")
    ap.add_argument("--width_px", type=float, default=None, help="text box width in px (legacy single-body mode)")
    ap.add_argument("--height_px", type=float, default=None, help="text box height in px (legacy single-body mode)")
    ap.add_argument("--font_px", type=float, default=16.0, help="base font size in px")
    ap.add_argument("--line_height", type=float, default=DEFAULT_LINE_HEIGHT, help="CSS line-height multiplier")
    ap.add_argument("--fill_ratio", type=float, default=DEFAULT_FILL_RATIO, help="how full to pack the box (0.5-1.0)")
    ap.add_argument("--min_chars", type=int, default=120)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--sections", type=int, default=0, help="number of main sections to generate (triggers staged mode)")
    ap.add_argument("--section_lengths", default=None, help="JSON array of target char counts per section")
    ap.add_argument("--section_para_count", type=int, default=DEFAULT_SECTION_PARA_COUNT, help="paragraphs per section in staged mode")
    ap.add_argument("--section_para_counts", default=None, help="JSON array of paragraph counts per section (overrides section_para_count)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rag_path", default="rag_corpus.jsonl", help="path to RAG corpus jsonl")
    ap.add_argument("--context_chunks", type=int, default=MAX_CONTEXT_CHUNKS)
    ap.add_argument("--context_strategy", choices=["random", "topic"], default="topic")
    ap.add_argument("--token", default=None, help="HF token; highest precedence")
    ap.add_argument("--token_file", default=None, help="path to file containing HF token (fallback if --token not set)")
    ap.add_argument("--out", default=None, help="optional path to write JSON output")
    ap.add_argument("--no_stdout", action="store_true", help="suppress stdout JSON output")
    ap.add_argument("--dry_prompt", action="store_true", help="print prompt/metadata instead of calling API")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    rag_path = Path(args.rag_path) if args.rag_path else None
    rag = load_rag(rag_path)
    context = pick_context(
        rag=rag,
        strategy=args.context_strategy,
        max_chunks=args.context_chunks,
        content_type=args.content_type,
    )

    token = args.token
    if not token and args.token_file:
        tf = Path(args.token_file)
        if tf.exists():
            token = tf.read_text(encoding="utf-8").strip()
    if not token:
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    # Staged mode: sections > 0
    if args.sections and args.sections > 0:
        lengths = parse_section_lengths(args.section_lengths, args.sections)
        para_counts = parse_section_para_counts(args.section_para_counts, args.sections, args.section_para_count)

        if args.dry_prompt:
            dummy_site = "PlaceholderCo"
            hero_prompt = f"Hero prompt would reference site '{dummy_site}'"
            plan_prompt = f"Plan prompt for {args.sections} sections"
            body_prompts = [f"Body prompt for section {i+1}" for i in range(args.sections)]
            output = {
                "mode": "staged",
                "model": args.model,
                "site_name": dummy_site,
                "hero_prompt": hero_prompt,
                "section_plan_prompt": plan_prompt,
                "section_body_prompts": body_prompts,
                "target_lengths": lengths,
                "context_used": context,
            }
            if args.out:
                Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            if not args.no_stdout:
                print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        site_name, name_prompt = generate_site_name(
            token=token,
            model_id=args.model,
            content_type=args.content_type,
            audience=args.audience,
            tone=args.tone,
            notes=args.notes,
            context=context,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        hero, hero_prompt = generate_hero(
            token=token,
            model_id=args.model,
            site_name=site_name,
            content_type=args.content_type,
            audience=args.audience,
            tone=args.tone,
            notes=args.notes,
            target_title_chars=DEFAULT_HERO_TITLE_CHARS,
            target_subtitle_chars=DEFAULT_HERO_SUBTITLE_CHARS,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        plan, plan_prompt = plan_sections(
            token=token,
            model_id=args.model,
            site_name=site_name,
            content_type=args.content_type,
            audience=args.audience,
            tone=args.tone,
            notes=args.notes,
            sections=args.sections,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if len(plan) < args.sections:
            # pad with generic headings
            for i in range(len(plan), args.sections):
                plan.append({"heading": f"Section {i+1}", "intent": ""})
        # sanitize headings
        for i, item in enumerate(plan):
            plan[i]["heading"] = sanitize_heading(item.get("heading", ""), fallback=f"Section {i+1}")

        sections_out = []
        prev_body: Optional[str] = None
        body_prompts: List[str] = []
        for idx, item in enumerate(plan, start=1):
            target_chars = lengths[idx - 1] if idx - 1 < len(lengths) else DEFAULT_SECTION_CHAR_MAX
            para_count = para_counts[idx - 1] if idx - 1 < len(para_counts) else args.section_para_count
            body, body_prompt = generate_section_body(
                token=token,
                model_id=args.model,
                site_name=site_name,
                content_type=args.content_type,
                audience=args.audience,
                tone=args.tone,
                heading=item.get("heading", f"Section {idx}"),
                intent=item.get("intent", ""),
                target_chars=target_chars,
                previous_body=prev_body,
                para_count=max(1, para_count),
                temperature=args.temperature,
                top_p=args.top_p,
            )
            sections_out.append(
                {
                    "heading": item.get("heading", f"Section {idx}"),
                    "intent": item.get("intent", ""),
                    "body_paragraphs": body.split("\n\n") if body else [],
                    "target_chars": target_chars,
                }
            )
            body_prompts.append(body_prompt)
            prev_body = body

        output = {
            "mode": "staged",
            "model": args.model,
            "content_type": args.content_type,
            "audience": args.audience,
            "tone": args.tone,
            "site_name": site_name,
            "hero": hero,
            "sections": sections_out,
            "context_used": context,
            "prompts": {
                "site_name": name_prompt,
                "hero": hero_prompt,
                "section_plan": plan_prompt,
                "section_bodies": body_prompts,
            },
        }
        rendered = json.dumps(output, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(rendered, encoding="utf-8")
        if not args.no_stdout:
            print(rendered)
        return

    # Legacy single-body mode (width/height -> char budget)
    if args.width_px is None or args.height_px is None:
        raise SystemExit("width_px and height_px are required when sections=0")

    char_budget = estimate_char_budget(
        width_px=args.width_px,
        height_px=args.height_px,
        font_size_px=args.font_px,
        line_height=args.line_height,
        char_width_factor=DEFAULT_CHAR_WIDTH_FACTOR,
        fill_ratio=args.fill_ratio,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    max_new_tokens = budget_to_max_tokens(char_budget)

    prompt = build_prompt(
        content_type=args.content_type,
        char_budget=char_budget,
        tone=args.tone,
        audience=args.audience,
        context=context,
        notes=args.notes,
    )

    output: Dict[str, Any] = {
        "mode": "single",
        "model": args.model,
        "content_type": args.content_type,
        "audience": args.audience,
        "tone": args.tone,
        "char_budget": char_budget,
        "max_new_tokens": max_new_tokens,
        "width_px": args.width_px,
        "height_px": args.height_px,
        "font_px": args.font_px,
        "line_height": args.line_height,
        "fill_ratio": args.fill_ratio,
        "context_used": context,
        "prompt": prompt,
    }

    if args.dry_prompt:
        if args.out:
            Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.no_stdout:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    text = call_hf_chat(
        prompt=prompt,
        model_id=args.model,
        token=token,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    output["text"] = text
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    if not args.no_stdout:
        print(rendered)


if __name__ == "__main__":
    main()

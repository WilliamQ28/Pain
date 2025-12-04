#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a RAG corpus (rag_corpus.jsonl) from the URLs listed in link*.txt files.

Heuristics:
- Hero title: first <h1> with reasonable length, not obviously footer/nav/legal.
- Hero subtitle: first <h2> or <p> after the hero <h1>.
- Main-body text: <p>, <h2>, <h3> after the hero, excluding footer/nav, with reasonable length.
- Footer/nav/legal are filtered by tag (<footer>, <nav>) and keyword patterns.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ------------------------
# Config
# ------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 12
SLEEP_BETWEEN_REQUESTS = 0.7  # seconds
MAX_SITES = 50                # cap how many URLs we process (for speed)
MAX_PARAGRAPHS_PER_SITE = 8   # main-body chunks per site

RAG_OUTPUT = "rag_corpus.jsonl"


# ------------------------
# Helpers
# ------------------------

def load_urls_from_link_files(pattern="link*.txt"):
    urls = set()
    for file in Path(".").glob(pattern):
        with file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                urls.add(line)
    return sorted(urls)


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_probably_english(text: str, min_ratio: float = 0.65) -> bool:
    """
    Lightweight language gate: require mostly ASCII and no obvious CJK/Hangul.
    """
    if not text:
        return False
    total = len(text)
    ascii_count = sum(1 for ch in text if ch.isascii())
    if ascii_count / max(1, total) < min_ratio:
        return False
    # Filter out CJK/Hangul blocks
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", text):
        return False
    return True


def get_domain_tokens(url: str):
    """
    From https://www.openai.com -> ['openai']
    From https://sub.shop.example.co.uk -> ['sub', 'shop', 'example']
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # Strip port if any
    host = host.split(":")[0]
    parts = host.split(".")
    # drop TLDs
    tlds = {"com", "net", "org", "io", "ai", "co", "uk", "de", "ca", "app"}
    tokens = [p for p in parts if p not in tlds and p != "www"]
    return tokens or parts


def strip_brand_tokens(text: str, domain_tokens):
    """
    Remove obvious domain tokens from the text (case-insensitive).
    This is a heuristic to avoid leaving brand names in the corpus.
    """
    if not text:
        return text

    new_text = text
    for tok in domain_tokens:
        if not tok:
            continue
        pattern = re.compile(re.escape(tok), re.IGNORECASE)
        new_text = pattern.sub("", new_text)
    # Collapse spaces again after removals
    return clean_text(new_text)


FOOTER_KEYWORDS = [
    "privacy policy",
    "terms of service",
    "terms & conditions",
    "cookies",
    "cookie policy",
    "newsletter",
    "subscribe",
    "©",
    "copyright",
    "all rights reserved",
    "support",
    "help center",
    "contact us",
    "contact support",
    "faq"
]

NAV_KEYWORDS = [
    "home", "pricing", "features", "about", "docs", "documentation",
    "blog", "login", "sign in", "sign up", "get started"
]


def looks_like_footer_text(text: str) -> bool:
    t = text.lower()
    if len(t) < 5:
        return True
    for kw in FOOTER_KEYWORDS:
        if kw in t:
            return True
    # address-like patterns: digits + street/road/etc.
    if re.search(r"\d{1,4}\s+\w+\s+(st|street|rd|road|ave|avenue|blvd|boulevard)", t):
        return True
    return False


def looks_like_nav_text(text: str) -> bool:
    # Very short (e.g., 'Home', 'Docs', 'Pricing')
    t = text.strip()
    if len(t.split()) <= 3 and len(t) <= 20:
        # if it matches typical nav words, treat as nav
        for kw in NAV_KEYWORDS:
            if t.lower() == kw:
                return True
    return False


def is_inside_tag_with_name_or_class(tag, names):
    """
    Check if a BeautifulSoup tag is inside any ancestor whose name
    is in 'names' or whose class/id contains those names.
    """
    if not tag:
        return False

    name_set = {n.lower() for n in names}
    for parent in tag.parents:
        if not hasattr(parent, "name"):
            continue
        pname = (parent.name or "").lower()
        if pname in name_set:
            return True
        # Check class/id
        classes = parent.get("class", [])
        cid = parent.get("id", "")
        joined = " ".join(classes + [cid]).lower()
        for n in name_set:
            if n in joined:
                return True
    return False


def is_footer_node(tag) -> bool:
    return is_inside_tag_with_name_or_class(tag, ["footer"])


def is_nav_or_header_node(tag) -> bool:
    return is_inside_tag_with_name_or_class(tag, ["nav", "header"])


def valid_hero_candidate(text: str) -> bool:
    length = len(text)
    if length < 8 or length > 120:
        return False
    # Filter out obvious nav/footer-like headings
    if looks_like_footer_text(text) or looks_like_nav_text(text):
        return False
    if not is_probably_english(text):
        return False
    return True


def valid_body_candidate(text: str) -> bool:
    length = len(text)
    if length < 30 or length > 400:
        return False
    if looks_like_footer_text(text):
        return False
    if not is_probably_english(text):
        return False
    return True


def fetch_html(url: str) -> str | None:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            print(f"[WARN] {url}: HTTP {resp.status_code}")
            return None
        return resp.text
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None


# ------------------------
# Extraction per site
# ------------------------

def extract_from_html(url: str, html: str):
    """
    Return dict with hero_title, hero_subtitle, body_paragraphs[]
    """
    soup = BeautifulSoup(html, "html.parser")
    domain_tokens = get_domain_tokens(url)

    # HERO
    hero_title = None
    hero_subtitle = None

    h1 = soup.find("h1")
    if h1:
        if not (is_footer_node(h1) or is_nav_or_header_node(h1)):
            txt = clean_text(h1.get_text())
            txt = strip_brand_tokens(txt, domain_tokens)
            if valid_hero_candidate(txt):
                hero_title = txt

                # hero subtitle: next h2 or p not in footer/nav
                sib = h1
                while sib:
                    sib = sib.find_next(["h2", "p"])
                    if not sib:
                        break
                    if is_footer_node(sib) or is_nav_or_header_node(sib):
                        continue
                    stxt = strip_brand_tokens(clean_text(sib.get_text()), domain_tokens)
                    if valid_body_candidate(stxt) or (15 <= len(stxt) <= 180):
                        hero_subtitle = stxt
                        break

    # MAIN BODY
    body_paras = []
    # Start scanning from hero (if exists), else from start of document
    start_node = h1 or soup.body or soup

    current = start_node
    visited = set()
    while current:
        current = current.find_next(["p", "h2", "h3"])
        if not current:
            break
        if current in visited:
            continue
        visited.add(current)

        if is_footer_node(current) or is_nav_or_header_node(current):
            continue

        text = strip_brand_tokens(clean_text(current.get_text()), domain_tokens)
        if not valid_body_candidate(text):
            continue
        if looks_like_nav_text(text):
            continue

        # dedupe
        if text in body_paras:
            continue

        body_paras.append(text)
        if len(body_paras) >= MAX_PARAGRAPHS_PER_SITE:
            break

    return {
        "hero_title": hero_title,
        "hero_subtitle": hero_subtitle,
        "body_paragraphs": body_paras,
    }


# ------------------------
# Main driver
# ------------------------

def main():
    urls = load_urls_from_link_files()
    if not urls:
        print("[ERROR] No URLs found in link*.txt")
        return

    if MAX_SITES:
        urls = urls[:MAX_SITES]

    print(f"[INFO] Found {len(urls)} URLs to process (capped).")

    out_path = Path(RAG_OUTPUT)
    out_f = out_path.open("w", encoding="utf-8")

    doc_count = 0

    for idx, url in enumerate(urls, start=1):
        print(f"[INFO] ({idx}/{len(urls)}) Fetching {url}")
        html = fetch_html(url)
        if not html:
            continue

        data = extract_from_html(url, html)
        hero_title = data["hero_title"]
        hero_subtitle = data["hero_subtitle"]
        body_paras = data["body_paragraphs"]

        parsed = urlparse(url)
        topic = parsed.netloc

        if hero_title:
            doc_count += 1
            rec = {
                "id": f"site{idx:03d}_hero_title",
                "band": "Header",
                "slot_type": "hero_title",
                "source_url": url,
                "topic": topic,
                "text": hero_title,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if hero_subtitle:
            doc_count += 1
            rec = {
                "id": f"site{idx:03d}_hero_subtitle",
                "band": "Header",
                "slot_type": "hero_subtitle",
                "source_url": url,
                "topic": topic,
                "text": hero_subtitle,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        for j, para in enumerate(body_paras, start=1):
            doc_count += 1
            rec = {
                "id": f"site{idx:03d}_body_{j:02d}",
                "band": "Main",
                "slot_type": "paragraph",
                "source_url": url,
                "topic": topic,
                "text": para,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"    -> hero_title={bool(hero_title)}, "
            f"hero_subtitle={bool(hero_subtitle)}, "
            f"body_paras={len(body_paras)}"
        )

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    out_f.close()
    print(f"[INFO] Wrote {doc_count} chunks to {RAG_OUTPUT}")


if __name__ == "__main__":
    main()

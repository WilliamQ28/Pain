#!/usr/bin/env python3
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

RG_INJECT = r"""
(() => {
  // -------- hash + utils --------
  const fnv1a = (str) => { let h=0x811c9dc5>>>0; for (let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=Math.imul(h,0x01000193)>>>0;} return h>>>0; };
  const domPath = (el) => {
    const parts = [];
    for (let e=el; e && e.nodeType===1 && e!==document; e=e.parentElement) {
      let seg = e.tagName.toLowerCase();
      if (e.id) seg += `#${e.id}`;
      if (e.classList && e.classList.length) seg += '.' + Array.from(e.classList).sort().join('.');
      let idx = 1; for (let s=e.previousElementSibling; s; s=s.previousElementSibling) if (s.tagName===e.tagName) idx++;
      seg += `:nth-of-type(${idx})`; parts.push(seg);
    }
    return parts.reverse().join('>');
  };
  const q8 = (v) => { v = v & ~7; if (v < 10) v = 10; if (v > 246) v = 246; return v; };
  const rectOf = (el) => el.getBoundingClientRect();
  const areaOf = (el) => { const r = rectOf(el); return Math.max(0, r.width * r.height); };

  // -------- type → G mapping --------
  const G_BG=0, G_TEXT=16, G_MEDIA=32, G_BUTTON=48, G_INPUT=64, G_SELECT=80, G_TEXTAREA=96, G_ROLEBTN=112;

  const typeIdG = (el) => {
    const tag = el.tagName;
    const role = (el.getAttribute && el.getAttribute('role')) || "";
    const cs = getComputedStyle(el);
    if (tag === 'IMG' || tag === 'PICTURE' || tag === 'VIDEO' || tag === 'SVG') return G_MEDIA;
    if (tag === 'BUTTON') return G_BUTTON;
    if (tag === 'INPUT')  return G_INPUT;
    if (tag === 'SELECT') return G_SELECT;
    if (tag === 'TEXTAREA') return G_TEXTAREA;
    if (role === 'button' || (tag === 'A' && (cs.display !== 'inline' || cs.cursor === 'pointer'))) return G_ROLEBTN;
    return G_BG; // default is true background, not text
  };

  const isFormish = (el) => ['FORM','FIELDSET','LABEL'].includes(el?.tagName);
  const looksLikeFieldContainer = (el) => {
    const s = (el?.id + ' ' + (el?.className || '')).toLowerCase();
    return /field|input|form|control|group|btn|button|box|wrap|container|search|bar|pill|card|item|row|col/.test(s);
  };
  const hasBoxSkin = (node) => {
    const s = getComputedStyle(node);
    const pad = ['Top','Right','Bottom','Left'].some(k => parseFloat(s['padding'+k]) > 0.5);
    const bor = ['Top','Right','Bottom','Left'].some(k => parseFloat(s['border'+k+'Width']) > 0.5);
    const bg = s.backgroundColor && s.backgroundColor !== 'rgba(0, 0, 0, 0)' && s.backgroundColor !== 'transparent';
    return pad || bor || bg;
  };

  const nearestPaintableContainer_relaxed = (el) => {
    const base = Math.max(1, areaOf(el)); let cur = el.parentElement, steps = 0;
    while (cur && steps < 4) {
      const s = getComputedStyle(cur); if (s.display==='none'||s.visibility==='hidden'||parseFloat(s.opacity)<0.01) break;
      if (s.display !== 'inline') {
        const growth = areaOf(cur)/base; const cc = cur.children?.length ?? 0;
        const ok = growth > 1.0 && growth <= 2.4;
        const wrappery = looksLikeFieldContainer(cur) || isFormish(cur) || hasBoxSkin(cur) || cc <= 3;
        if (ok && wrappery) return cur;
      }
      cur = cur.parentElement; steps++;
    }
    return null;
  };

  // text host detection
  const isBlockTextHost = (el) => /^(P|H1|H2|H3|H4|H5|H6|LI|DD|DT|BLOCKQUOTE|FIGCAPTION)$/.test(el.tagName);
  const hasVisibleText = (el) => {
    const cs = getComputedStyle(el);
    if (cs.display==='none'||cs.visibility==='hidden'||parseFloat(cs.opacity)<0.01) return false;
    if (!el.innerText || el.innerText.trim().length === 0) return false;
    const r = document.createRange(); r.selectNodeContents(el);
    const rects = r.getClientRects(); return rects.length > 0;
  };

  // -------- global CSS cleanup (hide text drawing effects) --------
  const styleEl = document.createElement('style');
  styleEl.textContent = `
    * { animation:none!important; transition:none!important; background-image:none!important; box-shadow:none!important; outline:none!important; border:none!important; }
    *, *::before, *::after { color:transparent!important; text-shadow:none!important; -webkit-text-stroke:0!important; }
    input, textarea, select { color:transparent!important; caret-color:transparent!important; }
    input::placeholder, textarea::placeholder { color:transparent!important; }
    svg text { fill:transparent!important; stroke:transparent!important; }
    ::selection { color:transparent!important; background:transparent!important; }
  `;
  document.head.appendChild(styleEl);

  // -------- paint pass (no container inflation for text) --------
  const els = Array.from(document.querySelectorAll('body *'));
  for (const el of els) {
    const cs = getComputedStyle(el);
    if (cs.display==='none'||cs.visibility==='hidden'||parseFloat(cs.opacity)<0.01) continue;
    const r = rectOf(el); if (r.width<=0||r.height<=0) continue;
    const area = r.width*r.height; if (area < 64) continue;

    // skip pure inline non-interactive texty tags
    const isInline = cs.display === 'inline';
    const isTexty = /^(A|SPAN|B|STRONG|I|EM|U|S|SMALL|SUB|SUP|ABBR|CODE|KBD|SAMP|MARK|TIME|VAR|CITE|Q)$/.test(el.tagName);
    const buttonish = (cs.display!=='inline' || cs.cursor==='pointer' || el.getAttribute('role')==='button' || ['BUTTON','INPUT','SELECT','TEXTAREA'].includes(el.tagName));
    if (isInline && isTexty && !buttonish) continue;

    let G = typeIdG(el);
    let target = el;

    // Controls keep wrapper logic and upward absorb
    if ([G_BUTTON,G_INPUT,G_SELECT,G_TEXTAREA,G_ROLEBTN].includes(G)) {
      const wrap = nearestPaintableContainer_relaxed(el);
      if (wrap) target = wrap;
    } else {
      // Only mark true text hosts
      if (isBlockTextHost(el) && hasVisibleText(el)) {
        G = G_TEXT;
      } else {
        // background: don't paint
        continue;
      }
    }

    // color: R from element path hash, G from final type
    const sig = domPath(el), h = fnv1a(sig);
    const R = q8(h & 0xff);
    const color = `rgb(${R},${q8(G)},0)`;

    target.style.setProperty('background-color', color, 'important');
    if (target !== el) el.style.setProperty('background-color', color, 'important');

    // Upward absorb ONLY for controls; skip for text to prevent container flooding
    if ([G_BUTTON,G_INPUT,G_SELECT,G_TEXTAREA,G_ROLEBTN].includes(G)) {
      let cur = target.parentElement, childA = areaOf(target), hops = 0;
      while (cur && hops < 3) {
        const s = getComputedStyle(cur);
        if (s.display==='none'||s.visibility==='hidden'||parseFloat(s.opacity)<0.01) break;
        if (s.display==='inline') break;
        const pA = areaOf(cur), growth = pA/Math.max(1, childA), cc = cur.children?.length ?? 0;
        const okSize = growth>1.0 && growth<=2.6;
        const wrappery = looksLikeFieldContainer(cur) || isFormish(cur) || hasBoxSkin(cur) || cc<=3;
        if (!(okSize && wrappery)) break;
        cur.style.setProperty('background-color', color, 'important');
        childA = pA; cur = cur.parentElement; hops++;
      }
    }
  }
})();
"""


async def main():
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("--url", required=True)
  p.add_argument("--vw", type=int, default=1440)
  p.add_argument("--vh", type=int, default=0, help="0=auto; still capture full_page=True")
  p.add_argument("--outdir", default="/home/kali/wfc/out")
  args = p.parse_args()

  outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

  async with async_playwright() as pwt:
    browser = await pwt.chromium.launch()
    ctx = await browser.new_context(viewport={"width": args.vw, "height": args.vh or 1000}, device_scale_factor=1)
    page = await ctx.new_page()
    await page.goto(args.url, wait_until="networkidle")
    await page.evaluate("window.scrollTo(0,0)")

    # pretty base
    shot_path = outdir / "screenshot.png"
    await page.screenshot(path=str(shot_path), full_page=True)

    # RG pass
    await page.evaluate(RG_INJECT)
    rg_path = outdir / "RG.png"
    await page.screenshot(path=str(rg_path), full_page=True)

    await browser.close()

  print(f"[OK] Base screenshot → {shot_path}")
  print(f"[OK] RG pass → {rg_path}")

if __name__ == "__main__":
  asyncio.run(main())

#!/usr/bin/env python3
# wfc_p1_allinone.py
# One script: capture (concurrent + timeouts + RG across iframes), aggregate, qa.
# Supports both LS JSON formats; use --ls2245 for the flat 22:45 export.

import argparse, asyncio, json, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

# ============ shared utils ============
def dbg(enabled: bool, *a): 
    if enabled: print("[DEBUG]", *a)

def pct_to_rows(y_pct: float, h_pct: float, H: int) -> Tuple[int,int]:
    y1 = int(round((y_pct/100.0)*H)); y2 = int(round(((y_pct+h_pct)/100.0)*H))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    if y2 < y1: y1, y2 = y2, y1
    return y1, y2

def strip_prefix_hex(name: str) -> str:
    s = re.sub(r"\.(png|jpg|jpeg)$", "", name, flags=re.I)
    parts = s.split("-", 1)
    if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{6,}", parts[0]): return parts[1]
    return s

def slug_to_domain(slug: str) -> str: return slug.replace("-", ".")
def reconstruct_url_from_fileupload(file_upload: str) -> Optional[str]:
    base = strip_prefix_hex(Path(file_upload).name.lower()); dom = slug_to_domain(base).strip("-")
    host = dom.split(".png")[0].split(".jpg")[0].split(".jpeg")[0]
    if "." not in host: return None
    return f"https://{host}/"

def domain_from_image_name(name: str) -> str:
    base = re.sub(r"^[0-9a-f]{6,}-", "", Path(name).stem); return base.replace("-", ".")

PREFERRED_URL_KEYS = ("url","page","website","src","source","image")
def preferred_url_from_task_2241(task: dict) -> Optional[str]:
    d = task.get("data", {}) or {}
    for k in PREFERRED_URL_KEYS:
        v = d.get(k)
        if isinstance(v,str) and v.startswith(("http://","https://")): return v.strip()
    return None
def preferred_url_from_item_2245(item: dict) -> Optional[str]:
    d = item.get("data", {}) or {}
    for k in PREFERRED_URL_KEYS:
        v = d.get(k)
        if isinstance(v,str) and v.startswith(("http://","https://")): return v.strip()
    return None
def iter_fileuploads_from_lsjson_2241(paths: List[str], debug=False) -> List[str]:
    seen=[]
    for p in paths:
        data=json.loads(Path(p).read_text(encoding="utf-8"))
        tasks=data if isinstance(data,list) else (data.get("tasks") or [data])
        for t in tasks:
            fu=t.get("file_upload") or (t.get("data",{}) or {}).get("image")
            if not fu: 
                dbg(debug,f"[{p}] task missing file_upload"); 
                continue
            fn=Path(fu).name
            if fn not in seen: seen.append(fn)
    return seen

# ============ capture (concurrent) ============
NORMALIZE_CSS = """
  * { scroll-behavior: auto !important; }
  :where(header, nav, [role="banner"], .header, .navbar),
  :where([class*="sticky"], [id*="sticky"], [class*="fixed"], [id*="fixed"]) {
    position: static !important; top:auto !important; bottom:auto !important; inset:auto !important;
  }
  :where([id*="cookie"], [class*="cookie"], [aria-label*="cookie"],
         [id*="consent"], [class*="consent"],
         [class*="chat"], [aria-label*="chat"], [id*="chat"],
         [class*="subscribe"], [id*="subscribe"],
         [role="dialog"], [aria-modal="true"]) { display:none !important; }
"""
UA=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# === FULL RG_INJECT with paint counter ===
RG_INJECT = r"""
(() => {
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
  const areaOf = (el) => { const r = rectOf(el); return Math.max(1, r.width * r.height); };

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
    return G_BG;
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
        const ok = growth > 1.0 && growth <= 2.6;
        const wrappery = looksLikeFieldContainer(cur) || isFormish(cur) || hasBoxSkin(cur) || cc <= 3;
        if (ok && wrappery) return cur;
      }
      cur = cur.parentElement; steps++;
    }
    return null;
  };

  const isBlockTextHost = (el) => /^(P|H1|H2|H3|H4|H5|H6|LI|DD|DT|BLOCKQUOTE|FIGCAPTION)$/.test(el.tagName);
  const hasVisibleText = (el) => {
    const cs = getComputedStyle(el);
    if (cs.display==='none'||cs.visibility==='hidden'||parseFloat(cs.opacity)<0.01) return false;
    if (!el.innerText || el.innerText.trim().length === 0) return false;
    const r = document.createRange(); r.selectNodeContents(el);
    const rects = r.getClientRects(); return rects.length > 0;
  };

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

  window.__rg_painted_count = 0;

  const els = Array.from(document.querySelectorAll('body *'));
  for (const el of els) {
    const cs = getComputedStyle(el);
    if (cs.display==='none'||cs.visibility==='hidden'||parseFloat(cs.opacity)<0.01) continue;
    const r = el.getBoundingClientRect(); if (r.width<=0||r.height<=0) continue;
    const area = r.width*r.height; if (area < 64) continue;

    const isInline = cs.display === 'inline';
    const isTexty = /^(A|SPAN|B|STRONG|I|EM|U|S|SMALL|SUB|SUP|ABBR|CODE|KBD|SAMP|MARK|TIME|VAR|CITE|Q)$/.test(el.tagName);
    const buttonish = (cs.display!=='inline' || cs.cursor==='pointer' || el.getAttribute('role')==='button' || ['BUTTON','INPUT','SELECT','TEXTAREA'].includes(el.tagName));
    if (isInline && isTexty && !buttonish) continue;

    let G = typeIdG(el);
    let target = el;

    if ([G_BUTTON,G_INPUT,G_SELECT,G_TEXTAREA,G_ROLEBTN].includes(G)) {
      const wrap = nearestPaintableContainer_relaxed(el);
      if (wrap) target = wrap;
    } else {
      if (isBlockTextHost(el) && hasVisibleText(el)) {
        G = G_TEXT;
      } else {
        continue;
      }
    }

    const sig = domPath(el), h = fnv1a(sig);
    const R = q8(h & 0xff);
    const color = `rgb(${R},${q8(G)},0)`;

    target.style.setProperty('background-color', color, 'important');
    window.__rg_painted_count++;
    if (target !== el) {
      el.style.setProperty('background-color', color, 'important');
      window.__rg_painted_count++;
    }

    if ([G_BUTTON,G_INPUT,G_SELECT,G_TEXTAREA,G_ROLEBTN].includes(G)) {
      let cur = target.parentElement, childA = area, hops = 0;
      while (cur && hops < 3) {
        const s = getComputedStyle(cur);
        if (s.display==='none'||s.visibility==='hidden'||parseFloat(s.opacity)<0.01) break;
        if (s.display==='inline') break;
        const pr = cur.getBoundingClientRect();
        const pA = Math.max(1, pr.width * pr.height);
        const growth = pA/Math.max(1, childA), cc = cur.children?.length ?? 0;
        const okSize = growth>1.0 && growth<=2.6;
        const wrappery = looksLikeFieldContainer(cur) || isFormish(cur) || hasBoxSkin(cur) || cc<=3;
        if (!(okSize && wrappery)) break;
        cur.style.setProperty('background-color', color, 'important');
        window.__rg_painted_count++;
        childA = pA; cur = cur.parentElement; hops++;
      }
    }
  }
  return window.__rg_painted_count || 0;
})();
"""

async def _robust_goto(page, url: str, timeout_ms: int, retries=1, debug=False):
    from playwright.async_api import TimeoutError as PWTimeout, Error as PWError
    last_err=None
    for attempt in range(retries+1):
        for wait_until in ("domcontentloaded","networkidle","load"):
            try:
                dbg(debug,f"goto {url} ({wait_until})")
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                return True, None
            except (PWTimeout, PWError) as e:
                last_err=e; dbg(debug,f"goto fail {wait_until} attempt {attempt}: {e}")
        if attempt<retries: await asyncio.sleep(0.4)
    return False, last_err

async def _scroll_to_bottom(page, max_steps=40, step_px=1200, pause_ms=200, debug=False):
    last=-1
    for i in range(max_steps):
        try:
            y=await page.evaluate("""(step)=>{const h=document.documentElement.scrollHeight||document.body.scrollHeight;
               const y0=window.scrollY||window.pageYOffset||0; const y1=Math.min(h,y0+step);
               window.scrollTo(0,y1); return y1;}""", step_px)
        except Exception as e:
            dbg(debug,f"scroll err: {e}"); return
        if y==last: break
        last=y
        await page.wait_for_timeout(pause_ms)

async def _normalize(page, debug=False):
    try:
        css = NORMALIZE_CSS.replace("`","\\`").replace("\\","\\\\")
        res = await page.evaluate(f"""
        (()=>{{try{{const s=document.createElement('style');s.textContent=`{css}`;
        (document.head||document.documentElement).appendChild(s);
        for(const el of Array.from(document.querySelectorAll('*'))){{
          try{{const cs=getComputedStyle(el);
            if(cs && (cs.position==='fixed'||cs.position==='sticky')){{el.style.position='static';el.style.inset='auto';}}
          }}catch(_){{}}
        }} return true;}}catch(e){{return {{__err__:String(e)}}}} }})()
        """)
        if isinstance(res,dict) and "__err__" in res: raise RuntimeError(res["__err__"])
        dbg(debug,"normalize OK"); return True,None
    except Exception as e1:
        dbg(debug,f"normalize eval failed: {e1}")
        try:
            await page.add_style_tag(content=NORMALIZE_CSS); return True,None
        except Exception as e2:
            return False, f"normalize failed {e1} / {e2}"

async def _inject_rg_all_frames(page, debug=False):
    total = 0
    try:
        for fr in page.frames:
            try:
                painted = await fr.evaluate(RG_INJECT)
                if isinstance(painted, (int, float)): total += int(painted)
            except Exception as e:
                if debug: print("[DEBUG] RG inject failed in frame:", e)
    except Exception as e:
        if debug: print("[DEBUG] enumerating frames failed:", e)
    return total

async def _capture_one_job(sema, browser, job, out_screens, out_rg, width,
                           nav_timeout, op_timeout, ss_timeout, jpeg, debug):
    fu, url = job
    async with sema:
        ctx = await browser.new_context(
            device_scale_factor=1, locale="en-US",
            user_agent=UA, ignore_https_errors=True, java_script_enabled=True
        )
        async def _run():
            page = await ctx.new_page()
            try:
                page.set_default_timeout(ss_timeout)
                await page.set_viewport_size({"width": width, "height": 900})
                ok, err = await _robust_goto(page, url, nav_timeout, retries=1, debug=debug)
                if not ok:
                    print(f"[ERR] {fu} ← {url} (goto_failed: {err})"); return

                await _normalize(page, debug=debug)
                await _scroll_to_bottom(page, debug=debug)
                try: await page.evaluate("window.scrollTo(0,0)"); await page.wait_for_timeout(200)
                except: pass

                # base screenshot (w/ explicit timeout + retry)
                base = (out_screens / fu)
                base = base.with_suffix(".jpg") if jpeg else (base.with_suffix(".png") if base.suffix.lower() not in (".png",".jpg",".jpeg") else base)
                try:
                    if jpeg:
                        await page.screenshot(path=str(base), full_page=True, type="jpeg", quality=95, timeout=ss_timeout)
                    else:
                        await page.screenshot(path=str(base), full_page=True, timeout=ss_timeout)
                except Exception as e:
                    if debug: print(f"[DEBUG] base screenshot retry ({fu}): {e}")
                    if jpeg:
                        await page.screenshot(path=str(base), full_page=True, type="jpeg", quality=95, timeout=int(ss_timeout*1.5))
                    else:
                        await page.screenshot(path=str(base), full_page=True, timeout=int(ss_timeout*1.5))
                if debug: print(f"[DEBUG] saved screen {base}")

                # RG injection across all frames + count
                painted_total = await _inject_rg_all_frames(page, debug=debug)
                if debug: print(f"[DEBUG] RG painted elements (approx): {painted_total}")

                # RG screenshot (w/ explicit timeout + retry)
                rgp = (out_rg / fu)
                rgp = rgp.with_suffix(".jpg") if jpeg else (rgp.with_suffix(".png") if rgp.suffix.lower() not in (".png",".jpg",".jpeg") else rgp)
                try:
                    if jpeg:
                        await page.screenshot(path=str(rgp), full_page=True, type="jpeg", quality=95, timeout=ss_timeout)
                    else:
                        await page.screenshot(path=str(rgp), full_page=True, timeout=ss_timeout)
                except Exception as e:
                    if debug: print(f"[DEBUG] RG screenshot retry ({fu}): {e}")
                    if jpeg:
                        await page.screenshot(path=str(rgp), full_page=True, type="jpeg", quality=95, timeout=int(ss_timeout*1.5))
                    else:
                        await page.screenshot(path=str(rgp), full_page=True, timeout=int(ss_timeout*1.5))
                print(f"[OK]  {fu} ← {url}  (RG painted≈{painted_total})")
            finally:
                try: await ctx.close()
                except: pass

        try:
            await asyncio.wait_for(_run(), timeout=op_timeout/1000.0)
        except asyncio.TimeoutError:
            print(f"[ERR] {fu} ← {url} (op_timeout {op_timeout} ms)")

async def run_capture(args):
    from playwright.async_api import async_playwright
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    out_screens = out_root / "screens"; out_screens.mkdir(exist_ok=True)
    out_rg = out_root / "rg"; out_rg.mkdir(exist_ok=True)

    # build jobs
    jobs=[]
    if args.ls2245:
        data=json.loads(Path(args.labels[0]).read_text(encoding="utf-8"))
        items=data if isinstance(data,list) else [data]
        for t in items:
            img=t.get("image") or (t.get("data",{}) or {}).get("image")
            if not img: 
                dbg(args.debug,"[22:45] skip item without image"); 
                continue
            fname=Path(img).name
            url=preferred_url_from_item_2245(t) or f"https://{domain_from_image_name(fname)}/"
            jobs.append((fname,url))
    else:
        file_uploads=iter_fileuploads_from_lsjson_2241(args.labels, debug=args.debug)
        for p in args.labels:
            data=json.loads(Path(p).read_text(encoding="utf-8"))
            tasks=data if isinstance(data,list) else (data.get("tasks") or [data])
            for t in tasks:
                fu=t.get("file_upload") or (t.get("data",{}) or {}).get("image")
                if not fu: continue
                fname=Path(fu).name
                url=preferred_url_from_task_2241(t) or reconstruct_url_from_fileupload(fname)
                if url: jobs.append((fname,url))

    # de-dupe; cap
    seen=set(); jobs=[j for j in jobs if not (j[0] in seen or seen.add(j[0]))]
    if args.limit>0: jobs=jobs[:args.limit]
    if not jobs:
        print("[ERR] no jobs built"); 
        return 1

    sema = asyncio.Semaphore(max(1, args.concurrency))
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            tasks=[_capture_one_job(sema, browser, j, out_screens, out_rg,
                                    args.width, args.nav_timeout, args.op_timeout,
                                    args.ss_timeout, args.jpeg, args.debug) for j in jobs]
            await asyncio.gather(*tasks)
        finally:
            await browser.close()
    return 0

# ============ aggregate ============
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _load_rg(p: Path): 
    return np.array(Image.open(p).convert("RGBA"))

def run_aggregate(args):
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rgdir=Path(args.rgdir)
    legend=None
    if args.legend and Path(args.legend).exists():
        legend={int(k):v for k,v in json.loads(Path(args.legend).read_text(encoding="utf-8")).items()}
        dbg(args.debug,f"legend {len(legend)} entries")
    rows=[]; jsonl=(outdir/"band_densities.jsonl").open("w",encoding="utf-8")

    def process_band(fname,y_pct,h_pct,label):
        rgpng=rgdir/fname
        if rgpng.suffix.lower() not in (".png",".jpg",".jpeg"): rgpng=rgpng.with_suffix(".png")
        if not rgpng.exists(): 
            print(f"[WARN] RG missing for {fname} at {rgpng}"); 
            return
        arr=_load_rg(rgpng); H,W,_=arr.shape; G=arr[:,:,1]
        y1,y2=pct_to_rows(float(y_pct),float(h_pct),H)
        if y2<=y1: 
            dbg(args.debug,f"{fname} {label} empty rows {y1}-{y2}"); 
            return
        vals=G[y1:y2,:].flatten().tolist()
        cnt=Counter(vals); total=int((y2-y1)*W)
        rec={"file_upload":fname,"H":int(H),"W":int(W),"band_label":label,
             "y1":int(y1),"y2":int(y2),"pixels_in_band":total,
             "g_hist":{str(k):int(v) for k,v in cnt.items()}}
        if legend:
            dec=defaultdict(int)
            for k,v in rec["g_hist"].items(): dec[legend.get(int(k),f"g{int(k)}")]+=int(v)
            rec["g_hist_decoded"]=dict(dec)
        jsonl.write(json.dumps(rec)+"\n")
        g_top,g_n=(0,0) if not cnt else max(cnt.items(), key=lambda kv: kv[1])
        row={"file_upload":fname,"band_label":label,"y1":y1,"y2":y2,
             "pixels_in_band":total,"g_top":int(g_top),"g_top_count":int(g_n),
             "g_top_pct":(g_n/total) if total else 0.0}
        if legend: row["g_top_name"]=legend.get(int(g_top),f"g{int(g_top)}")
        rows.append(row)

    for lbl in args.labels:
        data=json.loads(Path(lbl).read_text(encoding="utf-8"))
        if args.ls2245:
            items=data if isinstance(data,list) else [data]
            for t in items:
                img=t.get("image") or (t.get("data",{}) or {}).get("image")
                if not img: continue
                fname=Path(img).name
                for b in (t.get("band") or []):
                    label=b.get("rectanglelabels") or b.get("label") or "Unknown"
                    y=float(b.get("y",0)); h=float(b.get("height",0))
                    process_band(fname,y,h,label)
        else:
            tasks=data if isinstance(data,list) else (data.get("tasks") or [data])
            for t in tasks:
                fu=t.get("file_upload") or (t.get("data",{}) or {}).get("image")
                if not fu: continue
                fname=Path(fu).name
                for ann in (t.get("annotations") or []):
                    for r in (ann.get("result") or []):
                        if r.get("type")!="rectanglelabels": continue
                        v=r.get("value",{})
                        if "y" not in v or "height" not in v: continue
                        labels=v.get("rectanglelabels") or []
                        label=labels[0] if labels else "Unknown"
                        process_band(fname,float(v["y"]),float(v["height"]),label)

    jsonl.close()
    if args.emit_csv and rows:
        import csv
        csvp=outdir/"band_densities_summary.csv"
        with csvp.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[OK] CSV   → {csvp}")
    print(f"[OK] JSONL → {outdir/'band_densities.jsonl'}")
    return 0

# ============ qa ============
def run_qa(args):
    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    overlay_dir=outdir/"overlays"; overlay_dir.mkdir(exist_ok=True)
    rgdir=Path(args.rgdir); issues=[]

    def push(kind,site,detail): issues.append({"kind":kind,"site":site,"detail":detail})
    def overlay_one(fname,bands):
        rgp=rgdir/fname
        if rgp.suffix.lower() not in (".png",".jpg",".jpeg"): rgp=rgp.with_suffix(".png")
        if not rgp.exists(): push("rg_missing",fname,f"{rgp}"); return
        from PIL import Image
        img=Image.open(rgp).convert("RGB"); W,H=img.size; d=ImageDraw.Draw(img)
        for label,y_pct,h_pct in bands:
            if not (0<=y_pct<=100 and 0<=h_pct<=100) or y_pct+h_pct>100.0001:
                push("range_warn",fname,f"{label} y={y_pct} h={h_pct}")
            y1,y2=pct_to_rows(y_pct,h_pct,H)
            if y2<=y1: push("empty_band",fname,f"{label} {y1}-{y2}"); continue
            d.line([(0,y1),(W,y1)], fill=(0,255,0), width=2)
            d.line([(0,y2),(W,y2)], fill=(255,0,0), width=2)
            d.text((8,max(0,y1+3)), f"{label} ({y2-y1}px)", fill=(255,255,0))
        img.save((overlay_dir/fname).with_suffix(".png"))

    for lbl in args.labels:
        data=json.loads(Path(lbl).read_text(encoding="utf-8"))
        if args.ls2245:
            items=data if isinstance(data,list) else [data]
            for t in items:
                img=t.get("image") or (t.get("data",{}) or {}).get("image")
                if not img: continue
                fname=Path(img).name; bands=[]
                for b in (t.get("band") or []):
                    label=b.get("rectanglelabels") or b.get("label") or "Unknown"
                    bands.append((label, float(b.get("y",0)), float(b.get("height",0))))
                overlay_one(fname,bands)
        else:
            tasks=data if isinstance(data,list) else (data.get("tasks") or [data])
            for t in tasks:
                fu=t.get("file_upload") or (t.get("data",{}) or {}).get("image")
                if not fu: push("missing_file_upload","<unknown>",f"in {lbl}"); continue
                fname=Path(fu).name; bands=[]
                for ann in (t.get("annotations") or []):
                    for r in (ann.get("result") or []):
                        if r.get("type")!="rectanglelabels": continue
                        v=r.get("value",{})
                        if "y" not in v or "height" not in v: push("missing_coords",fname,"no y/height"); continue
                        labels=v.get("rectanglelabels") or []
                        label=labels[0] if labels else "Unknown"
                        bands.append((label,float(v["y"]),float(v["height"])))
                overlay_one(fname,bands)

    report={"issues_found":len(issues),"issues":issues[:200],"overlay_dir":str(overlay_dir)}
    (outdir/"qa_report.json").write_text(json.dumps(report, indent=2))
    print(f"[OK] QA report → {outdir/'qa_report.json'}")
    print(f"[OK] Overlays  → {overlay_dir}")
    if issues: print(f"[WARN] {len(issues)} issues found")
    else: print("[OK] No issues found")
    return 0

# ============ CLI ============
def main():
    ap=argparse.ArgumentParser(description="WFC P1 all-in-one — capture / aggregate / qa")
    sub=ap.add_subparsers(dest="cmd", required=True)

    ap_cap=sub.add_parser("capture", help="Normalize + RG-encode (concurrent)")
    ap_cap.add_argument("--labels", nargs="+", required=True)
    ap_cap.add_argument("--out", required=True)
    ap_cap.add_argument("--width", type=int, default=1440)
    ap_cap.add_argument("--nav_timeout", type=int, default=20000, help="ms for page.goto")
    ap_cap.add_argument("--op_timeout", type=int, default=90000, help="ms per-site hard cap")
    ap_cap.add_argument("--ss_timeout", type=int, default=90000, help="ms for full_page screenshots")
    ap_cap.add_argument("--concurrency", type=int, default=6, help="parallel pages")
    ap_cap.add_argument("--jpeg", action="store_true")
    ap_cap.add_argument("--limit", type=int, default=0)
    ap_cap.add_argument("--ls2245", action="store_true")
    ap_cap.add_argument("--debug", action="store_true")
    ap_cap.set_defaults(func=lambda a: asyncio.run(run_capture(a)))

    ap_agg=sub.add_parser("aggregate", help="Aggregate RG pixels by labeled bands")
    ap_agg.add_argument("--labels", nargs="+", required=True)
    ap_agg.add_argument("--rgdir", required=True)
    ap_agg.add_argument("--outdir", required=True)
    ap_agg.add_argument("--legend")
    ap_agg.add_argument("--emit_csv", action="store_true")
    ap_agg.add_argument("--ls2245", action="store_true")
    ap_agg.add_argument("--debug", action="store_true")
    ap_agg.set_defaults(func=run_aggregate)

    ap_qa=sub.add_parser("qa", help="Validate & overlay bands")
    ap_qa.add_argument("--labels", nargs="+", required=True)
    ap_qa.add_argument("--rgdir", required=True)
    ap_qa.add_argument("--outdir", required=True)
    ap_qa.add_argument("--max_preview_sites", type=int, default=10)  # kept for parity, not used here
    ap_qa.add_argument("--ls2245", action="store_true")
    ap_qa.add_argument("--debug", action="store_true")
    ap_qa.set_defaults(func=run_qa)

    args=ap.parse_args()
    rc=args.func(args); sys.exit(rc if isinstance(rc,int) else 0)

if __name__=="__main__":
    main()

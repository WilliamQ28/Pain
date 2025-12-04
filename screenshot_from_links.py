#!/usr/bin/env python
# Usage:
#   (.venv)> python screenshot_from_links_concurrent_patched.py --in C:\wfc\urls.txt --out C:\wfc\screens --concurrency 6

import argparse, asyncio, re, sys, traceback
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PWTimeout, Error as PWError

NORMALIZE_CSS = """
  * { scroll-behavior: auto !important; }
  /* neutralize sticky/fixed positions so headers/footers sit in DOM order */
  :where(header, nav, [role="banner"], .header, .navbar) {
    position: static !important; top:auto !important; bottom:auto !important; inset:auto !important;
  }
  :where([class*="sticky"], [id*="sticky"], [class*="fixed"], [id*="fixed"]) {
    position: static !important; top:auto !important; bottom:auto !important; inset:auto !important;
  }
  /* hide common overlays that pollute layout */
  :where([id*="cookie"], [class*="cookie"], [aria-label*="cookie"],
         [id*="consent"], [class*="consent"],
         [class*="chat"], [aria-label*="chat"], [id*="chat"],
         [class*="subscribe"], [id*="subscribe"],
         [role="dialog"], [aria-modal="true"]) {
    display: none !important;
  }
"""

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

def slugify(url: str) -> str:
    s = re.sub(r"^https?://", "", url.strip().lower())
    s = s.split("?")[0].split("#")[0]
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "site"

async def robust_goto(page, url: str, timeout_ms: int, retries=1):
    last_err = None
    for attempt in range(retries + 1):
        for i, wait_until in enumerate(("domcontentloaded", "networkidle", "load")):
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                return True, None
            except PWTimeout as e:
                last_err = e
                if i == 2:
                    break
            except PWError as e:
                last_err = e
                if "Execution context was destroyed" in str(e) and attempt < retries:
                    await asyncio.sleep(0.5)
                    break
                else:
                    break
        if attempt < retries:
            await asyncio.sleep(0.5)
    return False, last_err

async def scroll_to_bottom(page, max_steps=40, step_px=1200, pause_ms=200):
    last_y = -1
    for _ in range(max_steps):
        try:
            y = await page.evaluate("""(step)=>{const h = document.documentElement.scrollHeight||document.body.scrollHeight;
                const y0 = window.scrollY||window.pageYOffset||0; const next = Math.min(h, y0+step);
                window.scrollTo(0,next); return next;}""", step_px)
        except Exception:
            return
        if y == last_y:
            break
        last_y = y
        await page.wait_for_timeout(pause_ms)

async def normalize_layout(page):
    """
    CSP-safe style injection:
      1) prefer adding a <style> element via evaluate (DOM API)
      2) if that fails, fall back to page.add_style_tag (may be blocked by CSP)
    Also force computed fixed/sticky -> static as belt-and-suspenders.
    """
    try:
        css_escaped = NORMALIZE_CSS.replace("`", "\\`").replace("\\", "\\\\")
        js = f"""
        (() => {{
        try {{
            const css = `{css_escaped}`;
            const style = document.createElement('style');
            style.type = 'text/css';
            style.textContent = css;
            const head = document.head || document.getElementsByTagName('head')[0] || document.documentElement;
            head.appendChild(style);
            for (const el of Array.from(document.querySelectorAll('*'))) {{
            try {{
                const s = getComputedStyle(el);
                if (s && (s.position === 'fixed' || s.position === 'sticky')) {{
                el.style.position = 'static';
                el.style.top = 'auto'; el.style.bottom = 'auto'; el.style.inset = 'auto';
                }}
            }} catch(e){{}}
            }}
            return true;
        }} catch (e) {{
            return {{__err__: String(e)}};
        }}
        }})();
        """

        res = await page.evaluate(js)
        if isinstance(res, dict) and "__err__" in res:            raise Exception(res["__err__"])
        return True, None
    except Exception as e_eval:
        try:
            await page.add_style_tag(content=NORMALIZE_CSS)
            
            await page.evaluate("""
              for (const el of Array.from(document.querySelectorAll('*'))) {
                try {
                  const s = getComputedStyle(el);
                  if (s && (s.position === 'fixed' || s.position === 'sticky')) {
                    el.style.position = 'static';
                    el.style.top = 'auto'; el.style.bottom = 'auto'; el.style.inset = 'auto';
                  }
                } catch(e){}
              }
            """)
            return True, None
        except Exception as e_add:
            return False, f"evaluate_err={e_eval} add_style_tag_err={e_add}"

async def capture_once(ctx, url, out_path, width, timeout_ms, jpeg, goto_retries=1):
    page = await ctx.new_page()
    try:
        await page.set_viewport_size({"width": width, "height": 900})
        ok, err = await robust_goto(page, url, timeout_ms, retries=goto_retries)
        if not ok:
            return False, f"goto_failed: {err}"
        ok_norm, err_norm = await normalize_layout(page)
        if not ok_norm:
            return False, f"normalize_failed: {err_norm}"

        # trigger lazy load
        await scroll_to_bottom(page)
        try:
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(250)
        except Exception:
            pass

        if jpeg:
            await page.screenshot(path=str(out_path), full_page=True, type="jpeg", quality=95)
        else:
            await page.screenshot(path=str(out_path), full_page=True)
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        try:
            await page.close()
        except Exception:
            pass

async def worker(worker_id, pw, queue: asyncio.Queue, out_dir: Path, width: int,
                 timeout_ms: int, jpeg: bool, throttle_ms: int):
    browser = await pw.chromium.launch(headless=True)
    try:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            url, retries = item
            slug = slugify(url)
            out_path = out_dir / (slug + (".jpg" if jpeg else ".png"))
            if out_path.exists():
                print(f"[W{worker_id} SKIP] {slug}")
                queue.task_done()
                continue
            try:
                ctx = await browser.new_context(
                    device_scale_factor=1,
                    locale="en-US",
                    user_agent=UA,
                    ignore_https_errors=True,
                    java_script_enabled=True,
                )
                ok, err = await capture_once(ctx, url, out_path, width, timeout_ms, jpeg, goto_retries=1)
                await ctx.close()
                if ok:
                    print(f"[W{worker_id} OK ] {slug}")
                else:
                    msg = str(err or "")
                    if retries > 0:
                        print(f"[W{worker_id} RETRY] {slug} -> {msg}")
                        await asyncio.sleep(0.5)
                        queue.put_nowait((url, retries - 1))
                    else:
                        print(f"[W{worker_id} ERR] {slug} -> {msg}")
            finally:
                queue.task_done()
                if throttle_ms > 0:
                    await asyncio.sleep(throttle_ms / 1000)
    finally:
        await browser.close()

async def main():
    ap = argparse.ArgumentParser(description="Concurrent normalized full-page screenshots (CSP-safe).")
    ap.add_argument("--in", dest="infile", required=True, help="Text file with one URL per line")
    ap.add_argument("--out", dest="outdir", required=True, help="Output folder")
    ap.add_argument("--width", type=int, default=1440, help="Viewport width")
    ap.add_argument("--timeout", type=int, default=20000, help="Per-page nav timeout (ms)")
    ap.add_argument("--jpeg", action="store_true", help="Save JPEG (default PNG)")
    ap.add_argument("--concurrency", type=int, default=6, help="Number of parallel workers/browsers")
    ap.add_argument("--retries", type=int, default=1, help="Retries per URL on failure")
    ap.add_argument("--throttle", type=int, default=100, help="Delay (ms) between jobs per worker")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of URLs")
    args = ap.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(args.infile).read_text(encoding="utf-8").splitlines()
            if u.strip() and not u.strip().startswith("#")]
    if args.limit > 0:
        urls = urls[:args.limit]

    seen = set(); deduped = []
    for u in urls:
        try:
            pr = urlparse(u)
            key = (pr.scheme, pr.netloc, pr.path)
            if key not in seen:
                seen.add(key); deduped.append(u)
        except Exception:
            deduped.append(u)
    urls = deduped

    queue: asyncio.Queue = asyncio.Queue()
    for u in urls:
        queue.put_nowait((u, args.retries))

    async with async_playwright() as pw:
        workers = [asyncio.create_task(worker(i+1, pw, queue, out_dir, args.width,
                                             args.timeout, args.jpeg, args.throttle))
                   for i in range(max(1, args.concurrency))]
        for _ in workers:
            queue.put_nowait(None)
        await queue.join()
        await asyncio.gather(*workers, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)

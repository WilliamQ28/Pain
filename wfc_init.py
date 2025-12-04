#!/usr/bin/env python3
"""
Generate WFC-ready priors + adjacency matrices from:
  - band_area_fractions.csv
  - band_height_stats.csv
  - band_densities_summary.csv
  - band_densities.json / .jsonl  (very tolerant)
Outputs:
  - wfc_stats/probabilities.json
  - wfc_stats/adjacency.json
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd, numpy as np

# ---------- helpers ----------
def load_csv_loose(p: Path) -> pd.DataFrame:
    if not p.exists(): raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pickcol(df: pd.DataFrame, *names, required=True, default=None):
    for n in names:
        if n in df.columns: return n
    if required: raise ValueError(f"Missing required column; tried {names} in {list(df.columns)}")
    return default

def normalize_dict(d: Dict[str, float], eps=1e-9) -> Dict[str,float]:
    tot = float(sum(max(0.0,v) for v in d.values()))
    if tot<=eps: n=len(d) or 1; return {k:1.0/n for k in d}
    return {k:float(max(0.0,v)/tot) for k,v in d.items()}

def safe_float(v,default=0.0):
    try: return float(v)
    except: return float(default)

def to_numpy_density(x)->np.ndarray:
    if isinstance(x,np.ndarray): arr=x
    elif isinstance(x,list):
        if len(x)>0 and isinstance(x[0],list):
            arr=np.array(x,dtype=float)
        else:
            arr=np.array(x,dtype=float)[:,None]
    else: return None
    arr=np.nan_to_num(arr.astype(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    m=float(arr.max()) if arr.size else 1
    if m>0: arr/=m
    return arr

def corr_overlap_bottom_top(a,b,k):
    if a is None or b is None or a.size==0 or b.size==0: return 0.0
    Ha,Wa=a.shape; Hb,Wb=b.shape; k=int(max(1,min(k,Ha,Hb))); W=min(Wa,Wb)
    if W<=0: return 0.0
    A=a[Ha-k:Ha,:W].ravel(); B=b[0:k,:W].ravel()
    d=(np.linalg.norm(A)*np.linalg.norm(B))
    return 0.0 if d<=1e-8 else float(np.dot(A,B)/d)

def horizontal_neighbor_overlap(d):
    if d is None or d.size==0: return 0.0
    H,W=d.shape
    if W<2: return float(d.sum())
    L=d[:,:W-1]; R=d[:,1:]
    num=float((L*R).sum()); den=float(np.sqrt((L**2).sum())*np.sqrt((R**2).sum()))
    return 0.0 if den<=1e-8 else float(num/den)

# ---------- robust JSON/JSONL loader ----------
G2TYPE = {16:"text",32:"media",48:"button",64:"input",80:"select",96:"textarea",112:"role_button"}
CANON_G = set(G2TYPE.keys())

def _shape_from_hints(obj: Dict[str, Any]) -> Optional[Tuple[int,int]]:
    H = obj.get("H") or obj.get("h") or obj.get("height") or obj.get("tiles_y")
    W = obj.get("W") or obj.get("w") or obj.get("width")  or obj.get("tiles_x")
    try:
        if H is not None and W is not None:
            return int(H), int(W)
    except Exception:
        pass
    return None

def _find_matrix(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    # Try common keys
    keys = ["density","matrix","values","data","blocks","grid","mask","heatmap"]
    for k in keys:
        if k in obj:
            val = obj[k]
            if isinstance(val, list):
                # 2D or flat with shape hints
                if len(val)>0 and isinstance(val[0], list):
                    return to_numpy_density(val)
                shp = _shape_from_hints(obj)
                if shp and len(val) == int(shp[0]*shp[1]):
                    arr = np.array(val, dtype=float).reshape(shp)
                    return to_numpy_density(arr.tolist())
                return to_numpy_density(val)
    return None

def _band_name(obj: Dict[str, Any]) -> Optional[str]:
    # band_label might be a list like ["Header"]
    if "band_label" in obj:
        v = obj["band_label"]
        try:
            if isinstance(v, list) and v:
                return str(v[0]).strip()
            return str(v).strip()
        except Exception:
            pass
    for k in ("band","band_name","label","band_index","band_idx"):
        if k in obj:
            v = obj[k]
            try:
                return str(v).strip()
            except Exception:
                return None
    return None

def _type_name(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("type","elem_type","kind","t","etype","element_type"):
        if k in obj:
            return str(obj[k]).strip().lower()
    # numeric G?
    for k in ("g","g_top","gval","g_value"):
        if k in obj:
            try:
                return G2TYPE.get(int(obj[k]))
            except Exception:
                return None
    return None

def _merge_record(dens: Dict[str, Dict[str, Any]], rec: Dict[str, Any]) -> bool:
    # Case: dict-of-dicts (band -> type -> matrix)
    if all(isinstance(v, dict) for v in rec.values()) and not {"band","type","band_label"} <= set(rec.keys()):
        for band, tmap in rec.items():
            if isinstance(tmap, dict):
                b = str(band)
                dst = dens.setdefault(b, {})
                for t, mat in tmap.items():
                    arr = _find_matrix({"density": mat})
                    if arr is None and isinstance(mat, list): arr = to_numpy_density(mat)
                    if arr is None: continue
                    dst[str(t)] = arr.tolist()
        return True

    # Case: flat record with per-type matrix
    b = _band_name(rec); t = _type_name(rec)
    if b and t:
        arr = _find_matrix(rec)
        if arr is not None:
            dens.setdefault(b, {})[t] = arr.tolist()
            return True

    # Case: record with g_hist (histogram by G channel per band)
    # Example keys: {"band_label":["Header"], "g_hist":{"112": 73322, "48": 4262, ...}, "H":7607,"W":1440,"y1":...,"y2":...}
    if "g_hist" in rec and (_band_name(rec) is not None):
        b = _band_name(rec)
        gh = rec.get("g_hist") or {}
        if isinstance(gh, dict):
            dst = dens.setdefault(b, {})
            for gk, cnt in gh.items():
                try:
                    g = int(gk)
                except Exception:
                    continue
                if g in CANON_G:
                    tname = G2TYPE[g]
                    # synthesize a tiny placeholder density (1x1)
                    dst[tname] = [[1.0]]
            return True

    return False

def load_dens_any(path: str) -> Dict[str, Dict[str, List[List[float]]]]:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    txt = p.read_text(encoding="utf-8", errors="ignore").strip()

    # 1) Whole-file JSON
    try:
        obj = json.loads(txt)
        dens: Dict[str, Dict[str, List[List[float]]]] = {}
        if isinstance(obj, dict):
            _merge_record(dens, obj)
            if dens: return dens
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    _merge_record(dens, item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            _merge_record(dens, sub)
            if dens: return dens
    except Exception:
        pass

    # 2) JSONL
    dens: Dict[str, Dict[str, List[List[float]]]] = {}
    nonempty_lines: List[str] = []
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s: continue
            nonempty_lines.append(s)
            # Some JSONL lines may contain multiple objects; split simple cases
            parts = s.replace("}{", "}\n{").splitlines()
            for part in parts:
                part = part.strip()
                if not part: continue
                try:
                    obj = json.loads(part)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    _merge_record(dens, obj)
                elif isinstance(obj, list):
                    for sub in obj:
                        if isinstance(sub, dict):
                            _merge_record(dens, sub)

    if dens: return dens

    # 3) Diagnostics
    print("[ERROR] Could not parse densities file as JSON or JSONL:", p)
    for i, s in enumerate(nonempty_lines[:3]):
        print(f"  sample line {i+1}: {s[:240]}{'...' if len(s)>240 else ''}")
    raise SystemExit(1)

# ---------- normalization utilities ----------
def _clean_band_name(name: str) -> str:
    # collapse "['Header']" or " \"Main\" " -> "Header", "Main"
    return str(name).strip().strip("[]'\" ").replace('"','').replace("'", "").strip()

def _clean_top_level_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        nk = _clean_band_name(k)
        # merge if duplicate after cleaning
        if nk in out and isinstance(out[nk], dict) and isinstance(v, dict):
            # shallow merge; later normalization will handle sums where needed
            merged = dict(out[nk]); merged.update(v); out[nk] = merged
        else:
            out[nk] = v
    return out

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--area_csv",default="band_area_fractions.csv")
    ap.add_argument("--height_csv",default="band_height_stats.csv")
    ap.add_argument("--summary_csv",default="band_densities_summary.csv")
    ap.add_argument("--dens_json",default="band_densities.json")
    ap.add_argument("--out_dir",default="wfc_stats")
    ap.add_argument("--keep_tile_densities",action="store_true")
    ap.add_argument("--border_rows_pct",type=float,default=10.0)
    a=ap.parse_args()

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    A=load_csv_loose(Path(a.area_csv))
    H=load_csv_loose(Path(a.height_csv))
    S=load_csv_loose(Path(a.summary_csv))

    # ---- band priors ----
    bandcol=pickcol(A,"band","band_label","label")
    areacol=next((c for c in ("area_fraction","fraction","area","coverage") if c in A.columns),None)
    pri_raw={}
    if areacol:
        for _,r in A.iterrows():
            b=str(r[bandcol]).strip()
            pri_raw[b]=pri_raw.get(b,0)+safe_float(r[areacol])
    elif "px_total" in A.columns:
        for _,r in A.iterrows():
            b=str(r[bandcol]).strip()
            pri_raw[b]=pri_raw.get(b,0)+safe_float(r["px_total"])
    else:
        frac=[c for c in A.columns if c.startswith("frac_")]
        for _,r in A.iterrows():
            b=str(r[bandcol]).strip()
            val=sum(safe_float(r[c]) for c in frac if c!="frac_bg")
            pri_raw[b]=pri_raw.get(b,0)+max(0.0,val)
    # clean band names in priors
    priors = normalize_dict({_clean_band_name(k): v for k, v in pri_raw.items()})

    # ---- heights ----
    band_h=pickcol(H,"band","band_label","label")
    has_mean=any(c in H.columns for c in ("mean","height_mean","avg","avg_height"))
    hcol=None
    for c in ("band_h","height","h","px_h"):
        if c in H.columns: hcol=c; break
    heights={}
    if has_mean:
        mcol=next(c for c in ("mean","height_mean","avg","avg_height") if c in H.columns)
        pcol=next((c for c in ("p95","p90","height_p95") if c in H.columns),None)
        grp=H.groupby(band_h)
        m=grp[mcol].mean()
        if pcol: p=grp[pcol].mean()
        elif hcol: p=grp[hcol].quantile(0.95)
        else: p=m
        for b,v in m.items():
            heights[_clean_band_name(b)]={"mean":float(v),"p95":float(p.get(b,v))}
    elif hcol:
        grp=H.groupby(band_h)[hcol]
        m=grp.mean(); p=grp.quantile(0.95)
        for b,v in m.items():
            heights[_clean_band_name(b)]={"mean":float(v),"p95":float(p.get(b,v))}
    else:
        print("[WARN] no height info found")

    # ---- per-type weights ----
    bsum = pickcol(S, "band", "band_label", "label")
    weights: Dict[str, Dict[str, float]] = {}

    if any(c in S.columns for c in ("type","elem_type","kind","elem","etype")):
        tsum = pickcol(S, "type", "elem_type", "kind", "elem", "etype")
        msum = pickcol(S, "mean", "avg", "value", "mean_density", "pct", "percent", "weight",
                       required=False, default=None)
        if msum is None:
            msum = pickcol(S, "count", "n", "freq", required=False, default=None)
        for _, r in S.iterrows():
            b = _clean_band_name(str(r[bsum]).strip())
            t = str(r[tsum]).lower().strip()
            w = safe_float(r[msum], 1.0) if msum else 1.0
            weights.setdefault(b, {})[t] = weights.get(b, {}).get(t, 0.0) + w
    else:
        gcol = pickcol(S, "g", "g_top", "gval", "g_value")
        wcol = pickcol(S, "g_top_pct", "pct", "percent", "weight", "g_top_count",
                       "count", "freq", required=False, default=None)
        for _, r in S.iterrows():
            b = _clean_band_name(str(r[bsum]).strip())
            t = G2TYPE.get(int(safe_float(r[gcol], 0.0)))
            if not t: continue
            w = safe_float(r[wcol], 1.0) if wcol else 1.0
            weights.setdefault(b, {})[t] = weights.get(b, {}).get(t, 0.0) + w

    # clean and normalize type weights per band
    weights = _clean_top_level_keys(weights)
    for b in set(list(priors)+list(heights)+list(weights)):
        weights[b] = normalize_dict(weights.get(b, {}))

    # ---- load densities (robust) ----
    dens = load_dens_any(a.dens_json)
    dens = _clean_top_level_keys(dens)

    # ---- normalize per-tile densities ----
    tiles={}
    band_set = sorted(set(list(priors)+list(heights)+list(weights)+list(dens)))
    for b in band_set:
        tiles[b]={}
        tmap=dens.get(b,{})
        if not isinstance(tmap,dict): continue
        for t,raw in tmap.items():
            arr=to_numpy_density(raw)
            if arr is None or arr.size==0: continue
            s=float(arr.sum()); arr=(arr/s).astype(np.float32) if s>0 else arr
            tiles[b][t]={"shape":[*arr.shape],"density":arr.tolist() if a.keep_tile_densities else None}

    # ---- vertical adjacency ----
    def band_sum_matrix(b):
        tmap=tiles.get(b,{})
        if not tmap: return np.array([[1.0]],np.float32)
        target=list(tmap.values())[0]["shape"]; Ht,Wt=target
        mats=[]
        for info in tmap.values():
            if info.get("density") is None: mats.append(np.full(target,1.0/(Ht*Wt)))
            else: mats.append(np.array(info["density"],np.float32))
        M=np.zeros(target,np.float32)
        for m in mats: M+=m
        return M

    pct=max(0.1,min(50,a.border_rows_pct))/100.0
    bands=sorted(set(band_set))
    cache={b:band_sum_matrix(b) for b in bands}
    vert={a1:{} for a1 in bands}
    for a1 in bands:
        for b1 in bands:
            k=int(max(1,math.floor(pct*min(cache[a1].shape[0],cache[b1].shape[0]))))
            vert[a1][b1]=max(0.0,corr_overlap_bottom_top(cache[a1],cache[b1],k))
        vert[a1]=normalize_dict(vert[a1])

    # ---- within-band adjacency ----
    adj_in={}
    for b in bands:
        tmap=tiles.get(b,{})
        types=sorted(set(list(tmap)+list(weights.get(b,{}))))
        if not types: continue
        selfs={t:horizontal_neighbor_overlap(np.array(tmap[t]["density"],np.float32))
               if t in tmap and tmap[t]["density"] else weights[b].get(t,0.0)
               for t in types}
        row={}
        for t1 in types:
            row[t1]={t2:math.sqrt(max(1e-9,selfs[t1])*max(1e-9,selfs[t2])) for t2 in types}
            row[t1]=normalize_dict(row[t1])
        adj_in[b]=row

    # ---- write ----
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    probs={"bands":{b:{"prior":priors.get(b,0.0)} for b in bands},
           "band_heights":heights,"type_weights":weights,
           "tile_densities":tiles if a.keep_tile_densities else {}}
    adj={"vertical_band_to_band":vert,"within_band_type_to_type":adj_in}
    outp,outa=out/"probabilities.json",out/"adjacency.json"
    outp.write_text(json.dumps(probs,indent=2))
    outa.write_text(json.dumps(adj,indent=2))
    print(f"[OK] wrote {outp}\n[OK] wrote {outa}")

if __name__=="__main__": main()

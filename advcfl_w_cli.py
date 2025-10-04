#!/usr/bin/env python3
"""
advcfl_w_cli.py

Stream level-by-level through MITgcm big-endian real*4 binaries laid out as
13 faces per level (each face nx x nx), with nk vertical levels stacked.

This is a C-grid configuration:
- W is at interfaces (file has nk interface levels).
- hFacC is at centers (nk levels).
- DRF is cell-center thickness (nk values).
- RF is interface depths (nk+1 values).

What’s computed (depending on which files you provide)
------------------------------------------------------
1) advcfl_wvel_max  (non-hf; no partial cells)
   For interface k (1..nk):
     advcfl_wvel_max(k) = max(|W| over the level) * dt *
                          max( 1/drF_top, 1/drF_bot )
     where:
       drF_top = drF[k]   (center layer above interface k)
       drF_bot = drF[k-1] (center layer below interface k)
   Required files: W + dt + (DRF  or  RF to derive drF)

2) advcfl_W_hf_max  (hf; partial cells)
   For interface k (1..nk):
     advcfl_W_hf_max(k) = max over grid of:
                          |W| * dt * max( 1/(hFacC_top*drF_top),
                                           1/(hFacC_bot*drF_bot) )
     Required files: W + dt + hFacC + (DRF  or  RF to derive drF)

Input file formats (all big-endian real*4)
------------------------------------------
- W:     nk levels × (13 * nx * nx) elements
- hFacC: nk levels × (13 * nx * nx) elements
- DRF:   nk elements
- RF:    nk+1 elements  (used to derive DRF if DRF not provided)

Output (printed line by line, no locations)
-------------------------------------------
k=<k>  W_min=<...>  W_max=<...>
        advcfl_wvel_max=<...>  W_at_wvel=<...>
        advcfl_W_hf_max=<...>  W_at_W_hf=<...>

Usage example
-------------
python advcfl_w_cli.py --w W.0000000060.data --nx 8640 --nk 174 --dt 1200 --rf RF.data --hfacC hFacC.data --use-nan
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, Tuple
import numpy as np

# ------------------ IO helpers (big-endian real*4) ------------------ #

def open_memmap_be_f4(path: str) -> np.memmap:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.memmap(path, dtype=">f4", mode="r")

def load_vec_be_f4(path: str) -> np.ndarray:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.fromfile(path, dtype=">f4").astype("f8", copy=False)  # promote to float64

# ------------------ grid thickness logic ------------------ #

def drf_from_inputs(nk: int, drf_path: Optional[str], rf_path: Optional[str]) -> np.ndarray:
    """
    Return drF (length nk) using DRF if provided; else derive from RF (length nk+1).
    Conventions per user:
      - --nk is the number of vertical levels in the file
      - RF has nk+1 elements
      - DRF has nk elements
    """
    if drf_path:
        drf = load_vec_be_f4(drf_path).ravel()
        if drf.size < nk:
            raise ValueError(f"DRF length ({drf.size}) < nk ({nk})")
        return np.abs(drf[:nk])
    if rf_path:
        rf = load_vec_be_f4(rf_path).ravel()
        if rf.size < nk + 1:
            raise ValueError(f"RF length ({rf.size}) < nk+1 ({nk+1})")
        return np.abs(np.diff(rf[: nk + 1]))  # length nk
    raise ValueError("Need either --drf or --rf to compute drF.")

# ------------------ streaming reducers ------------------ #

def stream_level_minmax(mm: np.memmap, start: int, stop: int, chunk: int, use_nan: bool) -> Tuple[float, float]:
    vmin = np.nanmin if use_nan else np.min
    vmax = np.nanmax if use_nan else np.max
    w_min, w_max = np.inf, -np.inf
    off = 0
    while start + off < stop:
        n = min(chunk, stop - start - off)
        w = mm[start + off : start + off + n]
        w_min = float(min(w_min, vmin(w)))
        w_max = float(max(w_max, vmax(w)))
        off += n
    if w_min == np.inf:  w_min = float("nan")
    if w_max == -np.inf: w_max = float("nan")
    return w_min, w_max

def nanmax_with_index_1d(arr: np.ndarray) -> Tuple[float, int]:
    """Return (nan-aware max value, index) for 1-D float array; (-inf, -1) if all-NaN."""
    tmp = np.where(np.isnan(arr), -np.inf, arr)
    idx = int(tmp.argmax())
    val = float(tmp[idx])
    return (val, idx) if val != float("-inf") else (float("-inf"), -1)

def stream_absmax_and_W_at_arg(mm: np.memmap, start: int, stop: int, chunk: int, use_nan: bool) -> Tuple[float, float]:
    """
    Find max(|W|) over [start:stop) and return (max_absW, W_at_that_point).
    Nan-aware if requested (ignores NaN).
    """
    best_abs = -np.inf
    best_W = float("nan")
    off = 0
    while start + off < stop:
        n = min(chunk, stop - start - off)
        w = mm[start + off : start + off + n]
        wa = np.abs(w.astype("f8", copy=False))
        if use_nan:
            val, idx = nanmax_with_index_1d(wa)
        else:
            idx = int(wa.argmax())
            val = float(wa[idx])
        if not np.isfinite(val):
            off += n
            continue
        if val > best_abs:
            best_abs = val
            best_W = float(w[idx])  # signed W
        off += n
    if best_abs == -np.inf:
        return float("nan"), float("nan")
    return float(best_abs), float(best_W)

def stream_advcfl_W_hf_max_and_W(
    mmW: np.memmap, mmH: np.memmap,
    start_w: int, stop_w: int,
    start_top: Optional[int], drf_top: Optional[float],
    start_bot: Optional[int], drf_bot: Optional[float],
    dt: float, chunk: int, use_nan: bool
) -> Tuple[float, float]:
    """
    Compute per-level advcfl_W_hf_max and the W value at that maximum, streaming in chunks.
    advcfl_W_hf = |W| * dt * max( 1/(h_top*drF_top), 1/(h_bot*drF_bot) ).
    Missing top/bottom -> zero contribution.
    """
    best_cfl = -np.inf
    best_W = float("nan")
    off = 0
    while start_w + off < stop_w:
        n = min(chunk, stop_w - start_w - off)
        w = mmW[start_w + off : start_w + off + n]
        wa = np.abs(w.astype("f8", copy=False))

        inv_top = np.zeros(n, dtype="f8")
        inv_bot = np.zeros(n, dtype="f8")
        if start_top is not None and drf_top and drf_top > 0:
            h_top = mmH[start_top + off : start_top + off + n]
            denom_t = h_top.astype("f8", copy=False) * float(drf_top)
            mt = denom_t > 0.0
            inv_top[mt] = 1.0 / denom_t[mt]
        if start_bot is not None and drf_bot and drf_bot > 0:
            h_bot = mmH[start_bot + off : start_bot + off + n]
            denom_b = h_bot.astype("f8", copy=False) * float(drf_bot)
            mb = denom_b > 0.0
            inv_bot[mb] = 1.0 / denom_b[mb]

        inv_max = np.maximum(inv_top, inv_bot)
        cfl = wa * float(dt) * inv_max

        if use_nan:
            val, idx = nanmax_with_index_1d(cfl)
        else:
            idx = int(cfl.argmax())
            val = float(cfl[idx])

        if not np.isfinite(val):
            off += n
            continue
        if val > best_cfl:
            best_cfl = val
            best_W = float(w[idx])
        off += n

    if best_cfl == -np.inf:
        return float("nan"), float("nan")
    return float(best_cfl), float(best_W)

# ------------------ main ------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-level advective CFL (W) from big-endian real*4 MITgcm binaries (C-grid)."
    )
    ap.add_argument("--w", required=True, help="Path to W.*.data (interfaces, nk levels), big-endian real*4")
    ap.add_argument("--nx", required=True, type=int, help="LLC face size (e.g., 8640)")
    ap.add_argument("--nk", required=True, type=int, help="Number of vertical levels in the file (interfaces for W)")
    ap.add_argument("--dt", required=True, type=float, help="Timestep in seconds")

    # Optional grid files (all big-endian real*4)
    ap.add_argument("--drf", help="Path to DRF.data (centers, length nk)")
    ap.add_argument("--rf",  help="Path to RF.data  (interfaces, length nk+1); used if --drf not given")
    ap.add_argument("--hfacC", help="Path to hFacC.data (centers, nk levels, each 13*nx*nx)")

    # Behavior
    ap.add_argument("--use-nan", action="store_true", help="Ignore NaNs in reductions")
    ap.add_argument("--chunk", type=int, default=5_000_000, help="Chunk size in elements (default 5e6)")
    args = ap.parse_args()

    nx = int(args.nx)
    nk = int(args.nk)
    n_per_level = 13 * nx * nx  # faces laid back-to-back

    # Open W and sanity-check size
    mmW = open_memmap_be_f4(args.w)
    need_W = nk * n_per_level
    if mmW.size < need_W:
        raise ValueError(f"W too small: has {mmW.size} elements, need >= {need_W} (= nk*13*nx*nx).")

    # Build drF (length nk): from DRF if given, else from RF (nk+1)
    have_drf = False
    try:
        drF = drf_from_inputs(nk, args.drf, args.rf)
        have_drf = True
    except Exception as e:
        print(f"# NOTE: advcfl metrics will be skipped: {e}", file=sys.stderr)

    # Optional hFacC
    have_hfac = False
    if args.hfacC:
        try:
            mmH = open_memmap_be_f4(args.hfacC)
            need_H = nk * n_per_level
            if mmH.size < need_H:
                print(f"# NOTE: hFacC smaller than expected ({mmH.size} < {need_H}); advcfl_W_hf_max may be partial.", file=sys.stderr)
            have_hfac = True
        except Exception as e:
            mmH = None
            print(f"# NOTE: could not open hFacC -> advcfl_W_hf_max will be skipped ({e})", file=sys.stderr)
    else:
        mmH = None

    print(f"# nx={nx}  nk={nk}  dt={args.dt}  n_per_level={n_per_level}")
    print(f"# W={args.w}")
    print(f"# DRF={args.drf or 'NONE'}  RF={args.rf or 'NONE'}  hFacC={args.hfacC or 'NONE'}")
    print("# Per-level results:")

    # Main loop over levels 1..nk (interfaces)
    for k in range(1, nk + 1):
        start_w = (k - 1) * n_per_level
        stop_w  = start_w + n_per_level

        # W extrema on this level
        w_min, w_max = stream_level_minmax(mmW, start_w, stop_w, args.chunk, args.use_nan)

        # advcfl_wvel_max (non-hf): needs drF
        advcfl_wvel = float("nan")
        W_at_wvel   = float("nan")
        if have_drf:
            # Adjacent center layers to interface k: top=k, bottom=k-1 (1-based)
            inv_top = (1.0 / drF[k - 1]) if (1 <= k <= nk) else 0.0
            inv_bot = (1.0 / drF[k - 2]) if (k >= 2) else 0.0
            inv_scalar = max(inv_top, inv_bot)  # constant over the level
            max_absW, W_at_wvel = stream_absmax_and_W_at_arg(mmW, start_w, stop_w, args.chunk, args.use_nan)
            if np.isfinite(max_absW):
                advcfl_wvel = max_absW * float(args.dt) * inv_scalar

        # advcfl_W_hf_max (hf): needs drF + hFacC
        advcfl_W_hf = float("nan")
        W_at_W_hf   = float("nan")
        if have_drf and have_hfac:
            kt = k                  # top center layer index (1..nk)
            kb = k - 1 if k >= 2 else None  # bottom center layer index
            start_top = (kt - 1) * n_per_level if (1 <= kt <= nk) else None
            start_bot = (kb - 1) * n_per_level if (kb is not None) else None
            drf_top = float(drF[kt - 1]) if (1 <= kt <= nk) else None
            drf_bot = float(drF[kb - 1]) if (kb is not None) else None

            # Ensure hFacC coverage for these slices
            ok_top = True if (start_top is None) else (start_top + n_per_level <= (mmH.size if mmH is not None else 0))
            ok_bot = True if (start_bot is None) else (start_bot + n_per_level <= (mmH.size if mmH is not None else 0))
            if ok_top and ok_bot:
                advcfl_W_hf, W_at_W_hf = stream_advcfl_W_hf_max_and_W(
                    mmW, mmH, start_w, stop_w,
                    start_top, drf_top, start_bot, drf_bot,
                    args.dt, args.chunk, args.use_nan
                )

        # Print one compact line (no locations)
        print(
            f"k={k:4d}  W_min={w_min:.6g}  W_max={w_max:.6g}  "
            f"advcfl_wvel_max={advcfl_wvel:.6g}  W_at_wvel={W_at_wvel:.6g}  "
            f"advcfl_W_hf_max={advcfl_W_hf:.6g}  W_at_W_hf={W_at_W_hf:.6g}"
        )

if __name__ == "__main__":
    main()

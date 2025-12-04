#!/usr/bin/env python3
"""
field_stats_cli.py

Stream level-by-level through MITgcm LLC big-endian real*4 binaries laid out as
[nx, nx*13, nk], with nk vertical levels stacked.

Each level k = 1..nk is [nx, nx*13] (i, j).

For each level k, compute:
  - min
  - max
  - mean      (population mean)
  - std       (population standard deviation, sqrt(E[x^2] - E[x]^2))
  - (i_min, j_min): location of min within that level
  - (i_max, j_max): location of max within that level

Index conventions
-----------------
- Levels k are printed 1-based (1..nk).
- i, j indices are 0-based:
    i in [0, nx-1]
    j in [0, nx*13-1]
  This matches Fortran-style storage where i is the fastest index.

Input file format (big-endian real*4)
-------------------------------------
- field: [nx, nx*13, nk]  -> nk levels × (13 * nx * nx) elements in linear storage

Required arguments
------------------
--field : path to the binary file
--nx    : LLC face size (e.g., 8640)
--nk    : number of vertical levels

Optional
--------
--use-nan : ignore NaNs in all statistics
--chunk   : chunk size in elements (default 5e6)

Example
-------
python field_stats_cli.py --field T.0000000060.data --nx 8640 --nk 174 --use-nan
"""

from __future__ import annotations
import argparse
import os
from typing import Tuple
import numpy as np


# ------------------ IO helpers (big-endian real*4) ------------------ #

def open_memmap_be_f4(path: str) -> np.memmap:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.memmap(path, dtype=">f4", mode="r")


# ------------------ streaming reducers ------------------ #

def nanmin_with_index_1d(arr: np.ndarray) -> Tuple[float, int]:
    """Return (nan-aware min value, index) for 1-D float array; (+inf, -1) if all-NaN."""
    tmp = np.where(np.isnan(arr), np.inf, arr)
    idx = int(tmp.argmin())
    val = float(tmp[idx])
    return (val, idx) if val != float("inf") else (float("inf"), -1)


def nanmax_with_index_1d(arr: np.ndarray) -> Tuple[float, int]:
    """Return (nan-aware max value, index) for 1-D float array; (-inf, -1) if all-NaN."""
    tmp = np.where(np.isnan(arr), -np.inf, arr)
    idx = int(tmp.argmax())
    val = float(tmp[idx])
    return (val, idx) if val != float("-inf") else (float("-inf"), -1)


def stream_level_stats(
    mm: np.memmap,
    start: int,
    stop: int,
    chunk: int,
    use_nan: bool,
) -> Tuple[float, float, float, float, int, int]:
    """
    Stream over [start:stop) and compute:
      min, max, mean, std, idx_min_flat, idx_max_flat
    Flat indices are 0-based within the level (length = stop - start).

    If use_nan=True, NaNs are ignored in all stats. If all values are NaN,
    all stats are NaN and indices are -1.
    """

    # Accumulators for mean/std
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0

    # Track min/max and their global flat indices (within the level)
    best_min = np.inf
    best_max = -np.inf
    best_min_idx = -1
    best_max_idx = -1

    offset_in_level = 0  # how far into this level we are (0-based)

    while start < stop:
        n = min(chunk, stop - start)
        x = mm[start:start + n].astype("f8", copy=False)

        if use_nan:
            # NaN-aware min/max with indices (within chunk)
            cmin, cmin_idx = nanmin_with_index_1d(x)
            cmax, cmax_idx = nanmax_with_index_1d(x)

            # For sums, ignore NaNs
            mask = np.isfinite(x)
            if mask.any():
                x_valid = x[mask]
                cnt = x_valid.size
                s = float(x_valid.sum())
                ss = float((x_valid * x_valid).sum())
            else:
                cnt = 0
                s = 0.0
                ss = 0.0
        else:
            # No NaNs: straight min/max/sums
            cmin_idx = int(x.argmin())
            cmax_idx = int(x.argmax())
            cmin = float(x[cmin_idx])
            cmax = float(x[cmax_idx])
            cnt = x.size
            s = float(x.sum())
            ss = float((x * x).sum())

        # Update global accumulators
        total_count += cnt
        total_sum += s
        total_sumsq += ss

        # Update global min
        if np.isfinite(cmin) and cmin < best_min:
            best_min = cmin
            best_min_idx = offset_in_level + cmin_idx

        # Update global max
        if np.isfinite(cmax) and cmax > best_max:
            best_max = cmax
            best_max_idx = offset_in_level + cmax_idx

        start += n
        offset_in_level += n

    if total_count == 0:
        # Entire level was NaN or empty
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            -1,
            -1,
        )

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean * mean
    var = max(var, 0.0)  # guard against tiny negative from roundoff
    std = var ** 0.5

    return (
        float(best_min),
        float(best_max),
        float(mean),
        float(std),
        int(best_min_idx),
        int(best_max_idx),
    )


# ------------------ main ------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-level min, max, mean, std, and (i,j) indices from a big-endian real*4 MITgcm LLC field."
    )
    ap.add_argument("--field", required=True,
                    help="Path to field .data ([nx, nx*13, nk]), big-endian real*4")
    ap.add_argument("--nx", required=True, type=int,
                    help="LLC face size (e.g., 8640)")
    ap.add_argument("--nk", required=True, type=int,
                    help="Number of vertical levels in the file (nk)")
    ap.add_argument("--use-nan", action="store_true",
                    help="Ignore NaNs in computations")
    ap.add_argument("--chunk", type=int, default=5_000_000,
                    help="Chunk size in elements (default 5e6)")

    args = ap.parse_args()

    nx = int(args.nx)
    nk = int(args.nk)
    n_i = nx
    n_j = nx * 13
    n_per_level = n_i * n_j  # = 13 * nx * nx

    # Open field and sanity-check size
    mm = open_memmap_be_f4(args.field)
    need = nk * n_per_level
    if mm.size < need:
        raise ValueError(
            f"field too small: has {mm.size} elements, need >= {need} (= nk*13*nx*nx)."
        )

    print(f"# nx={nx}  nk={nk}  n_i={n_i}  n_j={n_j}  n_per_level={n_per_level}")
    print(f"# field={args.field}")
    print("# Per-level results:")
    print("# k  min  max  mean  std  i_min  j_min  i_max  j_max  (i,j are 0-based)")

    # Main loop over levels 1..nk
    for k in range(1, nk + 1):
        start = (k - 1) * n_per_level
        stop = start + n_per_level

        f_min, f_max, f_mean, f_std, idx_min, idx_max = stream_level_stats(
            mm, start, stop, args.chunk, args.use_nan
        )

        if idx_min >= 0:
            # Map flat index -> (i, j) with Fortran-style i-fastest:
            # idx = i + j * n_i
            i_min = idx_min % n_i
            j_min = idx_min // n_i
        else:
            i_min = j_min = -1

        if idx_max >= 0:
            i_max = idx_max % n_i
            j_max = idx_max // n_i
        else:
            i_max = j_max = -1

        print(
            f"k={k:4d}  "
            f"min={f_min:.6g}  max={f_max:.6g}  "
            f"mean={f_mean:.6g}  std={f_std:.6g}  "
            f"i_min={i_min:6d}  j_min={j_min:6d}  "
            f"i_max={i_max:6d}  j_max={j_max:6d}"
        )


if __name__ == "__main__":
    main()

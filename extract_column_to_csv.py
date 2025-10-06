#!/usr/bin/env python3
"""
Parallel (x,y) extractor -> CSV for <field>.NNNNNNNNNN.data files.

- File pattern:   <field>.0000038454.data     (10 digits)
- Layout:         [nx, nx*13, nz] with x fastest, then y, then z
- Stride:         nx * (nx*13) between same (x,y) in adjacent z
- Endianness:     big-endian on disk
- Output:         CSV with first column 'z_level' (0..nz-1), remaining columns named as filenames

This version:
- Adds --field (default: Salt). Example: --field Theta (matches Theta.##########.data)
- Everything else (sorting, bounds, parallelism, CSV format) is unchanged to keep results bit-identical.

USAGE
  python extract_column_parallel.py \
    --dir /path/to/dir \
    --field Theta \
    --nx 8640 --nz 1000 \
    --x 12 --y 34 \
    --out /path/to/output.csv \
    --workers 16 \
    --backend thread          # or: process
"""

from pathlib import Path
import argparse
import concurrent.futures as cf
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- Utilities ----------

def parse_dtype(dtype_str: str) -> np.dtype:
    try:
        dt = np.dtype(dtype_str)
    except TypeError as e:
        raise ValueError(f"Unrecognized dtype '{dtype_str}'") from e
    return dt.newbyteorder(">")  # force big-endian

def list_and_filter_files(
    d: Path,
    field: str,
    lower: Optional[int],
    upper: Optional[int],
) -> List[Tuple[Path, int]]:
    """
    Return a list of (path, N) for files matching: ^{field}.(\d{10}).data$
    Sorted by N ascending. Apply inclusive numeric filters lower/upper if provided.
    """
    pattern = re.compile(rf"^{re.escape(field)}\.(\d{{10}})\.data$")
    items: List[Tuple[Path, int]] = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        N = int(m.group(1))  # tolerant of leading zeros
        if (lower is not None and N < lower) or (upper is not None and N > upper):
            continue
        items.append((p, N))
    items.sort(key=lambda t: t[1])  # stable deterministic order
    return items

def compute_offsets(nx: int, nz: int, x: int, y: int) -> Tuple[int, int, int, int]:
    """
    Return (ny, plane_entries, start, step) for the 1D memmap slice.
    """
    ny = nx * 13
    if not (0 <= x < nx):
        raise ValueError(f"x index {x} out of bounds for nx={nx}")
    if not (0 <= y < ny):
        raise ValueError(f"y index {y} out of bounds for ny={ny} (ny = nx*13 = {ny})")
    plane_entries = nx * ny  # == nx * (nx*13)
    start = y * nx + x
    step  = plane_entries
    return ny, plane_entries, start, step

def read_one_file_column(
    path: Path,
    nz: int,
    start: int,
    step: int,
    plane_entries: int,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Memory-map and slice one file's (x,y) vertical column for all z levels.
    Cast to float64 for CSV consistency (bit-identical with the original script).
    """
    file_size = os.path.getsize(path)
    itemsize  = dtype.itemsize
    if file_size % itemsize != 0:
        print(f"Warning: {path.name} size {file_size} not multiple of dtype size {itemsize}.", file=sys.stderr)
    total_entries = file_size // itemsize
    expected_total_entries = plane_entries * nz
    if total_entries != expected_total_entries:
        print(
            f"Warning: {path.name} has {total_entries} entries; expected {expected_total_entries} (nx*nx*13*nz).",
            file=sys.stderr,
        )
    last_needed_index = start + (nz - 1) * step
    if total_entries <= last_needed_index:
        raise ValueError(
            f"{path.name} insufficient data: need index {last_needed_index}, have max {total_entries-1}."
        )

    mm = np.memmap(path, dtype=dtype, mode="r")
    vals = mm[start : start + step * nz : step].astype(np.float64, copy=True)  # explicit copy frees memmap
    del mm
    return vals  # shape (nz,)

# ---------- Main build ----------

def build_matrix_parallel(
    files_with_N: List[Tuple[Path, int]],
    nx: int,
    nz: int,
    x: int,
    y: int,
    dtype_str: str,
    workers: int,
    backend: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return:
      - a (nz, nfiles) float64 matrix of values
      - list of column names (filenames) in the same order
    """
    dtype = parse_dtype(dtype_str)
    _ny, plane_entries, start, step = compute_offsets(nx, nz, x, y)

    nfiles = len(files_with_N)
    out = np.empty((nz, nfiles), dtype=np.float64)
    colnames = [p.name for (p, _N) in files_with_N]

    def task(idx_pathN):
        idx, (path, _N) = idx_pathN
        vals = read_one_file_column(path, nz, start, step, plane_entries, dtype)
        return idx, vals

    if backend == "process":
        Executor = cf.ProcessPoolExecutor
    elif backend == "thread":
        Executor = cf.ThreadPoolExecutor
    else:
        raise ValueError("--backend must be 'thread' or 'process'")

    with Executor(max_workers=workers) as ex:
        futures = [ex.submit(task, (i, files_with_N[i])) for i in range(nfiles)]
        for fut in cf.as_completed(futures):
            idx, col = fut.result()
            out[:, idx] = col

    return out, colnames

def main():
    ap = argparse.ArgumentParser(description="Parallel (x,y) extractor -> CSV for <field>.NNNNNNNNNN.data files.")
    ap.add_argument("--dir", type=str, required=True, help="Directory containing the data files")
    ap.add_argument("--field", type=str, default="Salt", help="Field/prefix for files (default: Salt). Examples: Theta, U, V")
    ap.add_argument("--nx", type=int, required=True, help="X dimension (nx)")
    ap.add_argument("--nz", type=int, required=True, help="Z dimension (nz)")
    ap.add_argument("--x", type=int, required=True, help="x index (0-based by default)")
    ap.add_argument("--y", type=int, required=True, help="y index (0-based by default; must be < nx*13)")
    ap.add_argument("--dtype", type=str, default="float32", help="On-disk element dtype (default: float32)")
    ap.add_argument("--one-based-indices", action="store_true", help="Treat --x and --y as 1-based if set")
    ap.add_argument("--lower", type=int, default=None, help="Inclusive lower bound for the 10-digit N")
    ap.add_argument("--upper", type=int, default=None, help="Inclusive upper bound for the 10-digit N")
    ap.add_argument("--out", type=str, required=True, help="Path to write CSV output (required)")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8) + 4),
                    help="Parallel workers (default: min(32, CPU+4))")
    ap.add_argument("--backend", type=str, choices=["thread", "process"], default="thread",
                    help="Parallel backend: 'thread' (default) or 'process'")

    args = ap.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        ap.error(f"--dir '{data_dir}' is not a directory")

    x = args.x - 1 if args.one_based_indices else args.x
    y = args.y - 1 if args.one_based_indices else args.y

    files_with_N = list_and_filter_files(data_dir, args.field, args.lower, args.upper)
    if not files_with_N:
        bounds = []
        if args.lower is not None: bounds.append(f"lower={args.lower}")
        if args.upper is not None: bounds.append(f"upper={args.upper}")
        btxt = f" (filtered by {', '.join(bounds)})" if bounds else ""
        print(f"No files matching '{args.field}.##########.data' found in {data_dir}{btxt}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files_with_N)} file(s) for field '{args.field}' after sorting/filtering.", file=sys.stderr)
    print(f"Reading (x={x}, y={y}) over nz={args.nz} with {args.workers} {args.backend} worker(s)...", file=sys.stderr)

    matrix, colnames = build_matrix_parallel(
        files_with_N=files_with_N,
        nx=args.nx, nz=args.nz, x=x, y=y,
        dtype_str=args.dtype,
        workers=args.workers,
        backend=args.backend,
    )

    df = pd.DataFrame({"z_level": np.arange(args.nz, dtype=np.int32)})
    df_vals = pd.DataFrame(matrix, columns=colnames)
    df = pd.concat([df, df_vals], axis=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} columns to '{out_path}'.", file=sys.stderr)

if __name__ == "__main__":
    main()

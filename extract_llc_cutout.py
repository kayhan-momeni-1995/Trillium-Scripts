#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import llc


_FILE_RE_TEMPLATE = r"^{field}\.(\d{{10}})\.data$"


def find_matching_files(input_path: Path, field: str, ts_min: int, ts_max: int) -> List[Tuple[int, Path]]:
    rx = re.compile(_FILE_RE_TEMPLATE.format(field=re.escape(field)))
    matches: List[Tuple[int, Path]] = []
    for p in input_path.iterdir():
        if not p.is_file():
            continue
        m = rx.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        if ts_min <= n <= ts_max:
            matches.append((n, p))
    matches.sort(key=lambda t: t[0])
    return matches


def llc2rect_level(level_compact_yx: np.ndarray, nx: int) -> np.ndarray:
    """
    level_compact_yx: shape (ny, nx) where ny=13*nx, as read from file (y,x).
    llc.llc2rect expects (nx, 13*nx), so we transpose to (x,y) before calling.
    """
    ny = 13 * nx
    if level_compact_yx.shape != (ny, nx):
        raise ValueError(f"Unexpected level shape {level_compact_yx.shape}, expected ({ny}, {nx})")

    level_compact_xy = level_compact_yx.T  # (nx, 13*nx) -> what your llc2rect expects
    # Some implementations might require contiguous input; this is still cheap-ish relative to llc2rect output.
    if not level_compact_xy.flags["C_CONTIGUOUS"]:
        level_compact_xy = np.ascontiguousarray(level_compact_xy)

    return llc.llc2rect(level_compact_xy)


def process_one_file(
    infile: Path,
    outfile: Path,
    *,
    nx: int,
    nz: int,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
):
    ny = 13 * nx

    expected_bytes = nx * ny * nz * 4
    actual_bytes = infile.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch for {infile.name}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} bytes (nx={nx}, ny={ny}, nz={nz})"
        )

    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    if xlen <= 0 or ylen <= 0:
        raise ValueError("Invalid bounds: require xmin<=xmax and ymin<=ymax")

    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Read as (nz, ny, nx) so x is fastest in file (standard raw .data layout)
    inp = np.memmap(infile, dtype=">f4", mode="r", shape=(nz, ny, nx), order="C")
    out = np.memmap(outfile, dtype=">f4", mode="w+", shape=(nz, ylen, xlen), order="C")

    rect_shape_checked = False
    for k in range(nz):
        level_compact = inp[k, :, :]                 # (ny, nx)
        level_rect = llc2rect_level(level_compact, nx)  # whatever llc2rect returns

        if not rect_shape_checked:
            H, W = level_rect.shape
            if not (0 <= xmin <= xmax < W):
                raise ValueError(f"x bounds out of range for llc2rect output: W={W}, got [{xmin},{xmax}]")
            if not (0 <= ymin <= ymax < H):
                raise ValueError(f"y bounds out of range for llc2rect output: H={H}, got [{ymin},{ymax}]")
            rect_shape_checked = True

        cut = level_rect[ymin : ymax + 1, xmin : xmax + 1]
        out[k, :, :] = np.asarray(cut, dtype=">f4", order="C")

    out.flush()
    del out
    del inp


def main():
    ap = argparse.ArgumentParser(description="Cut out a rectangular (x,y) column from LLC compact .data files.")
    ap.add_argument("--input-path", required=True, type=Path)
    ap.add_argument("--output-path", required=True, type=Path)
    ap.add_argument("--field", required=True, type=str)

    ap.add_argument("--ts-min", required=True, type=int)
    ap.add_argument("--ts-max", required=True, type=int)

    ap.add_argument("--nx", required=True, type=int)
    ap.add_argument("--nz", required=True, type=int)

    ap.add_argument("--xmin", required=True, type=int)
    ap.add_argument("--xmax", required=True, type=int)
    ap.add_argument("--ymin", required=True, type=int)
    ap.add_argument("--ymax", required=True, type=int)

    args = ap.parse_args()

    if not args.input_path.is_dir():
        raise SystemExit(f"input-path is not a directory: {args.input_path}")

    files = find_matching_files(args.input_path, args.field, args.ts_min, args.ts_max)
    if not files:
        print("No matching files found.")
        return

    print(f"Found {len(files)} file(s) to process.")
    for n, f in files:
        outf = args.output_path / f.name
        print(f"Processing N={n:010d}: {f.name} -> {outf}")
        process_one_file(
            infile=f,
            outfile=outf,
            nx=args.nx,
            nz=args.nz,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )

    print("Done.")


if __name__ == "__main__":
    main()


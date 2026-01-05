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

    # Validate bounds for the *input compact* slice indexing intent.
    # (Bounds are applied AFTER llc2rect, but we validate later against llc2rect output shape.)
    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid bounds: require xmin<=xmax and ymin<=ymax")

    # Sanity check file size
    expected_bytes = nx * ny * nz * 4
    actual_bytes = infile.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch for {infile.name}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} bytes (nx={nx}, ny={ny}, nz={nz})"
        )

    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1

    outfile.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # We interpret the file as a Fortran-ordered array (x fastest, then y, then z),
    # with shape (nx, ny, nz) == (nx, 13*nx, nz).
    inp = np.memmap(infile, dtype=">f4", mode="r", shape=(nx, ny, nz), order="F")

    # Output in the same logical axis order: (x, y, z) == (xlen, ylen, nz)
    out = np.memmap(outfile, dtype=">f4", mode="w+", shape=(xlen, ylen, nz), order="F")

    rect_shape_checked = False
    for k in range(nz):
        # One horizontal level in compact LLC form: (nx, 13*nx)
        level_compact = inp[:, :, k]

        # Your llc implementation expects level_compact.shape[1] == 13*NX
        # Ensure it sees exactly (nx, 13*nx)
        if level_compact.shape != (nx, ny):
            raise ValueError(f"Unexpected compact level shape {level_compact.shape}, expected ({nx},{ny})")

        # Some llc implementations are picky about contiguity; make a cheap contiguous view if needed.
        if not level_compact.flags["F_CONTIGUOUS"] and not level_compact.flags["C_CONTIGUOUS"]:
            level_compact = np.asfortranarray(level_compact)

        # Convert to rectangular
        level_rect = llc.llc2rect(level_compact)

        # Apply bounds in the order YOU specified: (x, y)
        if not rect_shape_checked:
            rx, ry = level_rect.shape  # axis0=x, axis1=y per your convention
            if not (0 <= xmin <= xmax < rx):
                raise ValueError(f"x bounds out of range for llc2rect output: rx={rx}, got [{xmin},{xmax}]")
            if not (0 <= ymin <= ymax < ry):
                raise ValueError(f"y bounds out of range for llc2rect output: ry={ry}, got [{ymin},{ymax}]")
            rect_shape_checked = True

        cut = level_rect[xmin : xmax + 1, ymin : ymax + 1]

        # Write out as big-endian float32, preserving (x,y,z) axis order
        out[:, :, k] = np.asarray(cut, dtype=">f4", order="F")

    out.flush()
    del out
    del inp


def main():
    ap = argparse.ArgumentParser(
        description="Extract a rectangular (x,y) cutout from LLC compact .data files using llc.llc2rect(), level-by-level."
    )
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
    print(
        f"Input shape (x,y,z)=({args.nx},{13*args.nx},{args.nz}); "
        f"Cut x=[{args.xmin},{args.xmax}], y=[{args.ymin},{args.ymax}] -> "
        f"Output shape (x,y,z)=({args.xmax-args.xmin+1},{args.ymax-args.ymin+1},{args.nz})"
    )

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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

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

    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid bounds: require xmin<=xmax and ymin<=ymax")

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

    # Treat file as Fortran-ordered (x fastest, then y, then z): shape (nx, 13*nx, nz)
    inp = np.memmap(infile, dtype=">f4", mode="r", shape=(nx, ny, nz), order="F")
    out = np.memmap(outfile, dtype=">f4", mode="w+", shape=(xlen, ylen, nz), order="F")

    rect_shape_checked = False
    for k in range(nz):
        level_compact = inp[:, :, k]  # (nx, 13*nx)

        if level_compact.shape != (nx, ny):
            raise ValueError(f"Unexpected compact level shape {level_compact.shape}, expected ({nx},{ny})")

        # Ensure llc2rect sees a sensible contiguous array if it needs one
        if not level_compact.flags["F_CONTIGUOUS"] and not level_compact.flags["C_CONTIGUOUS"]:
            level_compact = np.asfortranarray(level_compact)

        level_rect = llc.llc2rect(level_compact)

        # Bounds apply as (x,y) on llc2rect output
        if not rect_shape_checked:
            rx, ry = level_rect.shape
            if not (0 <= xmin <= xmax < rx):
                raise ValueError(f"x bounds out of range for llc2rect output: rx={rx}, got [{xmin},{xmax}]")
            if not (0 <= ymin <= ymax < ry):
                raise ValueError(f"y bounds out of range for llc2rect output: ry={ry}, got [{ymin},{ymax}]")
            rect_shape_checked = True

        cut = level_rect[xmin : xmax + 1, ymin : ymax + 1]
        out[:, :, k] = np.asarray(cut, dtype=">f4", order="F")

    out.flush()
    del out
    del inp


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract a rectangular (x,y) cutout from LLC compact .data files using llc.llc2rect(), level-by-level.\n"
            "Modes:\n"
            "  1) Batch mode: scan input-path for {field}.{N:010d}.data within [ts-min, ts-max]\n"
            "  2) Single-file mode: use --file and process only that file (no naming convention required)"
        )
    )

    # Mode selector
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", type=Path, default=None, help="Single-file mode: process only this file.")
    mode.add_argument("--input-path", type=Path, default=None, help="Batch mode: directory to scan for files.")

    ap.add_argument("--output-path", required=True, type=Path, help="Directory to write outputs into.")
    ap.add_argument("--nx", required=True, type=int)
    ap.add_argument("--nz", required=True, type=int)
    ap.add_argument("--xmin", required=True, type=int)
    ap.add_argument("--xmax", required=True, type=int)
    ap.add_argument("--ymin", required=True, type=int)
    ap.add_argument("--ymax", required=True, type=int)

    # Batch-mode-only args (not required in single-file mode)
    ap.add_argument("--field", type=str, default=None, help="Batch mode only.")
    ap.add_argument("--ts-min", type=int, default=None, help="Batch mode only.")
    ap.add_argument("--ts-max", type=int, default=None, help="Batch mode only.")

    args = ap.parse_args()

    out_dir: Path = args.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Single-file mode
    # -------------------------
    if args.file is not None:
        if args.field is not None or args.ts_min is not None or args.ts_max is not None:
            raise SystemExit("In --file (single-file) mode, do not pass --field/--ts-min/--ts-max.")

        infile = args.file
        if not infile.is_file():
            raise SystemExit(f"--file is not a file: {infile}")

        outfile = out_dir / infile.name
        print(f"Single-file mode: {infile} -> {outfile}")
        print(
            f"Input shape (x,y,z)=({args.nx},{13*args.nx},{args.nz}); "
            f"Cut x=[{args.xmin},{args.xmax}], y=[{args.ymin},{args.ymax}] -> "
            f"Output shape (x,y,z)=({args.xmax-args.xmin+1},{args.ymax-args.ymin+1},{args.nz})"
        )

        process_one_file(
            infile=infile,
            outfile=outfile,
            nx=args.nx,
            nz=args.nz,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
        )
        print("Done.")
        return

    # -------------------------
    # Batch mode
    # -------------------------
    input_path = args.input_path
    if input_path is None or not input_path.is_dir():
        raise SystemExit(f"--input-path is not a directory: {input_path}")

    if args.field is None or args.ts_min is None or args.ts_max is None:
        raise SystemExit("Batch mode requires --field, --ts-min, and --ts-max.")

    files = find_matching_files(input_path, args.field, args.ts_min, args.ts_max)
    if not files:
        print("No matching files found.")
        return

    print(f"Batch mode: found {len(files)} file(s) to process.")
    print(
        f"Input shape (x,y,z)=({args.nx},{13*args.nx},{args.nz}); "
        f"Cut x=[{args.xmin},{args.xmax}], y=[{args.ymin},{args.ymax}] -> "
        f"Output shape (x,y,z)=({args.xmax-args.xmin+1},{args.ymax-args.ymin+1},{args.nz})"
    )

    for n, f in files:
        outf = out_dir / f.name
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

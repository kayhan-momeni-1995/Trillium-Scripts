#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CutoutMap:
    """Precomputed mapping for one (x,y) rectangle from llc2rect output back to compact linear indices."""
    offsets_in_level: np.ndarray  # int64, shape (xlen, ylen), 0-based offsets within a level
    valid_mask: np.ndarray        # bool, same shape; True where llc2rect is valid
    area_per_level: int           # nx * (13*nx)


def build_cutout_map(nx: int, xmin: int, xmax: int, ymin: int, ymax: int) -> CutoutMap:
    """
    Build mapping from rect-cutout cells -> compact linear offsets (within a single level),
    using llc.llc2rect() only once (actually twice: indices + mask), then reuse for all levels/files.
    """
    ny = 13 * nx
    area = nx * ny

    if xmin > xmax or ymin > ymax:
        raise ValueError("Invalid bounds: require xmin<=xmax and ymin<=ymax")

    # Mask: where does llc2rect place real data vs padding?
    # Using ones avoids confusing real zeros in the field.
    ones = np.ones((nx, ny), dtype=np.uint8, order="F")
    mask_rect = llc.llc2rect(ones)
    mask_rect = np.asarray(mask_rect)
    rx, ry = mask_rect.shape

    if not (0 <= xmin <= xmax < rx):
        raise ValueError(f"x bounds out of range for llc2rect output: rx={rx}, got [{xmin},{xmax}]")
    if not (0 <= ymin <= ymax < ry):
        raise ValueError(f"y bounds out of range for llc2rect output: ry={ry}, got [{ymin},{ymax}]")

    valid_mask = (mask_rect[xmin : xmax + 1, ymin : ymax + 1] != 0)

    # Index field: store 1-based linear indices in compact layout so 0 can remain "invalid".
    # IMPORTANT: float32 cannot represent all indices exactly for large grids, so use uint32.
    # (If llc2rect internally casts to float64, still exact. If it casts to float32, you'd lose precision.)
    idx_1based = (np.arange(area, dtype=np.uint32) + 1).reshape((nx, ny), order="F")
    idx_rect = llc.llc2rect(idx_1based)
    idx_rect = np.asarray(idx_rect)

    cut_idx = idx_rect[xmin : xmax + 1, ymin : ymax + 1]

    # Convert to int64 offsets (0-based). If cut_idx is float, rint it.
    if np.issubdtype(cut_idx.dtype, np.floating):
        nearest = np.rint(cut_idx)
        # sanity check: values should be extremely close to integers
        err = np.max(np.abs(cut_idx - nearest)) if cut_idx.size else 0.0
        if err > 1e-3:
            raise ValueError(
                "Index mapping lost integer precision inside llc2rect (likely cast to float32). "
                "In that case, we need a different mapping strategy."
            )
        cut_idx_i64 = nearest.astype(np.int64, copy=False)
    else:
        cut_idx_i64 = cut_idx.astype(np.int64, copy=False)

    # Convert 1-based -> 0-based offsets; fill invalids with 0 (we'll zero them using mask later)
    offsets = cut_idx_i64 - 1
    offsets[~valid_mask] = 0

    # Additional sanity
    if offsets.max(initial=0) >= area or offsets.min(initial=0) < 0:
        raise ValueError("Computed offsets are out of range; mapping is inconsistent.")

    return CutoutMap(offsets_in_level=offsets, valid_mask=valid_mask, area_per_level=area)


def process_one_file_mapped(
    infile: Path,
    outfile: Path,
    *,
    nx: int,
    nz: int,
    cmap: CutoutMap,
):
    ny = 13 * nx
    area = cmap.area_per_level

    expected_bytes = area * nz * 4
    actual_bytes = infile.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch for {infile.name}: got {actual_bytes} bytes, "
            f"expected {expected_bytes} bytes (nx={nx}, ny={ny}, nz={nz})"
        )

    xlen, ylen = cmap.offsets_in_level.shape
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # 1D view of the whole file in Fortran-linear order:
    # linear index = x + nx*y + (nx*ny)*z
    flat = np.memmap(infile, dtype=">f4", mode="r", shape=(area * nz,))

    # output in (x,y,z) order, Fortran-contiguous for x-fastest storage
    out = np.memmap(outfile, dtype=">f4", mode="w+", shape=(xlen, ylen, nz), order="F")

    offsets = cmap.offsets_in_level.astype(np.int64, copy=False)
    mask = cmap.valid_mask

    # Reuse an index buffer to avoid allocating base+offsets each level
    idx_buf = np.empty_like(offsets, dtype=np.int64)

    for k in range(nz):
        base = k * area
        np.add(offsets, base, out=idx_buf)

        vals = flat[idx_buf]          # shape (xlen, ylen), dtype >f4
        vals[~mask] = 0               # enforce llc2rect padding behavior
        out[:, :, k] = vals

    out.flush()
    del out
    del flat


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract a rectangular (x,y) cutout from LLC compact .data files.\n"
            "Fast mode: calls llc.llc2rect() only to build a mapping once, then gathers values per level."
        )
    )

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", type=Path, default=None, help="Single-file mode: process only this file.")
    mode.add_argument("--input-path", type=Path, default=None, help="Batch mode: directory to scan for files.")

    ap.add_argument("--output-path", required=True, type=Path)

    ap.add_argument("--nx", required=True, type=int)
    ap.add_argument("--nz", required=True, type=int)

    ap.add_argument("--xmin", required=True, type=int)
    ap.add_argument("--xmax", required=True, type=int)
    ap.add_argument("--ymin", required=True, type=int)
    ap.add_argument("--ymax", required=True, type=int)

    ap.add_argument("--field", type=str, default=None, help="Batch mode only.")
    ap.add_argument("--ts-min", type=int, default=None, help="Batch mode only.")
    ap.add_argument("--ts-max", type=int, default=None, help="Batch mode only.")

    args = ap.parse_args()

    out_dir: Path = args.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping ONCE (reused for all files and all levels)
    print("Building llc2rect mapping once...")
    cmap = build_cutout_map(args.nx, args.xmin, args.xmax, args.ymin, args.ymax)
    xlen, ylen = cmap.offsets_in_level.shape
    print(f"Mapping built. Output per file will be (x,y,z)=({xlen},{ylen},{args.nz})")

    # Single-file mode
    if args.file is not None:
        if args.field is not None or args.ts_min is not None or args.ts_max is not None:
            raise SystemExit("In --file mode, do not pass --field/--ts-min/--ts-max.")

        infile = args.file
        if not infile.is_file():
            raise SystemExit(f"--file is not a file: {infile}")

        outfile = out_dir / infile.name
        print(f"Single-file mode: {infile} -> {outfile}")
        process_one_file_mapped(infile, outfile, nx=args.nx, nz=args.nz, cmap=cmap)
        print("Done.")
        return

    # Batch mode
    input_path = args.input_path
    if input_path is None or not input_path.is_dir():
        raise SystemExit(f"--input-path is not a directory: {input_path}")

    if args.field is None or args.ts_min is None or args.ts_max is None:
        raise SystemExit("Batch mode requires --field, --ts-min, and --ts-max.")

    files = find_matching_files(input_path, args.field, args.ts_min, args.ts_max)
    if not files:
        print("No matching files found.")
        return

    print(f"Batch mode: found {len(files)} file(s).")
    for n, f in files:
        outf = out_dir / f.name
        print(f"Processing N={n:010d}: {f.name} -> {outf}")
        process_one_file_mapped(f, outf, nx=args.nx, nz=args.nz, cmap=cmap)

    print("Done.")


if __name__ == "__main__":
    main()

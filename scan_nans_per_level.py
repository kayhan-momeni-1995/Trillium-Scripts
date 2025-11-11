#!/usr/bin/env python3
import numpy as np
import argparse
import os
import sys


def scan_nans_per_level(filename, nx, ny, nz):
    """
    Stream through a huge binary file containing big-endian float32 data
    with logical shape [nx, ny, nz] laid out level-by-level in nz,
    and report NaNs per level.
    """
    dtype = np.dtype('>f4')  # big-endian float32
    nxy = nx * ny

    # Optional sanity check on file size
    expected_bytes = nxy * nz * dtype.itemsize
    actual_bytes = os.path.getsize(filename)
    if actual_bytes != expected_bytes:
        print(
            f"Warning: file size ({actual_bytes} bytes) does not match "
            f"expected size ({expected_bytes} bytes) for "
            f"{nx} x {ny} x {nz} big-endian float32."
        )

    nan_levels = []

    with open(filename, 'rb') as f:
        for k in range(nz):
            data = np.fromfile(f, dtype=dtype, count=nxy)

            if data.size != nxy:
                print(
                    f"Level {k}: expected {nxy} values, got {data.size}. "
                    "File ended early?"
                )
                break

            n_nans = np.isnan(data).sum()

            if n_nans > 0:
                print(f"Level {k}: found {int(n_nans)} NaNs")
                nan_levels.append((k, int(n_nans)))
            else:
                print(f"Level {k}: no NaNs")

    print("\nSummary:")
    if nan_levels:
        for k, n in nan_levels:
            print(f"  Level {k}: {n} NaNs")
    else:
        print("  No NaNs found in any fully-read level.")

    return nan_levels


def main():
    parser = argparse.ArgumentParser(
        description="Scan a big-endian real*4 (float32) 3D binary file "
                    "for NaNs level-by-level."
    )
    parser.add_argument("filename", help="Path to binary file")
    parser.add_argument("nx", type=int, help="First dimension size (e.g. 4320)")
    parser.add_argument("ny", type=int, help="Second dimension size (e.g. 4320*13)")
    parser.add_argument("nz", type=int, help="Number of levels (third dimension)")

    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print(f"Error: file '{args.filename}' does not exist.", file=sys.stderr)
        sys.exit(1)

    scan_nans_per_level(args.filename, args.nx, args.ny, args.nz)


if __name__ == "__main__":
    main()


import quikplot_llc, quikread_llc, numpy as np, readbin, writebin, llc2rect, rect2llc
from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple, Union
import numpy as np

PathLike = Union[str, Path]

def _natural_key_parts(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _natural_key_path(p: Path):
    # Natural sort across the full path: folder names and filenames
    key = []
    for part in p.parts:
        key.extend(_natural_key_parts(part))
    return key


def find_eta_files(directories: Iterable[PathLike],
                   pattern: str = "Eta.*.data",
                   recursive: bool = False) -> List[Path]:
    """
    Collect and globally natural-sort all matching files across the given directories.
    """
    candidates: List[Path] = []
    for d in directories:
        d = Path(d)
        if not d.is_dir():
            raise NotADirectoryError(f"{d} is not a directory.")
        globber = d.rglob if recursive else d.glob
        candidates.extend([p for p in globber(pattern) if p.is_file()])

    # Resolve + dedupe while preserving order
    resolved = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            resolved.append(rp)

    return sorted(resolved, key=_natural_key_path)

def stack_eta_across_directories(directories: Iterable[PathLike],
                                 pattern: str = "Eta.*.data",
                                 *,
                                 shape: Optional[Tuple[int, int]] = None,
                                 dtype: Optional[np.dtype] = None,
                                 prefer: str = "auto",
                                 recursive: bool = False,
                                 strict_shapes: bool = True):
    """
    Returns (stacked, files):
      - stacked: single array of shape (n, m, t_total)
      - files: ordered list of Paths used (so you know which slice is which)
    """
    files = find_eta_files(directories, pattern=pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in the provided directories.")

    # Load first to establish shape & output dtype
    first = readbin.readbin(files[0], shape)
    if first.ndim != 2:
        raise ValueError(f"First file {files[0]} did not load as 2D.")
    n, m = first.shape
    out_dtype = first.dtype

    t_total = len(files)
    stacked = np.empty((n, m, t_total), dtype=out_dtype)
    stacked[..., 0] = first

    for k, f in enumerate(files[1:], start=1):
        arr = readbin.readbin(f, [n, m])
        if arr.shape != (n, m):
            if strict_shapes:
                raise ValueError(f"Shape mismatch for {f}. Got {arr.shape}, expected {(n, m)}.")
            if arr.size == n * m:
                arr = arr.reshape(n, m)
            else:
                raise ValueError(f"Size mismatch for {f}. Got {arr.size}, expected {n*m}.")
        # Cast to output dtype if necessary
        if arr.dtype != out_dtype:
            arr = arr.astype(out_dtype, copy=False)
        stacked[..., k] = arr

    return stacked, files

if __name__ == "__main__":
    # ===== EDIT THESE DIRECTORIES =====
    directories = [
        "/scratch/dmenemen/llc_1080/run_1080_hr000_hr024",
        "/scratch/dmenemen/llc_1080/run_1080_hr024_hr162",
        "/scratch/dmenemen/llc_1080/run_1080_hr162_dy020",
        "/scratch/dmenemen/llc_1080/run_1080_dy020_dy021",
        "/scratch/dmenemen/llc_1080/run_1080_dy021_dy022",
        "/scratch/dmenemen/llc_1080/run_1080_dy022_hr541",
    ]

    n, m = 1080, 1080*13
    stacked_bin, files_bin = stack_eta_across_directories(
        directories,
        prefer="binary",
        shape=(n, m),
        dtype=np.float32,
        recursive=False,
    )
    print(f"Binary stacked array shape: {stacked_bin.shape}")


    print(stacked_bin[603, 13913, :])
    print("-----------------------------")
    print(stacked_bin[604, 13895, :])
    #stacked_bin = stacked_bin.astype(float, copy=False)
    #mask = (np.abs(stacked_bin) > 20)
    #stacked_bin[mask] = np.nan 

    #minimum_eta = np.nanmin(stacked_bin, axis=2) 
    #maximum_eta = np.nanmax(stacked_bin, axis=2) 
    #std_eta     = np.nanstd(stacked_bin, axis=2) 

    #writebin.writebin("minimum_eta.bin", minimum_eta)
    #writebin.writebin("maximum_eta.bin", maximum_eta)
    #writebin.writebin("std_eta.bin", std_eta)
    #writebin.writebin("stacked_Eta.bin", stacked_bin)

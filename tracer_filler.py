#!/usr/bin/env python3
# Parallel LLC level processor:
# - Single reader (parent) maps input .data (big-endian real*4, Fortran layout)
# - Parent creates SharedMemory for per-level input+output
# - Workers compute per-level in parallel (attach to SHM; no creation/unlink)
# - Parent writes levels in strict order to the output file (big-endian real*4, Fortran order)
# - Parent closes+unlinks SHMs (no leaked SHM / resource_tracker warnings)
import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
from multiprocessing.shared_memory import SharedMemory
from scipy import ndimage as ndi

import llc 


# ===========================
# Interpolation utilities (identical math)
# ===========================

@lru_cache(maxsize=32)
def _row_weights(n: int, k: int):
    rfine = 0.5 + 1.0/(2*k) + np.arange(n * k, dtype=np.float64) / k
    i0 = np.floor(rfine).astype(np.intp)
    di = rfine - i0
    return i0, i0 + 1, di

@lru_cache(maxsize=32)
def _col_weights(m: int, k: int):
    cfine = 0.5 + 1.0/(2*k) + np.arange(m * k, dtype=np.float64) / k
    j0 = np.floor(cfine).astype(np.intp)
    dj = cfine - j0
    return j0, j0 + 1, dj


def _upsample_periodic_staggered_separable(a_ext: np.ndarray,
                                           k: int,
                                           *,
                                           max_bytes: int = 512 << 20) -> np.ndarray:
    """Separable bilinear upsample (cols then rows), chunked by columns for memory bound."""
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)

    n2, m2 = a_ext.shape
    n, m = n2 - 2, m2 - 2
    if n <= 0 or m <= 0:
        raise ValueError("Input must be (n+2)×(m+2) with n,m >= 1")

    A = np.asarray(a_ext, dtype=np.float64, order='C')

    i0, i1, di = _row_weights(n, k)
    j0_full, j1_full, dj_full = _col_weights(m, k)

    nk = n * k
    mk = m * k
    out = np.empty((nk, mk), dtype=np.float64, order='C')

    denom = max(n2 + 3 * nk, 1)
    chunk = max(1, min(mk, max_bytes // (denom * 8)))
    chunk = int(max(chunk, 512))

    for c0 in range(0, mk, chunk):
        c1 = min(c0 + chunk, mk)

        j0 = j0_full[c0:c1]
        j1 = j1_full[c0:c1]
        dj = dj_full[c0:c1]

        A0 = np.take(A, j0, axis=1)
        A1 = np.take(A, j1, axis=1)
        A0 *= (1.0 - dj)
        A1 *= dj
        C = A0
        C += A1
        del A1

        Y0 = np.take(C, i0, axis=0)
        Y1 = np.take(C, i1, axis=0)
        Y0 *= (1.0 - di)[:, None]
        Y1 *= di[:, None]
        Y0 += Y1
        out[:, c0:c1] = Y0

        del C, Y0, Y1

    return out


def upsample_periodic_staggered(a_ext: np.ndarray, k: int) -> np.ndarray:
    return _upsample_periodic_staggered_separable(a_ext, k)


def fillna_with_nearest(a: np.ndarray) -> np.ndarray:
    """Fill NaNs with nearest non-NaN (Euclidean), identical to SciPy EDT behavior."""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Expected a 2D array")
    mask = np.isnan(a)
    if not mask.any():
        return a.copy()

    idx = ndi.distance_transform_edt(mask, return_distances=False, return_indices=True)
    i_idx = np.asarray(idx[0], dtype=np.int32, order='C')
    j_idx = np.asarray(idx[1], dtype=np.int32, order='C')

    out = np.array(a, copy=True, order='C')
    out[mask] = a[i_idx[mask], j_idx[mask]]
    return out


# ===========================
# Boundary builders (no np.insert storms)
# ===========================

def _build_face1245_with_bc(face1245: np.ndarray, face3: np.ndarray) -> np.ndarray:
    r, c = face1245.shape
    nx = face3.shape[0]
    ext = np.empty((r + 2, c + 2), dtype=face1245.dtype)
    ext[1:-1, 1:-1] = face1245
    ext[1:-1, 0]    = face1245[:, 0]
    ext[1:1+nx,            -1] = face3[0, ::-1]
    ext[1+nx:1+2*nx,       -1] = face3[:, 0]
    ext[1+2*nx:1+3*nx,     -1] = face3[-1, :]
    ext[1+3*nx: -1,        -1] = face3[::-1, -1]
    ext[0, 1:-1]  = face1245[-1, :]
    ext[-1, 1:-1] = face1245[0, :]
    ext[0, 0]   = face1245[-1, 0]
    ext[-1, 0]  = face1245[0, 0]
    fr = face3[0, -1]
    ext[0, -1]  = fr
    ext[-1, -1] = fr
    return ext


def _build_face3_with_bc(face3: np.ndarray, face1245_trim: np.ndarray, nx: int) -> np.ndarray:
    ext = np.empty((nx + 2, nx + 2), dtype=face3.dtype)
    ext[1:-1, 1:-1] = face3
    ext[0, 1:-1]  = face1245_trim[0:nx, -1][::-1]
    ext[-1, 1:-1] = face1245_trim[2*nx:3*nx, -1]
    ext[1:-1, 0]  = face1245_trim[nx:2*nx, -1]
    ext[1:-1, -1] = face1245_trim[3*nx:4*nx, -1][::-1]
    ext[0, 0]   = 0.5 * (ext[0, 1]   + ext[1, 0])
    ext[0, -1]  = 0.5 * (ext[0, -2]  + ext[1, -1])
    ext[-1, 0]  = 0.5 * (ext[-2, 0]  + ext[-1, 1])
    ext[-1, -1] = 0.5 * (ext[-2, -1] + ext[-1, -2])
    return ext


# ===========================
# Per-level processing (same output as single-process)
# ===========================

def interpolate_level(fld: np.ndarray, nx: int, k: int,
                      *,
                      interp_mem_budget: int = 512 << 20) -> np.ndarray:
    # LLC -> rectangular
    fld = llc.llc2rect(fld)

    # Remove zeros => NaN
    fld = fld.copy()
    fld[fld == 0] = np.nan

    # Split out Arctic and the rest
    face3 = fld[nx:2*nx, 3*nx:4*nx]   # (nx, nx)
    face1245 = fld[:, 0:3*nx]         # (4*nx, 3*nx)

    # Rectangle with periodic borders
    face1245_ext = _build_face1245_with_bc(face1245, face3)
    face1245_ext[face1245_ext == 0] = np.nan
    face1245_ext = fillna_with_nearest(face1245_ext)
    face1245_upsampled = _upsample_periodic_staggered_separable(face1245_ext, k, max_bytes=interp_mem_budget)

    # Arctic with borders
    face3_ext = _build_face3_with_bc(face3, face1245, nx)
    face3_ext[face3_ext == 0] = np.nan
    face3_ext = fillna_with_nearest(face3_ext)
    face3_upsampled = _upsample_periodic_staggered_separable(face3_ext, k, max_bytes=interp_mem_budget)

    # Paste final rect
    Rk = face1245_upsampled.shape[0]
    Ck_left = face1245_upsampled.shape[1]
    Ck_right = nx * k

    final = np.empty((Rk, Ck_left + Ck_right), dtype=np.float64)
    final[:, :Ck_left] = face1245_upsampled
    final[:nx*k, Ck_left:] = 0.0
    final[nx*k:nx*k + face3_upsampled.shape[0], Ck_left:] = face3_upsampled
    final[nx*k + face3_upsampled.shape[0]:, Ck_left:] = 0.0

    # Fill NaNs
    final[final == 0] = np.nan
    final = fillna_with_nearest(final)

    # Back to LLC
    final = llc.rect2llc(final)
    return final  # float64


# ===========================
# Worker: compute one level using parent-owned SHM
# ===========================

def _worker_process_level(idx: int,
                          shm_in_name: str, in_shape: tuple[int, int],
                          shm_out_name: str, out_shape: tuple[int, int],
                          nx: int, k: int,
                          interp_mem_budget: int) -> int:
    """
    Worker attaches to parent-created SHMs:
      - reads native float32 input (in_shape)
      - computes float64 result, casts once to float32
      - writes to output SHM (out_shape)
    Returns the level index.
    """
    shm_in = SharedMemory(name=shm_in_name)
    shm_out = SharedMemory(name=shm_out_name)
    try:
        in_arr  = np.ndarray(in_shape,  dtype=np.float32, buffer=shm_in.buf)   # native f32
        out_arr = np.ndarray(out_shape, dtype=np.float32, buffer=shm_out.buf)  # native f32

        out64 = interpolate_level(in_arr, nx, k, interp_mem_budget=interp_mem_budget)
        out32 = out64.astype(np.float32, copy=False)  # single rounding, bit-identical to write-stage cast
        np.copyto(out_arr, out32, casting="no")
        # free intermediates ASAP
        del out64, out32
        return idx
    finally:
        shm_in.close()
        shm_out.close()


# ===========================
# Writer helper (SHM -> file, ordered)
# ===========================

def _write_shm_to_file(fout, shm_name: str, shape: tuple[int, int], col_chunk: int = 2048) -> None:
    """
    Stream shared-memory float32 array to fout as big-endian float32 in Fortran order.
    """
    shm = SharedMemory(name=shm_name)
    try:
        arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        nrows, ncols = shape
        for c0 in range(0, ncols, col_chunk):
            c1 = min(c0 + col_chunk, ncols)
            blk_be = arr[:, c0:c1].astype('>f4', copy=False)  # endian-convert this small slice
            fout.write(blk_be.tobytes(order='F'))
    finally:
        shm.close()


# ===========================
# Parallel driver
# ===========================

def main(src_path: str, dst_path: str, nx: int, k_proc: int,
         *,
         workers: int,
         inflight: int,
         interp_mem_budget: int = 512 << 20,
         writer_chunk_cols: int = 2048) -> None:
    # On-disk format: big-endian float32, levels are (nx, nx*13) Fortran layout
    dtype_be = np.dtype(">f4")

    # Sizes
    in_rows, in_cols = nx, nx * 13
    in_bytes = in_rows * in_cols * 4

    out_rows = nx * k_proc
    out_cols = nx * 13 * k_proc
    out_bytes = out_rows * out_cols * 4

    fsize = os.path.getsize(src_path)
    if fsize % in_bytes != 0:
        raise ValueError(f"File size {fsize} is not a multiple of level size {in_bytes}")
    n_levels = fsize // in_bytes
    if n_levels == 0:
        raise ValueError("No full levels found in the input file")

    workers = max(1, workers)
    inflight = max(1, inflight)

    print(f"[info] levels={n_levels}, nx={nx}, k={k_proc}, workers={workers}, inflight={inflight}")
    print(f"[info] per-level sizes: input={in_bytes/1e6:.1f} MB, output={out_bytes/1e6:.1f} MB")

    # Single input memmap
    mm = np.memmap(src_path, dtype='u1', mode="r")

    # Writer (single)
    with open(dst_path, "wb", buffering=io.DEFAULT_BUFFER_SIZE) as fout, \
         ProcessPoolExecutor(max_workers=workers) as pool:

        pending = {}        # future -> (idx, shm_in_name, shm_out_name)
        ready   = {}        # idx -> (shm_in_name, shm_out_name)
        next_to_write = 0
        submitted = 0

        def submit_level(idx: int):
            nonlocal submitted
            off = idx * in_bytes

            # Vectorized read view (big-endian f4) over mmapped file at this level
            lvl_be = np.ndarray(shape=(in_rows, in_cols),
                                dtype=">f4",
                                buffer=mm,
                                offset=off,
                                strides=(4, 4*in_rows))  # Fortran layout on disk

            # Parent creates input SHM (native f32) and copies the level
            shm_in = SharedMemory(create=True, size=in_bytes)
            in_arr = np.ndarray((in_rows, in_cols), dtype=np.float32, buffer=shm_in.buf)
            np.copyto(in_arr, lvl_be, casting="unsafe")  # endian + dtype convert in one pass

            # Parent creates output SHM (native f32) for worker to fill
            shm_out = SharedMemory(create=True, size=out_bytes)

            # Dispatch worker (attaches to SHMs by name; no unlink in worker)
            fut = pool.submit(_worker_process_level,
                              idx,
                              shm_in.name, (in_rows, in_cols),
                              shm_out.name, (out_rows, out_cols),
                              nx, k_proc,
                              interp_mem_budget)
            pending[fut] = (idx, shm_in.name, shm_out.name)
            submitted += 1

        # Prime initial inflight
        while submitted < min(inflight, n_levels):
            submit_level(submitted)

        written = 0
        while pending:
            # Wait for any completion
            for fut in as_completed(list(pending.keys())):
                idx, shm_in_name, shm_out_name = pending.pop(fut)
                idx_done = fut.result()  # propagate worker exceptions
                assert idx_done == idx
                ready[idx] = (shm_in_name, shm_out_name)

                # Maintain inflight: submit next if any left
                if submitted < n_levels:
                    submit_level(submitted)

                # Write in strict order as far as possible
                while next_to_write in ready:
                    in_name, out_name = ready.pop(next_to_write)

                    # Stream write this level from output SHM
                    _write_shm_to_file(fout, out_name, (out_rows, out_cols), col_chunk=writer_chunk_cols)

                    # Cleanup both SHMs (parent created them → parent unlinks)
                    try:
                        shm_out = SharedMemory(name=out_name); shm_out.close(); shm_out.unlink()
                    except FileNotFoundError:
                        pass
                    try:
                        shm_in = SharedMemory(name=in_name); shm_in.close(); shm_in.unlink()
                    except FileNotFoundError:
                        pass

                    written += 1
                    next_to_write += 1
                    print(f"Level {written}/{n_levels} written.")
                break  # re-enter as_completed with updated 'pending'

    # Close input memmap
    del mm

    print(f"Finished. Wrote {n_levels} levels to {dst_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parallel LLC level processor (bit-identical output).")
    ap.add_argument("--nx", type=int, required=True, help="Tile size (e.g., 270 or 1080)")
    ap.add_argument("--k", type=int, default=1, help="Upsampling factor")
    ap.add_argument("--input", required=True, help="Path to input .data (big-endian real*4)")
    ap.add_argument("--output", required=True, help="Path to output .data (will be overwritten)")
    ap.add_argument("--mem", type=str, default="4G",
                    help="Approx per-interpolator working memory for interpolation chunks, e.g. 256M, 1G, 2G")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count(), 173),
                    help="Worker processes (<=192 on your node)")
    ap.add_argument("--inflight", type=int, default=min(os.cpu_count(), 16),
                    help="Max in-flight levels (controls total SHM RAM)")
    ap.add_argument("--writer-cols", type=int, default=98304,
                    help="Writer column chunk size for BE f4 streaming, 0 means setting it to ncols (a one-shot conversion.)")
    args = ap.parse_args()



    # --workers = how many processes do interpolation at the same time (CPU parallelism).
    # --inflight = how many levels the parent keeps alive at once (each with an input SHM and a big output SHM).
    #This caps RAM usage and also provides an out-of-order buffer so the writer can stay busy.

    # If you made them the same automatically, you’d either:
    #Over-allocate RAM (when workers is high but you don’t want that many multi-GB outputs alive), or
    #Starve CPUs (when you need small inflight to fit memory, but could still run more workers if levels are small).

    #Memory math for your case (nx=1080, k=8):
    #Per-level input (nx × 13nx × f4): ~57.9 MiB
    #Per-level output ((nx·k) × (13nx·k) × f4): ~3.62 GiB
    #Total per level in SHM: ~3.68 GiB

    #So RAM ≈ inflight × 3.68 GiB (plus overhead).
    #Examples:
    #inflight=8 → ~29.4 GiB
    #inflight=32 → ~118 GiB
    #inflight=64 → ~235 GiB
    #On a 755 GiB node you can afford large inflight,
    #but you may not want to: EDT/NumPy scratch, OS cache, and other processes need headroom.




    # Parse mem string for the upsampler chunking
    s = args.mem.strip().upper()
    if s.endswith("G"):
        mem_budget = int(float(s[:-1]) * (1 << 30))
    elif s.endswith("M"):
        mem_budget = int(float(s[:-1]) * (1 << 20))
    else:
        mem_budget = int(s)

    # Quick safety: huge outputs → keep inflight moderate unless you have RAM
    out_rows = args.nx * args.k
    out_cols = args.nx * 13 * args.k
    per_level_out_gb = (out_rows * out_cols * 4) / (1024**3)
    if per_level_out_gb > 2.0 and args.inflight > 32:
        print(f"[warn] Each level output ≈ {per_level_out_gb:.2f} GiB; --inflight {args.inflight} "
              f"may stress RAM. Consider lower inflight.", file=sys.stderr)

    main(args.input, args.output,
         nx=args.nx, k_proc=args.k,
         workers=args.workers, inflight=args.inflight,
         interp_mem_budget=mem_budget,
         writer_chunk_cols=args.writer_cols)

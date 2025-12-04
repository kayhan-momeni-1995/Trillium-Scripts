#!/usr/bin/env python3
"""
Parallel, SharedMemory-based, ordered-writer driver for LLC U/V velocity interpolation.

Design pattern:

1) Parent memmaps inputs, reads each level and BE->native f32 converts into input SHM (U,V).
2) Worker attaches to input SHM, computes, writes native f32 outputs into output SHM (U,V).
3) Parent writes output SHMs to disk in strict level order using a single writer per file,
   streaming big-endian f32 in Fortran order with a configurable column chunk size.
"""

import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

import llc


# ===========================
# Interpolation utilities (bit-identical; tiled for memory/speed)
# ===========================

def upsample_periodic_staggered(a_ext, k, return_padded=False, *, mem_bytes=None):
    """
    Bilinear upsample by factor k on a 2D cell-centered grid, using a *staggered* fine grid.
    Treat a_ext (shape (n+2, m+2)) as the full coarse grid to resample.

    This version is bit-identical to the original single-call implementation but
    streams evaluation in row tiles to cap memory. It builds the flattened points
    in the exact same order as meshgrid(..., indexing="ij") + ravel().
    """
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    k = int(k)

    N_rows, N_cols = a_ext.shape
    if N_rows < 2 or N_cols < 2:
        raise ValueError("a_ext must be at least 2x2")

    # Coarse index coordinates (float64)
    r_coarse = np.arange(1, N_rows + 1, dtype=float)
    c_coarse = np.arange(1, N_cols + 1, dtype=float)

    interp = RegularGridInterpolator(
        (r_coarse, c_coarse), a_ext,
        method="linear",
        bounds_error=False,
        fill_value=None  # linear extrapolation at half-cells
    )

    # Staggered fine coordinates (float64)
    rfine = 0.5 + 1.0/(2*k) + np.arange(N_rows * k, dtype=float) / k
    cfine = 0.5 + 1.0/(2*k) + np.arange(N_cols * k, dtype=float) / k

    out = np.empty((rfine.size, cfine.size), dtype=a_ext.dtype)

    # Choose tile height based on memory budget (default ~64MB)
    if mem_bytes is None:
        mem_bytes = 64 << 20
    ncf = cfine.size
    item = np.dtype(float).itemsize  # points are float64
    # per point: two float64 values in pts + one output element (dtype of a_ext)
    per_point_bytes = 2*item + np.dtype(a_ext.dtype).itemsize
    max_points = max(1, mem_bytes // per_point_bytes)
    tile_h = max(1, int(min(rfine.size, max_points // ncf)))

    for r0 in range(0, rfine.size, tile_h):
        r1 = min(r0 + tile_h, rfine.size)
        rblk = rfine[r0:r1]
        # Build flattened points in the exact same order as meshgrid(..., indexing="ij")
        R = np.repeat(rblk, cfine.size)
        C = np.tile(cfine, rblk.size)
        pts = np.empty((R.size, 2), dtype=float)
        pts[:, 0] = R
        pts[:, 1] = C

        vals = interp(pts).reshape(rblk.size, cfine.size)
        out[r0:r1, :] = vals

    if return_padded:
        return out

    if out.shape[0] <= 2*k or out.shape[1] <= 2*k:
        raise ValueError("Grid too small to strip k layers on each side.")
    return out[k:-k, k:-k]


def fillna_with_nearest(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("Expected a 2D array")

    mask = np.isnan(a)
    if not mask.any():
        return a.copy()
    if (~mask).sum() == 0:
        return a.copy()

    nearest_idx = ndi.distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    filled = a[tuple(nearest_idx)]
    return filled


# ===========================
# Velocity interpolation (bit-identical; fewer copies)
# ===========================

def interpolate_level(U: np.ndarray, V: np.ndarray, nx: int, k: int,
                      *,
                      interp_mem_budget: int = 512 << 20):
    # LLC -> rectangular
    U = llc.llc2rect(U)
    V = llc.llc2rect(V)

    # Views (no copies)
    U12 = U[:2*nx, :3*nx]
    U3_core = U[nx:2*nx, 3*nx:4*nx]
    U45 = U[2*nx:4*nx, :3*nx]

    V12 = V[:2*nx, :3*nx]
    V3_core = V[nx:2*nx, 3*nx:4*nx]
    V45 = V[2*nx:4*nx, :3*nx]

    # --- U45 "roll" & boundary ---
    U45_boundary = U45[:, -1].copy()
    U45_shifted = np.empty_like(U45)
    U45_shifted[:, 0]  = U45[:, 0]
    U45_shifted[:, 1]  = U45[:, 0]
    U45_shifted[:, 2:] = U45[:, 1:-1]

    # --------------------------
    # Build U3 (size: (nx+4, nx+4))
    # --------------------------
    U3 = np.empty((nx+4, nx+4), dtype=U.dtype)
    U3.fill(0)

    # interior
    U3[2:-2, 2:-2] = U3_core

    # inner ring
    U3[1,      2:-2] = np.flip(V12[:nx, -1])
    U3[-2,     2:-2] = U45_boundary[:nx]
    U3[2:-2,   1   ] = U12[nx:2*nx, -1]
    U3[2:-2,  -2   ] = np.flip(-V45[nx:2*nx, -1])

    # outer ring
    U3[0,      2:-2] = np.flip(V12[:nx, -2])
    U3[-1,     2:-2] = U45[:nx, -1]
    U3[2:-2,   0   ] = U12[nx:2*nx, -2]
    U3[2:-2,  -1   ] = np.flip(-V45[nx:2*nx, -2])

    #U3[U3 == 0] = np.nan
    U3 = fillna_with_nearest(U3)

    # --------------------------
    # Build V3 (size: (nx+4, nx+4))
    # --------------------------
    V3 = np.empty((nx+4, nx+4), dtype=V.dtype)
    V3[:] = np.nan

    # interior
    V3[2:-2, 2:-2] = V3_core

    # inner ring
    V3[1,      2:-2] = np.flip(-U12[:nx, -1])
    V3[-2,     2:-2] = V45[:nx, -1]
    V3[2:-2,   1   ] = V12[nx:2*nx, -1]
    V3[2:-2,  -2   ] = np.flip(U45[nx:2*nx, -1])

    # outer ring
    V3[0,      2:-2] = np.flip(-U12[:nx, -2])
    V3[-1,     2:-2] = V45[:nx, -2]
    V3[2:-2,   0   ] = V12[nx:2*nx, -2]
    V3[2:-2,  -1   ] = np.flip(U45[nx:2*nx, -2])

    #V3[V3 == 0] = np.nan
    V3 = fillna_with_nearest(V3)

    # upsample + crop (unchanged)
    U3_upsampled = upsample_periodic_staggered(U3, k, True, mem_bytes=interp_mem_budget)
    V3_upsampled = upsample_periodic_staggered(V3, k, True, mem_bytes=interp_mem_budget)
    U3 = U3[2:-2, 2:-2]
    V3 = V3[2:-2, 2:-2]

    # --------------------------
    # Build U_new (size: (4nx+4, 3nx+4))
    # --------------------------
    U_core = np.empty((4*nx, 3*nx), dtype=U.dtype)
    U_core[:2*nx, :] = U12
    U_core[2*nx:, :] = V45

    U_right1 = np.empty(4*nx, dtype=U.dtype)
    U_right1[:nx]       = np.flip(-V3[0, :])
    U_right1[nx:2*nx]   = U3[:, 0]
    U_right1[2*nx:3*nx] = V3[-1, :]
    U_right1[3*nx:]     = np.flip(-U3[:, -1])

    U_right2 = np.empty(4*nx, dtype=U.dtype)
    U_right2[:nx]       = np.flip(-V3[1, :])
    U_right2[nx:2*nx]   = U3[:, 1]
    U_right2[2*nx:3*nx] = V3[-2, :]
    U_right2[3*nx:]     = np.flip(-U3[:, -2])

    U_rows = np.empty((4*nx, 3*nx + 4), dtype=U.dtype)
    U_rows[:, 2:2+3*nx] = U_core
    U_rows[:, 2+3*nx]   = U_right1
    U_rows[:, 3*nx+3]   = U_right2
    U_rows[:, 0]        = U_core[:, 0]
    U_rows[:, 1]        = U_core[:, 0]

    U_new = np.empty((4*nx + 4, 3*nx + 4), dtype=U.dtype)
    U_new[2:-2, :] = U_rows
    U_new[0,  :]   = U_rows[-2, :]
    U_new[1,  :]   = U_rows[-1, :]
    U_new[-2, :]   = U_rows[0,  :]
    U_new[-1, :]   = U_rows[1,  :]

    #U_new[U_new == 0] = np.nan
    U_new = fillna_with_nearest(U_new)

    # --------------------------
    # Build V_new (size: (4nx+4, 3nx+4))
    # --------------------------
    V_core = np.empty((4*nx, 3*nx), dtype=V.dtype)
    V_core[:2*nx, :] = V12
    V_core[2*nx:, :] = -U45_shifted

    V_right1 = np.empty(4*nx, dtype=V.dtype)
    V_right1[:nx]       = np.flip(U3[0, :])
    V_right1[nx:2*nx]   = V3[:, 0]
    V_right1[2*nx:]     = -U45_boundary

    V_right2 = np.empty(4*nx, dtype=V.dtype)
    V_right2[:nx]       = np.flip(U3[1, :])
    V_right2[nx:2*nx]   = V3[:, 1]
    V_right2[2*nx:3*nx] = -U3[-1, :]
    V_right2[3*nx:]     = np.flip(-V3[:, -1])

    V_rows = np.empty((4*nx, 3*nx + 4), dtype=V.dtype)
    V_rows[:, 2:2+3*nx] = V_core
    V_rows[:, 2+3*nx]   = V_right1
    V_rows[:, 3*nx+3]   = V_right2
    V_rows[:, 0]        = V_core[:, 0]
    V_rows[:, 1]        = V_core[:, 0]

    V_new = np.empty((4*nx + 4, 3*nx + 4), dtype=V.dtype)
    V_new[2:-2, :] = V_rows
    V_new[0,  :]   = V_rows[-2, :]
    V_new[1,  :]   = V_rows[-1, :]
    V_new[-2, :]   = V_rows[0,  :]
    V_new[-1, :]   = V_rows[1,  :]

    #V_new[V_new == 0] = np.nan
    V_new = fillna_with_nearest(V_new)

    # upsample + crop (unchanged)
    U_new_upsampled = upsample_periodic_staggered(U_new, k, True, mem_bytes=interp_mem_budget)
    V_new_upsampled = upsample_periodic_staggered(V_new, k, True, mem_bytes=interp_mem_budget)
    U_new_upsampled = U_new_upsampled[2*k:-2*k, 2*k:-2*k]
    V_new_upsampled = V_new_upsampled[2*k:-2*k, 2*k:-2*k]

    # --------------------------
    # Final assembly (avoid np.roll and extra temporaries)
    # --------------------------
    V1245_upsampled = np.empty_like(V_new_upsampled)
    V1245_upsampled[:2*k*nx, :] = V_new_upsampled[:2*k*nx, :]
    V1245_upsampled[2*k*nx:, :] = U_new_upsampled[2*k*nx:, :]

    low = V_new_upsampled[2*k*nx:4*k*nx, :]
    U45_upsampled = np.empty_like(low)
    U45_upsampled[:, :-1]    = low[:, 1:]
    U45_upsampled[:nx*k, -1] = -U3_upsampled[-1, 2*k:-2*k]
    U45_upsampled[nx*k:, -1] = -np.flip(V3_upsampled[2*k:-2*k, -1])

    U1245_upsampled = np.empty_like(U_new_upsampled)
    U1245_upsampled[:2*k*nx, :] = U_new_upsampled[:2*k*nx, :]
    U1245_upsampled[2*k*nx:, :] = -U45_upsampled

    right_pad_U = np.concatenate(
        [np.zeros((nx*k, nx*k), dtype=U.dtype),
         U3_upsampled[2*k:-2*k, 2*k:-2*k],
         np.zeros((2*nx*k, nx*k), dtype=U.dtype)],
        axis=0
    )
    final_U = np.concatenate([U1245_upsampled, right_pad_U], axis=1)

    right_pad_V = np.concatenate(
        [np.zeros((nx*k, nx*k), dtype=V.dtype),
         V3_upsampled[2*k:-2*k, 2*k:-2*k],
        np.zeros((2*nx*k, nx*k), dtype=V.dtype)],
        axis=0
    )
    final_V = np.concatenate([V1245_upsampled, right_pad_V], axis=1)

    # Fill NaNs
    #final_U[final_U == 0] = np.nan
    final_U = fillna_with_nearest(final_U)

    #final_V[final_V == 0] = np.nan
    final_V = fillna_with_nearest(final_V)
                         
    final_U = llc.rect2llc(final_U)
    final_V = llc.rect2llc(final_V)

    return final_U, final_V


# ===========================
# I/O helpers (column-chunk Fortran-order writer from SHM)
# ===========================

def _write_shm_to_file(fout, shm_name: str, shape: Tuple[int, int], *, col_chunk: int = 2048) -> None:
    """
    Stream an array stored in SharedMemory (native float32) to fout
    as big-endian float32 in **Fortran order**, column-chunked.

    - fout: an open file object (binary) whose offset is already positioned for sequential write.
    - shm_name: name of the SharedMemory block containing the 2D float32 array.
    - shape: (nrows, ncols)
    - col_chunk: number of columns per chunk (defaults to 2048). If <=0 or >ncols, uses ncols.
    """
    shm = SharedMemory(name=shm_name)
    try:
        nrows, ncols = shape
        arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        # normalize chunk size
        if col_chunk <= 0 or col_chunk > ncols:
            col_chunk = ncols

        c0 = 0
        while c0 < ncols:
            c1 = min(c0 + col_chunk, ncols)
            blk = arr[:, c0:c1]                       # native f32
            blk_be = blk.astype('>f4', copy=False)    # big-endian f32 view/copy
            # Write this contiguous column block in Fortran order
            fout.write(blk_be.tobytes(order='F'))
            c0 = c1
    finally:
        shm.close()  # parent owns unlink elsewhere


# ===========================
# Worker: attach SHMs, compute, write outputs into SHMs
# ===========================

def _worker_process_level(
    idx: int,
    shm_u_in_name: str, in_shape: Tuple[int, int],
    shm_v_in_name: str,               # same in_shape
    shm_u_out_name: str, out_shape: Tuple[int, int],
    shm_v_out_name: str,              # same out_shape
    nx: int, k_proc: int,
    interp_mem_budget: int
) -> int:
    """
    Worker attaches to input SHMs (U,V), runs interpolate_level, and stores
    outputs (U,V) into output SHMs. Returns the level index.
    """
    shm_u_in = SharedMemory(name=shm_u_in_name)
    shm_v_in = SharedMemory(name=shm_v_in_name)
    shm_u_out = SharedMemory(name=shm_u_out_name)
    shm_v_out = SharedMemory(name=shm_v_out_name)

    try:
        in_rows, in_cols = in_shape
        out_rows, out_cols = out_shape

        U_in = np.ndarray((in_rows, in_cols), dtype=np.float32, buffer=shm_u_in.buf)
        V_in = np.ndarray((in_rows, in_cols), dtype=np.float32, buffer=shm_v_in.buf)

        # Run exact interpolation kernel (bit-identical math path)
        U_out_np, V_out_np = interpolate_level(U_in, V_in, nx, k_proc,
                                               interp_mem_budget=interp_mem_budget)

        if U_out_np.shape != (out_rows, out_cols) or V_out_np.shape != (out_rows, out_cols):
            raise RuntimeError(
                f"Unexpected output shape at level {idx}: "
                f"U{U_out_np.shape} V{V_out_np.shape} vs expected {(out_rows, out_cols)}"
            )

        # Store into output SHMs as float32 (serial writer casts to f32; match that)
        U_out = np.ndarray((out_rows, out_cols), dtype=np.float32, buffer=shm_u_out.buf)
        V_out = np.ndarray((out_rows, out_cols), dtype=np.float32, buffer=shm_v_out.buf)
        np.copyto(U_out, U_out_np.astype(np.float32, copy=False), casting="unsafe")
        np.copyto(V_out, V_out_np.astype(np.float32, copy=False), casting="unsafe")

        return idx
    finally:
        # Workers only close; parent unlinks
        shm_u_in.close()
        shm_v_in.close()
        shm_u_out.close()
        shm_v_out.close()


# ===========================
# Driver (parallel, SHM-based, ordered writer)
# ===========================

def main(u_src_path: str, v_src_path: str,
         u_dst_path: str, v_dst_path: str,
         nx: int, k_proc: int,
         *,
         workers: int,
         inflight: int,
         interp_mem_budget: int = 512 << 20,
         writer_chunk_cols: int = 2048) -> None:
    """
    Parallel, SHM-based, ordered writer — mirrors the user's 1-in/1-out driver,
    but handles U and V in lockstep and writes two outputs sequentially.

    On-disk format for inputs: big-endian float32, levels are (nx, nx*13) Fortran layout.
    Outputs are written as big-endian float32, levels (nx*k, nx*13*k) Fortran layout.
    """

    dtype_be = np.dtype(">f4")

    # Sizes
    in_rows, in_cols = nx, nx * 13
    in_bytes = in_rows * in_cols * 4

    out_rows = nx * k_proc
    out_cols = nx * 13 * k_proc
    out_bytes = out_rows * out_cols * 4

    # Validate input sizes and level counts
    fsize_u = os.path.getsize(u_src_path)
    fsize_v = os.path.getsize(v_src_path)
    if fsize_u % in_bytes != 0:
        raise ValueError(f"U file size {fsize_u} is not a multiple of level size {in_bytes}")
    if fsize_v % in_bytes != 0:
        raise ValueError(f"V file size {fsize_v} is not a multiple of level size {in_bytes}")

    n_levels_u = fsize_u // in_bytes
    n_levels_v = fsize_v // in_bytes
    if n_levels_u == 0 or n_levels_v == 0:
        raise ValueError("No full levels found in one or both input files")
    if n_levels_u != n_levels_v:
        raise ValueError(f"Level count mismatch: U={n_levels_u}, V={n_levels_v}")
    n_levels = n_levels_u

    workers = max(1, workers)
    inflight = max(1, inflight)

    print(f"[info] levels={n_levels}, nx={nx}, k={k_proc}, workers={workers}, inflight={inflight}")
    print(f"[info] per-level sizes: input={in_bytes/1e6:.1f} MB x2, output={out_bytes/1e6:.1f} MB x2")
    print(f"[info] writer_chunk_cols={writer_chunk_cols}")
    sys.stdout.flush()

    # Single input memmaps (parent)
    mm_u = np.memmap(u_src_path, dtype='u1', mode="r")
    mm_v = np.memmap(v_src_path, dtype='u1', mode="r")

    # Open writers (parent, single writer per file; strict sequential order)
    with open(u_dst_path, "wb", buffering=io.DEFAULT_BUFFER_SIZE) as fu, \
         open(v_dst_path, "wb", buffering=io.DEFAULT_BUFFER_SIZE) as fv, \
         ProcessPoolExecutor(max_workers=workers) as pool:

        pending = {}   # future -> (idx, shm names tuple)
        ready   = {}   # idx -> (shm names tuple)
        next_to_write = 0
        submitted = 0

        def submit_level(idx: int):
            nonlocal submitted
            off = idx * in_bytes

            # Vectorized BE f4 views (Fortran layout on disk)
            lvl_be_u = np.ndarray(shape=(in_rows, in_cols),
                                  dtype=">f4",
                                  buffer=mm_u,
                                  offset=off,
                                  strides=(4, 4 * in_rows))
            lvl_be_v = np.ndarray(shape=(in_rows, in_cols),
                                  dtype=">f4",
                                  buffer=mm_v,
                                  offset=off,
                                  strides=(4, 4 * in_rows))

            # Parent creates input SHMs (native f32) and copies the levels
            shm_u_in = SharedMemory(create=True, size=in_bytes)
            shm_v_in = SharedMemory(create=True, size=in_bytes)

            U_in = np.ndarray((in_rows, in_cols), dtype=np.float32, buffer=shm_u_in.buf)
            V_in = np.ndarray((in_rows, in_cols), dtype=np.float32, buffer=shm_v_in.buf)
            # endian + dtype convert in one pass
            np.copyto(U_in, lvl_be_u, casting="unsafe")
            np.copyto(V_in, lvl_be_v, casting="unsafe")

            # Parent creates output SHMs (native f32) for worker to fill
            shm_u_out = SharedMemory(create=True, size=out_bytes)
            shm_v_out = SharedMemory(create=True, size=out_bytes)

            # Dispatch worker (attaches to SHMs by name; no unlink in worker)
            fut = pool.submit(
                _worker_process_level,
                idx,
                shm_u_in.name, (in_rows, in_cols),
                shm_v_in.name,
                shm_u_out.name, (out_rows, out_cols),
                shm_v_out.name,
                nx, k_proc,
                interp_mem_budget
            )
            pending[fut] = (idx, (shm_u_in.name, shm_v_in.name, shm_u_out.name, shm_v_out.name))
            submitted += 1

        # Prime initial inflight
        while submitted < min(inflight, n_levels):
            submit_level(submitted)

        written = 0
        while pending:
            # Wait for any completion
            for fut in as_completed(list(pending.keys())):
                idx, shm_names = pending.pop(fut)
                idx_done = fut.result()  # propagate worker exceptions
                assert idx_done == idx
                ready[idx] = shm_names

                # Maintain inflight: submit next if any left
                if submitted < n_levels:
                    submit_level(submitted)

                # Write in strict order as far as possible
                while next_to_write in ready:
                    shm_u_in_name, shm_v_in_name, shm_u_out_name, shm_v_out_name = ready.pop(next_to_write)

                    # Stream U then V for this level (separate files, ordered)
                    _write_shm_to_file(fu, shm_u_out_name, (out_rows, out_cols), col_chunk=writer_chunk_cols)
                    _write_shm_to_file(fv, shm_v_out_name, (out_rows, out_cols), col_chunk=writer_chunk_cols)

                    # Cleanup SHMs (parent created them → parent unlinks)
                    # Output SHMs
                    try:
                        shm = SharedMemory(name=shm_u_out_name); shm.close(); shm.unlink()
                    except FileNotFoundError:
                        pass
                    try:
                        shm = SharedMemory(name=shm_v_out_name); shm.close(); shm.unlink()
                    except FileNotFoundError:
                        pass
                    # Input SHMs
                    try:
                        shm = SharedMemory(name=shm_u_in_name); shm.close(); shm.unlink()
                    except FileNotFoundError:
                        pass
                    try:
                        shm = SharedMemory(name=shm_v_in_name); shm.close(); shm.unlink()
                    except FileNotFoundError:
                        pass

                    written += 1
                    next_to_write += 1
                    print(f"Level {written}/{n_levels} written.")
                    sys.stdout.flush()
                break  # re-enter as_completed with updated 'pending'

    # Close input memmaps
    del mm_u
    del mm_v

    print(f"Finished. Wrote {n_levels} levels to {u_dst_path} and {v_dst_path}")


# ===========================
# CLI
# ===========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Parallel per-level interpolation for LLC velocity (U,V) binaries using SharedMemory; "
                    "bit-identical to the serial path; strict ordered writer."
    )
    ap.add_argument("--nx", type=int, required=True, help="Tile size (e.g., 270 or 1080)")
    ap.add_argument("--k", type=int, default=1, help="Upsampling factor passed to processing")

    ap.add_argument("--input-u", required=True, help="Path to U input .data (big-endian real*4)")
    ap.add_argument("--input-v", required=True, help="Path to V input .data (big-endian real*4)")
    ap.add_argument("--output-u", required=True, help="Path to U output .data (will be overwritten)")
    ap.add_argument("--output-v", required=True, help="Path to V output .data (will be overwritten)")

    # Match the user's SHM-driver style: interp_mem_budget is a function arg (no CLI parsing here).
    ap.add_argument("--workers", type=int, default=max(os.cpu_count(), 173),
                    help="Worker processes")
    ap.add_argument("--inflight", type=int, default=max(1, min(os.cpu_count() or 1, 16)),
                    help="Max in-flight levels (controls total SHM RAM)")
    ap.add_argument("--writer-chunk-cols", type=int, default=98304,
                    help="Writer column chunk size for BE f4 streaming; "
                         "0 or >ncols means one-shot")

    args = ap.parse_args()

    # Optional: allow an env var to override memory per interpolator without changing CLI.
    # If not set, stays at default 512 MiB.
    mem_env = os.environ.get("INTERP_MEM_BUDGET")
    if mem_env is not None:
        try:
            # Accept plain integers (bytes), or with M/G suffix (e.g., "256M", "2G")
            s = mem_env.strip().upper()
            if s.endswith("G"):
                interp_mem_budget = int(float(s[:-1]) * (1 << 30))
            elif s.endswith("M"):
                interp_mem_budget = int(float(s[:-1]) * (1 << 20))
            else:
                interp_mem_budget = int(s)
        except Exception:
            print(f"[warn] Failed to parse INTERP_MEM_BUDGET={mem_env!r}; using default 512 MiB", file=sys.stderr)
            interp_mem_budget = 512 << 20
    else:
        interp_mem_budget = 512 << 20

    main(args.input_u, args.input_v,
         args.output_u, args.output_v,
         nx=args.nx, k_proc=args.k,
         workers=args.workers,
         inflight=args.inflight,
         interp_mem_budget=interp_mem_budget,
         writer_chunk_cols=args.writer_chunk_cols)

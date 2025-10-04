import numpy as np

def llc2rect(LLC: np.ndarray) -> np.ndarray:
    """
    Convert an LLC mosaic (NX x 13*NX) to a 4NX x 4NX rectangular layout,
    matching MATLAB llc2rect.m (float64 output). Optimized to avoid large
    temporaries and order-dependent reshapes.
    """
    if LLC.ndim != 2:
        raise ValueError("LLC must be a 2D array")
    NX, NY = LLC.shape
    if NY != 13 * NX:
        raise ValueError(f"Expected LLC.shape[1] == 13*NX; got NX={NX}, NY={NY}")

    # Preallocate; we'll explicitly zero only the rightmost NX columns (then overwrite face 7).
    RECT = np.empty((NX * 4, NX * 4), dtype=np.float64)
    RECT[:, 3*NX:4*NX] = 0.0  # bands 1,3,4 remain zero here; band 2 gets face 7 below.

    # Top band: faces 1–3 (straight copy)
    RECT[0:NX, 0:3*NX] = LLC[:, 0:3*NX]

    # Second band: faces 4–6 (left) and 7 (rightmost)
    RECT[NX:2*NX, 0:3*NX] = LLC[:, 3*NX:6*NX]
    RECT[NX:2*NX, 3*NX:4*NX] = LLC[:, 6*NX:7*NX]

    # Helper to place bands 8–10 and 11–13 using a closed-form mapping
    def _place_band(band_row_start: int, src_col_start: int) -> None:
        # blk: (NX, 3*NX)
        blk = LLC[:, src_col_start:src_col_start + 3*NX]
        blk_rev_rows = blk[::-1, :]  # view; rows reversed

        dest = RECT[band_row_start:band_row_start + NX, 0:3*NX]  # (NX, 3*NX)
        # We fill in three N-wide column groups (q=0,1,2)
        idxN = np.arange(NX, dtype=np.intp)
        for q in range(3):
            # For each output row i, take column (3*i + 2 - q) of blk_rev_rows and write it across N columns.
            cols = (3 * idxN + (2 - q))  # shape (NX,)
            M = blk_rev_rows[:, cols]     # (NX, NX) – gather all at once
            dest[:, q*NX:(q+1)*NX] = M.T  # (NX, NX)

    # Third band: faces 8–10
    _place_band(2*NX, 7*NX)
    # Fourth band: faces 11–13
    _place_band(3*NX, 10*NX)

    return RECT


def llc2rect_nd(LLC: np.ndarray) -> np.ndarray:
    """
    Vectorized LLC->rect transform.
    - If LLC has shape (NX, 13*NX), returns (4*NX, 4*NX).
    - If LLC has shape (NX, 13*NX, K), returns (4*NX, 4*NX, K).
    Matches MATLAB llc2rect.m layout; output is float64.
    Optimized to avoid order-dependent copies.
    """
    if LLC.ndim not in (2, 3):
        raise ValueError("LLC must be a 2D or 3D array")

    orig_2d = (LLC.ndim == 2)
    if orig_2d:
        LLC = LLC[..., None]  # (NX, 13*NX, 1)

    NX, NY, K = LLC.shape
    if NY != 13 * NX:
        raise ValueError(f"Expected LLC.shape[1] == 13*NX; got NX={NX}, NY={NY}")

    RECT = np.empty((4*NX, 4*NX, K), dtype=np.float64)
    RECT[:, 3*NX:4*NX, :] = 0.0  # zero the rightmost NX columns for all bands/K

    # Top band: faces 1–3
    RECT[0:NX, 0:3*NX, :] = LLC[:, 0:3*NX, :]

    # Second band: faces 4–6 and 7
    RECT[NX:2*NX, 0:3*NX, :] = LLC[:, 3*NX:6*NX, :]
    RECT[NX:2*NX, 3*NX:4*NX, :] = LLC[:, 6*NX:7*NX, :]

    def _place_band_nd(band_row_start: int, src_col_start: int) -> None:
        blk = LLC[:, src_col_start:src_col_start + 3*NX, :]      # (NX, 3N, K)
        blk_rev_rows = blk[::-1, :, :]                           # (NX, 3N, K)
        dest = RECT[band_row_start:band_row_start + NX, 0:3*NX, :]  # (NX, 3N, K)
        idxN = np.arange(NX, dtype=np.intp)
        for q in range(3):
            cols = (3 * idxN + (2 - q))                          # (NX,)
            # take along axis=1 → (NX, NX, K), then transpose rows/cols
            M = np.take(blk_rev_rows, cols, axis=1)              # (NX, NX, K)
            dest[:, q*NX:(q+1)*NX, :] = np.transpose(M, (1, 0, 2))

    _place_band_nd(2*NX, 7*NX)    # faces 8–10
    _place_band_nd(3*NX, 10*NX)   # faces 11–13

    return RECT[..., 0] if orig_2d else RECT

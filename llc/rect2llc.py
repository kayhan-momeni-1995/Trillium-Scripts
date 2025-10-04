import numpy as np

def rect2llc(RECT: np.ndarray) -> np.ndarray:
    """
    Convert a 4NX x 4NX rectangular mosaic back to an LLC field (NX x 13*NX),
    matching MATLAB rect2llc.m exactly (float64 output). Optimized (no
    reshape(order='F') / copies; uses closed-form reindex).
    """
    if RECT.ndim != 2:
        raise ValueError("RECT must be a 2D array")

    nx, ny = RECT.shape
    if nx % 4 != 0:
        raise ValueError(f"RECT.shape[0] must be divisible by 4; got {nx}")
    NX = nx // 4
    if ny != 4 * NX:
        raise ValueError(f"Expected RECT.shape[1] == {4*NX}; got {ny}")

    LLC = np.empty((NX, NX * 13), dtype=np.float64)

    # Faces 1–7 (straight copies)
    LLC[:, 0:3*NX]    = RECT[0:NX,    0:3*NX]
    LLC[:, 3*NX:6*NX] = RECT[NX:2*NX, 0:3*NX]
    LLC[:, 6*NX:7*NX] = RECT[NX:2*NX, 3*NX:4*NX]

    # Helper to invert the band mapping for faces 8–10 and 11–13
    def _pull_band(band_row_start: int, dst_col_start: int) -> None:
        band = RECT[band_row_start:band_row_start + NX, 0:3*NX]  # (NX, 3N)
        blk = np.empty((NX, 3*NX), dtype=np.float64)             # (NX, 3N)
        idxN = np.arange(NX, dtype=np.intp)
        for t in range(3):  # t = v % 3
            start = (2 - t) * NX
            sub = band[:, start:start + NX]          # (N, N)
            cols = (3 * idxN + t)                    # (N,)
            # Each column v=3*i+t gets sub[i, ::-1] along rows
            blk[:, cols] = sub[:, ::-1].T
        LLC[:, dst_col_start:dst_col_start + 3*NX] = blk

    _pull_band(2*NX, 7*NX)    # faces 8–10
    _pull_band(3*NX, 10*NX)   # faces 11–13

    return LLC
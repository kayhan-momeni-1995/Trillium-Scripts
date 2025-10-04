import os
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .read_llc_fkij import read_llc_fkij

def quikread(
    fnam: str,
    nx: int = 270,
    kx: Union[int, Sequence[int]] = 1,
    prec: str = "real*4",
    gdir: Optional[str] = None,
    minlat: Optional[float] = None,
    maxlat: Optional[float] = None,
    minlon: Optional[float] = None,
    maxlon: Optional[float] = None,
) -> Tuple[Union[np.ndarray, Dict[int, np.ndarray]], List[int], Union[List[int], Dict[int, List[int]]], Union[List[int], Dict[int, List[int]]]]:
    """
    Optimized, bit-identical quikread_llc.m.
    Full-field path uses memmap + Fortran-stride view per k-level.
    Regional path uses read_llc_fkij (bit-identical).
    """
    def _to_list_strict(v) -> List[int]:
        if v is None: return []
        if np.isscalar(v): return [int(v)]
        if isinstance(v, (list, tuple, range)): return [int(x) for x in v]
        if isinstance(v, np.ndarray):
            return [int(x) for x in v.ravel()]
        return [int(v)]
    kx_list: List[int] = _to_list_strict(kx)

    # precision mapping
    prec_lut = {
        "integer*1": "int8", "integer*2": "int16", "integer*4": "int32", "integer*8": "int64",
        "real*4": "single", "float32": "single", "real*8": "double", "float64": "double",
        "int8": "int8", "int16": "int16", "uint16": "uint16",
        "int32": "int32", "uint32": "uint32", "single": "single",
        "int64": "int64", "uint64": "uint64", "double": "double",
    }
    prec_norm = prec_lut.get(prec.lower())
    if prec_norm is None: raise ValueError(f"Unsupported precision: {prec}")

    base_to_np = {
        "int8": ("i1", 1),
        "int16": ("i2", 2), "uint16": ("u2", 2),
        "int32": ("i4", 4), "uint32": ("u4", 4), "single": ("f4", 4),
        "int64": ("i8", 8), "uint64": ("u8", 8), "double": ("f8", 8),
    }
    code, preclength = base_to_np[prec_norm]
    dtype_be = np.dtype(">" + code)
    dtype_native = np.dtype(code).newbyteorder("=")

    # ---------- FULL FIELD (like nargin<6) ----------
    if minlat is None:
        max_k = max(kx_list) if kx_list else 1
        need_bytes = max_k * nx * nx * 13 * preclength
        have_bytes = os.path.getsize(fnam)
        if need_bytes > have_bytes:
            raise EOFError("past end of file")

        fld = np.zeros((nx, nx * 13, len(kx_list)), dtype=dtype_native)
        mm = np.memmap(fnam, dtype=dtype_be, mode="r")

        for kk, klev in enumerate(kx_list):
            lvl_off = (klev - 1) * nx * nx * 13
            lvl = np.ndarray(
                shape=(nx, nx * 13),
                dtype=dtype_be,
                buffer=mm,
                offset=lvl_off * preclength,
                strides=(preclength, preclength * nx),
            )
            fld[:, :, kk] = lvl.astype(dtype_native, copy=False)

        del mm
        return fld, [], [], []

    # ---------- REGION READ (like nargin>=6) ----------
    if gdir is None: gdir = ""
    if maxlon is None: maxlon = 180.0
    if minlon is None: minlon = -180.0
    if maxlat is None: maxlat = 90.0
    if minlat is None: minlat = -90.0

    if minlon < -180:
        raise ValueError("minlon<-180 not yet implemented: please email menemenlis@jpl.nasa.gov")
    if maxlon > 360:
        raise ValueError("maxlon>360 not yet implemented: please email menemenlis@jpl.nasa.gov")
    if minlat >= maxlat:
        raise ValueError("maxlat must be greater than minlat")
    if minlon >= maxlon:
        raise ValueError("maxlon must be greater than minlon")

    yc_path = os.path.join(gdir, "YC.data")
    xc_path = os.path.join(gdir, "XC.data")

    fld_multi: Dict[int, np.ndarray] = {}
    ix_multi: Dict[int, List[int]] = {}
    jx_multi: Dict[int, List[int]] = {}
    fc: List[int] = []

    for f in (1, 2, 3, 4, 5):
        yc = read_llc_fkij(yc_path, nx, f)[:, :, 0]
        xc = read_llc_fkij(xc_path, nx, f)[:, :, 0]
        if maxlon > 180:
            xc = xc.copy()
            xc[xc < 0] += 360.0

        mask = (yc >= minlat) & (yc <= maxlat) & (xc >= minlon) & (xc <= maxlon)
        ii, jj = np.where(mask)
        if ii.size == 0:
            continue

        fc.append(f)
        imin, imax = int(ii.min()) + 1, int(ii.max()) + 1  # 1-based
        jmin, jmax = int(jj.min()) + 1, int(jj.max()) + 1
        ix_face = list(range(imin, imax + 1))
        jx_face = list(range(jmin, jmax + 1))
        ix_multi[f] = ix_face
        jx_multi[f] = jx_face

        face_arr = np.zeros((len(ix_face), len(jx_face), len(kx_list)), dtype=dtype_native)
        for kk, klev in enumerate(kx_list):
            tmp = read_llc_fkij(fnam, nx, f, klev, ix_face, jx_face, prec_norm)
            face_arr[:, :, kk] = tmp[:, :, 0] if (tmp.ndim == 3 and tmp.shape[2] == 1) else tmp
        fld_multi[f] = face_arr

    if len(fc) == 1:
        f = fc[0]
        return fld_multi[f], fc, ix_multi[f], jx_multi[f]
    else:
        return fld_multi, fc, ix_multi, jx_multi



def minmax_levels(fnam, nx, levels, prec="real*4", use_nan=True):
    # Precision → numpy dtype (big-endian)
    prec_lut = {
        "integer*1": "i1", "integer*2": "i2", "integer*4": "i4", "integer*8": "i8",
        "real*4": "f4", "float32": "f4", "real*8": "f8", "float64": "f8",
        "int8": "i1", "int16": "i2", "uint16": "u2",
        "int32": "i4", "uint32": "u4", "single": "f4",
        "int64": "i8", "uint64": "u8", "double": "f8",
    }
    base = prec_lut[prec.lower()]
    dtype_be = np.dtype(">" + base)  # file is big-endian

    # Elements per vertical level (faces 1..13 laid out back-to-back)
    n_per_level = nx * nx * 13

    # One memmap for the whole file; slicing is O(1)
    mm = np.memmap(fnam, dtype=dtype_be, mode="r")

    # Choose reducers
    vmin = np.nanmin if use_nan else np.min
    vmax = np.nanmax if use_nan else np.max

    try:
        for k in levels:
            start = (k - 1) * n_per_level
            stop  = start + n_per_level
            lvl_1d = mm[start:stop]            # 1-D view, no copy
            # Reductions work fine on non-native byteorder dtypes
            umin = vmin(lvl_1d).item()
            umax = vmax(lvl_1d).item()
            yield k, umin, umax
    finally:
        # Ensure the mapping is closed promptly on Windows
        del mm


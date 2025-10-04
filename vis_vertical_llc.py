#!/usr/bin/env python3
# llc_section_visualizer.py — advanced vertical-section viewer (box-zoom, hover, top strip)
# -----------------------------------------------------------------------------
# Same CLI & behavior as your original (sea-ice / ice-shelf / bathy supported).
# Internal change: low-RAM, level-by-level streaming + cluster-friendly fixed-Y I/O.
#
# Optional tuning via env vars (no CLI changes):
#   LLC_THREADS   : threads for fixed-Y bottom bands (default: min(8, max(1, cpu//2)))
#   DOCK_PAD_PX   : bottom padding px for window fit (default 100)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RectangleSelector

import llc

# -----------------------------
# Matplotlib layout defaults (like your sample)
# -----------------------------
mpl.rcParams.update({
    "figure.subplot.left":   0.0,
    "figure.subplot.bottom": 0.0,
    "figure.subplot.top":    1.0,
    "figure.subplot.right":  1.0,
    "figure.subplot.wspace": 0.0,
    "figure.subplot.hspace": 0.0,
})

# -----------------------------
# Work-area sizing (avoid macOS Dock)
# -----------------------------
def _query_x11_workarea():
    """Return (x,y,w,h) EWMH work area if available (often not on XQuartz)."""
    try:
        from Xlib import display, X  # type: ignore
        d = display.Display()
        root = d.screen().root
        NET_WORKAREA = d.intern_atom('_NET_WORKAREA')
        prop = root.get_full_property(NET_WORKAREA, X.AnyPropertyType)
        if prop and len(prop.value) >= 4:
            x, y, w, h = prop.value[:4]
            return int(x), int(y), int(w), int(h)
    except Exception:
        pass
    return None


def _refresh_after_resize(fig):
    """Nudge so the canvas actually fills the resized window immediately."""
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        try:
            t = fig.canvas.new_timer(interval=160)
            try:
                t.single_shot = True
            except Exception:
                pass
            t.add_callback(lambda: (fig.canvas.draw_idle(), fig.canvas.flush_events()))
            t.start()
        except Exception:
            pass
        try:
            plt.pause(0.05)
        except Exception:
            pass
    except Exception:
        pass


def _fit_to_workarea(fig, dock_pad_px=None):
    """Resize window to the screen's usable area (not true fullscreen)."""
    mgr = fig.canvas.manager
    backend = mpl.get_backend().lower()

    pad = dock_pad_px
    if pad is None:
        try:
            pad = int(os.environ.get("DOCK_PAD_PX", "100"))
        except Exception:
            pad = 100

    try:
        if 'qt' in backend:
            win = mgr.window
            try:
                geo = win.screen().availableGeometry()
                x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
            except Exception:
                x, y, w, h = 0, 0, 1600, 900
            try: win.showMaximized()
            except Exception: pass
            try: win.setGeometry(x, y, w, h)
            except Exception: pass
            try: mgr.resize(w, h)
            except Exception: pass
            dpi = fig.get_dpi()
            fig.set_size_inches(w / dpi, h / dpi, forward=True)
            try:
                from PyQt5 import QtWidgets  # type: ignore
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
            _refresh_after_resize(fig); return

        if 'tkagg' in backend:
            root = mgr.window
            wa = _query_x11_workarea()
            if wa is not None:
                x, y, w, h = wa
            else:
                try:
                    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                except Exception:
                    sw, sh = 1600, 900
                w, h = sw, max(200, sh - pad)
            try:
                root.state('zoomed')
            except Exception:
                try: root.attributes('-zoomed', True)
                except Exception: pass
            try: root.geometry(f"{w}x{h}+0+0")
            except Exception: pass
            try: root.update_idletasks()
            except Exception: pass
            try:
                sw = root.winfo_width()
                sh = root.winfo_height()
                dpi = fig.get_dpi()
                fig.set_size_inches(max(1, sw) / dpi, max(1, sh) / dpi, forward=True)
                try: mgr.resize(sw, sh)
                except Exception: pass
            except Exception:
                pass
            _refresh_after_resize(fig); return

        if 'wx' in backend:
            try:
                import wx  # type: ignore
                frame = mgr.frame
                w, h = wx.GetDisplaySize()
                h = max(200, h - pad)
                frame.SetPosition((0, 0))
                frame.SetSize((w, h))
                try: frame.Layout()
                except Exception: pass
                try: mgr.resize(w, h)
                except Exception: pass
                dpi = fig.get_dpi()
                fig.set_size_inches(w / dpi, h / dpi, forward=True)
                _refresh_after_resize(fig); return
            except Exception:
                pass

        if 'macosx' in backend:
            try:
                mgr.full_screen_toggle(); mgr.full_screen_toggle()
            except Exception:
                pass
            try:
                dpi = fig.get_dpi()
                w, h = 1680, 950
                fig.set_size_inches(w / dpi, h / dpi, forward=True)
            except Exception:
                pass
            _refresh_after_resize(fig); return

        if 'gtk' in backend:
            try: mgr.window.maximize()
            except Exception: pass
            _refresh_after_resize(fig); return

        # Last resort: DPI-based sizing
        sw, sh = 1920, 1080
        try:
            sw = mgr.window.winfo_screenwidth()
            sh = mgr.window.winfo_screenheight()
        except Exception:
            pass
        dpi = fig.get_dpi()
        fig.set_size_inches(sw / dpi, (sh - pad) / dpi, forward=True)
        try: mgr.resize(sw, sh)
        except Exception: pass
        _refresh_after_resize(fig)
    except Exception:
        pass


def _ensure_fit_sequence(fig, dock_pad_px):
    """Repeatedly try to fit after the window is realized; helps stubborn WMs/backends."""
    try:
        _fit_to_workarea(fig, dock_pad_px=dock_pad_px)
        delays = (30, 120, 400, 900)
        for ms in delays:
            try:
                t = fig.canvas.new_timer(interval=int(ms))
                try:
                    t.single_shot = True
                except Exception:
                    pass
                t.add_callback(lambda: _fit_to_workarea(fig, dock_pad_px=dock_pad_px))
                t.start()
            except Exception:
                pass
        fired = {"d": False}
        def _on_first_draw(evt):
            if evt.canvas is fig.canvas and not fired["d"]:
                fired["d"] = True
                _fit_to_workarea(fig, dock_pad_px=dock_pad_px)
        fig.canvas.mpl_connect('draw_event', _on_first_draw)
    except Exception:
        pass


# -----------------------------
# Top strip: title (left) + longer/thinner colorbar (right)
# -----------------------------
def apply_top_strip(ax, mappable, title_text: str | None,
                    *, strip_h=0.06, pad=0.01,
                    cbar_rel_w=0.46,          # wider (longer)
                    cbar_height_frac=0.33,    # thinner
                    decimals=2, bg_alpha=0.98):
    """Create a top strip outside the data canvas with a left title and a compact horizontal colorbar."""
    fig = ax.figure
    pos = ax.get_position()

    new_h = pos.height - (strip_h + pad)
    if new_h <= 0: new_h = max(0.5 * pos.height, 0.1)
    ax.set_position([pos.x0, pos.y0, pos.width, new_h])

    y = pos.y0 + new_h + pad
    gap = pad
    title_w = pos.width * (1.0 - cbar_rel_w) - gap
    cbar_w  = pos.width * cbar_rel_w

    title_ax = fig.add_axes([pos.x0, y, max(0.0, title_w), strip_h], zorder=10)
    title_ax.set_facecolor("white"); title_ax.patch.set_alpha(float(bg_alpha)); title_ax.axis('off')

    cbar_bg_x0 = pos.x0 + max(0.0, title_w) + gap
    cbar_bg_ax = fig.add_axes([cbar_bg_x0, y, cbar_w, strip_h], zorder=9)
    cbar_bg_ax.set_facecolor("white"); cbar_bg_ax.patch.set_alpha(float(bg_alpha)); cbar_bg_ax.axis('off')

    cbar_h = max(0.05, strip_h * float(cbar_height_frac))
    cbar_y = y + (strip_h - cbar_h) / 2.0
    cax = fig.add_axes([cbar_bg_x0, cbar_y, cbar_w, cbar_h], zorder=11); cax.set_facecolor("none")

    fig.canvas.draw()
    dpi = fig.get_dpi()
    fig_h_px = fig.get_size_inches()[1] * dpi
    strip_h_px = strip_h * fig_h_px
    title_px = np.clip(strip_h_px * 0.58, 12, 64)
    title_pt = float(title_px * 72.0 / dpi)
    if title_text:
        title_ax.text(0.02, 0.5, str(title_text), ha='left', va='center', color='black', fontsize=title_pt)

    cb = plt.colorbar(mappable, cax=cax, orientation='horizontal')
    vmin, vmax = mappable.get_clim()
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        cb.set_ticks(ticks)
        fmt = f"%.{decimals}f"
        cb.set_ticklabels([fmt % t for t in ticks])

    label_px = np.clip(strip_h_px * 0.44, 9, 44)
    label_pt = float(label_px * 72.0 / dpi)
    tick_len_px = np.clip(strip_h_px * 0.28, 3, 14)
    tick_len_pt = float(tick_len_px * 72.0 / dpi)
    tick_w_pt = float(np.clip(strip_h_px * 0.045, 1.0, 3.0))

    cb.ax.tick_params(colors="black", length=tick_len_pt, width=tick_w_pt, labelsize=label_pt)
    plt.setp(cb.ax.get_xticklabels(), color="black")
    try:
        cb.outline.set_edgecolor("black"); cb.outline.set_linewidth(1.0)
    except Exception:
        pass
    for spine in cax.spines.values(): spine.set_visible(False)
    return cb


# -----------------------------
# Low-level raw LLC I/O (cluster-friendly)
# -----------------------------
def _prec_to_np(prec: str):
    lut = {
        "integer*1": ("i1", 1), "int8": ("i1", 1),
        "integer*2": ("i2", 2), "int16": ("i2", 2), "uint16": ("u2", 2),
        "integer*4": ("i4", 4), "int32": ("i4", 4), "uint32": ("u4", 4),
        "single": ("f4", 4), "real*4": ("f4", 4), "float32": ("f4", 4),
        "integer*8": ("i8", 8), "int64": ("i8", 8), "uint64": ("u8", 8),
        "double": ("f8", 8), "real*8": ("f8", 8), "float64": ("f8", 8),
    }
    code, nbytes = lut[prec.lower()]
    return np.dtype(">" + code), nbytes  # big-endian dtype + element size


def _pread_into_fd(fd: int, offset: int, out_arr: np.ndarray):
    """Fill 'out_arr' bytes from fd at 'offset' using os.pread (safe on parallel FS)."""
    mv = memoryview(out_arr).cast('B')
    nbytes = mv.nbytes
    pos = 0
    while pos < nbytes:
        chunk = os.pread(fd, nbytes - pos, offset + pos)
        if not chunk:
            raise EOFError(f"short pread at {offset+pos}")
        mv[pos:pos+len(chunk)] = chunk
        pos += len(chunk)


class RawLLC:
    """
    Column-contiguous reader for LLC mosaic:
      - file stores levels back-to-back (each level has nx*nx*13 elements)
      - each 'column' (nx values) is contiguous on disk
    """
    def __init__(self, path: str, nx: int, prec: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.nx = int(nx)
        self.dtype_be, self.elsize = _prec_to_np(prec)
        self.level_stride_elems = self.nx * self.nx * 13
        self._colbytes = self.nx * self.elsize

    def close(self):
        try: os.close(self.fd)
        except Exception: pass

    # ---- contiguous reads ----
    def read_column(self, level: int, j_abs: int) -> np.ndarray:
        """Return full column (nx elements) at absolute column j_abs for level."""
        arr = np.empty(self.nx, dtype=self.dtype_be)
        base_elem = level * self.level_stride_elems + self.nx * j_abs
        _pread_into_fd(self.fd, base_elem * self.elsize, arr)
        return arr

    def read_block_cols(self, level: int, start_col: int, width_cols: int) -> np.ndarray:
        """Return block (nx x width_cols) contiguous columns starting at start_col (Fortran layout)."""
        nx = self.nx
        flat = np.empty(nx * width_cols, dtype=self.dtype_be)
        base_elem = level * self.level_stride_elems + nx * start_col
        _pread_into_fd(self.fd, base_elem * self.elsize, flat)
        return flat.reshape((nx, width_cols), order="F")


def _default_threads() -> int:
    try:
        env = int(os.environ.get("LLC_THREADS", "0"))
        if env > 0: return env
    except Exception:
        pass
    try:
        c = os.cpu_count() or 4
    except Exception:
        c = 4
    return max(1, min(8, c // 2))


# -----------------------------
# Rectified row/col extractors (no RAM blowup)
# -----------------------------
def _extract_rect_row_level(reader: RawLLC, level: int, rect_row_idx: int) -> np.ndarray:
    """
    One rectified ROW (length 4*NX) for this level (fixed X).
    - Bands 1–2 & face 7 use big contiguous reads (fast).
    - Bands 8–13 read 3 contiguous columns and flip (cheap).
    Returns float64 to match llc2rect_nd behavior.
    """
    nx = reader.nx
    out = np.zeros(4 * nx, dtype=np.float64)

    # Band 1: faces 1–3
    if 0 <= rect_row_idx < nx:
        band = reader.read_block_cols(level, 0, 3*nx)  # (nx, 3nx)
        out[0:3*nx] = band[rect_row_idx, :].astype(np.float64, copy=False)
        return out

    # Band 2 + face 7
    if nx <= rect_row_idx < 2*nx:
        r = rect_row_idx - nx
        band = reader.read_block_cols(level, 3*nx, 3*nx)
        out[0:3*nx] = band[r, :].astype(np.float64, copy=False)
        face7 = reader.read_block_cols(level, 6*nx, nx)
        out[3*nx:4*nx] = face7[r, :].astype(np.float64, copy=False)
        return out

    # Band 3: faces 8–10 (flip columns)
    if 2*nx <= rect_row_idx < 3*nx:
        r = rect_row_idx - 2*nx
        j0 = 7*nx + (3*r + 0)
        j1 = 7*nx + (3*r + 1)
        j2 = 7*nx + (3*r + 2)
        c0 = reader.read_column(level, j0)[::-1].astype(np.float64, copy=False)
        c1 = reader.read_column(level, j1)[::-1].astype(np.float64, copy=False)
        c2 = reader.read_column(level, j2)[::-1].astype(np.float64, copy=False)
        out[0:nx]      = c2
        out[nx:2*nx]   = c1
        out[2*nx:3*nx] = c0
        return out

    # Band 4: faces 11–13 (flip columns)
    if 3*nx <= rect_row_idx < 4*nx:
        r = rect_row_idx - 3*nx
        j0 = 10*nx + (3*r + 0)
        j1 = 10*nx + (3*r + 1)
        j2 = 10*nx + (3*r + 2)
        c0 = reader.read_column(level, j0)[::-1].astype(np.float64, copy=False)
        c1 = reader.read_column(level, j1)[::-1].astype(np.float64, copy=False)
        c2 = reader.read_column(level, j2)[::-1].astype(np.float64, copy=False)
        out[0:nx]      = c2
        out[nx:2*nx]   = c1
        out[2*nx:3*nx] = c0
        return out

    raise IndexError("rect_row_idx out of range")


def _gather_band_row_every3_parallel(path: str, nx: int, level: int, band_col0: int, i0: int, j0: int) -> np.ndarray:
    """
    Fixed-Y bottom bands: read exactly NX columns (contiguous per column).
    Parallelized with small thread pool; each worker uses its own fd/buffer.
    """
    nthreads = _default_threads()
    parts = [[] for _ in range(nthreads)]
    for r in range(nx):
        parts[r % nthreads].append(r)

    def worker(r_list):
        rdr = RawLLC(path, nx, "real*4")  # 3D fields are typically real*4
        try:
            buf = np.empty(nx, dtype=rdr.dtype_be)
            out_vals = np.empty(len(r_list), dtype=np.float64)
            mv = memoryview(buf).cast('B')
            for idx, rpos in enumerate(r_list):
                col = band_col0 + j0 + 3 * rpos
                base_elem = level * rdr.level_stride_elems + nx * col
                _pread_into_fd(rdr.fd, base_elem * rdr.elsize, buf)
                out_vals[idx] = float(buf[i0])
            return (r_list, out_vals)
        finally:
            rdr.close()

    out = np.empty(nx, dtype=np.float64)
    if nthreads <= 1:
        rr, vv = worker(list(range(nx)))
        out[np.array(rr, dtype=int)] = vv
        return out

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        futs = [ex.submit(worker, rlist) for rlist in parts if rlist]
        for fut in as_completed(futs):
            rr, vv = fut.result()
            out[np.array(rr, dtype=int)] = vv
    return out


def _extract_rect_col_level(path: str, nx: int, level: int, rect_col_idx: int) -> np.ndarray:
    """
    One rectified COLUMN (length 4*NX) for this level (fixed Y).
    Top bands: two full contiguous columns.
    Bottom bands: exactly NX columns (parallel), minimal bytes.
    """
    out = np.zeros(4 * nx, dtype=np.float64)
    top = RawLLC(path, nx, "real*4")
    try:
        if 0 <= rect_col_idx < 3*nx:
            v1 = top.read_column(level, rect_col_idx).astype(np.float64, copy=False)
            v2 = top.read_column(level, rect_col_idx + 3*nx).astype(np.float64, copy=False)
            out[0:nx]    = v1
            out[nx:2*nx] = v2

            jfix = 3*nx - 1 - rect_col_idx
            i0 = int(jfix % nx)
            j0 = int(jfix // nx)  # 0,1,2

            out[2*nx:3*nx] = _gather_band_row_every3_parallel(path, nx, level, 7*nx,  i0, j0)
            out[3*nx:4*nx] = _gather_band_row_every3_parallel(path, nx, level, 10*nx, i0, j0)
            return out

        if 3*nx <= rect_col_idx < 4*nx:
            c = rect_col_idx - 3*nx
            v7 = top.read_column(level, 6*nx + c).astype(np.float64, copy=False)
            out[nx:2*nx] = v7
            return out
    finally:
        top.close()

    raise IndexError("rect_col_idx out of range")


# -----------------------------
# Stream the 3D section level-by-level (no big arrays)
# -----------------------------
def _stream_section_x(path: str, nx: int, nk: int, rect_row_idx: int, trace_every: int) -> np.ndarray:
    rdr = RawLLC(path, nx, "real*4")
    try:
        A2D = np.empty((nk, 4*nx), dtype=np.float64)
        for k in range(nk):
            A2D[k, :] = _extract_rect_row_level(rdr, k, rect_row_idx)
            if trace_every > 0 and ((k + 1) % trace_every == 0 or (k + 1) == nk):
                print(f"[read] level {k+1}/{nk}", flush=True)
        return A2D
    finally:
        rdr.close()


def _stream_section_y(path: str, nx: int, nk: int, rect_col_idx: int, trace_every: int) -> np.ndarray:
    A2D = np.empty((nk, 4*nx), dtype=np.float64)
    for k in range(nk):
        A2D[k, :] = _extract_rect_col_level(path, nx, k, rect_col_idx)
        if trace_every > 0 and ((k + 1) % trace_every == 0 or (k + 1) == nk):
            print(f"[read] level {k+1}/{nk}", flush=True)
    return A2D


# -----------------------------
# 2D overlays: read only the needed line (row or column)
# -----------------------------
def _detect_2d_prec(path: str, nx: int) -> str:
    """Detect 2D file precision by size; default to real*4."""
    try:
        size = os.path.getsize(path)
        denom = nx * nx * 13
        if denom > 0 and size % denom == 0:
            bpe = size // denom
            if bpe == 8: return "real*8"
            if bpe == 4: return "real*4"
    except Exception:
        pass
    return "real*4"


def _read_line_2d_row(path: str, nx: int, rect_row_idx: int) -> np.ndarray:
    """Return a rectified ROW (length 4*nx) from a 2D LLC file."""
    prec = _detect_2d_prec(path, nx)
    rdr = RawLLC(path, nx, prec)
    try:
        return _extract_rect_row_level(rdr, 0, rect_row_idx)
    finally:
        rdr.close()


def _gather_band_row_every3_parallel_2d(path: str, nx: int, prec: str, level: int, band_col0: int, i0: int, j0: int):
    nthreads = _default_threads()
    parts = [[] for _ in range(nthreads)]
    for r in range(nx):
        parts[r % nthreads].append(r)

    def worker(r_list):
        rdr = RawLLC(path, nx, prec)
        try:
            buf = np.empty(nx, dtype=rdr.dtype_be)
            out_vals = np.empty(len(r_list), dtype=np.float64)
            for idx, rpos in enumerate(r_list):
                col = band_col0 + j0 + 3 * rpos
                base_elem = level * rdr.level_stride_elems + nx * col
                _pread_into_fd(rdr.fd, base_elem * rdr.elsize, buf)
                out_vals[idx] = float(buf[i0])
            return (r_list, out_vals)
        finally:
            rdr.close()

    out = np.empty(nx, dtype=np.float64)
    if nthreads <= 1:
        rr, vv = worker(list(range(nx)))
        out[np.array(rr, dtype=int)] = vv
        return out
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        futs = [ex.submit(worker, rlist) for rlist in parts if rlist]
        for fut in as_completed(futs):
            rr, vv = fut.result()
            out[np.array(rr, dtype=int)] = vv
    return out


def _read_line_2d_col(path: str, nx: int, rect_col_idx: int) -> np.ndarray:
    """Return a rectified COLUMN (length 4*nx) from a 2D LLC file."""
    prec = _detect_2d_prec(path, nx)
    out = np.zeros(4 * nx, dtype=np.float64)
    top = RawLLC(path, nx, prec)
    try:
        c = rect_col_idx
        if 0 <= c < 3*nx:
            v1 = top.read_column(0, c).astype(np.float64, copy=False)
            v2 = top.read_column(0, c + 3*nx).astype(np.float64, copy=False)
            out[0:nx]    = v1
            out[nx:2*nx] = v2
            jfix = 3*nx - 1 - c
            i0 = int(jfix % nx)
            j0 = int(jfix // nx)
            out[2*nx:3*nx] = _gather_band_row_every3_parallel_2d(path, nx, prec, 0, 7*nx,  i0, j0)
            out[3*nx:4*nx] = _gather_band_row_every3_parallel_2d(path, nx, prec, 0, 10*nx, i0, j0)
            return out
        if 3*nx <= c < 4*nx:
            cc = c - 3*nx
            v7 = top.read_column(0, 6*nx + cc).astype(np.float64, copy=False)
            out[nx:2*nx] = v7
            return out
    finally:
        top.close()
    raise IndexError("rect_col_idx out of range")


# -----------------------------
# RF reader (as before)
# -----------------------------
def _read_rf_edges(path: str, K_expected: int) -> np.ndarray:
    """Read RF edges as length K_expected+1 via llc.readbin, fallback to dtype guesses."""
    try:
        RF = llc.readbin(path, [K_expected + 1])
        RF = np.asarray(RF).reshape(-1)
        if RF.size == K_expected + 1:
            return RF
    except Exception:
        pass
    for dtype in (">f4", ">f8", "<f4", "<f8", "f4", "f8"):
        try:
            arr = np.fromfile(path, dtype=dtype).reshape(-1)
            if arr.size == K_expected + 1:
                return arr
        except Exception:
            continue
    raise RuntimeError(f"RF length mismatch in {path}: need {K_expected+1} edges")


# -----------------------------
# Plotting the section (advanced layout + interactions)
# -----------------------------
def plot_section(AK_by_N: np.ndarray, RF: np.ndarray, *,
                 sea_ice: np.ndarray | None,
                 ice_shelf: np.ndarray | None,
                 bathymetry: np.ndarray | None,
                 xlabel: str,
                 ylabel: str,
                 title_text: str | None,
                 cmap: str,
                 vmin, vmax,
                 horiz_name: str,
                 yscale: str = "linear",
                 linthresh: float = 1e-2,
                 dock_pad_px: int | None = None):
    """
    AK_by_N: (K, N) field slice for pcolormesh (K rows = vertical cells, N columns = horiz cells)
    RF:      (K+1,) edges (0 at surface, negative downward)
    """
    A = np.asarray(AK_by_N)
    K, N = A.shape
    RF = np.asarray(RF).reshape(-1)
    if RF.size != K + 1:
        raise ValueError("RF must have length K+1")

    # Create figure/axes
    fig, ax = plt.subplots(constrained_layout=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    # Main image
    x_edges = np.arange(N + 1, dtype=float)
    m = ax.pcolormesh(x_edges, RF, A, shading="auto", cmap=cmap)
    if (vmin is not None) or (vmax is not None):
        m.set_clim(vmin, vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Optional y-scale
    if str(yscale).lower() == "symlog":
        ax.set_yscale("symlog", linthresh=max(1e-12, float(linthresh)), linscale=1.0)

    # Place colorbar + title in a top strip outside the canvas (wider/longer but thinner)
    apply_top_strip(ax, m, title_text, strip_h=0.065, pad=0.012, cbar_rel_w=0.46, cbar_height_frac=0.33)

    # Precompute vertical cell-edge helpers
    y_top    = np.maximum(RF[:-1], RF[1:])
    y_bottom = np.minimum(RF[:-1], RF[1:])

    AZ = (A == 0)

    def overlay_sea_ice(thickness_pos: np.ndarray | None, color: str):
        """SEA ICE: draw from negative draft up to 0 wherever thickness>0 (ignore 0). Overrides field."""
        if thickness_pos is None:
            return
        si = np.asarray(thickness_pos, dtype=float).reshape(-1)
        if si.size != N:
            raise ValueError(f"Sea-ice length {si.size} != N {N}")
        valid_cols = np.isfinite(si) & (si > 0.0)
        if not np.any(valid_cols):
            return
        D = np.full((1, N), np.nan, dtype=float)
        D[0, valid_cols] = -np.abs(si[valid_cols])
        mask_span = (y_top[:, None] >= D) & (y_bottom[:, None] <= 0.0)
        if not np.any(mask_span):
            return
        Z = np.ma.masked_where(~mask_span, np.ones_like(A, dtype=float))
        ax.pcolormesh(x_edges, RF, Z, shading="auto",
                      cmap=ListedColormap(["#e9f4ff"]), antialiased=False)

    def overlay_top_cap(draft_neg: np.ndarray | None, color: str):
        """ICE SHELF: negative draft; draw above D to 0 only where A==0."""
        if draft_neg is None:
            return
        arr = np.asarray(draft_neg, dtype=float).reshape(-1)
        if arr.size != N:
            raise ValueError(f"Ice-shelf length {arr.size} != N {N}")
        valid_cols = np.isfinite(arr) & (arr < 0.0)
        if not np.any(valid_cols):
            return
        D = np.full((1, N), np.nan, dtype=float)
        D[0, valid_cols] = arr[valid_cols]
        mask_span = (y_top[:, None] >= D) & (y_bottom[:, None] <= 0.0)
        mask = mask_span & AZ
        if not np.any(mask):
            return
        Z = np.ma.masked_where(~mask, np.ones_like(A, dtype=float))
        ax.pcolormesh(x_edges, RF, Z, shading="auto",
                      cmap=ListedColormap([color]), antialiased=False)

    def overlay_bottom_cap(depth_neg: np.ndarray | None, color: str):
        """BATHYMETRY: draw below D only where A==0."""
        if depth_neg is None:
            return
        arr = np.asarray(depth_neg, dtype=float).reshape(-1)
        if arr.size != N:
            raise ValueError(f"Bathymetry length {arr.size} != N {N}")
        valid_cols = np.isfinite(arr)
        if not np.any(valid_cols):
            return
        D = np.full((1, N), np.nan, dtype=float)
        D[0, valid_cols] = arr[valid_cols]
        mask_span = (y_bottom[:, None] <= D)
        mask = mask_span & AZ
        if not np.any(mask):
            return
        Z = np.ma.masked_where(~mask, np.ones_like(A, dtype=float))
        ax.pcolormesh(x_edges, RF, Z, shading="auto",
                      cmap=ListedColormap(["burlywood"]), antialiased=False)

    ICE_SHELF_COLOR = "steelblue"

    overlay_bottom_cap(bathymetry, "burlywood")
    overlay_top_cap(ice_shelf, ICE_SHELF_COLOR)
    overlay_sea_ice(sea_ice, "#e9f4ff")

    # ---------------- Interactions ----------------
    tip = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    tip.set_visible(False)
    y_mid = 0.5 * (RF[:-1] + RF[1:])

    def on_move(evt):
        if evt.inaxes is not ax or evt.xdata is None or evt.ydata is None:
            tip.set_visible(False); fig.canvas.draw_idle(); return
        i = int(np.floor(evt.xdata))
        if i < 0 or i >= N:
            tip.set_visible(False); fig.canvas.draw_idle(); return
        y = evt.ydata
        k_hits = np.where((y <= y_top) & (y >= y_bottom))[0]
        if k_hits.size == 0:
            k = int(np.argmin(np.abs(y_mid - y)))
        else:
            k = int(k_hits[0])
        k = max(0, min(K - 1, k))
        val = A[k, i]
        sval = "nan" if (isinstance(val, float) and np.isnan(val)) else f"{float(val):.6g}"
        tip.set_text(f"{horiz_name}={i} (0-based)\n"
                     f"k={k}\nval={sval}\ndepth≈{y_mid[k]:.6g}")
        tip.set_visible(True); fig.canvas.draw_idle()

    def on_click(evt):
        if evt.inaxes is ax and evt.xdata is not None and evt.ydata is not None:
            i = int(np.floor(evt.xdata))
            if 0 <= i < N:
                y = evt.ydata
                k_hits = np.where((y <= y_top) & (y >= y_bottom))[0]
                if k_hits.size == 0:
                    k = int(np.argmin(np.abs(y_mid - y)))
                else:
                    k = int(k_hits[0])
                k = max(0, min(K - 1, k))
                print(f"{horiz_name}={i}\tk={k}\tdepth≈{y_mid[k]:.6g}\tval={A[k,i]}")

    selector_state = {"active": False, "selector": None, "xlim0": ax.get_xlim(), "ylim0": ax.get_ylim()}

    def toggle_selector(active: bool):
        if active and selector_state["selector"] is None:
            sel = RectangleSelector(
                ax,
                onselect=lambda eclick, erelease: _apply_zoom(eclick, erelease, ax, fig),
                drawtype='box', useblit=True, interactive=False,
                button=[1], minspanx=0.0, minspany=0.0, spancoords='data'
            )
            selector_state["selector"] = sel
        if selector_state["selector"] is not None:
            selector_state["selector"].set_active(active)
        selector_state["active"] = active

    def _apply_zoom(eclick, erelease, ax, fig):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if (x0 is None) or (y0 is None) or (x1 is None) or (y1 is None):
            return
        ax.set_xlim(min(x0, x1), max(x0, x1))
        ax.set_ylim(min(y0, y1), max(y0, y1))
        fig.canvas.draw_idle()

    def on_key(evt):
        k = (evt.key or "").lower()
        if k in ("q", "escape"):
            plt.close(fig)
        elif k == "f":
            _fit_to_workarea(fig, dock_pad_px=dock_pad_px)
        elif k == "z":
            toggle_selector(not selector_state["active"])
        elif k == "r":
            ax.set_xlim(selector_state["xlim0"])
            ax.set_ylim(selector_state["ylim0"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    _ensure_fit_sequence(fig, dock_pad_px)

    plt.show()


# -----------------------------
# CLI (unchanged + --trace-every)
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Advanced vertical-section visualizer for MITgcm LLC fields.")
    p.add_argument("field3d", help="Path to the 3D field to visualize (e.g., Theta*.data).")
    p.add_argument("--sea-ice", help="Path to 2D sea-ice THICKNESS file (positive).", default=None)
    p.add_argument("--ice-shelf", help="Path to 2D ice-shelf DRAFT file (negative).", default=None)
    p.add_argument("--bathy", help="Path to 2D bathymetry file (negative).", default=None)
    p.add_argument("--rf", required=True, help="Path to RF file (vertical EDGES, length NK+1).")

    # nx controls
    p.add_argument("--nx", type=int, required=True, help="LLC tile size for the 3D field (e.g., 432, 1080).")
    p.add_argument("--nx-sea-ice", type=int, default=None, help="LLC tile size for sea-ice file (overrides --nx).")
    p.add_argument("--nx-ice-shelf", type=int, default=None, help="LLC tile size for ice-shelf file (overrides --nx).")
    p.add_argument("--nx-bathy", type=int, default=None, help="LLC tile size for bathy file (overrides --nx).")

    # vertical levels
    p.add_argument("--nk", type=int, default=173, help="Number of vertical levels NK to read (default 173).")

    # 0-based section indices
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--x", type=int, help="0-based fixed X index; vary Y. fld = fld[x, :, :]. X-axis label 'longitude'.")
    g.add_argument("--y", type=int, help="0-based fixed Y index; vary X. fld = fld[:, y, :]. X-axis label 'latitude'.")

    # visuals
    p.add_argument("--cmap", default="viridis", help="Colormap for the main field.")
    p.add_argument("--vmin", type=float, default=None, help="Min of color scale (optional).")
    p.add_argument("--vmax", type=float, default=None, help="Max of color scale (optional).")
    p.add_argument("--ylabel", default="Depth (m)", help="Y-axis label.")
    p.add_argument("--title", default="", help="Title shown above the plot (outside canvas).")
    p.add_argument("--dock-pad", type=int, default=None, help="Bottom padding (px) above Dock when sizing window.")
    p.add_argument("--yscale", choices=["linear", "symlog"], default="linear", help="Y axis scale.")
    p.add_argument("--linthresh", type=float, default=1e-2, help="linthresh for symlog (ignored if linear).")

    # progress
    p.add_argument("--trace-every", type=int, default=5, help="Print a progress line every N levels (0=off).")

    return p.parse_args()


# -----------------------------
# main: same UX, different internals (streaming)
# -----------------------------
def main():
    args = parse_args()

    nx = int(args.nx)
    nk = int(args.nk)
    trace_every = int(args.trace_every)

    # Build the 2D section line-by-line, level-by-level (no full 3D in RAM)
    if args.x is not None:
        ii = int(args.x)
        if not (0 <= ii < 4*nx):
            raise SystemExit(f"--x must be in 0..{4*nx-1}; got {args.x}")
        A2D = _stream_section_x(args.field3d, nx, nk, ii, trace_every)
        horiz_label = "longitude"
        horiz_name  = "y"
    else:
        jj = int(args.y)
        if not (0 <= jj < 4*nx):
            raise SystemExit(f"--y must be in 0..{4*nx-1}; got {args.y}")
        A2D = _stream_section_y(args.field3d, nx, nk, jj, trace_every)
        horiz_label = "latitude"
        horiz_name  = "x"

    # Optional 2D overlays (read only the needed 1-D line; keep identical semantics)
    nx_sea   = args.nx_sea_ice   or nx
    nx_shelf = args.nx_ice_shelf or nx
    nx_bathy = args.nx_bathy     or nx

    sea_v = shelf_v = bathy_v = None
    if args.x is not None:
        if args.sea_ice:   sea_v   = _read_line_2d_row(args.sea_ice,   nx_sea,   int(args.x))
        if args.ice_shelf: shelf_v = _read_line_2d_row(args.ice_shelf, nx_shelf, int(args.x))
        if args.bathy:     bathy_v = _read_line_2d_row(args.bathy,     nx_bathy, int(args.x))
    else:
        if args.sea_ice:   sea_v   = _read_line_2d_col(args.sea_ice,   nx_sea,   int(args.y))
        if args.ice_shelf: shelf_v = _read_line_2d_col(args.ice_shelf, nx_shelf, int(args.y))
        if args.bathy:     bathy_v = _read_line_2d_col(args.bathy,     nx_bathy, int(args.y))

    # RF edges must match NK+1
    RF = _read_rf_edges(args.rf, K_expected=nk)

    # Default title if none provided
    title = args.title or f"{os.path.basename(args.field3d)} | " + (f"x={args.x}" if args.x is not None else f"y={args.y}")

    # Plot with advanced UI (unchanged)
    plot_section(A2D, RF,
                 sea_ice=sea_v, ice_shelf=shelf_v, bathymetry=bathy_v,
                 xlabel=horiz_label, ylabel=args.ylabel, title_text=title,
                 cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
                 horiz_name=horiz_name, yscale=args.yscale, linthresh=args.linthresh,
                 dock_pad_px=args.dock_pad)


if __name__ == "__main__":
    main()

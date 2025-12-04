#!/usr/bin/env python3
# llc_view.py
# -----------------------------------------------------------------------------
# LLC horizontal slice visualizer with optional depth-aware overlays
#
# Now uses:
#   - Tracer: llc.llc2rect() to build a 4*nx x 4*nx rectangular view
#   - Vector: C-grid aware LLC->rect transform so arrows remain physically correct
#   - Optional depth-aware masks (bathy, ice-shelf, sea-ice) applied in rect space
#
# Supports:
#   - Tracer (one file)
#   - Vector (two files U and V, C-grid) with speed background and/or arrows
#   - Optional depth-aware masks (bathy, ice-shelf, sea-ice) using RF
#
# Usage examples
# --------------
# Tracer (one file):
#   ./llc_view.py temp.data --nx 432 --level 1 --cmap Blues --vmin -2 --vmax 30
#
# Vector (two files U and V):
#   ./llc_view.py U.data V.data --nx 432 --level 1 --mode both --cmap viridis --vmin 0 --vmax 2
#   ./llc_view.py U.data V.data --nx 1080 --level 5 --mode arrows --quiver-step 10 --quiver-color white
#
# Depth-aware masks (optional files, applied using RF[level]):
#   ./llc_view.py temp.data --nx 432 --level 5 \
#       --rf RF.data --bathy bathy.data --ice-shelf draft.data --sea-ice hi.data
#
# Save without showing a window:
#   ./llc_view.py temp.data --nx 432 --no-show -o out.png
#
# Title behavior
# --------------
# If you pass all four of: --startyr --startmo --startdy --deltaT, and the FIRST
# input filename matches "*<10 digits>.data" (where those 10 digits are "ts"),
# the script sets the plot title to llc.ts2dte(ts, deltat=deltaT, startyr=..., startmo=..., startdy=...).
# The title is drawn in a WHITE banner OUTSIDE the plot frame (top-center) and autosized.
#
# Notes
# -----
# * Color limits (--vmin/--vmax) apply to the plotted image (tracer or vector speed).
# * No normalization is applied; imshow uses your raw values.
# * Hover shows raw value at the cursor (tracer value or speed SQRT(U^2 + V^2)).
# * Overlays are drawn on the SAME rectangular canvas you display (llc.llc2rect()).
#   - bathy (m, negative): if bathy >= z(level) AND field==0 → burlywood
#   - ice-shelf (m, negative): if ice_shelf <= z(level) AND field==0 → steelblue
#   - sea-ice thickness (m, positive): if thickness>0 AND -thickness <= z(level) → #e9f4ff
# -----------------------------------------------------------------------------

import os
import re
import argparse
import numpy as np

import llc  # expected to provide: quikread, readbin, ts2dte, llc2rect

# These will be set by _setup_matplotlib(...)
mpl = None
plt = None

# -----------------------------
# Matplotlib setup (lazy, backend-aware)
# -----------------------------
def _setup_matplotlib(no_show: bool):
    """
    Import matplotlib lazily. If no_show is True, select a headless backend (Agg)
    BEFORE importing pyplot to avoid any GUI creation and event-loop overhead.
    """
    global mpl, plt
    if mpl is not None and plt is not None:
        return

    if no_show:
        import matplotlib as _mpl
        # Must select backend BEFORE importing pyplot
        _mpl.use("Agg")
        import matplotlib.pyplot as _plt
        _plt.ioff()  # ensure interactive mode is off
        mpl, plt = _mpl, _plt
    else:
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        mpl, plt = _mpl, _plt

    # Layout defaults (unchanged from original)
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
    """Best-effort nudge so the canvas actually fills the resized window immediately."""
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        try:
            t = fig.canvas.new_timer(interval=160)
            try:
                t.single_shot = True  # not on all backends
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
    """
    Resize window to the screen's usable area (not true fullscreen), then force the
    Matplotlib canvas to match that size immediately (no manual nudge needed).
    """
    mgr = fig.canvas.manager
    backend = mpl.get_backend().lower()

    # Fallback pad if we can't detect a work area
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
            try:
                win.showMaximized()
            except Exception:
                pass
            try:
                win.setGeometry(x, y, w, h)
            except Exception:
                pass
            try:
                mgr.resize(w, h)
            except Exception:
                pass
            dpi = fig.get_dpi()
            fig.set_size_inches(w / dpi, h / dpi, forward=True)
            try:
                from PyQt5 import QtWidgets  # type: ignore
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
            _refresh_after_resize(fig)
            return

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
                try:
                    root.attributes('-zoomed', True)
                except Exception:
                    pass
            try:
                root.geometry(f"{w}x{h}+0+0")
            except Exception:
                pass
            try:
                root.update_idletasks()
            except Exception:
                pass
            try:
                sw = root.winfo_width()
                sh = root.winfo_height()
                dpi = fig.get_dpi()
                fig.set_size_inches(max(1, sw) / dpi, max(1, sh) / dpi, forward=True)
                try:
                    mgr.resize(sw, sh)
                except Exception:
                    pass
            except Exception:
                pass
            _refresh_after_resize(fig)
            return

        if 'wx' in backend:
            try:
                import wx  # type: ignore
                frame = mgr.frame
                w, h = wx.GetDisplaySize()
                h = max(200, h - pad)
                frame.SetPosition((0, 0))
                frame.SetSize((w, h))
                try:
                    frame.Layout()
                except Exception:
                    pass
                try:
                    mgr.resize(w, h)
                except Exception:
                    pass
                dpi = fig.get_dpi()
                fig.set_size_inches(w / dpi, h / dpi, forward=True)
                _refresh_after_resize(fig)
                return
            except Exception:
                pass

        if 'macosx' in backend:
            try:
                mgr.full_screen_toggle()
                mgr.full_screen_toggle()
            except Exception:
                pass
            try:
                dpi = fig.get_dpi()
                w, h = 1680, 950
                fig.set_size_inches(w / dpi, h / dpi, forward=True)
            except Exception:
                pass
            _refresh_after_resize(fig)
            return

        if 'gtk' in backend:
            try:
                mgr.window.maximize()
            except Exception:
                pass
            _refresh_after_resize(fig)
            return

        # Last resort: DPI-based sizing
        sw, sh = 1920, 1080
        try:
            sw = mgr.window.winfo_screenwidth()
            sh = mgr.window.winfo_screenheight()
        except Exception:
            pass
        dpi = fig.get_dpi()
        fig.set_size_inches(sw / dpi, (sh - pad) / dpi, forward=True)
        try:
            mgr.resize(sw, sh)
        except Exception:
            pass
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
# Helpers to call llc APIs robustly
# -----------------------------
def _llc_quikread(path, nx, level, prec):
    """Call llc.quikread with a few possible parameter names (level is 1-based)."""
    try:
        return llc.quikread(path, nx=nx, kx=level, prec=prec)
    except TypeError:
        try:
            return llc.quikread(path, nx=nx, k=level, prec=prec)
        except TypeError:
            return llc.quikread(path, nx=nx, prec=prec)


def _level_to_kidx(A: np.ndarray, level_1based: int) -> int:
    """Clamp 1-based level to 0-based index if 3D; else return 0."""
    A = np.asarray(A)
    if A.ndim == 3:
        return max(0, min(A.shape[2] - 1, int(level_1based) - 1))
    return 0

# -----------------------------
# Title helpers (optional)
# -----------------------------
def _extract_ts_from_filename(path: str) -> int | None:
    b = os.path.basename(path)
    m = re.search(r'(\d{10})\.data$', b)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def _compute_title(files, startyr, startmo, startdy, deltaT) -> str | None:
    if (startyr is None) or (startmo is None) or (startdy is None) or (deltaT is None):
        return None
    ts = _extract_ts_from_filename(files[0])
    if ts is None:
        return None
    try:
        title = llc.ts2dte(
            ts,
            deltat=deltaT,
            startyr=int(startyr),
            startmo=int(startmo),
            startdy=int(startdy),
        )
        return str(title)
    except Exception:
        return None

# -----------------------------
# Value lookup from AxesImage
# -----------------------------
def _value_from_axesimage(im, xdata, ydata):
    """Map data coords (xdata,ydata) to nearest (col,row) in imshow array and return value."""
    arr = im.get_array()
    data = np.asanyarray(arr)  # instead of np.asarray(arr)
    ny, nx = data.shape[:2]
    xmin, xmax, ymin, ymax = im.get_extent()  # (x0, x1, y0, y1)
    tx = (xdata - xmin) / (xmax - xmin) if xmax != xmin else 0.0
    ty = (ydata - ymin) / (ymax - ymin) if ymax != ymin else 0.0
    col = int(round(tx * (nx - 1)))
    row = int(round(ty * (ny - 1)))
    col = max(0, min(nx - 1, col))
    row = max(0, min(ny - 1, row))
    v = data[row, col]
    if np.ma.isMaskedArray(arr) and arr.mask is not np.ma.nomask:
        if np.ma.getmaskarray(arr)[row, col]:
            v = np.nan
    return col, row, v

# -----------------------------
# Cropping utilities
# -----------------------------
def _apply_crop_to_image(ax, im, xmin=None, xmax=None, ymin=None, ymax=None):
    if all(v is None for v in (xmin, xmax, ymin, ymax)):
        return 0, 0

    # Preserve MaskedArray if present
    arr = im.get_array()
    arr = np.asanyarray(arr)  # keeps MaskedArray subclass intact

    ny, nx = arr.shape[:2]
    xmin = 0 if xmin is None else int(xmin)
    ymin = 0 if ymin is None else int(ymin)
    xmax = nx if xmax is None else int(xmax)
    ymax = ny if ymax is None else int(ymax)
    xmin = max(0, min(nx, xmin)); xmax = max(0, min(nx, xmax))
    ymin = max(0, min(ny, ymin)); ymax = max(0, min(ny, ymax))
    if xmax <= xmin or ymax <= ymin:
        return 0, 0

    cropped = arr[ymin:ymax, xmin:xmax]  # stays MaskedArray if input was
    im.set_data(cropped)
    im.set_extent((xmin, xmax - 1, ymin, ymax - 1))
    ax.set_xlim(xmin, xmax - 1)
    ax.set_ylim(ymin, ymax - 1)
    return xmin, ymin


# -----------------------------
# Colorbar helpers
# -----------------------------
def apply_top_strip(
    ax, im, title_text: str | None, *,
    strip_h=0.05, pad=0.05, cbar_rel_w=0.4, cbar_height_frac=0.2,
    decimals=2, bg_alpha=0.98
):
    """
    Top strip outside the data canvas with a left title and right colorbar.
    In Agg/headless mode, font sizes auto-scale from the data pixel height (ny).
    """
    if strip_h <= 0:
      return None
    import matplotlib as _mpl
    fig = ax.figure
    pos = ax.get_position()
    cbar_h = max(0.05, strip_h * float(cbar_height_frac))
    #new_h = pos.height - (strip_h + pad)
    occupied = max(strip_h, cbar_h)  # reserve the true amount used above the canvas
    new_h = pos.height - (occupied + pad)
    if new_h <= 0:
        new_h = max(0.5 * pos.height, 0.1)
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

    cbar_y = y + (strip_h - cbar_h) / 2.0
    cax = fig.add_axes([cbar_bg_x0, cbar_y, cbar_w, cbar_h], zorder=11)
    cax.set_facecolor("none")

    # --- Sizing (points) ----------------------------------------------------
    fig.canvas.draw()  # needed to get correct sizes
    dpi = fig.get_dpi()
    fig_h_px = fig.get_size_inches()[1] * dpi
    strip_h_px = strip_h * fig_h_px
    is_headless = _mpl.get_backend().lower() == "agg"

    if is_headless:
        # Use displayed data height (ny pixels) as reference; cap by strip height
        ny, nx = im.get_array().shape[:2]
        # Pixel targets; tweak factors if you want larger/smaller
        title_px     = np.clip(ny * 0.06, 16, min(80, strip_h_px * 0.75))
        label_px     = np.clip(ny * 0.045, 12, min(64, strip_h_px * 0.60))
        tick_len_px  = np.clip(max(nx, ny) * 0.012, 4, 18)
        tick_w_px    = np.clip(max(nx, ny) * 0.0018, 1.0, 3.0)
    else:
        # Original behavior (scale from strip height)
        title_px     = np.clip(strip_h_px * 0.58, 12, 64)
        label_px     = np.clip(strip_h_px * 0.50, 10, 48)
        tick_len_px  = np.clip(strip_h_px * 0.35, 4, 16)
        tick_w_px    = np.clip(strip_h_px * 0.05, 1.0, 3.0)

    title_pt   = float(title_px * 72.0 / dpi)*1
    label_pt   = float(label_px * 72.0 / dpi)*1
    tick_len_pt= float(tick_len_px * 72.0 / dpi)*1
    tick_w_pt  = float(tick_w_px * 72.0 / dpi)*1

    # --- Draw title and colorbar -------------------------------------------
    if title_text:
        title_ax.text(0.02, 0.5, str(title_text),
                      ha='left', va='center', color='black', fontsize=title_pt)

    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    vmin, vmax = im.get_clim()
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        cb.set_ticks(ticks)
        fmt = f"%.{decimals}f"
        cb.set_ticklabels([fmt % t for t in ticks])

    cb.ax.tick_params(colors="black", length=tick_len_pt, width=tick_w_pt, labelsize=label_pt)
    plt.setp(cb.ax.get_xticklabels(), color="black")
    try:
        cb.outline.set_edgecolor("black"); cb.outline.set_linewidth(1.0)
    except Exception:
        pass

    for spine in cax.spines.values():
        spine.set_visible(False)

    return cb


# -----------------------------
# NEW: optional masks (RF/bathy/ice-shelf/sea-ice) in RECT space
# -----------------------------
def _read_llc_2d(path: str, nx: int) -> np.ndarray | None:
    """Read a 2D LLC field (nx, 13*nx)."""
    if not path:
        return None
    try:
        return np.asarray(llc.readbin(path, [nx, 13 * nx]))
    except Exception as e:
        print(f"[mask] Failed to read {path}: {e}")
        return None


def _read_rf(path: str, nk: int) -> np.ndarray | None:
    if not path:
        return None
    try:
        rf = np.asarray(llc.readbin(path, [nk])).reshape(-1)
        return rf
    except Exception as e:
        print(f"[mask] Failed to read RF {path}: {e}")
        return None


def _mask_overlay_rect(
    ax,
    base_im,
    *,
    level_1based: int,
    field_rect: np.ndarray,   # (4*nx, 4*nx) rectangular field used for display
    nx: int,
    rf_path: str | None,
    nk: int,
    bathy_path: str | None,
    iceshelf_path: str | None,
    seaice_path: str | None,
    xmin: int | None,
    xmax: int | None,
    ymin: int | None,
    ymax: int | None,
):
    # ---- depth from RF ----
    zlev = None
    if rf_path:
        rf = _read_rf(rf_path, nk)
        if rf is not None and rf.size > 0:
            kidx = max(0, min(rf.size - 1, int(level_1based) - 1))
            try:
                zlev = float(rf[kidx])
            except Exception:
                zlev = None
    if zlev is None and (bathy_path or iceshelf_path or seaice_path):
        print("[mask] RF not provided/valid; depth-based masks skipped.")
        return

    # ---- read LLC masks and map to rectangular layout ----
    def _read_rect(path):
        if not path: return None
        A = _read_llc_2d(path, nx)
        return None if A is None else llc.llc2rect(A)

    Brect = _read_rect(bathy_path)
    ISrect = _read_rect(iceshelf_path)
    HIrect = _read_rect(seaice_path)

    H, W = field_rect.shape
    overlay = np.zeros((H, W, 4), dtype=float)
    zero_mask = (field_rect == 0) | (~np.isfinite(field_rect))

    from matplotlib.colors import to_rgba
    c_bathy = np.array(to_rgba("burlywood"))
    c_ice   = np.array(to_rgba("steelblue"))
    c_hice  = np.array(to_rgba("#e9f4ff"))
    eps = 1e-12

    if Brect is not None:
        m = (Brect >= zlev) & zero_mask
        overlay[m] = c_bathy
    if ISrect is not None:
        m = (ISrect <= zlev) & zero_mask & (np.abs(ISrect) > eps)
        overlay[m] = c_ice
    if HIrect is not None:
        m = (HIrect > eps) & ((-HIrect) <= zlev)
        overlay[m] = c_hice

    # To display on top of imshow(F.T, origin="lower"), transpose rows/cols
    disp = np.transpose(overlay, (1, 0, 2))

    # Apply same crop (display coords)
    if not all(v is None for v in (xmin, xmax, ymin, ymax)):
        nrows, ncols = disp.shape[:2]
        x0 = 0 if xmin is None else max(0, min(ncols, int(xmin)))
        x1 = ncols if xmax is None else max(0, min(ncols, int(xmax)))
        y0 = 0 if ymin is None else max(0, min(nrows, int(ymin)))
        y1 = nrows if ymax is None else max(0, min(nrows, int(ymax)))
        if (x1 > x0) and (y1 > y0):
            disp = disp[y0:y1, x0:x1, :]

    ax.imshow(disp, origin="lower", interpolation="nearest", aspect="equal",
              resample=False, extent=base_im.get_extent(), zorder=6)

# -----------------------------
# NEW: Vector-aware LLC->rect mapping for C-grid U,V
# -----------------------------
def _llc2rect_vectors_cgrid(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map C-grid U,V (NX x 13*NX) onto rectangular (4*NX x 4*NX) and rotate components
    appropriately for the cap faces so arrows remain physically correct.

    Faces 1–7 copy directly into their rectangular slots.
    Faces 8–13 follow the same reshape/transpose and column-flip as llc.llc2rect()
    for scalars, with a component swap (u<->v) to account for the 90° rotation.
    """
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2D arrays")
    NX, NY = U.shape
    if V.shape != (NX, NY) or NY != 13 * NX:
        raise ValueError(f"Expected shape (NX, 13*NX); got U{U.shape}, V{V.shape}")

    RU = np.zeros((4*NX, 4*NX), dtype=np.float64)
    RV = np.zeros((4*NX, 4*NX), dtype=np.float64)

    # Faces 1–3 (top band): direct
    RU[0:NX, 0:3*NX] = U[:, 0:3*NX]
    RV[0:NX, 0:3*NX] = V[:, 0:3*NX]

    # Faces 4–6 (second band, left 3*NX): direct
    RU[NX:2*NX, 0:3*NX] = U[:, 3*NX:6*NX]
    RV[NX:2*NX, 0:3*NX] = V[:, 3*NX:6*NX]

    # Face 7 (second band, rightmost NX): direct
    RU[NX:2*NX, 3*NX:4*NX] = U[:, 6*NX:7*NX]
    RV[NX:2*NX, 3*NX:4*NX] = V[:, 6*NX:7*NX]

    # Faces 8–10 (third band): reshape/transpose like llc2rect, then swap components,
    # then apply the same column flip as the scalar path.
    blkU = np.asfortranarray(U[:, 7*NX:10*NX])
    blkV = np.asfortranarray(V[:, 7*NX:10*NX])
    tmpU = blkU.reshape((3*NX, NX), order="F").T      # (NX, 3*NX)
    tmpV = blkV.reshape((3*NX, NX), order="F").T
    RU[2*NX:3*NX, 0:3*NX] =  tmpV[:, ::-1]            # u <- v, then flip columns
    RV[2*NX:3*NX, 0:3*NX] =  tmpU[:, ::-1]            # v <- u

    # Faces 11–13 (fourth band): same treatment
    blkU = np.asfortranarray(U[:, 10*NX:13*NX])
    blkV = np.asfortranarray(V[:, 10*NX:13*NX])
    tmpU = blkU.reshape((3*NX, NX), order="F").T
    tmpV = blkV.reshape((3*NX, NX), order="F").T
    RU[3*NX:4*NX, 0:3*NX] =  tmpV[:, ::-1]
    RV[3*NX:4*NX, 0:3*NX] =  tmpU[:, ::-1]

    return RU, RV

# -----------------------------
# Plotters
# -----------------------------
def plot_tracer(A, *, level_1based: int, cmap: str, vmin, vmax, dock_pad_px, add_cbar: bool,
                outfile: str | None, no_show: bool, title_text: str | None = None,
                xmin: int | None = None, xmax: int | None = None, ymin: int | None = None, ymax: int | None = None,
                # NEW:
                nx_for_masks: int | None = None, rf_path: str | None = None, nk: int = 173,
                bathy_path: str | None = None, iceshelf_path: str | None = None, seaice_path: str | None = None):
    # Ensure matplotlib is set up appropriately for headless/interactive
    _setup_matplotlib(no_show)

    fig, ax = plt.subplots(constrained_layout=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    kidx0 = _level_to_kidx(A, level_1based)

    # ---- Rectangular tracer image via llc2rect
    R = llc.llc2rect(A[..., kidx0] if A.ndim == 3 else A)  # (4*nx, 4*nx)
    im = ax.imshow(R.T, origin="lower", interpolation="nearest", aspect="equal", resample=False)
    F = R  # keep a handle named F so the rest of the code stays unchanged

    xoff, yoff = _apply_crop_to_image(ax, im, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    if cmap:
        im.set_cmap(cmap)
    if (vmin is not None) or (vmax is not None):
        im.set_clim(vmin, vmax)
    ax.set_axis_off()

    # Optional depth-aware overlay masks in RECT space
    if (nx_for_masks is not None) and (rf_path or bathy_path or iceshelf_path or seaice_path):
        try:
            field_rect = np.asarray(F)
            _mask_overlay_rect(
                ax, im,
                level_1based=level_1based,
                field_rect=field_rect,
                nx=nx_for_masks,
                rf_path=rf_path, nk=nk,
                bathy_path=bathy_path, iceshelf_path=iceshelf_path, seaice_path=seaice_path,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            )
        except Exception as e:
            print(f"[mask] overlay failed: {e}")

    if add_cbar:
        apply_top_strip(ax, im, title_text, cbar_rel_w=0.4, cbar_height_frac=0.2)

    # --- UI/window workarea adjustments ONLY when showing a window
    if not no_show:
        _ensure_fit_sequence(fig, dock_pad_px)

    if not no_show:
        _ensure_fit_sequence(fig, dock_pad_px)
        tip = ax.text(5, 5, "", va="top", ha="left",
                      bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))

        def on_move(evt):
            if not evt.inaxes or evt.xdata is None or evt.ydata is None:
                tip.set_visible(False); fig.canvas.draw_idle(); return
            c, r, v = _value_from_axesimage(im, evt.xdata, evt.ydata)
            c, r = c + xoff, r + yoff
            sval = "nan" if (isinstance(v, float) and np.isnan(v)) else f"{float(v):.6g}"
            tip.set_text(f"x={c}, y={r}, val={sval}")
            tip.set_position((evt.xdata + 10, evt.ydata + 10))
            tip.set_visible(True); fig.canvas.draw_idle()

        def on_click(evt):
            if evt.inaxes and evt.xdata is not None and evt.ydata is not None:
                c, r, v = _value_from_axesimage(im, evt.xdata, evt.ydata)
                c, r = c + xoff, r + yoff
                print(f"{c}\t{r}\t{v}")

        def on_key(evt):
            k = (evt.key or "").lower()
            if k in ("q", "escape"):
                plt.close(fig)
            elif k == "f":
                _fit_to_workarea(fig, dock_pad_px=dock_pad_px)

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        try:
            t = fig.canvas.new_timer(interval=200)
            try:
                t.single_shot = True
            except Exception:
                pass
            t.add_callback(lambda: _fit_to_workarea(fig, dock_pad_px=dock_pad_px))
            t.start()
        except Exception:
            pass

    if outfile:
        #fig.savefig(outfile, dpi=600, bbox_inches="tight", pad_inches=0)
        ny, nx = im.get_array().shape[:2]    # after cropping; im shows R.T
        pos = ax.get_position()              # fraction of the figure occupied by the axes
        dpi = fig.get_dpi()
        fig.set_size_inches(nx / (dpi * pos.width), ny / (dpi * pos.height), forward=True)
        fig.savefig(outfile, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
        print(f"Saved figure to {outfile}")

    if not no_show:
        plt.show()
    else:
        # headless fast path: free resources immediately
        plt.close(fig)


def plot_vector(U, V, *, level_1based: int, mode: str, cmap: str, vmin, vmax,
                quiver_step: int, quiver_scale, quiver_color: str,
                dock_pad_px, add_cbar: bool, outfile: str | None,
                pad_to_square: bool, mask_zeros: bool, no_show: bool,
                title_text: str | None = None,
                xmin: int | None = None, xmax: int | None = None, ymin: int | None = None, ymax: int | None = None,
                # NEW:
                nx_for_masks: int | None = None, rf_path: str | None = None, nk: int = 173,
                bathy_path: str | None = None, iceshelf_path: str | None = None, seaice_path: str | None = None):
    """
    mode: 'bg' | 'arrows' | 'both'
    vmin/vmax: color limits for the background speed (if shown)
    """
    # Ensure matplotlib is set up appropriately for headless/interactive
    _setup_matplotlib(no_show)

    fig, ax = plt.subplots(constrained_layout=False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    if pad_to_square:
        print("[info] --pad-to-square is a no-op with rectangular llc2rect layout.")

    kidx0 = _level_to_kidx(np.asarray(U), level_1based)

    show_bg = (mode in ("bg", "both"))
    draw_ar = (mode in ("arrows", "both"))

    # ---- Select 2D level
    U2 = np.asarray(U[..., kidx0] if U.ndim == 3 else U)
    V2 = np.asarray(V[..., kidx0] if V.ndim == 3 else V)

    # ---- Vector-aware LLC->rect (keeps arrows sane in rectangular layout)
    Ur, Vr = _llc2rect_vectors_cgrid(U2, V2)   # (4*nx, 4*nx) each

    # ---- Background speed in display space
    speed_rect = np.hypot(Ur, Vr)
    if mask_zeros:
        bg = np.ma.masked_where(np.isnan(speed_rect) | (speed_rect == 0), speed_rect)
    else:
        bg = speed_rect

    im = ax.imshow(
        bg.T, origin="lower", interpolation="nearest", aspect="equal",
        resample=False, cmap=cmap, vmin=vmin, vmax=vmax, alpha=(1.0 if show_bg else 0.0)
    )
    ax.set_axis_off()

    # ---- Draw arrows (downsampled)
    q = None
    if draw_ar:
        UD = Ur.T  # match imshow orientation
        VD = Vr.T
        H, W = UD.shape
        step = max(1, int(quiver_step))
        xs = np.arange(0, W, step)
        ys = np.arange(0, H, step)
        X, Y = np.meshgrid(xs, ys)
        scale_val = None if (quiver_scale is None or str(quiver_scale).lower() == "auto") else float(quiver_scale)
        q = ax.quiver(
            X, Y,
            UD[ys][:, xs], VD[ys][:, xs],
            angles="xy", scale_units="xy", scale=scale_val,
            color=quiver_color, width=0.002
        )

    xoff, yoff = _apply_crop_to_image(ax, im, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    # ---- Optional depth-aware overlay masks on the rectangular canvas
    if (nx_for_masks is not None) and (rf_path or bathy_path or iceshelf_path or seaice_path):
        try:
            _mask_overlay_rect(
                ax, im,
                level_1based=level_1based,
                field_rect=speed_rect,
                nx=nx_for_masks,
                rf_path=rf_path, nk=nk,
                bathy_path=bathy_path, iceshelf_path=iceshelf_path, seaice_path=seaice_path,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            )
        except Exception as e:
            print(f"[mask] overlay failed: {e}")

    if add_cbar and show_bg:
        apply_top_strip(ax, im, title_text, cbar_rel_w=0.4, cbar_height_frac=0.2)

    # --- UI/window workarea adjustments ONLY when showing a window
    if not no_show:
        _ensure_fit_sequence(fig, dock_pad_px)

    if not no_show:
        _ensure_fit_sequence(fig, dock_pad_px)
        tip = ax.text(5, 5, "", va="top", ha="left",
                      bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.85))

        def on_move(evt):
            if not evt.inaxes or evt.xdata is None or evt.ydata is None:
                tip.set_visible(False); fig.canvas.draw_idle(); return
            c, r, v = _value_from_axesimage(im, evt.xdata, evt.ydata)
            c, r = c + xoff, r + yoff
            sval = "nan" if (isinstance(v, float) and np.isnan(v)) else f"{float(v):.6g}"
            tip.set_text(f"x={c}, y={r}, Speed={sval}")
            tip.set_position((evt.xdata + 10, evt.ydata + 10))
            tip.set_visible(True); fig.canvas.draw_idle()

        def on_click(evt):
            if evt.inaxes and evt.xdata is not None and evt.ydata is not None:
                c, r, v = _value_from_axesimage(im, evt.xdata, evt.ydata)
                c, r = c + xoff, r + yoff
                print(f"{c}\t{r}\t{v}")

        def on_key(evt):
            k = (evt.key or "").lower()
            if k in ("q", "escape"):
                plt.close(fig)
            elif k == "f":
                _fit_to_workarea(fig, dock_pad_px=dock_pad_px)

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        try:
            t = fig.canvas.new_timer(interval=200)
            try:
                t.single_shot = True
            except Exception:
                pass
            t.add_callback(lambda: _fit_to_workarea(fig, dock_pad_px=dock_pad_px))
            t.start()
        except Exception:
            pass

    if outfile:
        #fig.savefig(outfile, dpi=600, bbox_inches="tight", pad_inches=0)
        ny, nx = im.get_array().shape[:2]    # after cropping; im shows R.T
        pos = ax.get_position()              # fraction of the figure occupied by the axes
        dpi = fig.get_dpi()
        fig.set_size_inches(nx / (dpi * pos.width), ny / (dpi * pos.height), forward=True)
        fig.savefig(outfile, dpi=fig.dpi, bbox_inches="tight", pad_inches=0)

        print(f"Saved figure to {outfile}")

    if not no_show:
        plt.show()
    else:
        # headless fast path: free resources immediately
        plt.close(fig)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="LLC viewer for scalar tracer (1 file) or C-grid vectors (2 files). "
                    "Shows raw values (no normalization). Optional depth-aware masking."
    )
    ap.add_argument("files", nargs="+", help="Path(s) to LLC .data file(s): 1 for tracer, 2 for vector (U V)")
    ap.add_argument("--nx", type=int, required=True, help="Tile dimension (e.g., 432, 1080)")
    ap.add_argument("--level", type=int, default=1, help="Vertical level (1-based)")
    ap.add_argument("--prec", default="real*4", help="Precision for llc.quikread (e.g., real*4, real*8)")
    # color/limits (apply to tracer or vector-speed background)
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap for image")
    ap.add_argument("--vmin", type=float, default=None, help="Min of color axis (raw values)")
    ap.add_argument("--vmax", type=float, default=None, help="Max of color axis (raw values)")
    # vector draw options
    ap.add_argument("--mode", choices=["bg", "arrows", "both"], default="both",
                    help="Vector mode: colorful SQRT(U^2 + V^2) (speed) background, arrows only, or both (only used when 2 files given)")
    ap.add_argument("--quiver-step", type=int, default=8, help="Subsample factor for arrows (larger -> fewer arrows)")
    ap.add_argument("--quiver-scale", default=None,
                    help="Quiver scale (float) or 'auto' (default). Controls arrow size in pixels per data unit.")
    ap.add_argument("--quiver-color", default="black", help="Arrow color")
    # misc
    ap.add_argument("--output", "-o", help="Output PNG path (saved after viewing, or immediately with --no-show)")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window; just save the figure")
    ap.add_argument("--dock-pad", type=int, default=None,
                    help="Bottom padding (px) above Dock when sizing window (fallback). Default 100 or env DOCK_PAD_PX.")
    ap.add_argument("--pad-to-square", action="store_true",
                    help="(Rect mode: no-op) Pad right side so width=4*nx for a square canvas (legacy for mosaic layout)")
    ap.add_argument("--no-mask-zeros", action="store_true",
                    help="By default, zeros are masked as white in vector-speed backgrounds. Pass to keep zeros visible.")
    ap.add_argument("--xmin", type=int, default=None, help="Crop: min x (column) index, inclusive (display coords)")
    ap.add_argument("--xmax", type=int, default=None, help="Crop: max x (column) index, exclusive (display coords)")
    ap.add_argument("--ymin", type=int, default=None, help="Crop: min y (row) index, inclusive (display coords)")
    ap.add_argument("--ymax", type=int, default=None, help="Crop: max y (row) index, exclusive (display coords)")
    # optional time-origin inputs for title
    ap.add_argument("--startyr", type=int, default=None, help="Start year (used with --startmo/--startdy/--deltaT to title plot)")
    ap.add_argument("--startmo", type=int, default=None, help="Start month (used with --startyr/--startdy/--deltaT to title plot)")
    ap.add_argument("--startdy", type=int, default=None, help="Start day (used with --startyr/--startmo/--deltaT to title plot)")
    ap.add_argument("--deltaT", type=float, default=None, help="Time step (same units expected by llc.ts2dte; used for title)")

    # NEW: optional depth/mask inputs
    ap.add_argument("--rf", type=str, default=None, help="Path to RF (vertical levels, negative depths). Enables depth-based masks.")
    ap.add_argument("--nk", type=int, default=173, help="Number of vertical levels in RF (default: 173)")
    ap.add_argument("--bathy", type=str, default=None, help="Path to bathymetry (m, negative). Rule: bathy >= z(level) & field==0 -> burlywood")
    ap.add_argument("--ice-shelf", dest="iceshelf", type=str, default=None, help="Path to ice-shelf draft (m, negative). Rule: draft <= z(level) & field==0 -> steelblue")
    ap.add_argument("--sea-ice", dest="seaice", type=str, default=None, help="Path to sea-ice thickness (m, positive). Rule: thickness>0 & -thickness <= z(level) -> #e9f4ff")

    args = ap.parse_args()
    mask_zeros = not args.no_mask_zeros
    title_text = _compute_title(args.files, args.startyr, args.startmo, args.startdy, args.deltaT)

    # IMPORTANT: set up matplotlib once we know if we're headless
    _setup_matplotlib(args.no_show)

    # --- read files
    if len(args.files) == 1:
        f = args.files[0]
        fld, *_ = _llc_quikread(f, nx=args.nx, level=args.level, prec=args.prec)
        A = np.asarray(fld)
        if A.ndim not in (2, 3):
            raise ValueError(f"Unexpected array shape from llc.quikread: {A.shape}")

        if args.no_show:
            outname = args.output or (os.path.splitext(os.path.basename(f))[0] + f"_lvl{args.level}.png")
            plot_tracer(A, level_1based=args.level, cmap=args.cmap,
                        vmin=args.vmin, vmax=args.vmax,
                        dock_pad_px=args.dock_pad, add_cbar=True,
                        outfile=outname, no_show=True, title_text=title_text,
                        xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
                        nx_for_masks=args.nx, rf_path=args.rf, nk=args.nk,
                        bathy_path=args.bathy, iceshelf_path=args.iceshelf, seaice_path=args.seaice)
            return

        plot_tracer(A, level_1based=args.level, cmap=args.cmap,
                    vmin=args.vmin, vmax=args.vmax,
                    dock_pad_px=args.dock_pad, add_cbar=True,
                    outfile=args.output, no_show=False, title_text=title_text,
                    xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
                    nx_for_masks=args.nx, rf_path=args.rf, nk=args.nk,
                    bathy_path=args.bathy, iceshelf_path=args.iceshelf, seaice_path=args.seaice)
        return

    elif len(args.files) == 2:
        fu, fv = args.files
        U, *_ = _llc_quikread(fu, nx=args.nx, level=args.level, prec=args.prec)
        V, *_ = _llc_quikread(fv, nx=args.nx, level=args.level, prec=args.prec)
        U = np.asarray(U); V = np.asarray(V)
        if U.shape[:2] != V.shape[:2]:
            raise ValueError(f"U and V leading shapes differ: {U.shape} vs {V.shape}")

        if args.no_show:
            outname = args.output or (f"{os.path.splitext(os.path.basename(fu))[0]}_{os.path.splitext(os.path.basename(fv))[0]}_lvl{args.level}.png")
            plot_vector(U, V,
                        level_1based=args.level,
                        mode=args.mode,
                        cmap=args.cmap,
                        vmin=args.vmin, vmax=args.vmax,
                        quiver_step=args.quiver_step,
                        quiver_scale=args.quiver_scale,
                        quiver_color=args.quiver_color,
                        dock_pad_px=args.dock_pad,
                        add_cbar=True,
                        outfile=outname,
                        pad_to_square=args.pad_to_square,
                        mask_zeros=mask_zeros,
                        no_show=True,
                        xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
                        nx_for_masks=args.nx, rf_path=args.rf, nk=args.nk,
                        bathy_path=args.bathy, iceshelf_path=args.iceshelf, seaice_path=args.seaice,
                        title_text=title_text)
            return

        plot_vector(U, V,
                    level_1based=args.level,
                    mode=args.mode,
                    cmap=args.cmap,
                    vmin=args.vmin, vmax=args.vmax,
                    quiver_step=args.quiver_step,
                    quiver_scale=args.quiver_scale,
                    quiver_color=args.quiver_color,
                    dock_pad_px=args.dock_pad,
                    add_cbar=True,
                    outfile=args.output,
                    pad_to_square=args.pad_to_square,
                    mask_zeros=mask_zeros,
                    no_show=False,
                    xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax,
                    nx_for_masks=args.nx, rf_path=args.rf, nk=args.nk,
                    bathy_path=args.bathy, iceshelf_path=args.iceshelf, seaice_path=args.seaice,
                    title_text=title_text)
        return

    else:
        raise SystemExit("Pass exactly 1 file (tracer) or 2 files (U V for vectors).")


if __name__ == "__main__":
    main()

"""Zarr Viewer — browse/plot any MIDAS ``.zarr.zip`` (tree of groups/arrays;
1-D/2-D/3-D plots; R bin/2θ/Q/d axis conversion via the file's own REtaMap;
metadata/attributes inspector).

Ported from ``mpe_wf_saxs_waxs/gui_view_zarr.py``'s ``ZarrViewer`` +
``PlotCanvas`` — same file-format assumptions (root ``REtaMap`` array, shape
``(5, nR, nEta)``, channel order Radius/2θ/Eta/BinArea/Q; optional
``REtaMap_corrected``; ``InstrumentParameters/Lam`` for wavelength), same
control set and plotting logic, method-for-method. That format is produced by
``midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip`` — the same writer
Batch Integrate's own "zarr" output format and ``midas_gui/gsas_export.py``
already call — so a file this tab opens needs no format-detection or
conversion step.

This is MIDAS_GUI's second embedded matplotlib canvas (after
``peak_fit_panel.py``, for the same reason: no existing pyqtgraph-based zarr
tree/attribute browser to build on, and matplotlib is already an environment
dependency).

Deliberate simplifications vs. the source:
  - Drops the standalone ``QMainWindow`` shell — window title, font-size
    combo, Exit button, and the ``apply_font_size()``/widget-list bookkeeping
    that fed it. As a tab, the app's own global font/HiDPI scaling
    (``constants.DEFAULT_UI_SCALE``) already applies, and there's no
    separate window to title or exit.
  - Drops the ``PySide6``/``QT_BACKEND`` fallback — PyQt5 only, matching
    every other tab in this app.
  - No ``closeEvent``-driven store cleanup: a tab widget embedded in the
    main window's ``QTabWidget`` never reliably receives its own
    ``closeEvent`` (only top-level windows do), so that would be dead code.
    The meaningful cleanup — closing the previous ``zarr.ZipStore`` before
    opening the next one — already happens in ``_load_file`` itself, exactly
    as in the source.
  - No cross-tab wiring: opened via its own "Open zarr.zip…" file dialog
    (seeded from the last file this tab opened, via
    ``helpers.browse_start_dir``, rather than the source's always-blank
    starting directory) — not fed a mask/calibration/project context. A
    "View in Zarr Viewer" button on Batch Integrate's finished-run panel
    would be a natural follow-up; not built here.
  - No saved-project state: none of the plot/display state is meaningful to
    persist into a Project file (it's keyed to whatever file the user last
    browsed, which may not even belong to that project), so this tab defines
    no ``get_state()``/``set_state()`` — ``MainWindow._serialize_workspace``
    already skips any tab lacking one.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from midas_gui.helpers import browse_start_dir

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

try:
    import zarr
except ImportError:  # pragma: no cover — zarr is an environment.yml pin;
    zarr = None       # only unavailable in a broken/partial env.

CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "gray", "hot", "jet", "turbo", "RdBu_r",
]


def _zarr_full_path(item: QtWidgets.QTreeWidgetItem) -> str:
    """Walk up the QTreeWidget to reconstruct the zarr key path."""
    parts = []
    while item is not None:
        parts.append(item.text(0).split("  ")[0])   # strip shape annotation
        item = item.parent()
    parts.reverse()
    return "/".join(parts[1:])   # skip invisible root label


def _build_tree(node, parent_item: QtWidgets.QTreeWidgetItem):
    """Recursively populate a QTreeWidgetItem from a zarr group."""
    for key in sorted(node.keys()):
        child = node[key]
        if hasattr(child, "keys"):          # group
            item = QtWidgets.QTreeWidgetItem(parent_item, [key])
            item.setData(0, QtCore.Qt.UserRole, "group")
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            _build_tree(child, item)
        else:                               # array
            label = f"{key}  {list(child.shape)} {child.dtype}"
            item = QtWidgets.QTreeWidgetItem(parent_item, [label])
            item.setData(0, QtCore.Qt.UserRole, "array")


# ─────────────────────────────────────────────────────────────────────────
# Matplotlib canvas widget
# ─────────────────────────────────────────────────────────────────────────

def _draw_eta_bands(ax, x_coords, y_coords, z, eta_width, *, cmap, norm,
                    pack=False):
    """Draw the cake with each eta row at its TRUE height.

    ``pcolormesh`` treats its coordinates as cell CENTRES and infers the
    edges from the spacing between them. That is right for a contiguous
    caking, and badly wrong for a sparse one: four panels caked 10 deg
    wide but centred 90 deg apart were drawn as four 90-deg bands, so the
    eta axis ran -225 to +135 and the picture claimed nine times the
    azimuthal coverage actually measured. An axis reading -220 deg on a
    detector that spans +-180 was how it got noticed.

    With a known ``eta_width`` narrower than the spacing the rows are
    drawn one at a time at their real extent, leaving the gaps visibly
    empty -- which is the honest picture of what was collected. A
    contiguous caking takes the single-quadmesh path as before, because
    it is both correct there and much faster for 35+ rows.
    """
    import numpy as _np
    y = _np.asarray(y_coords, dtype=float)
    if eta_width and y.size > 1:
        gaps = _np.abs(_np.diff(y))
        spacing = float(_np.median(gaps)) if gaps.size else 0.0
        sparse = spacing > float(eta_width) * 1.5
    else:
        sparse = False

    def _edges(c):
        """Cell edges from centres. Flat shading needs n+1 on BOTH axes;
        giving y as edges and x as centres is a TypeError, and it is the
        one mistake this function invites twice."""
        c = _np.asarray(c, dtype=float)
        if c.size < 2:
            return _np.array([c[0] - 0.5, c[0] + 0.5])
        mid = 0.5 * (c[1:] + c[:-1])
        return _np.concatenate(([c[0] - (mid[0] - c[0])], mid,
                                [c[-1] + (c[-1] - mid[-1])]))

    if sparse and pack:
        # Rows adjacent, one per caked wedge. The y-axis is no longer
        # linear in eta, so it is ticked with the real angles and SAYS so
        # -- a packed axis that still claimed to be degrees would be the
        # same lie as the stretched bands, just in the other direction.
        im = ax.pcolormesh(_edges(x_coords), _np.arange(y.size + 1) - 0.5,
                           z, cmap=cmap, norm=norm, shading="flat")
        ax.set_yticks(_np.arange(y.size))
        ax.set_yticklabels([f"{v:g}" for v in y])
        # The caller styles the axes afterwards and would overwrite a
        # label set here, so hand it back instead of setting it.
        return im, "Eta (deg) — rows packed, gaps removed"

    if not sparse:
        return ax.pcolormesh(x_coords, y, z, cmap=cmap, norm=norm,
                             shading="auto"), None

    # Explicit EDGES on both axes: giving y as two edges selects flat
    # shading, which then requires x as edges too (n+1), not centres.
    xe = _edges(x_coords)

    half = float(eta_width) / 2.0
    im = None
    for i, c in enumerate(y):
        im = ax.pcolormesh(xe, _np.array([c - half, c + half]),
                           z[i:i + 1, :], cmap=cmap, norm=norm,
                           shading="flat")
    ax.set_ylim(float(y.min()) - half * 3, float(y.max()) + half * 3)
    return im, None


def _unwrap_eta(y):
    """``(axis, was_unwrapped)`` — an eta axis matplotlib can derive cell
    edges from.

    Needed only for archives written before ``hydra_zarr_combine`` sorted
    the merged axis; those come out monotonic and are returned untouched
    by the first test below. What follows describes the older ones.

    A four-panel combined archive concatenates its panels in panel order,
    and ge1's caking range at 1-ID-E is written PAST the cut (EtaMin=120,
    EtaMax=210, and ``eta_wrap`` labels its columns with those requested
    centres, 125..205 -- not wrapped into [-180, 180) as the other three
    panels happen to be). So the merged axis reads 125..205, then jumps to
    -145, then -55, then 45 -- physically one continuous sweep,
    numerically a sawtooth. pcolormesh
    treats its coordinates as cell CENTRES and infers the edges from the
    spacing, so at each jump it inferred a 350 deg cell and said so:

        UserWarning: The input coordinates to pcolormesh are interpreted
        as cell centers, but are not monotonically increasing or
        decreasing.

    Unwrapping turns it back into the sweep it is (125 -> 475), which is
    both monotonic and the physically honest picture: the panels really do
    continue into one another. An axis that is already monotonic is
    returned untouched, so a single-panel archive plots exactly as before.
    """
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1 or arr.size < 3:
        return y, False
    d = np.diff(arr)
    if np.all(d > 0) or np.all(d < 0):
        return y, False
    out = np.unwrap(arr, period=360.0)
    d2 = np.diff(out)
    if np.all(d2 > 0) or np.all(d2 < 0):
        return out, True
    return y, False       # not an angle wrap; leave it and let mpl warn


class PlotCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        self.toolbar = NavToolbar(self.canvas, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self._im = None
        self._cb = None
        self._plot_font_size = 11

    def get_current_xlim(self):
        if not self.fig.axes:
            return None
        try:
            return self.fig.axes[0].get_xlim()
        except Exception:
            return None

    def get_current_ylim(self):
        if not self.fig.axes:
            return None
        try:
            return self.fig.axes[0].get_ylim()
        except Exception:
            return None

    @staticmethod
    def _apply_xlim(ax, xlim):
        if xlim is None:
            return
        try:
            lo, hi = xlim
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                ax.set_xlim(lo, hi)
        except Exception:
            pass

    @staticmethod
    def _apply_ylim(ax, ylim):
        if ylim is None:
            return
        try:
            lo, hi = ylim
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                ax.set_ylim(lo, hi)
        except Exception:
            pass

    def set_plot_font_size(self, size: int):
        self._plot_font_size = max(8, int(size))

    @staticmethod
    def _needs_decimal_formatter(values: np.ndarray) -> bool:
        if values is None:
            return False
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return False
        return not np.allclose(arr, np.round(arr), atol=1e-8)

    @staticmethod
    def _decimal_tick_formatter(value, _pos):
        return f"{value:.5f}".rstrip("0").rstrip(".")

    def _configure_axis_ticks(self, ax, x_values=None, y_values=None):
        if self._needs_decimal_formatter(x_values):
            ax.xaxis.set_major_formatter(FuncFormatter(self._decimal_tick_formatter))
        if self._needs_decimal_formatter(y_values):
            ax.yaxis.set_major_formatter(FuncFormatter(self._decimal_tick_formatter))
        ax.xaxis.get_offset_text().set_fontsize(max(8, self._plot_font_size - 1))
        ax.yaxis.get_offset_text().set_fontsize(max(8, self._plot_font_size - 1))

    def _style_axes(self, ax, title: str, xlabel: str, ylabel: str, legend=None):
        title_size = self._plot_font_size + 1
        label_size = self._plot_font_size
        tick_size = max(8, self._plot_font_size - 1)
        ax.set_title(title, fontsize=title_size)
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        if legend is not None:
            legend_size = max(8, self._plot_font_size - 2)
            for text in legend.get_texts():
                text.set_fontsize(legend_size)

    def clear(self):
        self.fig.clear()
        self._im = None
        self._cb = None
        self.canvas.draw_idle()

    def plot_1d(self, data: np.ndarray, title: str, xlabel: str = "index",
                ylabel: str = None, x_values: np.ndarray = None,
                xlim=None, ylim=None, draw_style: str = "Lines + points",
                scale_mode: str = "Linear"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        line_style, marker_style = self._style_kwargs(draw_style)
        y = self._transform_series(np.asarray(data, dtype=float), scale_mode)
        x = np.arange(len(y), dtype=float) if x_values is None else np.asarray(x_values, dtype=float)
        ax.plot(x, y, linestyle=line_style, marker=marker_style, markersize=3)
        if ylabel is None:
            ylabel = self._series_label("value", scale_mode)
        else:
            ylabel = self._series_label(ylabel, scale_mode)
        self._style_axes(ax, title, xlabel, ylabel)
        self._configure_axis_ticks(ax, x_values=x, y_values=y)
        ax.grid(True, alpha=0.3)
        self._apply_xlim(ax, xlim)
        self._apply_ylim(ax, ylim)
        self.canvas.draw_idle()

    def plot_multiline(self, x: np.ndarray, ys: np.ndarray,
                        labels: list, title: str,
                        xlabel: str = "x", ylabel: str = "Intensity",
                        scale_mode: str = "Linear",
                        xlim=None, ylim=None, draw_style: str = "Lines + points"):
        """Plot multiple I vs x lines.
        x   : 1-D array, shape (nR,)
        ys  : 2-D array, shape (n_eta, nR) — one row per azimuth bin
        labels: list of str, length n_eta
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cmap = matplotlib.colormaps.get_cmap("tab20")
        n = len(ys)
        line_style, marker_style = self._style_kwargs(draw_style)
        for i, (row, lbl) in enumerate(zip(ys, labels)):
            y = self._transform_series(row.astype(float), scale_mode)
            color = cmap(i / max(n - 1, 1))
            ax.plot(
                x, y, label=lbl, color=color, linewidth=1.2,
                linestyle=line_style, marker=marker_style, markersize=3,
            )
        ax.grid(True, alpha=0.3)
        legend = None
        if n <= 20:
            legend = ax.legend(loc="best")
        self._style_axes(ax, title, xlabel, self._series_label(ylabel, scale_mode), legend=legend)
        self._configure_axis_ticks(ax, x_values=x, y_values=ys)
        self._apply_xlim(ax, xlim)
        self._apply_ylim(ax, ylim)
        self.canvas.draw_idle()

    @staticmethod
    def _style_kwargs(draw_style: str):
        if draw_style == "Lines":
            return "-", None
        if draw_style == "Points":
            return "None", "o"
        return "-", "o"

    @staticmethod
    def _series_label(base_label: str, scale_mode: str):
        if scale_mode == "Log":
            return f"log({base_label})"
        if scale_mode == "Sqrt":
            return f"sqrt({base_label})"
        return base_label

    @staticmethod
    def _transform_series(data: np.ndarray, scale_mode: str):
        if scale_mode == "Log":
            return np.where(data > 0, np.log10(data), np.nan)
        if scale_mode == "Sqrt":
            return np.where(data >= 0, np.sqrt(data), np.nan)
        return data

    def plot_2d(self, data: np.ndarray, title: str,
                cmap: str = "viridis", scale_mode: str = "Linear",
                clim_lo: float = None, clim_hi: float = None,
                xlabel: str = "column", ylabel: str = "row",
                x_coords: np.ndarray = None, y_coords: np.ndarray = None,
                xlim=None, eta_width: float = None,
                pack_eta: bool = False):
        """Plot 2-D array.  If x_coords/y_coords are given, use pcolormesh with
        physical axis values; otherwise fall back to imshow with bin indices."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        d = data.astype(float)
        vmin = clim_lo if clim_lo is not None else np.nanpercentile(d, 1)
        vmax = clim_hi if clim_hi is not None else np.nanpercentile(d, 99)
        if scale_mode == "Log":
            d = np.where(d > 0, d, np.nan)
            norm = mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-12))
        elif scale_mode == "Sqrt":
            d = np.where(d >= 0, d, np.nan)
            norm = mcolors.PowerNorm(gamma=0.5, vmin=max(vmin, 0.0), vmax=max(vmax, 0.0))
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        if x_coords is not None and y_coords is not None:
            # pcolormesh expects Z[row, col] = Z[y_idx, x_idx]
            # data shape is (nR, nEta); x_coords→R axis, y_coords→eta axis
            y_coords, unwrapped = _unwrap_eta(y_coords)
            if unwrapped:
                ylabel = f"{ylabel} (unwrapped)"
            self._im, _relabel = _draw_eta_bands(
                ax, x_coords, y_coords, d.T, eta_width,
                cmap=cmap, norm=norm, pack=pack_eta)
            if _relabel:
                ylabel = _relabel
        else:
            self._im = ax.imshow(
                d.T, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                interpolation="nearest",
            )
        # Colorbar intentionally omitted — the cmap dropdown + Log/Sqrt
        # scale combo tell the user what's what, and the strip stole a
        # noticeable chunk of horizontal space on narrow windows.
        self._style_axes(ax, title, xlabel, ylabel)
        self._apply_xlim(ax, xlim)
        self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────
# Tab
# ─────────────────────────────────────────────────────────────────────────

class ZarrViewerTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._root = None       # zarr root group
        self._store = None      # zarr ZipStore (kept open)
        self._cur_data = None   # currently selected array data
        self._cur_path = ""     # zarr key of selected array
        self._retamap = None    # REtaMap array (5, nR, nEta)
        self._retamap_corr = None  # REtaMap_corrected (5, nR, nEta) if present
        self._lam = None        # wavelength in Angstroms
        self._last_plot_signature = None
        self._last_plot_x = None
        self._last_dir = ""     # seeds the next "Open zarr.zip…" dialog

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # ── Top bar ──────────────────────────────────────────────────────
        top = QtWidgets.QHBoxLayout()

        self._btn_open = QtWidgets.QPushButton("Open zarr.zip…")
        self._btn_open.clicked.connect(self._on_open)
        top.addWidget(self._btn_open)

        self._lbl_path = QtWidgets.QLabel("No file loaded")
        self._lbl_path.setWordWrap(False)
        top.addWidget(self._lbl_path, stretch=1)

        main_layout.addLayout(top)

        # ── Splitter: tree | plot+controls ───────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left: tree
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        tree_label = QtWidgets.QLabel("Zarr contents")
        tree_label.setStyleSheet("font-weight: 600;")
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_tree_click)
        left_layout.addWidget(tree_label)
        left_layout.addWidget(self._tree)
        splitter.addWidget(left_widget)

        # Right: plot + controls + attrs
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = PlotCanvas()
        right_layout.addWidget(self._canvas, stretch=1)

        # Controls row
        ctrl_box = QtWidgets.QGroupBox("Display controls")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_box)

        ctrl_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        self._cmap_cb = QtWidgets.QComboBox()
        self._cmap_cb.addItems(CMAPS)
        self._cmap_cb.setCurrentText("viridis")
        self._cmap_cb.currentTextChanged.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._cmap_cb)

        ctrl_layout.addWidget(QtWidgets.QLabel("Scale:"))
        self._scale_cb = QtWidgets.QComboBox()
        self._scale_cb.addItems(["Linear", "Log", "Sqrt"])
        self._scale_cb.currentTextChanged.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._scale_cb)

        ctrl_layout.addWidget(QtWidgets.QLabel("Clim lo:"))
        self._clim_lo = QtWidgets.QLineEdit()
        self._clim_lo.setPlaceholderText("auto")
        self._clim_lo.setFixedWidth(70)
        self._clim_lo.returnPressed.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._clim_lo)

        ctrl_layout.addWidget(QtWidgets.QLabel("Clim hi:"))
        self._clim_hi = QtWidgets.QLineEdit()
        self._clim_hi.setPlaceholderText("auto")
        self._clim_hi.setFixedWidth(70)
        self._clim_hi.returnPressed.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._clim_hi)

        # Lock clim entries across array changes so successive images share
        # the same color range. When toggled on with empty fields, the current
        # image's 1st/99th percentiles are captured so the user can see (and
        # tweak) the locked range.
        self._clim_lock = QtWidgets.QCheckBox("Lock")
        self._clim_lock.setToolTip(
            "When checked, Clim lo/hi values are preserved when switching arrays\n"
            "or slices, so successive images share the same color scale."
        )
        self._clim_lock.toggled.connect(self._on_clim_lock_toggled)
        ctrl_layout.addWidget(self._clim_lock)

        ctrl_layout.addWidget(QtWidgets.QLabel("Xlim lo:"))
        self._xlim_lo = QtWidgets.QLineEdit()
        self._xlim_lo.setPlaceholderText("auto")
        self._xlim_lo.setFixedWidth(70)
        self._xlim_lo.returnPressed.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._xlim_lo)

        ctrl_layout.addWidget(QtWidgets.QLabel("Xlim hi:"))
        self._xlim_hi = QtWidgets.QLineEdit()
        self._xlim_hi.setPlaceholderText("auto")
        self._xlim_hi.setFixedWidth(70)
        self._xlim_hi.returnPressed.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._xlim_hi)

        # Lock the current zoom level (xlim + ylim) across array changes.
        # Re-captures the current view on every refresh so the user can pan/zoom
        # further with matplotlib's nav toolbar and have the new view stick too.
        self._zoom_lock = QtWidgets.QCheckBox("Lock zoom")
        self._zoom_lock.setToolTip(
            "When checked, the current x/y view limits are preserved across\n"
            "array, slice, and option changes. Pan/zoom with the matplotlib\n"
            "toolbar to update the locked view."
        )
        self._zoom_lock.toggled.connect(self._on_zoom_lock_toggled)
        ctrl_layout.addWidget(self._zoom_lock)
        self._zoom_lock_xlim = None
        self._zoom_lock_ylim = None

        ctrl_layout.addWidget(QtWidgets.QLabel("X axis:"))
        self._xaxis_cb = QtWidgets.QComboBox()
        self._xaxis_cb.addItems(["R bin", "2θ (deg)", "Q (Å⁻¹)", "d (Å)"])
        self._xaxis_cb.currentTextChanged.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._xaxis_cb)

        ctrl_layout.addWidget(QtWidgets.QLabel("REtaMap:"))
        self._rmap_cb = QtWidgets.QComboBox()
        self._rmap_cb.addItems(["REtaMap"])   # may gain "REtaMap_corrected" on load
        self._rmap_cb.currentTextChanged.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._rmap_cb)

        self._pack_eta = QtWidgets.QCheckBox("Pack eta")
        self._pack_eta.setToolTip(
            "Draw the eta rows adjacent to each other instead of at their "
            "true angular positions.\n\n"
            "A sparse caking -- four panels 10° wide but 90° apart -- leaves "
            "most of the plot empty, and the rows are too thin to read. This "
            "closes the gaps so the data fills the axis.\n\n"
            "The y-axis is then NOT linear in eta: each row is one caked "
            "wedge, labelled with its own angle, and the spacing between "
            "labels means nothing. The axis says so while this is on.\n\n"
            "No effect on a contiguous caking, where there are no gaps.")
        self._pack_eta.stateChanged.connect(self._refresh_plot)
        ctrl_layout.addWidget(self._pack_eta)

        btn_replot = QtWidgets.QPushButton("Apply")
        btn_replot.clicked.connect(self._refresh_plot)
        ctrl_layout.addWidget(btn_replot)

        ctrl_layout.addStretch()
        right_layout.addWidget(ctrl_box)

        # ── 1-D line plot controls (integration arrays) ───────────────────
        line_box = QtWidgets.QGroupBox("1-D plot (integration arrays)")
        line_layout = QtWidgets.QHBoxLayout(line_box)

        line_layout.addWidget(QtWidgets.QLabel("Plot mode:"))
        self._plot_mode_cb = QtWidgets.QComboBox()
        self._plot_mode_cb.addItems(["2-D map", "1-D lines"])
        self._plot_mode_cb.currentTextChanged.connect(self._refresh_plot)
        line_layout.addWidget(self._plot_mode_cb)

        line_layout.addWidget(QtWidgets.QLabel("1-D style:"))
        self._line_style_cb = QtWidgets.QComboBox()
        self._line_style_cb.addItems(["Lines", "Points", "Lines + points"])
        self._line_style_cb.setCurrentText("Lines + points")
        self._line_style_cb.currentTextChanged.connect(self._refresh_plot)
        line_layout.addWidget(self._line_style_cb)

        line_layout.addWidget(QtWidgets.QLabel("Azimuth bins:"))
        self._eta_edit = QtWidgets.QLineEdit()
        self._eta_edit.setPlaceholderText("e.g. 0,5,10  or  all")
        self._eta_edit.setMinimumWidth(160)
        self._eta_edit.setToolTip(
            "Comma-separated eta bin indices (0-based), or 'all' for every bin"
        )
        self._eta_edit.returnPressed.connect(self._refresh_plot)
        line_layout.addWidget(self._eta_edit, stretch=1)

        lbl_eta_hint = QtWidgets.QLabel("(eta bin indices or 'all')")
        lbl_eta_hint.setStyleSheet("color: #777777;")
        line_layout.addWidget(lbl_eta_hint)

        right_layout.addWidget(line_box)

        # Slice selector (for 3-D arrays)
        slice_box = QtWidgets.QGroupBox("Slice selector (3-D arrays)")
        slice_layout = QtWidgets.QHBoxLayout(slice_box)
        self._slice_lbl = QtWidgets.QLabel("axis-0 index:")
        self._slice_spin = QtWidgets.QSpinBox()
        self._slice_spin.setMinimum(1)
        self._slice_spin.setMaximum(1)
        self._slice_spin.setValue(1)
        self._slice_spin.valueChanged.connect(self._on_slice_change)
        self._slice_count_lbl = QtWidgets.QLabel("/ 1")
        slice_layout.addWidget(self._slice_lbl)
        slice_layout.addWidget(self._slice_spin)
        slice_layout.addWidget(self._slice_count_lbl)
        slice_layout.addStretch(1)
        right_layout.addWidget(slice_box)

        # Metadata / attributes panel — h5web-style: tree of attribute
        # names on the left, JSON-pretty-printed full value on the right.
        # Clickable for groups, arrays, and the root item.
        attr_box = QtWidgets.QGroupBox("Metadata & attributes")
        attr_layout = QtWidgets.QVBoxLayout(attr_box)

        self._attr_header = QtWidgets.QLabel("(select a group or array in the tree)")
        self._attr_header.setWordWrap(True)
        self._attr_header.setStyleSheet("color: #444; padding: 2px;")
        attr_layout.addWidget(self._attr_header)

        attr_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._attr_tree = QtWidgets.QTreeWidget()
        self._attr_tree.setColumnCount(2)
        self._attr_tree.setHeaderLabels(["Attribute", "Value"])
        self._attr_tree.setAlternatingRowColors(True)
        self._attr_tree.setUniformRowHeights(False)
        self._attr_tree.setRootIsDecorated(True)
        self._attr_tree.itemClicked.connect(self._on_attr_item_clicked)
        attr_split.addWidget(self._attr_tree)

        self._attr_text = QtWidgets.QTextEdit()
        self._attr_text.setReadOnly(True)
        self._attr_text.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self._attr_text.setPlaceholderText(
            "Select an attribute on the left to see its full value here."
        )
        attr_split.addWidget(self._attr_text)
        attr_split.setStretchFactor(0, 1)
        attr_split.setStretchFactor(1, 2)
        attr_split.setSizes([320, 600])
        attr_layout.addWidget(attr_split, stretch=1)

        attr_box.setMinimumHeight(220)
        right_layout.addWidget(attr_box, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([280, 1020])

    # ── File loading ─────────────────────────────────────────────────────

    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open zarr.zip", browse_start_dir(self._last_dir),
            "Zarr zip files (*.zip *.zarr.zip);;All files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
        try:
            self._store = zarr.ZipStore(path, mode="r")
            self._root = zarr.open(self._store, mode="r")
        except Exception as exc:
            self._attr_text.setPlainText(f"ERROR opening file:\n{exc}")
            return
        self._lbl_path.setText(path)
        self._last_dir = path
        # Cache REtaMap and wavelength for axis conversions
        self._retamap = None
        self._retamap_corr = None
        self._lam = None
        try:
            self._retamap = self._root["REtaMap"][...]   # (5, nR, nEta)
        except (KeyError, Exception):
            pass
        try:
            self._retamap_corr = self._root["REtaMap_corrected"][...]
        except (KeyError, Exception):
            pass
        try:
            self._lam = float(self._root["InstrumentParameters"]["Lam"][0])
        except (KeyError, Exception):
            pass
        # Populate REtaMap selector based on what's available
        self._rmap_cb.blockSignals(True)
        self._rmap_cb.clear()
        if self._retamap is not None:
            self._rmap_cb.addItem("REtaMap")
        if self._retamap_corr is not None:
            self._rmap_cb.addItem("REtaMap_corrected")
        if self._retamap_corr is not None:
            self._rmap_cb.setCurrentText("REtaMap_corrected")  # prefer corrected
        self._rmap_cb.blockSignals(False)
        self._populate_tree()
        self._cur_data = None
        self._last_plot_signature = None
        self._last_plot_x = None
        self._canvas.clear()
        self._attr_text.clear()
        self._attr_tree.clear()
        self._attr_header.setText("(select a group or array in the tree)")

    # ── Tree ─────────────────────────────────────────────────────────────

    def _populate_tree(self):
        self._tree.clear()
        root_item = QtWidgets.QTreeWidgetItem(self._tree, [os.path.basename(self._lbl_path.text())])
        root_item.setData(0, QtCore.Qt.UserRole, "root")
        font = root_item.font(0)
        font.setBold(True)
        root_item.setFont(0, font)
        _build_tree(self._root, root_item)
        self._tree.expandAll()

    def _on_tree_click(self, item: QtWidgets.QTreeWidgetItem, _col: int):
        role = item.data(0, QtCore.Qt.UserRole)
        if role == "root":
            # Root group: show its attrs but do not change the current plot.
            if self._root is not None:
                self._show_metadata(
                    path="/",
                    node=self._root,
                    role="root",
                )
            return
        if role == "group":
            key = _zarr_full_path(item)
            if not key or self._root is None:
                return
            try:
                grp = self._root[key]
            except KeyError:
                self._attr_header.setText(f"Key not found: {key}")
                self._attr_tree.clear()
                self._attr_text.clear()
                return
            self._show_metadata(path=key, node=grp, role="group")
            return
        if role != "array":
            return
        key = _zarr_full_path(item)
        if not key:
            return
        try:
            arr = self._root[key]
        except KeyError:
            self._attr_header.setText(f"Key not found: {key}")
            self._attr_tree.clear()
            self._attr_text.clear()
            return

        self._cur_path = key
        self._cur_data = arr[...]       # load into memory

        # Metadata panel (header + attrs tree + raw value pane)
        self._show_metadata(path=key, node=arr, role="array")

        # Reset clim fields unless the user has locked them across array
        # changes (so successive images can be compared on the same scale).
        if not self._clim_lock.isChecked():
            self._clim_lo.clear()
            self._clim_hi.clear()

        ndim = self._cur_data.ndim
        if ndim == 3:
            n = self._cur_data.shape[0]
            self._slice_spin.blockSignals(True)
            self._slice_spin.setMinimum(1)
            self._slice_spin.setMaximum(max(n, 1))
            self._slice_spin.setValue(1)
            self._slice_spin.blockSignals(False)
            self._slice_count_lbl.setText(f"/ {n}")
            self._slice_spin.setEnabled(True)
        else:
            self._slice_spin.blockSignals(True)
            self._slice_spin.setMinimum(1)
            self._slice_spin.setMaximum(1)
            self._slice_spin.setValue(1)
            self._slice_spin.blockSignals(False)
            self._slice_count_lbl.setText("/ 1")
            self._slice_spin.setEnabled(False)

        self._refresh_plot()

    # ── Metadata pane ────────────────────────────────────────────────────

    def _show_metadata(self, path: str, node, role: str) -> None:
        """Refresh the metadata header, attribute tree, and raw value pane
        for the selected tree node (root, group, or array)."""
        # Header summary
        if role == "array":
            shape = getattr(node, "shape", "?")
            dtype = getattr(node, "dtype", "?")
            chunks = getattr(node, "chunks", "?")
            self._attr_header.setText(
                f"<b>Array</b> &nbsp; {path} &nbsp;&nbsp; "
                f"shape={shape} &nbsp; dtype={dtype} &nbsp; chunks={chunks}"
            )
        elif role == "group":
            keys = list(node.keys()) if hasattr(node, "keys") else []
            self._attr_header.setText(
                f"<b>Group</b> &nbsp; {path} &nbsp;&nbsp; "
                f"{len(keys)} child{'ren' if len(keys) != 1 else ''}"
            )
        else:  # root
            keys = list(node.keys()) if hasattr(node, "keys") else []
            self._attr_header.setText(
                f"<b>Root group</b> &nbsp; / &nbsp;&nbsp; "
                f"{len(keys)} top-level entr{'ies' if len(keys) != 1 else 'y'}"
            )

        # Attribute tree
        attrs = dict(node.attrs) if getattr(node, "attrs", None) else {}
        self._populate_attrs_tree(attrs)

        # Default raw view: full attrs dump (JSON-pretty when possible)
        if not attrs:
            self._attr_text.setPlainText("(no attributes)")
        else:
            self._attr_text.setPlainText(self._json_dumps(attrs))

    def _populate_attrs_tree(self, attrs: dict) -> None:
        self._attr_tree.clear()
        if not attrs:
            placeholder = QtWidgets.QTreeWidgetItem(self._attr_tree, ["(no attributes)", ""])
            placeholder.setDisabled(True)
            return
        for k, v in attrs.items():
            self._add_attr_node(self._attr_tree.invisibleRootItem(), str(k), v)
        self._attr_tree.resizeColumnToContents(0)

    def _add_attr_node(self, parent: QtWidgets.QTreeWidgetItem, key: str, value) -> None:
        """Recursively add (key, value) into the attribute tree. Container
        values get expandable children; scalars get a one-line preview."""
        if isinstance(value, dict):
            item = QtWidgets.QTreeWidgetItem(parent, [key, f"{{...}} ({len(value)} keys)"])
            item.setData(0, QtCore.Qt.UserRole, value)
            for sub_k, sub_v in value.items():
                self._add_attr_node(item, str(sub_k), sub_v)
        elif isinstance(value, (list, tuple)):
            item = QtWidgets.QTreeWidgetItem(parent, [key, f"[...] ({len(value)} items)"])
            item.setData(0, QtCore.Qt.UserRole, list(value))
            for i, sub_v in enumerate(value):
                self._add_attr_node(item, f"[{i}]", sub_v)
        else:
            preview = self._scalar_preview(value)
            item = QtWidgets.QTreeWidgetItem(parent, [key, preview])
            item.setData(0, QtCore.Qt.UserRole, value)
            item.setToolTip(1, preview)

    @staticmethod
    def _scalar_preview(value) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        s = str(value)
        # Single-line preview only — show a hint when truncated.
        s_one = s.replace("\n", " ⏎ ")
        if len(s_one) > 140:
            return s_one[:137] + "…"
        return s_one

    @staticmethod
    def _json_dumps(value) -> str:
        """Pretty-print value as JSON; fall back to str() for non-JSON
        types (e.g. numpy scalars, bytes)."""
        try:
            return json.dumps(value, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def _on_attr_item_clicked(self, item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
        # Show the full JSON of whichever value (or sub-value) was clicked.
        value = item.data(0, QtCore.Qt.UserRole)
        # Build the dotted-path label by walking up the attribute tree.
        parts = []
        node = item
        while node is not None and node is not self._attr_tree.invisibleRootItem():
            parts.append(node.text(0))
            node = node.parent()
        parts.reverse()
        path = ".".join(parts) if parts else ""
        body = self._json_dumps(value) if value is not None else item.text(1)
        if path:
            self._attr_text.setPlainText(f"# {path}\n\n{body}")
        else:
            self._attr_text.setPlainText(body)

    def _on_slice_change(self, val: int):
        if self._cur_data is not None and self._cur_data.ndim == 3:
            self._refresh_plot()

    def _on_clim_lock_toggled(self, checked: bool) -> None:
        """Lock/unlock clim fields across array changes.

        On enable: if both fields are empty, pre-fill them with the current
        image's 1st/99th percentiles so the user can see (and edit) the
        locked range. On disable: leave fields as-is so the user can still
        adjust them; the next array selection will clear them naturally.
        """
        if not checked:
            return
        has_lo = bool(self._clim_lo.text().strip())
        has_hi = bool(self._clim_hi.text().strip())
        if has_lo and has_hi:
            return
        if self._cur_data is None:
            return
        try:
            d = np.asarray(self._cur_data[...], dtype=float)
        except Exception:
            return
        if d.size == 0 or not np.isfinite(d).any():
            return
        if not has_lo:
            self._clim_lo.setText(f"{float(np.nanpercentile(d, 1)):.6g}")
        if not has_hi:
            self._clim_hi.setText(f"{float(np.nanpercentile(d, 99)):.6g}")
        self._refresh_plot()

    def _get_current_value_label(self, ndim: int, slice_index: int = None) -> str:
        if self._cur_path == "REtaMap" or self._cur_path == "REtaMap_corrected":
            channel_labels = ["Radius (px)", "2θ (deg)", "Eta (deg)", "BinArea (px²)", "Q (Å⁻¹)"]
            if ndim == 3 and slice_index is not None and 0 <= slice_index < len(channel_labels):
                return channel_labels[slice_index]
        if ndim == 2 and self._plot_mode_cb.currentText() == "1-D lines":
            return "Intensity"
        if ndim == 1:
            attrs = {}
            try:
                attrs = dict(self._root[self._cur_path].attrs)
            except Exception:
                attrs = {}
            header = attrs.get("Header")
            units = attrs.get("Units")
            if isinstance(header, str) and isinstance(units, str) and units.strip():
                return f"{header} ({units})"
            if isinstance(header, str) and header.strip():
                return header
        return "value"

    # ── Plotting ─────────────────────────────────────────────────────────

    def _refresh_plot(self):
        """Wraps the dispatch with zoom-lock capture and re-apply.

        When the lock is on, the current view limits are captured BEFORE the
        plot is redrawn (so any user pan/zoom carries over) and reapplied
        AFTER (so the new image lands at the same view).
        """
        if self._cur_data is None:
            return
        if self._zoom_lock.isChecked():
            pre_x = self._canvas.get_current_xlim()
            pre_y = self._canvas.get_current_ylim()
            if pre_x is not None:
                self._zoom_lock_xlim = pre_x
            if pre_y is not None:
                self._zoom_lock_ylim = pre_y

        self._dispatch_plot()

        if self._zoom_lock.isChecked() and self._canvas.fig.axes:
            ax = self._canvas.fig.axes[0]
            if self._zoom_lock_xlim is not None:
                self._canvas._apply_xlim(ax, self._zoom_lock_xlim)
            if self._zoom_lock_ylim is not None:
                self._canvas._apply_ylim(ax, self._zoom_lock_ylim)
            self._canvas.canvas.draw_idle()

    def _on_zoom_lock_toggled(self, checked: bool) -> None:
        """On lock: capture the current zoom so the next refresh keeps it.
        On unlock: forget the saved view so future plots auto-fit again."""
        if checked:
            self._zoom_lock_xlim = self._canvas.get_current_xlim()
            self._zoom_lock_ylim = self._canvas.get_current_ylim()
        else:
            self._zoom_lock_xlim = None
            self._zoom_lock_ylim = None

    def _dispatch_plot(self):
        data = self._cur_data
        title = self._cur_path
        cmap = self._cmap_cb.currentText()
        scale_mode = self._scale_cb.currentText()

        def _parse_clim(widget):
            txt = widget.text().strip()
            try:
                return float(txt) if txt else None
            except ValueError:
                return None

        clim_lo = _parse_clim(self._clim_lo)
        clim_hi = _parse_clim(self._clim_hi)
        ylim = None
        if clim_lo is not None or clim_hi is not None:
            base_ylim = self._canvas.fig.axes[0].get_ylim() if self._canvas.fig.axes else None
            if base_ylim is None:
                if clim_lo is not None and clim_hi is not None:
                    ylim = (clim_lo, clim_hi)
            else:
                lo = base_ylim[0] if clim_lo is None else clim_lo
                hi = base_ylim[1] if clim_hi is None else clim_hi
                if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                    ylim = (lo, hi)

        def _resolve_xlim(auto_xlim):
            xlim_lo = _parse_clim(self._xlim_lo)
            xlim_hi = _parse_clim(self._xlim_hi)
            if xlim_lo is None and xlim_hi is None:
                return auto_xlim

            base = auto_xlim
            if base is None:
                base = self._canvas.get_current_xlim()

            if base is None:
                if xlim_lo is None or xlim_hi is None:
                    return auto_xlim
                lo, hi = xlim_lo, xlim_hi
            else:
                lo = base[0] if xlim_lo is None else xlim_lo
                hi = base[1] if xlim_hi is None else xlim_hi

            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                return (lo, hi)
            return auto_xlim

        draw_style = self._line_style_cb.currentText()

        ndim = data.ndim
        plot_mode = self._plot_mode_cb.currentText() if ndim in (2, 3) else ""
        plot_signature = (title, ndim, plot_mode)
        current_xlim = self._canvas.get_current_xlim() if plot_signature == self._last_plot_signature else None

        # Non-numeric (string / bytes): show values as text, skip plotting
        if data.dtype.kind in ('S', 'U', 'O'):
            self._canvas.clear()
            vals = data.flatten().tolist()
            decoded = []
            for v in vals:
                if isinstance(v, (bytes, np.bytes_)):
                    decoded.append(v.decode('utf-8', errors='replace'))
                else:
                    decoded.append(str(v))
            if len(decoded) == 1:
                self._attr_text.append(f"\nValue: {decoded[0]}")
            else:
                lines = "\n".join(f"  [{i}]: {v}" for i, v in enumerate(decoded))
                self._attr_text.append(f"\nValues ({len(decoded)}):\n{lines}")
            self._last_plot_signature = None
            self._last_plot_x = None
            return

        if ndim == 0:
            self._canvas.clear()
            self._attr_text.append(f"\nValue: {data[()]}")
            self._last_plot_signature = None
            self._last_plot_x = None
            return

        if ndim == 1:
            new_x = np.arange(data.shape[0], dtype=float)
            xlim = _resolve_xlim(self._remap_xlim(current_xlim, self._last_plot_x, new_x))
            self._canvas.plot_1d(
                data, title, ylabel=self._get_current_value_label(ndim),
                x_values=new_x, xlim=xlim, ylim=ylim, draw_style=draw_style,
                scale_mode=scale_mode,
            )
            self._last_plot_signature = plot_signature
            self._last_plot_x = new_x
            return

        if ndim == 2:
            # Check if 1-D line mode is requested for integration arrays
            if self._plot_mode_cb.currentText() == "1-D lines":
                x_coords, xlabel, eta_coords = self._get_integration_axes(data.shape)
                if x_coords is None:
                    # fall back to bin indices on x
                    x_coords = np.arange(data.shape[0])
                    xlabel = "R bin"
                eta_in_deg = eta_coords is not None
                if not eta_in_deg:
                    eta_coords = np.arange(data.shape[1])
                x_coords = np.asarray(x_coords, dtype=float)
                eta_bins = self._parse_eta_bins(data.shape[1])
                ys = data[:, eta_bins].T          # shape (n_selected, nR)
                if eta_in_deg:
                    labels = [f"η bin {b} ({eta_coords[b]:.2f}°)" if b < len(eta_coords)
                              else f"η bin {b}" for b in eta_bins]
                else:
                    labels = [f"η bin {b}" for b in eta_bins]
                xlim = _resolve_xlim(self._remap_xlim(current_xlim, self._last_plot_x, x_coords))
                self._canvas.plot_multiline(
                    x_coords, ys, labels, title,
                    xlabel=xlabel, ylabel=self._get_current_value_label(ndim),
                    scale_mode=scale_mode,
                    xlim=xlim, ylim=ylim, draw_style=draw_style,
                )
                self._last_plot_signature = plot_signature
                self._last_plot_x = x_coords
                return
            x_coords, xlabel, y_coords = self._get_integration_axes(data.shape)
            plot_x = None if x_coords is None else np.asarray(x_coords, dtype=float)
            if plot_x is None:
                plot_x = np.arange(data.shape[0], dtype=float)
            xlim = _resolve_xlim(self._remap_xlim(current_xlim, self._last_plot_x, plot_x))
            # When we have physical eta on y but R-bin indices on x, hand
            # plot_x in as x_coords so pcolormesh runs (otherwise the imshow
            # fallback would label row indices as degrees).
            x_coords_arg = x_coords if x_coords is not None else (
                plot_x if y_coords is not None else None
            )
            self._canvas.plot_2d(
                data, title, cmap=cmap, scale_mode=scale_mode,
                clim_lo=clim_lo, clim_hi=clim_hi,
                xlabel=xlabel, ylabel="Eta (deg)" if y_coords is not None else "Eta bin",
                x_coords=x_coords_arg, y_coords=y_coords, xlim=xlim,
                eta_width=self._eta_bin_width(),
                pack_eta=self._pack_eta.isChecked(),
            )
            self._last_plot_signature = plot_signature
            self._last_plot_x = plot_x
            return

        if ndim == 3:
            idx = max(0, self._slice_spin.value() - 1)
            slice_data = data[idx]
            slice_title = f"{title}  [slice {idx + 1}/{data.shape[0]}]"
            channel_label = self._get_current_value_label(ndim, idx)
            x_coords, xlabel, y_coords = self._get_integration_axes(slice_data.shape)
            if self._plot_mode_cb.currentText() == "1-D lines":
                plot_x = (np.asarray(x_coords, dtype=float)
                          if x_coords is not None
                          else np.arange(slice_data.shape[0], dtype=float))
                eta_bins = self._parse_eta_bins(slice_data.shape[1])
                ys = slice_data[:, eta_bins].T
                if y_coords is not None:
                    labels = [f"η bin {b} ({y_coords[b]:.2f}°)" if b < len(y_coords)
                              else f"η bin {b}" for b in eta_bins]
                else:
                    labels = [f"η bin {b}" for b in eta_bins]
                xlim = _resolve_xlim(self._remap_xlim(current_xlim, self._last_plot_x, plot_x))
                self._canvas.plot_multiline(
                    plot_x, ys, labels, slice_title,
                    xlabel=xlabel, ylabel=channel_label,
                    scale_mode=scale_mode,
                    xlim=xlim, ylim=ylim, draw_style=draw_style,
                )
                self._last_plot_signature = plot_signature
                self._last_plot_x = plot_x
                return
            plot_x = (np.asarray(x_coords, dtype=float)
                      if x_coords is not None
                      else np.arange(slice_data.shape[0], dtype=float))
            xlim = _resolve_xlim(self._remap_xlim(current_xlim, self._last_plot_x, plot_x))
            # Same trick as the 2-D branch: when eta-on-y is available but x
            # is R-bin indices, hand plot_x to pcolormesh so y is rendered in
            # degrees instead of falling back to imshow (which would ignore
            # y_coords and mislabel row indices as degrees).
            x_coords_arg = x_coords if x_coords is not None else (
                plot_x if y_coords is not None else None
            )
            self._canvas.plot_2d(
                slice_data, slice_title, cmap=cmap, scale_mode=scale_mode,
                clim_lo=clim_lo, clim_hi=clim_hi,
                xlabel=xlabel,
                ylabel="Eta (deg)" if y_coords is not None else "Eta bin",
                x_coords=x_coords_arg, y_coords=y_coords, xlim=xlim,
                eta_width=self._eta_bin_width(),
                pack_eta=self._pack_eta.isChecked(),
            )
            self._last_plot_signature = plot_signature
            self._last_plot_x = plot_x
            return

        # Higher-D: flatten last two dims and show as 2-D
        flat = data.reshape(-1, data.shape[-1])
        new_x = np.arange(flat.shape[0], dtype=float)
        xlim = _resolve_xlim(self._remap_xlim(current_xlim, self._last_plot_x, new_x))
        self._canvas.plot_2d(
            flat, f"{title} (flattened)", cmap=cmap, scale_mode=scale_mode,
            clim_lo=clim_lo, clim_hi=clim_hi,
            xlim=xlim,
        )
        self._last_plot_signature = plot_signature
        self._last_plot_x = new_x

    # ── Axis coordinate helpers ──────────────────────────────────────────

    def _parse_eta_bins(self, n_eta: int) -> list:
        """Parse the azimuth bin field. Returns a sorted list of valid indices."""
        txt = self._eta_edit.text().strip().lower()
        if not txt or txt == "all":
            return list(range(n_eta))
        bins = []
        for tok in txt.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx = int(tok)
                if 0 <= idx < n_eta:
                    bins.append(idx)
            except ValueError:
                pass
        return sorted(set(bins)) if bins else list(range(n_eta))

    @staticmethod
    def _remap_xlim(current_xlim, old_x, new_x):
        """Map a visible x-range from one x-coordinate system to another.

        This preserves the same radial-bin window when switching between
        R-bin, 2theta, Q, d, or REtaMap sources.
        """
        if current_xlim is None:
            return None
        if old_x is None or new_x is None:
            return current_xlim

        old_x = np.asarray(old_x, dtype=float)
        new_x = np.asarray(new_x, dtype=float)
        if old_x.ndim != 1 or new_x.ndim != 1 or len(old_x) != len(new_x) or len(old_x) < 2:
            return current_xlim

        valid = np.isfinite(old_x) & np.isfinite(new_x)
        if valid.sum() < 2:
            return current_xlim

        idx = np.arange(len(old_x), dtype=float)[valid]
        old_valid = old_x[valid]
        new_valid = new_x[valid]

        order_old = np.argsort(old_valid)
        old_sorted = old_valid[order_old]
        idx_sorted = idx[order_old]

        lo_idx = np.interp(current_xlim[0], old_sorted, idx_sorted, left=idx_sorted[0], right=idx_sorted[-1])
        hi_idx = np.interp(current_xlim[1], old_sorted, idx_sorted, left=idx_sorted[0], right=idx_sorted[-1])

        order_idx = np.argsort(idx)
        idx_for_new = idx[order_idx]
        new_sorted = new_valid[order_idx]
        new_lo = np.interp(lo_idx, idx_for_new, new_sorted, left=new_sorted[0], right=new_sorted[-1])
        new_hi = np.interp(hi_idx, idx_for_new, new_sorted, left=new_sorted[0], right=new_sorted[-1])
        return (new_lo, new_hi)

    def _eta_bin_width(self):
        """``EtaBinSize`` this archive was caked with, or None.

        Read from the archive's own provenance rather than inferred from
        the axis: the spacing between eta columns is NOT the bin width
        when the caking is sparse, and conflating the two is what drew
        four 10-degree bands as four 90-degree ones. A combined archive
        carries one entry per panel; they are normally equal, and the
        widest is the safe choice if they are not, since drawing a band
        too tall is the failure being fixed.
        """
        try:
            hist = (self._root.attrs.get("provenance_history") or []
                    if self._root is not None else [])
        except Exception:
            return None
        widths = []
        for entry in hist:
            try:
                w = (entry.get("cake_params") or {}).get("EtaBinSize")
            except AttributeError:
                continue
            if w:
                widths.append(float(w))
        return max(widths) if widths else None

    def _get_integration_axes(self, shape):
        """Return (x_coords, xlabel, y_coords) for an integration result array.

        x_coords and y_coords are None when REtaMap is unavailable or the shape
        doesn't match, falling back to bin indices.
        """
        xax = self._xaxis_cb.currentText()

        # Pick the active REtaMap
        rmap_choice = self._rmap_cb.currentText()
        if rmap_choice == "REtaMap_corrected" and self._retamap_corr is not None:
            rmap = self._retamap_corr
        elif self._retamap is not None:
            rmap = self._retamap
        else:
            rmap = None

        if rmap is None or len(shape) != 2:
            return None, "R bin", None

        nR, nEta = rmap.shape[1], rmap.shape[2]
        if shape[0] != nR or shape[1] != nEta:
            return None, "R bin", None

        # y-axis: mean eta angle across R bins → shape (nEta,)
        eta_coords = rmap[2].mean(axis=0)   # channel 2 = Eta

        if xax == "R bin":
            return None, "R bin", eta_coords

        if xax == "2θ (deg)":
            x = rmap[1].mean(axis=1)        # channel 1 = 2Theta, mean over eta
            return x, "2θ (deg)", eta_coords

        if xax == "Q (Å⁻¹)":
            x = rmap[4].mean(axis=1)        # channel 4 = Q
            return x, "Q (Å⁻¹)", eta_coords

        if xax == "d (Å)":
            q = rmap[4].mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(q > 0, 2.0 * np.pi / q, np.nan)
            return x, "d (Å)", eta_coords

        return None, "R bin", None

"""The MIDAS lab frame, pinned end to end.

Every one of these is a link in a single chain: the array the backend
integrates, the η it assigns to each pixel, the overlays that annotate η, the
screen the frame is painted on, and the azimuth the polarization correction is
applied at.  A sign error anywhere in that chain is invisible — the picture
still looks like a diffraction pattern and the profile still looks like a
profile — so the chain is asserted here rather than left to inspection.

The reference is the MIDAS lab frame viewed *looking downstream* (beam into the
screen, ⊗ at the beam centre)::

        +Y_Lab (+Z_MIDAS)
             η = 0°
               ↑
    η = −90°   |
  +X_Lab  ←────⊗────→  η = +90°
 (+Y_MIDAS)    |      (+Z_Lab = +X_MIDAS = beam, into the screen)
               ↓
            η = 180°

Two facts anchor it.  η is ``atan2(-Yc, Zc)``, measured from VERTICAL, so η = 0
is up and η = ±90 is horizontal.  And ``lattice_to_phys`` sets
``Yc = (BC_y - Y_pix) * pxY``, so the array's COLUMN axis runs along −Y_MIDAS:
a higher column index is screen-right.
"""
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


BC = 100.0


def _spec():
    from types import SimpleNamespace
    from midas_gui.helpers import _build_spec
    return _build_spec(SimpleNamespace(
        Lsd=200000.0, BC_y=BC, BC_z=BC, tx=0.0, ty=0.0, tz=0.0, distortion={},
        pxY=200.0, pxZ=200.0, NrPixelsY=200, NrPixelsZ=200,
        wavelength_A=0.1729), r_bin=2.0, eta_bin=45.0)


# ── 1. the backend's own η ───────────────────────────────────────────────────

@pytest.mark.parametrize("col, row, eta", [
    (BC + 20, BC,      +90.0),   # screen-right
    (BC - 20, BC,      -90.0),   # screen-left  (+Y_MIDAS = +X_Lab)
    (BC,      BC + 20,   0.0),   # screen-up    (+Z_MIDAS = +Y_Lab)
    (BC,      BC - 20, 180.0),   # screen-down
])
def test_backend_eta_at_the_four_cardinals(col, row, eta):
    torch = pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    from midas_integrate_v2.forward.pixels import pixel_to_REta_from_spec

    out = pixel_to_REta_from_spec(torch.tensor(col, dtype=torch.float64),
                                  torch.tensor(row, dtype=torch.float64), _spec())
    got = float(out.eta_deg)
    assert math.isclose(abs(got) if eta == 180.0 else got, eta, abs_tol=1e-6)


# ── 2. the screen the frame is painted on ────────────────────────────────────

def test_viewer_paints_rows_up_and_columns_right(app):
    """``disp = d.T`` puts the array's column index on pyqtgraph's x axis and
    ``invertY(False)`` puts row 0 at the bottom, so a higher row index is
    screen-UP and a higher column index is screen-RIGHT — matching η = 0 up and
    η = +90 right above."""
    from midas_gui.widgets import ImageViewer

    v = ImageViewer()
    frame = np.zeros((16, 24), dtype=np.float32)   # (rows = Z, cols = Y)
    frame[5, 9] = 1.0
    v.set_image(frame)

    shown = v._iv.imageItem.image
    assert shown.shape == (24, 16), "x axis must be the COLUMN axis"
    # Compared by position, not value: the viewer's log scaling is on by
    # default, so the bright pixel is the brightest one rather than a 1.0.
    assert np.unravel_index(int(np.argmax(shown)), shown.shape) == (9, 5), \
        "array (row, col) must land at pyqtgraph (x, y) = (col, row)"
    assert v._iv.getView().getViewBox().yInverted() is False, "row 0 at the bottom"
    assert v._iv.getView().getViewBox().xInverted() is False, "col 0 at the left"


# ── 3. the overlays that annotate η ──────────────────────────────────────────

def _text_items(items):
    import pyqtgraph as pg
    out = []
    for it in items:
        if isinstance(it, pg.TextItem):
            try:
                out.append((it.textItem.toPlainText(), it.pos().x(), it.pos().y()))
            except Exception:
                pass
    return out


def test_lab_frame_compass_agrees_with_the_backend(app):
    """The compass is the overlay a user checks ImTransOpt against, so its η
    ticks have to land where ``pixel_to_REta`` actually puts them."""
    from midas_gui.widgets import ImageViewer, build_lab_frame_axes_items

    v = ImageViewer()
    v.set_image(np.zeros((200, 200), dtype=np.float32))
    labels = _text_items(build_lab_frame_axes_items(v._iv, (200, 200), BC, BC))
    assert labels, "compass drew no labels"

    def pos(needle):
        hits = [(x, y) for txt, x, y in labels if needle in txt]
        assert len(hits) == 1, f"{needle!r} matched {len(hits)} labels"
        return hits[0]

    x, y = pos("η=+90")
    assert x > BC and abs(y - BC) < 1e-6, "η=+90 must sit screen-RIGHT"
    x, y = pos("η=180")
    assert y < BC and abs(x - BC) < 1e-6, "η=180 must sit screen-DOWN"
    # η=0 / η=−90 are folded into the +Y_Lab / +X_Lab axis labels.
    x, y = pos("+YLab")
    assert y > BC, "+Y_Lab (η=0) must point screen-UP"
    x, y = pos("+XLab")
    assert x < BC, "+X_Lab (η=−90, +Y_MIDAS) must point screen-LEFT"


def test_bin_grid_spokes_agree_with_the_backend(app):
    """'Show bin grid' draws η spokes as bc + r·(sin η, cos η).

    The two mistakes worth catching each map the spoke set onto ITSELF for the
    obvious choice of parameters, which is how a toothless version of this test
    gets written:

      * swapping sin for cos sends η to 90° − η, so any bin size dividing 90
        (45°, 30°, the four cardinals) draws a set that is invariant under it;
      * flipping the sign of the Y term sends η to −η, so any η range that is
        symmetric about 0 — including the natural −180…180 — is invariant too.

    So: 20° bins over the asymmetric range −10…170. The η = 30° spoke would have
    to land on η = 60° under the swap and η = −30° under the flip, and neither
    is in that set; likewise η = 90° would need η = 0° or η = −90°.
    """
    import pyqtgraph as pg
    from midas_gui.widgets import ImageViewer
    from midas_gui.helpers import draw_polar_bin_overlay

    v = ImageViewer()
    v.set_image(np.zeros((200, 200), dtype=np.float32))
    items = []
    draw_polar_bin_overlay(
        v, items, bc_y=BC, bc_z=BC, r_min=10.0, r_max=40.0, r_bin=10.0,
        e_bin=20.0, show_grid=True, eta_min=-10.0, eta_max=170.0)
    try:
        ends = [(float(it.xData[-1]), float(it.yData[-1])) for it in items
                if isinstance(it, pg.PlotDataItem) and it.xData is not None
                and len(it.xData) == 2]
        assert ends, "no spokes drawn"

        def has(x, y):
            return any(abs(ex - x) < 1e-6 and abs(ey - y) < 1e-6
                       for ex, ey in ends)

        # η = 30°: mostly up, a little right.
        assert has(BC + 40.0 * math.sin(math.radians(30.0)),
                   BC + 40.0 * math.cos(math.radians(30.0))), \
            "η=30 spoke is not at bc + r·(sin η, cos η)"
        # η = +90°: straight screen-right, the cardinal the compass agrees on.
        assert has(BC + 40.0, BC), "η=+90 spoke must point screen-right"
    finally:
        for it in items:
            v._iv.removeItem(it)


# ── 4. the azimuth the polarization correction is applied at ─────────────────

def test_polarization_plane_defaults_to_the_horizontal_ring_plane(app):
    """η is measured from vertical, so the storage ring's X_Lab–Z_Lab plane is
    η = 90.  A default of 0 puts the correction a quarter turn away, which adds
    the cos(2η) modulation it exists to remove."""
    from midas_gui.constants import POL_PLANE_HORIZONTAL_ETA_DEG
    from midas_gui.widgets import CorrectionFlagsWidget
    from midas_gui.batch_cli import _build_arg_parser

    assert POL_PLANE_HORIZONTAL_ETA_DEG == 90.0

    w = CorrectionFlagsWidget()
    assert w.pol_plane.value() == pytest.approx(90.0)
    w.polar_check.setChecked(True)
    pol, _sa = w.build_corrections()
    assert float(pol.pol_plane_eta_deg) == pytest.approx(90.0)

    ns = _build_arg_parser().parse_args(
        ["--calib-file", "x", "--source-type", "hdf5", "--out-dir", "/tmp/x"])
    assert ns.pol_plane == pytest.approx(90.0)


def test_a_saved_project_keeps_its_own_plane(app):
    """The new default must not silently rewrite a value a user deliberately
    saved — including the old 0.0, which the run log calls out instead."""
    from midas_gui.widgets import CorrectionFlagsWidget

    w = CorrectionFlagsWidget()
    w.set_state({"pol_plane": 0.0, "pol_fraction": 0.5})
    assert w.pol_plane.value() == pytest.approx(0.0)
    assert w.pol_fraction.value() == pytest.approx(0.5)


def test_the_bin_grid_arcs_stop_at_the_eta_limits(app):
    """Reported: "caking scheme plotting functionality still plots the full
    rings when I am integrating limited eta range."

    The grid depicts the region being integrated, so an arc has to stop
    where the region does. It used to sweep a whole turn regardless, which
    drew a grid over the entire detector for a run binning a 110° wedge.

    Arcs are the items with many points; spokes have exactly two. Checked
    in the same bc + r*(sin η, cos η) convention the spokes use — the old
    full-circle path swept (cos, sin), which traces the same circle and so
    was harmless only while the sweep was complete.
    """
    import pyqtgraph as pg
    from midas_gui.widgets import ImageViewer
    from midas_gui.helpers import draw_polar_bin_overlay

    eta_lo, eta_hi, r_max = -100.0, 10.0, 40.0
    v = ImageViewer()
    v.set_image(np.zeros((200, 200), dtype=np.float32))
    items = []
    draw_polar_bin_overlay(
        v, items, bc_y=BC, bc_z=BC, r_min=10.0, r_max=r_max, r_bin=10.0,
        e_bin=20.0, show_grid=True, eta_min=eta_lo, eta_max=eta_hi)
    try:
        arcs = [it for it in items if isinstance(it, pg.PlotDataItem)
                and it.xData is not None and len(it.xData) > 2]
        assert arcs, "no arcs drawn"
        for arc in arcs:
            r = np.hypot(np.asarray(arc.xData) - BC,
                         np.asarray(arc.yData) - BC)
            eta = np.degrees(np.arctan2(np.asarray(arc.xData) - BC,
                                        np.asarray(arc.yData) - BC))
            assert eta.min() >= eta_lo - 1e-6, f"arc runs past η min: {eta.min()}"
            assert eta.max() <= eta_hi + 1e-6, f"arc runs past η max: {eta.max()}"
            # and it really does span the wedge, not collapse to a point
            assert eta.max() - eta.min() > 100.0
            assert np.ptp(r) < 1e-6, "an arc must stay at one radius"
    finally:
        for it in items:
            v._iv.removeItem(it)


def test_a_full_azimuth_still_draws_a_closed_ring(app):
    """The default must not regress: with no η limit the arcs are circles."""
    import pyqtgraph as pg
    from midas_gui.widgets import ImageViewer
    from midas_gui.helpers import draw_polar_bin_overlay

    v = ImageViewer()
    v.set_image(np.zeros((200, 200), dtype=np.float32))
    items = []
    draw_polar_bin_overlay(
        v, items, bc_y=BC, bc_z=BC, r_min=10.0, r_max=40.0, r_bin=10.0,
        e_bin=20.0, show_grid=True, eta_min=-180.0, eta_max=180.0)
    try:
        arcs = [it for it in items if isinstance(it, pg.PlotDataItem)
                and it.xData is not None and len(it.xData) > 2]
        assert arcs
        for arc in arcs:
            x, y = np.asarray(arc.xData), np.asarray(arc.yData)
            assert abs(x[0] - x[-1]) < 1e-6 and abs(y[0] - y[-1]) < 1e-6, \
                "a full-azimuth ring must close on itself"
    finally:
        for it in items:
            v._iv.removeItem(it)

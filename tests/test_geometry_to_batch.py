"""Rough geometry can go straight from the Data Viewer to Batch Integrate.

Asked for: "There has to be a way to send the instrument parameters
directly to batch integrate, for cases where i do not want to calibrate
and just use the rough instrument parameters from data viewer."

Plenty of runs only need a nominal Lsd and a picked beam centre. The only
routes to the integrator were to save a paramstest and load it back in the
other tab, or to run a calibration nobody wanted.

Deliberately a separate signal from the existing "Send ->", because the
payloads differ in kind: Calibrate wants SEED fields (lambda, pixel, Lsd,
BC) while the integrator needs the whole forward model -- tilts,
distortion, detector size and the Transforms checkboxes. Reusing
pushGeometry would have sent the integrator a geometry missing exactly the
parts it integrates with.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import pytest

pytestmark = pytest.mark.forked

#: Everything the integration-spec builders read off a calibration.
FORWARD_MODEL = ("wavelength_A", "Lsd", "BC_y", "BC_z", "tx", "ty", "tz",
                 "pxY", "pxZ", "NrPixelsY", "NrPixelsZ", "distortion",
                 "im_trans")


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def _no_modals(monkeypatch):
    """A modal QMessageBox blocks forever under a headless test runner, so
    the whole file hangs with no output rather than failing -- which is
    exactly what happened while writing it. Stubbed for every test here,
    and the one that cares reads `warned` instead."""
    from PyQt5 import QtWidgets
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2] if len(a) > 2 else "")))
    return warned


@pytest.fixture
def view(app, tmp_path):
    """A Data Viewer with a frame actually loaded.

    The detector size in the payload comes from the image, so a tab that
    has not loaded one has no geometry to send -- the constructor's default
    file is not guaranteed to be in place synchronously, and relying on it
    made these tests exercise the refusal path instead of the feature.
    """
    import numpy as np
    h5py = pytest.importorskip("h5py")
    from midas_gui.tab_view import DataViewerTab
    f = tmp_path / "frame_00001.h5"
    with h5py.File(str(f), "w") as fh:
        fh.create_dataset("exchange/data",
                          data=np.ones((2, 48, 64), np.float32))
    v = DataViewerTab()
    v._loader.set_path(str(f), dataset="exchange/data")
    app.processEvents()
    if v._cur is None:
        # The card's image provider is `lambda: self._cur` (tab_view.py:403),
        # so setting it drives the real production path. Done explicitly
        # because the loader does not populate _cur synchronously under a
        # headless runner, and the subject here is the hand-off, not loading.
        v._cur = np.ones((48, 64), np.float32)
    assert v._geom_card._export_geom(), "fixture must have a usable geometry"
    return v


def test_the_button_sends_the_whole_forward_model(view, app):
    got = []
    view.pushGeometryToBatch.connect(got.append)
    view._geom_card._lsd.setValue(3298.309)
    view._geom_card._tx.setValue(210.0)
    view._geom_card._to_batch_btn.click()
    app.processEvents()

    assert len(got) == 1, "the button should emit exactly once"
    g = got[0]
    missing = [k for k in FORWARD_MODEL if k not in g]
    assert not missing, f"the integrator needs these and they are absent: {missing}"
    assert g["Lsd"] == pytest.approx(3298309.0)     # mm in the box, um in the dict
    assert g["tx"] == pytest.approx(210.0)
    # Detector size comes from the image, not from any typed field.
    assert (g["NrPixelsY"], g["NrPixelsZ"]) == (64, 48)


def test_the_seed_hand_off_is_not_what_gets_sent(view, app):
    """"Send ->" targets Calibrate and carries seed fields only. If this
    button reused it, the integrator would silently lose the tilts and
    distortion it integrates with."""
    seed, full = [], []
    view.pushGeometry.connect(seed.append)
    view.pushGeometryToBatch.connect(full.append)
    view._geom_card._to_batch_btn.click()
    app.processEvents()
    assert full and not seed, "the Calibrate hand-off must not also fire"
    assert "distortion" in full[0] and "NrPixelsY" in full[0]


def test_what_is_sent_is_what_the_integrator_can_resolve(view, app):
    """The payload has to survive the conversion Batch Integrate actually
    performs on it, not merely look complete."""
    from midas_gui import project
    from midas_gui.helpers import resolve_calibration_fields
    got = []
    view.pushGeometryToBatch.connect(got.append)
    view._geom_card._lsd.setValue(2390.0)
    view._geom_card._to_batch_btn.click()
    app.processEvents()

    ns = project.calibration_namespace(got[0])
    fields, note = resolve_calibration_fields(ns, False, "")
    assert fields, f"the integrator rejected it: {note}"
    assert float(fields["Lsd"]) == pytest.approx(2390000.0)


def test_it_refuses_rather_than_sending_a_geometry_with_no_detector(
        view, app, monkeypatch, _no_modals):
    """Without an image there is no NrPixels, and the spec builders would
    fail downstream with something far less obvious than saying so here."""
    warned = _no_modals
    monkeypatch.setattr(view._geom_card, "_export_geom", lambda: None)
    got = []
    view.pushGeometryToBatch.connect(got.append)
    view._geom_card._to_batch_btn.click()
    app.processEvents()
    assert not got, "nothing should be sent"
    assert warned and "detector size" in warned[0]

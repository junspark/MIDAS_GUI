"""Qt-level wiring test for the Calibrate tab's manual d-spacing ring-picking
fit mode (non-crystalline calibrants like AgBH) — exercises the
PICK_DSPACING pick mode, the live per-ring summary, the Fit button gating,
and that a fitted result flows through the same _on_done pipeline as a
regular midas-calibrate-v2 result (ring overlay uses the new `_d_list`
branch of `_predict_ring_radii`, not the CeO2 fallback).
"""
from types import SimpleNamespace

import pytest
from PyQt5 import QtCore, QtWidgets


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeManualDspacingCalibWorker(QtCore.QObject):
    """No-op-thread fake: finishes on the next event-loop tick with a fixed
    known result, instead of actually running least_squares."""
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, picks, wavelength_A, pxY, pxZ, seed, NY, NZ,
                 material_name, d_list, parent=None):
        super().__init__(parent)
        self._wavelength_A = wavelength_A
        self._pxY = pxY
        self._pxZ = pxZ
        self._NY = NY
        self._NZ = NZ
        self._material_name = material_name
        self._d_list = d_list

    def start(self):
        QtCore.QTimer.singleShot(0, self._finish)

    def isRunning(self) -> bool:
        return False

    def requestInterruption(self):
        pass

    def _finish(self):
        result = SimpleNamespace(
            Lsd=300000.0, BC_y=512.0, BC_z=498.0, tx=0.0, ty=0.0, tz=0.0,
            distortion={}, pxY=self._pxY, pxZ=self._pxZ or self._pxY,
            NrPixelsY=self._NY, NrPixelsZ=self._NZ,
            wavelength_A=self._wavelength_A, post_residual_strain_uE=None,
            _calibrant_name=self._material_name, _d_list=list(self._d_list))
        self.finished.emit(result)


class _FakeIntegrationWorker(QtCore.QObject):
    """No-op-thread fake for the real ``IntegrationWorker`` that ``_on_done``
    kicks off automatically after any successful fit (manual or not) — the
    Calibrate tab auto-loads ``DEFAULT_CALIBRANT_TIF`` at construction, so
    ``_calib_image()`` is never None and a real background thread (importing
    torch/midas_calibrate_v2) would otherwise be left running past test/
    interpreter teardown."""
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, result, image, dark, im_trans, r_bin, eta_bin, mask=None,
                 parent=None, bright=None, background=None, bright_mode="divide",
                 weighted=True):
        super().__init__(parent)
        self._result = result

    def start(self):
        QtCore.QTimer.singleShot(0, self._finish)

    def isRunning(self) -> bool:
        return False

    def requestInterruption(self):
        pass

    def _finish(self):
        import numpy as np
        r_axis = np.linspace(0, 100, 50)
        profile = np.ones_like(r_axis)
        eta_axis = np.linspace(-180, 180, 36)
        cake = np.ones((len(eta_axis), len(r_axis)))
        self.finished.emit({"r_axis_px": r_axis, "profile": profile,
                            "wavelength_A": self._result.wavelength_A,
                            "lsd_um": self._result.Lsd, "px_um": self._result.pxY,
                            "cake_2d": cake, "eta_axis_deg": eta_axis})


def test_manual_fit_pick_summary_and_button_gating(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()

    idx = tab._dsp_material.findText("AgBH (silver behenate)")
    assert idx >= 0
    tab._dsp_material.setCurrentIndex(idx)
    assert not tab._dsp_custom_ed.isVisible()
    assert not tab._dsp_fit_btn.isEnabled()

    view = tab._img_view
    view._dsp_ring_spin.setValue(1)
    for x, y in [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)

    # 3 points on a single ring already satisfies the ">=3 total" gate.
    assert tab._dsp_fit_btn.isEnabled()
    assert "Ring 1" in tab._dsp_summary.text()
    assert "58.380" in tab._dsp_summary.text()

    view._dsp_ring_spin.setValue(2)
    for x, y in [(50.0, 0.0), (0.0, 50.0), (-50.0, 0.0)]:
        view._add_dspacing_point(512.0 + x, 498.0 + y)
    assert "Ring 2" in tab._dsp_summary.text()
    assert "29.190" in tab._dsp_summary.text()

    # A ring index beyond the material's d-spacing count is flagged, not silently dropped.
    view._dsp_ring_spin.setValue(20)
    view._add_dspacing_point(700.0, 700.0)
    assert "invalid" in tab._dsp_summary.text()


def test_manual_fit_result_flows_through_on_done_with_correct_rings(app, monkeypatch):
    import midas_gui.tab_calibrate as tab_calibrate_mod
    from midas_gui.helpers import simulate_rings_from_dspacings
    monkeypatch.setattr(tab_calibrate_mod, "ManualDspacingCalibWorker",
                        _FakeManualDspacingCalibWorker)
    monkeypatch.setattr(tab_calibrate_mod, "IntegrationWorker",
                        _FakeIntegrationWorker)
    tab = tab_calibrate_mod.CalibrationTab()

    idx = tab._dsp_material.findText("AgBH (silver behenate)")
    tab._dsp_material.setCurrentIndex(idx)
    view = tab._img_view
    for ring_idx, r in ((1, 100.0), (2, 60.0)):
        view._dsp_ring_spin.setValue(ring_idx)
        for x, y in [(r, 0.0), (0.0, r), (-r, 0.0)]:
            view._add_dspacing_point(512.0 + x, 498.0 + y)

    assert tab._dsp_fit_btn.isEnabled()
    tab._run_manual_fit()
    assert tab._worker is not None
    assert not tab._dsp_fit_btn.isEnabled()   # disabled while running

    loop = QtCore.QEventLoop()
    tab.calibrationDone.connect(lambda *_: loop.quit())
    QtCore.QTimer.singleShot(2000, loop.quit)   # safety timeout
    loop.exec_()

    assert tab._result is not None
    assert tab._result._calibrant_name == "AgBH (silver behenate)"
    assert tab._save_json_btn.isEnabled()
    assert tab._save_ps_btn.isEnabled()
    assert tab._dsp_fit_btn.isEnabled()   # re-enabled after the fake worker finishes

    d_list = sorted(tab._result._d_list, reverse=True)
    expected_radii = sorted({round(r["radius_px"], 3) for r in simulate_rings_from_dspacings(
        d_list, tab._result.wavelength_A, tab._result.Lsd, tab._result.pxY)})
    from midas_gui.helpers import _predict_ring_radii
    assert _predict_ring_radii(tab._result) == expected_radii

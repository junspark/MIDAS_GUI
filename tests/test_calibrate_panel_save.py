"""CalibrationTab._save_json must (re)write a co-located _panelshifts.txt
sidecar for a multi-panel result, instead of leaving the saved
calibration.json's panel_shifts_path pointing at whatever transient file
was live when Fit finished (see .context/DECISIONS.md for the bug this
fixes).
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _panel_result(panel_shifts_path):
    ns = SimpleNamespace(
        Lsd=200000.0, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={"iso_R2": 0.1}, pxY=200.0, pxZ=200.0,
        NrPixelsY=2048, NrPixelsZ=2048, wavelength_A=0.1729,
        panel_layout={"n_y": 1, "n_z": 2, "sy": 100, "sz": 100},
        panel_shifts_path=panel_shifts_path,
    )
    ns._calibrant_name = "CeO2"
    ns._panel_unpacked = {
        "panel_delta_yz": np.array([[0.1, -0.2], [0.3, 0.4]]),
        "panel_delta_theta": np.array([0.001, -0.002]),
        "panel_delta_lsd": np.array([1.5, -1.5]),
        "panel_delta_p2": np.array([0.0, 0.0]),
    }
    return ns


def test_save_json_rewrites_colocated_panel_shifts_sidecar(app, tmp_path, monkeypatch):
    from PyQt5 import QtWidgets
    from midas_gui.tab_calibrate import CalibrationTab

    # Simulates the ephemeral tempfile calib._attach_panel_result falls back
    # to when no Output folder was set during Fit.
    tempfile_shifts = tmp_path / "elsewhere" / "some_tmp_panel_shifts.txt"
    tempfile_shifts.parent.mkdir()
    tempfile_shifts.write_text("stale")

    tab = CalibrationTab()
    tab._result = _panel_result(str(tempfile_shifts))

    out_path = tmp_path / "calibration.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                         staticmethod(lambda *a, **k: (str(out_path), "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                         staticmethod(lambda *a, **k: None))

    tab._save_json()

    sidecar = out_path.with_name("calibration_panelshifts.txt")
    assert sidecar.is_file()
    saved = json.loads(out_path.read_text())
    assert saved["panel_shifts_path"] == str(sidecar)
    assert not saved["panel_shifts_path"].startswith(str(tempfile_shifts.parent))
    lines = sidecar.read_text().splitlines()
    assert len(lines) == 2


def _crystalline_result():
    return SimpleNamespace(
        Lsd=200000.0, BC_y=1024.0, BC_z=1024.0, tx=0.15, ty=0.0, tz=0.0,
        distortion={"iso_R2": 0.1}, pxY=200.0, pxZ=200.0,
        NrPixelsY=2048, NrPixelsZ=2048, wavelength_A=0.1729,
        _calibrant_name="CeO2", im_trans=[])


def test_populate_param_grid_marks_unrefined_geometry_params_as_fixed(app):
    """Parallel check to the manual-fit "(fixed)" annotation test — same
    _populate_param_grid method, driven by a crystalline (CeO2-shaped)
    result and a refine dict with Distortion/tx unchecked."""
    from PyQt5 import QtWidgets
    from midas_gui.tab_calibrate import CalibrationTab
    from midas_gui.helpers import paramstest_pairs

    tab = CalibrationTab()
    result = _crystalline_result()
    refine_flags = {"Lsd": True, "BC": True, "tx": False, "ty": True,
                     "tz": True, "Wavelength": True, "Distortion": False,
                     "distortion_coeffs": set()}
    tab._populate_param_grid(paramstest_pairs(result, selected=set()),
                             refine_flags=refine_flags)

    grid = tab._param_grid
    labels = [grid.itemAt(i).widget().text() for i in range(grid.count())
              if isinstance(grid.itemAt(i).widget(), QtWidgets.QLabel)
              and grid.itemAt(i).widget().text().endswith(":")]
    assert any(lbl.startswith("tx") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("Lsd") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("ty") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("tz") and "(fixed)" in lbl for lbl in labels)
    assert not any(lbl.startswith("Wavelength") and "(fixed)" in lbl for lbl in labels)


# ── Parameter windows (tol*) reaching the crystalline backend ────────────────

_V1_KW = dict(wavelength=0.1729, pxY=200.0, pxZ=200.0, calibrant="CeO2",
              NY=2048, NZ=2048, refine={}, n_iter=4, device="cpu")
_V1_SEED = {"BC_y": 1024.0, "BC_z": 1024.0, "Lsd": 1_000_000.0}


def test_build_v1_params_tols_override_only_what_is_given():
    """midas_calibrate/param_vector.py:bounds() turns these into hard (lo, hi)
    box constraints, so they are the crystalline bounding mechanism. Omitted
    keys must keep the dataclass default, and tols=None must reproduce exactly
    the object this built before tolerances were plumbed through."""
    import dataclasses as dc
    from midas_gui.calib import build_v1_params, tol_defaults, TOL_FIELDS

    base = build_v1_params(_V1_SEED, **_V1_KW)
    defaults = tol_defaults()
    assert {f: getattr(base, f) for f in TOL_FIELDS} == defaults

    tight = build_v1_params(_V1_SEED, **_V1_KW,
                            tols={"tolLsd": 2000.0, "tolBC": 5.0})
    assert tight.tolLsd == pytest.approx(2000.0)
    assert tight.tolBC == pytest.approx(5.0)
    assert tight.tolTilts == pytest.approx(defaults["tolTilts"])
    # Nothing else moved.
    differing = [f.name for f in dc.fields(base)
                 if getattr(base, f.name) != getattr(tight, f.name)]
    assert sorted(differing) == ["tolBC", "tolLsd"]


def test_tols_are_default_drives_the_one_shot_reroute():
    """calibrate() builds its own CalibrationParams and hardcodes
    Refine={"Lsd": True, "BC": True} (auto.py:619), so it can express neither a
    custom window nor a held Lsd/BC. tols_are_default is what decides whether
    the plain path is still usable."""
    from midas_gui.calib import tols_are_default, tol_defaults
    d = tol_defaults()

    assert tols_are_default(None)
    assert tols_are_default({})
    assert tols_are_default({"tolBC": d["tolBC"]})
    assert not tols_are_default({"tolBC": 5.0})
    assert not tols_are_default({"tolLsd": 2000.0})


def test_crystalline_tols_maps_rows_to_backend_windows(app):
    """The UI rows are coarser than the manual fit's: one window for both
    centre coordinates, one for both tilts, one for all distortion slots."""
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(1000.0)
    tab._seed_ty.setValue(0.0)

    cb, spin, combo = tab._limit_widgets["Lsd"]
    spin.setValue(3.0); combo.setCurrentText("mm")
    cb, spin, combo = tab._limit_widgets["ty"]
    spin.setValue(1.5); combo.setCurrentText("°")

    tols = tab._crystalline_tols()
    assert tols["tolLsd"] == pytest.approx(3000.0)     # mm entered, µm stored
    assert tols["tolTilts"] == pytest.approx(1.5)      # covers ty and tz
    assert set(tols) <= {"tolLsd", "tolBC", "tolTilts",
                          "tolWavelength", "tolDistortion"}

    # A d-spacing calibrant has no crystalline run to bound.
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    assert tab._crystalline_tols() is None


def _route_taken(monkeypatch, *, refine=None, tols=None):
    """Which backend entry point run_pipeline('one_shot') actually calls."""
    import numpy as np
    import midas_gui.calib as calib_mod
    import midas_calibrate_v2
    import midas_calibrate_v2.pipelines.single as single_mod

    calls = []

    def _stub(name):
        # A namespace, not a bare object: the reroute branch tags the result
        # with _residual_bin_path on the way out.
        def run(*_a, **_k):
            calls.append(name)
            return SimpleNamespace()
        return run

    monkeypatch.setattr(midas_calibrate_v2, "calibrate", _stub("calibrate"))
    monkeypatch.setattr(single_mod, "autocalibrate", _stub("autocalibrate"))
    cfg = {"wavelength": 0.1729, "pxY": 200.0, "pxZ": 200.0, "calibrant": "CeO2",
           "refine": refine if refine is not None else {}, "n_iter": 1,
           "lm_max_iter": 10, "device": "cpu", "im_trans": (),
           "build_residual_corr": False, "tols": tols,
           "manual_seed": {"BC_y": 1024.0, "BC_z": 1024.0, "Lsd": 1_000_000.0}}
    calib_mod.run_pipeline("one_shot", np.zeros((256, 256), dtype=np.float32),
                           None, cfg)
    return calls[0]


def test_one_shot_stays_on_calibrate_when_nothing_needs_rerouting(monkeypatch):
    assert _route_taken(monkeypatch) == "calibrate"
    assert _route_taken(monkeypatch, tols={"tolBC": 20.0}) == "calibrate"


def test_one_shot_reroutes_for_what_calibrate_cannot_express(monkeypatch):
    """calibrate() has no tol* kwarg and hardcodes Refine Lsd/BC True, and its
    single refine_tilts bool cannot refine exactly one of ty/tz — each of those
    has to go through the v1 route instead, or the GUI would be lying about
    what it asked for."""
    # A tightened window.
    assert _route_taken(monkeypatch, tols={"tolLsd": 2000.0}) == "autocalibrate"
    # Lsd or BC held fixed.
    assert _route_taken(monkeypatch, refine={"Lsd": False}) == "autocalibrate"
    assert _route_taken(monkeypatch, refine={"BC": False}) == "autocalibrate"
    # Exactly one tilt refined.
    assert _route_taken(monkeypatch,
                        refine={"ty": True, "tz": False}) == "autocalibrate"


def test_crystalline_at_limit_flags_a_fit_stopped_by_its_window(app):
    """A bounded fit that stops on its bound is reporting the bound, not a
    measurement — the windows being visible is not on its own enough."""
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    tab._manual_seed_check.setChecked(True)
    tab._seed_lsd.setValue(1000.0)          # mm in the UI, µm in the seed slot
    tab._seed_bcy.setValue(1024.0); tab._seed_bcz.setValue(1024.0)
    tab._seed_ty.setValue(0.0); tab._seed_tz.setValue(0.0)
    _cb, spin, combo = tab._limit_widgets["Lsd"]
    spin.setValue(2.0); combo.setCurrentText("mm")

    def at_limit(**over):
        base = dict(Lsd=1_000_000.0, BC_y=1024.0, BC_z=1024.0, ty=0.0, tz=0.0,
                    wavelength_A=tab._wl.value())
        base.update(over)
        return tab._crystalline_at_limit(SimpleNamespace(**base))

    assert at_limit() == set()                                # mid-window
    assert at_limit(Lsd=1_002_000.0) == {"Lsd"}               # on the ±2 mm bound
    assert at_limit(Lsd=1_001_500.0) == set()                 # just inside
    assert at_limit(BC_z=1044.0) == {"BC_z"}                  # on the ±20 px bound
    assert at_limit(tz=3.0) == {"tz"}                         # on the ±3° bound

    # Held-fixed parameters never moved, so they cannot have been stopped.
    tab._ref_lsd.setChecked(False)
    tab._last_refine_flags = tab._refine_flags()
    assert at_limit(Lsd=1_002_000.0) == set()

    # Without a manual seed the window's centre is unknown; guessing would be
    # worse than saying nothing.
    tab._ref_lsd.setChecked(True)
    tab._last_refine_flags = tab._refine_flags()
    tab._manual_seed_check.setChecked(False)
    assert at_limit(Lsd=1_002_000.0) == set()

    # A d-spacing calibrant reports through the manual fit's own at_limit.
    tab._cal.setCurrentIndex(tab._cal.findText("AgBH (silver behenate)"))
    assert tab._crystalline_at_limit(SimpleNamespace(Lsd=1_002_000.0)) == set()


# ── Default names for saved calibrations ─────────────────────────────────────


def test_default_save_stem_joins_expid_and_calibration_image(app, tmp_path):
    """The user's own convention: <expid>_<image name>.instr.txt/.json, so a
    saved instrument file says at a glance which experiment and which
    calibration exposure it came from."""
    from midas_gui.tab_calibrate import CalibrationTab

    img = tmp_path / "ceo2_000123.tif"
    img.write_bytes(b"")
    tab = CalibrationTab()
    tab.set_expid_provider(lambda: "park_may26")
    tab._loader.set_path(str(img), load=False)

    assert tab._default_save_stem() == "park_may26_ceo2_000123"
    assert tab._default_save_path(".instr.json").endswith(
        "park_may26_ceo2_000123.instr.json")


def test_default_save_stem_drops_whichever_half_is_missing(app, tmp_path):
    """Neither half is guaranteed: the Exp ID header is free-form and may be
    blank, and the Data path may not be set yet. A missing half is dropped
    rather than joined as an empty string — "_.instr.txt" would be a worse
    suggestion than either half alone."""
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    tab._loader.set_path("", load=False)
    assert tab._default_save_stem() == "calibration"

    tab.set_expid_provider(lambda: "  park_may26 ")
    assert tab._default_save_stem() == "park_may26"

    img = tmp_path / "ceo2.h5"
    img.write_bytes(b"")
    tab.set_expid_provider(lambda: "   ")
    tab._loader.set_path(str(img), load=False)
    assert tab._default_save_stem() == "ceo2"


def test_default_save_path_prefers_the_output_dir_then_the_image_folder(app, tmp_path):
    """A bare filename would open the save dialog on the process CWD — i.e.
    wherever the app was launched from, which is never where the user wants
    the calibration to land."""
    from pathlib import Path
    from midas_gui.tab_calibrate import CalibrationTab

    img_dir = tmp_path / "raw"; img_dir.mkdir()
    img = img_dir / "ceo2.tif"; img.write_bytes(b"")
    out_dir = tmp_path / "analysis"; out_dir.mkdir()

    tab = CalibrationTab()
    tab.set_expid_provider(lambda: "park_may26")
    tab._loader.set_path(str(img), load=False)

    tab._out_ed.setText("")
    assert Path(tab._default_save_path(".instr.txt")).parent == img_dir

    tab._out_ed.setText(str(out_dir))
    assert Path(tab._default_save_path(".instr.txt")).parent == out_dir


def test_save_paramstest_dialog_prefills_the_suggested_name(app):
    """_SaveParamstestDialog types its own output path rather than going
    through a file dialog, so the suggestion has to arrive as a constructor
    argument. It stays optional: hydra_calib_widgets constructs the same
    dialog with no suggestion to offer."""
    from midas_gui.dialogs import _SaveParamstestDialog

    dlg = _SaveParamstestDialog(default_out="/tmp/park_may26_ceo2.instr.txt")
    assert dlg.out_path() == "/tmp/park_may26_ceo2.instr.txt"
    assert _SaveParamstestDialog().out_path() == ""


# ── "Feed result back to seed" only promotes refined parameters ──────────────

def _feedback_result(**over):
    base = dict(Lsd=3_227_008.5, BC_y=2265.66, BC_z=2080.73,
                tx=180.0, ty=-5.2542, tz=24.4086, distortion={},
                wavelength_A=0.15381)
    base.update(over)
    return SimpleNamespace(**base)


def test_seed_from_result_skips_unrefined_tx(app):
    """An unrefined parameter's value in the result is a fixed constant the
    fit was given, not something it determined — feeding it back would
    silently lock a future run to whatever that constant happened to be
    (the connoly_oct26 Hydra bug: tx held fixed at a different nonsense
    value per panel, see .context/DECISIONS.md)."""
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    tab._cal.setCurrentIndex(tab._cal.findText("CeO2"))
    tab._feedback_check.setChecked(True)
    assert not tab._seed_en_tx.isChecked()
    stale_tx = tab._seed_tx.value()

    tab._ref_tx.setChecked(False)
    tab._last_refine_flags = tab._refine_flags()   # Lsd/BC/ty/tz True, tx False by default
    tab._seed_from_result(_feedback_result())

    assert tab._seed_en_tx.isChecked() is False
    assert tab._seed_tx.value() == pytest.approx(stale_tx)
    assert tab._seed_en_lsd.isChecked() and tab._seed_en_bc.isChecked()
    assert tab._seed_lsd.value() == pytest.approx(3227.0085, rel=1e-6)
    assert "tx" not in tab._seed_note.text()

    # Refine tx this time — now it's real new information and should be fed back.
    tab._ref_tx.setChecked(True)
    tab._last_refine_flags = tab._refine_flags()
    tab._seed_from_result(_feedback_result(tx=12.5))
    assert tab._seed_en_tx.isChecked() is True
    assert tab._seed_tx.value() == pytest.approx(12.5)


def test_seed_from_result_with_nothing_refined_leaves_seed_untouched(app):
    from midas_gui.tab_calibrate import CalibrationTab

    tab = CalibrationTab()
    tab._last_refine_flags = {"Lsd": False, "BC": False, "tx": False,
                              "ty": False, "tz": False, "Wavelength": False,
                              "Distortion": False, "distortion_coeffs": set()}
    tab._seed_from_result(_feedback_result())
    assert not any(cb.isChecked() for cb in
                   (tab._seed_en_lsd, tab._seed_en_bc, tab._seed_en_tx,
                    tab._seed_en_ty, tab._seed_en_tz))
    assert "no seedable parameters" in tab._seed_note.text()

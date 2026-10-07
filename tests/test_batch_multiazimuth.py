"""Tests for Batch Integrate's opt-in "Multi-azimuth output (cake)" mode:
BatchWorker keeps a real (n_eta, n_r) cake per frame instead of collapsing to
one full-circle profile, only when explicitly requested — off by default,
byte-identical to prior behavior otherwise (regression guard)."""
from types import SimpleNamespace

import numpy as np
import pytest


def _tiny_calib_result(**overrides):
    fields = dict(
        Lsd=200000.0, BC_y=32.0, BC_z=32.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={}, pxY=200.0, pxZ=200.0,
        NrPixelsY=64, NrPixelsZ=64, wavelength_A=0.1729,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _make_tiff_frames(tmp_path, n=2, size=64):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.tif"
        tifffile.imwrite(str(p), (rng.random((size, size)) * 100 + 10).astype(np.float32))
        paths.append(str(p))
    return paths


@pytest.fixture(scope="module")
def app():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _run_batch_worker(app, tmp_path, *, multi_azimuth, eta_bin, n_frames=2):
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    paths = _make_tiff_frames(tmp_path, n=n_frames)
    spec = _build_spec(_tiny_calib_result(), r_bin=2.0, eta_bin=eta_bin)

    worker = wk.BatchWorker(
        spec, {"type": "tiff_list", "paths": paths}, None, None, [], "subpixel2",
        (None, None), None, multi_azimuth=multi_azimuth)
    results = {}
    failures = []
    worker.finished.connect(lambda data: results.update(data))
    worker.failed.connect(failures.append)
    worker.run()   # direct call, not .start() — no real QThread spawned
    assert not failures, failures[0] if failures else ""
    return results, spec


def test_multi_azimuth_off_keeps_2d_profiles(app, tmp_path):
    """Off by default — unaffected, byte-shape-identical to prior behavior."""
    results, _ = _run_batch_worker(app, tmp_path, multi_azimuth=False, eta_bin=5.0)
    profiles = results["profiles"]
    assert profiles.ndim == 2
    assert profiles.shape[0] == 2
    assert results["multi_azimuth"] is False
    assert results["eta_axis"] is None
    assert results["sigmas"].shape == profiles.shape


def test_multi_azimuth_on_keeps_full_cake(app, tmp_path):
    """Checked — every azimuthal sector survives as a real output dimension."""
    results, spec = _run_batch_worker(app, tmp_path, multi_azimuth=True, eta_bin=120.0)
    n_eta = spec.n_eta_bins
    assert n_eta == 3   # 360 / 120
    profiles = results["profiles"]
    assert profiles.ndim == 3
    assert profiles.shape == (2, n_eta, spec.n_r_bins)
    assert results["multi_azimuth"] is True
    assert results["eta_axis"] is not None
    assert len(results["eta_axis"]) == n_eta
    assert results["sigmas"].shape == profiles.shape


def test_multi_azimuth_rejects_q_uniform_combo(app, tmp_path):
    """The UI blocks this combination; BatchWorker itself also refuses it as
    defense-in-depth (rebin_R_to_grid only handles a 1-D profile).

    Deliberately built with the LEGACY Q-only cfg spelling, which is also
    what every project saved before 2θ existed carries — so this doubles as
    coverage that rebin_cfg_parts still reads it."""
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    paths = _make_tiff_frames(tmp_path, n=1)
    spec = _build_spec(_tiny_calib_result(), r_bin=2.0, eta_bin=120.0)
    worker = wk.BatchWorker(
        spec, {"type": "tiff_list", "paths": paths}, None, None, [], "subpixel2",
        (None, None), None, multi_azimuth=True,
        q_cfg={"QMin": 0.1, "QMax": 1.0, "QBinSize": 0.01})
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert failures and "Q-uniform" in failures[0]


def test_write_all_profiles_handles_3d_cake_profiles(tmp_path):
    """(n_frames, n_eta, n_r) profiles → one file per (frame, eta), not a
    malformed single write."""
    pytest.importorskip("midas_integrate_v2")
    import midas_gui.workers as wk

    n_frames, n_eta, n_r = 2, 3, 5
    r_axis = np.linspace(1.0, 10.0, n_r)
    profiles = np.abs(np.random.rand(n_frames, n_eta, n_r)) + 1.0
    sigmas = np.sqrt(profiles)
    frame_ids = [f"frame_{i:03d}" for i in range(n_frames)]

    paths = wk.write_all_profiles(
        tmp_path, ["csv", "h5"], r_axis, profiles, sigmas, frame_ids,
        lsd=200000.0, px=200.0, wl=0.2)

    # <froot>_<NNNNNN> output convention — see workers.frame_output_base.
    for i in range(n_frames):
        for k in range(n_eta):
            assert (tmp_path / f"frame_{i:06d}_eta{k:03d}.csv").exists()
    # h5 is skipped for 3-D profiles (write_h5 expects one profile per frame)
    assert not (tmp_path / "integrated.h5").exists()
    assert len(paths) == n_frames * n_eta

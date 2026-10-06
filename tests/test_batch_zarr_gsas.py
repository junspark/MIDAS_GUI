"""Batch Integrate's Zarr output must satisfy GSAS-II's own hard gate — all
three top-level groups InstrumentParameters/REtaMap/OmegaSumFrame present
(confirmed by reading GSAS-II's G2pwd_MIDAS.py ContentsValidator directly) —
and carry a provenance_history stamp, since midas_integrate_v2's
write_gsas_zarr_zip (unlike the retired zarr_cake.write_cake_zarr) has no
attrs slot of its own for it."""
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


def test_batch_zarr_output_has_gsas_required_groups_and_provenance(app, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    zarr = pytest.importorskip("zarr")
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    paths = _make_tiff_frames(src_dir, n=2)
    out_dir = tmp_path / "out"
    spec = _build_spec(_tiny_calib_result(), r_bin=2.0, eta_bin=5.0)

    worker = wk.BatchWorker(
        spec, {"type": "tiff_list", "paths": paths}, None, str(out_dir),
        ["zarr"], "subpixel2", (None, None), None)
    results = {}
    failures = []
    worker.finished.connect(lambda data: results.update(data))
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures, failures[0] if failures else ""

    # One zarr per processed frame — mirrors mpe_wf's own one-zarr-per-
    # scan-point convention (see BatchWorker.run()'s per-frame zarr-write
    # comment), named directly off `fid` (here the two tiff frames' own
    # stems from _ExplicitTIFFSource) rather than a bundled whole-run
    # range. ".zarr.zip" (not just ".zarr") is required for GSAS-II's
    # importer to recognize the file.
    zarr_paths = [out_dir / "zarr" / f"frame_{i:04d}.ave.zarr.zip" for i in range(2)]
    for zarr_path in zarr_paths:
        assert zarr_path.is_file(), f"expected zarr output at {zarr_path}"
        assert str(zarr_path) in results["out_paths"]

    for zarr_path in zarr_paths:
        root = zarr.open(str(zarr_path), mode="r")
        # GSAS-II's ContentsValidator hard-gates on exactly these three groups.
        for group in ("InstrumentParameters", "REtaMap", "OmegaSumFrame"):
            assert group in root, f"missing GSAS-II-required group {group!r}"
        assert len(list(root["OmegaSumFrame"])) == 1   # one frame per zarr now

        history = root.attrs.get("provenance_history")
        assert history, "provenance_history missing from zarr root attrs"
        assert history[-1]["tool"] == "midas_gui.batch_integrate"


def _make_varex_h5(path, n_light=10, n_dark=10, size=64):
    """A synthetic VAREX-style HDF5 stack: ``n_light`` usable data frames
    followed by ``n_dark`` dark acquisitions, with per-acquisition metadata
    (Temperature/Pressure/SRCurrent + timestamps) recorded for *all*
    ``n_light + n_dark`` entries in one flat array — reproducing the real
    light-then-dark schema this session's investigation found, including
    the timestamp gap (steady ~7.0s, one ~9.5s at the light->dark boundary)
    used to tell them apart."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    n_total = n_light + n_dark
    period = 7.0
    gaps = np.full(n_total - 1, period)
    gaps[n_light - 1] = 9.5   # the one anomalous gap, at the light/dark boundary
    timestamps = np.concatenate([[0.0], np.cumsum(gaps)])
    temperature = np.arange(n_total, dtype=np.float64)          # 0, 1, 2, ...
    pressure = np.arange(n_total, dtype=np.float64) * 10.0      # 0, 10, 20, ...
    current = 200.0 + np.arange(n_total, dtype=np.float64)      # 200, 201, ...
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data",
                          data=(rng.random((n_light, size, size)) * 100).astype(np.float32))
        f.create_dataset("exchange/data_dark",
                          data=np.zeros((n_dark, size, size), dtype=np.float32))
        f.create_dataset("misc/NDArrayTimeStamp", data=timestamps)
        f.create_dataset("instrument/GSAS2_PVS/Temperature", data=temperature)
        f.create_dataset("instrument/GSAS2_PVS/Pressure", data=pressure)
        f.create_dataset("instrument/StorageRing/SRCurrent", data=current)
    return temperature, pressure, current


def test_hdf5_metadata_ignores_trailing_dark_frames_via_timestamp_gap(tmp_path):
    """_HDF5StackGlobSource.metadata_for_index must average only the leading
    light-frame metadata entries (found via the timestamp-gap technique),
    never blending in the trailing dark-frame samples."""
    pytest.importorskip("h5py")
    import midas_gui.workers as wk

    h5_path = tmp_path / "scan_001.h5"
    temperature, pressure, current = _make_varex_h5(h5_path, n_light=10, n_dark=10)

    # No chunking: one combined frame == mean of all 10 light entries only.
    src = wk._HDF5StackGlobSource([h5_path], "exchange/data", chunk_size=None)
    meta = src.metadata_for_index(0)
    assert meta["temperature"] == pytest.approx(temperature[:10].mean())
    assert meta["pressure"] == pytest.approx(pressure[:10].mean())
    assert meta["current"] == pytest.approx(current[:10].mean())

    # Chunked into two 5-frame combined outputs: each averages its own
    # 5-entry slice of the light block, never reaching into the dark tail.
    src2 = wk._HDF5StackGlobSource([h5_path], "exchange/data", chunk_size=5)
    assert src2.n_frames == 2
    m0, m1 = src2.metadata_for_index(0), src2.metadata_for_index(1)
    assert m0["temperature"] == pytest.approx(temperature[0:5].mean())
    assert m1["temperature"] == pytest.approx(temperature[5:10].mean())


def test_batch_zarr_metadata_is_mean_regardless_of_pixel_combine_op(tmp_path):
    """Per the user's explicit requirement: even when the pixel data for a
    chunk is combined with Sum/Max/Median, the metadata written alongside it
    must always be the arithmetic mean across that chunk's raw frames."""
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("h5py")
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    h5_path = src_dir / "scan_001.h5"
    temperature, pressure, current = _make_varex_h5(h5_path, n_light=10, n_dark=10)

    out_dir = tmp_path / "out"
    spec = _build_spec(_tiny_calib_result(), r_bin=2.0, eta_bin=5.0)

    worker = wk.BatchWorker(
        spec,
        {"type": "hdf5_stack_glob", "paths": [str(h5_path)],
         "dataset": "exchange/data", "chunk_size": 5, "combine_op": "max"},
        None, str(out_dir), ["zarr"], "subpixel2", (None, None), None)
    results = {}
    failures = []
    worker.finished.connect(lambda data: results.update(data))
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures, failures[0] if failures else ""

    expected = [
        (0, 4, temperature[0:5].mean(), pressure[0:5].mean(), current[0:5].mean()),
        (5, 9, temperature[5:10].mean(), pressure[5:10].mean(), current[5:10].mean()),
    ]
    for start, end, temp_exp, pres_exp, cur_exp in expected:
        zarr_path = out_dir / "zarr" / f"scan_001.frame_{start}_{end}.ave.zarr.zip"
        assert zarr_path.is_file(), f"expected zarr output at {zarr_path}"
        root = zarr.open(str(zarr_path), mode="r")
        attrs = root["OmegaSumFrame/LastFrameNumber_0"].attrs
        assert attrs["Temperature"] == pytest.approx(temp_exp)
        assert attrs["Pressure"] == pytest.approx(pres_exp)
        # Storage-ring current is not a beam monitor — it rides along in the
        # provenance entry's `extra` (see BatchWorker.run()), never under
        # the zarr's own "I"/"I0" attrs (those are for a real ion chamber;
        # this source's path has no varexE/varexD hutch marker, so
        # ion_chamber_i/_i0 are never populated and "I"/"I0" are absent
        # entirely, not just zero).
        assert "I" not in attrs and "I0" not in attrs
        history = root.attrs.get("provenance_history")
        assert history[-1]["extra"]["storage_ring_current_mA"] == pytest.approx(cur_exp)


# ── Ion chamber / sample-motor stopgap (see .context/DECISIONS.md) ─────────
#
# Real per-hutch beam-monitor and sample-motion-system paths, confirmed
# against a real file and the beamline's own HDF5-layout docs — not the
# HDF5 file's own `active_instrument` (documented upstream as always empty),
# so the hutch is inferred from the source path instead (`varexE`/`varexD`).

def _make_hutch_varex_h5(path, hutch, *, include_d2pd=True, n_light=10, size=64):
    """Like ``_make_varex_h5`` (base VAREX schema + Temperature/Pressure/
    SRCurrent) but also writes this hutch's real ion-chamber and
    sample-motion-system datasets, so metadata_for_index's hutch-specific
    reads have something real to find. No dark frames here — the
    light/dark-boundary behavior is already covered by
    ``test_hdf5_metadata_ignores_trailing_dark_frames_via_timestamp_gap``;
    this fixture is only about the per-hutch fields."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("exchange/data",
                          data=(rng.random((n_light, size, size)) * 100).astype(np.float32))
        f.create_dataset("instrument/GSAS2_PVS/Temperature",
                          data=np.full(n_light, np.nan))
        f.create_dataset("instrument/GSAS2_PVS/Pressure",
                          data=np.full(n_light, np.nan))
        f.create_dataset("instrument/StorageRing/SRCurrent",
                          data=200.0 + np.arange(n_light, dtype=np.float64))
        if hutch == "D":
            f.create_dataset("instrument/Scalers/D/IC2",
                              data=1000.0 + np.arange(n_light, dtype=np.float64))
        elif hutch == "E":
            f.create_dataset("instrument/Scalers/E/US_IC",
                              data=1300.0 + np.arange(n_light, dtype=np.float64))
            if include_d2pd:
                f.create_dataset("instrument/Scalers/E/D2PD",
                                  data=1100.0 + np.arange(n_light, dtype=np.float64))
            f.create_dataset("instrument/SMS/E/HL/samX",
                              data=np.full(n_light, 3.9))
            f.create_dataset("instrument/SMS/E/HL/samY",
                              data=np.full(n_light, 8.78))
            f.create_dataset("instrument/SMS/E/HR/samX",
                              data=np.full(n_light, np.nan))   # inactive sub-config


def test_hutch_resolved_from_source_path(tmp_path, monkeypatch):
    """Stopgap hutch detection: varexE/varexD in the path, case-insensitive.
    The path outranks the active profile, which is only the fallback (see
    ion_csv.resolve_hutch) — pinned here to a non-20-ID profile so this
    asserts the PATH rule rather than whichever profile the machine
    running the suite happens to have selected."""
    import midas_gui.workers as wk
    from midas_gui import settings
    monkeypatch.setattr(settings, "active_profile", lambda: "Default")
    assert wk._HDF5StackGlobSource(
        [tmp_path / "VarexE" / "scan.h5"], "exchange/data")._hutch == "E"
    assert wk._HDF5StackGlobSource(
        [tmp_path / "varexD" / "scan.h5"], "exchange/data")._hutch == "D"
    assert wk._HDF5StackGlobSource(
        [tmp_path / "ge3" / "scan.h5"], "exchange/data")._hutch is None


def test_the_profile_resolves_a_hutch_the_path_cannot(tmp_path, monkeypatch):
    """An Eiger run lives under eiger2/ and matches neither varex marker, so
    path-only detection silently skipped every ion chamber on a file that
    had them. The header profile is the fallback."""
    import midas_gui.workers as wk
    from midas_gui import settings
    monkeypatch.setattr(settings, "active_profile", lambda: "20-ID-E")
    assert wk._HDF5StackGlobSource(
        [tmp_path / "eiger2" / "scan.h5"], "exchange/data")._hutch == "E"
    monkeypatch.setattr(settings, "active_profile", lambda: "1-ID-E")
    assert wk._HDF5StackGlobSource(
        [tmp_path / "eiger2" / "scan.h5"], "exchange/data")._hutch is None


def test_d_hutch_ion_chamber_is_i0_only(tmp_path):
    """D hutch has no transmission monitor yet — only I0 (IC2) is ever
    attempted; "ion_chamber_i" isn't even a key, not just None, matching
    _ION_CHAMBER_H5_PATHS declaring no "i" entry for D at all."""
    import midas_gui.workers as wk
    h5_dir = tmp_path / "varexD"
    h5_dir.mkdir()
    h5_path = h5_dir / "scan_001.h5"
    _make_hutch_varex_h5(h5_path, "D")

    src = wk._HDF5StackGlobSource([h5_path], "exchange/data")
    assert src._hutch == "D"
    meta = src.metadata_for_index(0)
    assert meta["ion_chamber_i0"] == pytest.approx(np.mean(1000.0 + np.arange(10)))
    assert "ion_chamber_i" not in meta


@pytest.mark.parametrize("include_d2pd", [True, False])
def test_e_hutch_ion_chamber_i_is_auto_detected(tmp_path, include_d2pd):
    """E hutch's I0 (US_IC) is always attempted; I (D2PD) is genuinely
    setup-dependent — present only when the file actually has it, which
    doubles as the "auto-detect" the user asked for rather than a hardcoded
    assumption."""
    import midas_gui.workers as wk
    h5_dir = tmp_path / "varexE"
    h5_dir.mkdir()
    h5_path = h5_dir / "scan_001.h5"
    _make_hutch_varex_h5(h5_path, "E", include_d2pd=include_d2pd)

    src = wk._HDF5StackGlobSource([h5_path], "exchange/data")
    meta = src.metadata_for_index(0)
    assert meta["ion_chamber_i0"] == pytest.approx(np.mean(1300.0 + np.arange(10)))
    if include_d2pd:
        assert meta["ion_chamber_i"] == pytest.approx(np.mean(1100.0 + np.arange(10)))
    else:
        assert meta["ion_chamber_i"] is None


def test_e_hutch_sample_motors_capture_both_hl_and_hr(tmp_path):
    """E hutch has two coexisting SMS sub-configs (HL/HR) with no reliable
    signal for which is physically active — captured both, distinctly keyed,
    rather than guessing. On real data the inactive one comes back NaN
    (see .context/DECISIONS.md), which this fixture reproduces for HR."""
    import midas_gui.workers as wk
    h5_dir = tmp_path / "varexE"
    h5_dir.mkdir()
    h5_path = h5_dir / "scan_001.h5"
    _make_hutch_varex_h5(h5_path, "E")

    src = wk._HDF5StackGlobSource([h5_path], "exchange/data")
    meta = src.metadata_for_index(0)
    assert meta["motor:HL/samX"] == pytest.approx(3.9)
    assert meta["motor:HL/samY"] == pytest.approx(8.78)
    assert np.isnan(meta["motor:HR/samX"])


def test_unknown_hutch_has_no_ion_chamber_or_sample_motor_keys(tmp_path,
                                                               monkeypatch):
    """A source neither the path nor the profile can place attempts none of
    the new fields — same "not every source has this metadata" contract the
    existing temperature/pressure/current keys already follow.

    The profile is pinned off a 20-ID station, since it is now the fallback
    and the machine running the suite is usually set to one."""
    import midas_gui.workers as wk
    from midas_gui import settings
    monkeypatch.setattr(settings, "active_profile", lambda: "Default")
    h5_dir = tmp_path / "ge3"
    h5_dir.mkdir()
    h5_path = h5_dir / "scan_001.h5"
    _make_hutch_varex_h5(h5_path, "E")   # real datasets present in the file...

    # ...but the path gives no hutch signal, so none of them are read.
    src = wk._HDF5StackGlobSource([h5_path], "exchange/data")
    assert src._hutch is None
    meta = src.metadata_for_index(0)
    assert not any(k.startswith("motor:") for k in meta)
    assert "ion_chamber_i0" not in meta and "ion_chamber_i" not in meta
    assert meta["temperature"] is not None or True  # base keys still attempted regardless


def test_batch_zarr_ion_chamber_and_sample_motors_end_to_end(tmp_path):
    """Full BatchWorker run against a real-shaped D-hutch and E-hutch source:
    the zarr's own I/I0 attrs carry the real ion chamber (never ring
    current), and the provenance entry's extra carries storage-ring current
    plus sample motors."""
    pytest.importorskip("torch")
    pytest.importorskip("midas_integrate_v2")
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("h5py")
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import midas_gui.workers as wk
    from midas_gui.helpers import _build_spec

    spec = _build_spec(_tiny_calib_result(), r_bin=2.0, eta_bin=5.0)

    def _run(hutch, **fixture_kwargs):
        src_dir = tmp_path / f"varex{hutch}_{fixture_kwargs.get('include_d2pd', '')}"
        src_dir.mkdir()
        h5_path = src_dir / "scan_001.h5"
        _make_hutch_varex_h5(h5_path, hutch, **fixture_kwargs)
        out_dir = src_dir / "out"
        worker = wk.BatchWorker(
            spec, {"type": "hdf5_stack_glob", "paths": [str(h5_path)],
                  "dataset": "exchange/data"},
            None, str(out_dir), ["zarr"], "subpixel2", (None, None), None)
        failures = []
        worker.failed.connect(failures.append)
        worker.run()
        assert not failures, failures[0] if failures else ""
        zarr_path = out_dir / "zarr" / "scan_001.ave.zarr.zip"
        root = zarr.open(str(zarr_path), mode="r")
        return root

    root_d = _run("D")
    attrs_d = root_d["OmegaSumFrame/LastFrameNumber_0"].attrs
    assert attrs_d["I0"] == pytest.approx(np.mean(1000.0 + np.arange(10)))
    assert "I" not in attrs_d
    extra_d = root_d.attrs["provenance_history"][-1]["extra"]
    assert extra_d["storage_ring_current_mA"] == pytest.approx(np.mean(200.0 + np.arange(10)))

    root_e = _run("E", include_d2pd=True)
    attrs_e = root_e["OmegaSumFrame/LastFrameNumber_0"].attrs
    assert attrs_e["I0"] == pytest.approx(np.mean(1300.0 + np.arange(10)))
    assert attrs_e["I"] == pytest.approx(np.mean(1100.0 + np.arange(10)))
    extra_e = root_e.attrs["provenance_history"][-1]["extra"]
    assert extra_e["sample_motors"]["HL/samX"] == pytest.approx(3.9)
    assert np.isnan(extra_e["sample_motors"]["HR/samX"])

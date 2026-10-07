"""Batch Integrate's "Eta-R cakes" tab, and the cake x-axis unit selector.

A multi-azimuth run keeps each frame's (η, R) cake instead of the η-collapsed
profile, so its result is ``(n_frames, n_eta, n_r)``. Before this, the tab
computed those, wrote them to disk, embedded them in the project — and showed
none of them; reopening such a project actually *raised*, because the restore
path fed 2-D cake rows to the 1-D waterfall buffer. What's pinned here:

* the R → 2θ / d / Q conversion, including d's divergence at R = 0;
* ``CakeViewer``'s unit selector staying hidden (and the axis untouched) until
  a caller supplies geometry — that's what keeps Calibrate/Hydra unchanged;
* ``CakeStackViewer`` scrubbing frames without discarding the user's zoom;
* ``BatchTab`` filling the tab from a live run and from a stored attempt, and
  clearing it for a non-multi-azimuth run rather than showing stale cakes.

Builds pyqtgraph widgets, hence forked, and defers every Qt / midas_gui GUI
import into the fixtures — see STATE.md's rule for new Qt test files
(``tests/test_set_raw_frame.py`` is the reference).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked

LSD, PX, WL = 200000.0, 200.0, 0.1729   # µm, µm, Å
N_ETA, N_R = 24, 40


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widgets(app):
    from midas_gui import widgets
    return widgets


def _axes():
    return (np.arange(N_R, dtype=float),
            np.linspace(-180.0, 180.0, N_ETA))


def _stack(n_frames=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n_frames, N_ETA, N_R)).astype(np.float32)


# ── radial-unit conversion ───────────────────────────────────────────

def test_convert_radial_round_trips_every_unit(widgets):
    r = np.array([10.0, 100.0, 500.0, 1000.0])
    for unit in ("2th", "d", "Q"):
        out = widgets._convert_radial(r, LSD, PX, WL, "R", unit)
        back = widgets._convert_radial(out, LSD, PX, WL, unit, "R")
        assert back == pytest.approx(r, rel=1e-9), unit


def test_d_spacing_diverges_at_the_beam_centre_and_falls_with_r(widgets):
    r = np.array([0.0, 10.0, 100.0, 500.0])
    d = widgets._convert_radial(r, LSD, PX, WL, "R", "d")
    assert not np.isfinite(d[0])                  # d → ∞ as 2θ → 0
    assert np.all(np.diff(d[1:]) < 0)             # and decreases with R


def test_convert_radial_is_identity_without_geometry(widgets):
    r = np.array([1.0, 2.0, 3.0])
    assert widgets._convert_radial(r, None, PX, WL, "R", "Q") == pytest.approx(r)
    assert widgets._convert_radial(r, LSD, PX, WL, "R", "R") == pytest.approx(r)


# ── CakeViewer x-axis selector ───────────────────────────────────────

def test_x_unit_selector_is_hidden_until_a_geometry_is_supplied(widgets):
    cv = widgets.CakeViewer()
    cv.set_cake(_stack(1)[0], *_axes())
    # isHidden(), not isVisible(): nothing here is ever shown on screen, so
    # isVisible() is False for every widget regardless of setVisible().
    assert cv._xunit.isHidden()
    assert cv._xaxis_item._convert is None       # axis behaves as a plain one
    cv.set_axis_context(LSD, PX, WL)
    assert not cv._xunit.isHidden()
    assert cv.x_unit() == "R"                    # ...still defaulting to R


def test_switching_unit_relabels_ticks_without_touching_the_data(widgets):
    cv = widgets.CakeViewer()
    r_axis, eta_axis = _axes()
    cake = _stack(1)[0]
    cv.set_cake(cake, r_axis, eta_axis)
    cv.set_axis_context(LSD, PX, WL)
    ticks = [0.0, 10.0, 20.0]

    assert cv._xaxis_item.tickStrings(ticks, 1, 10) == ["0", "10", "20"]

    cv._xunit.setCurrentIndex(cv._xunit.findData("2th"))
    tth = cv._xaxis_item.tickStrings(ticks, 1, 10)
    assert tth != ["0", "10", "20"]
    assert float(tth[1]) == pytest.approx(
        np.degrees(np.arctan(10.0 * PX / LSD)), rel=1e-3)

    cv._xunit.setCurrentIndex(cv._xunit.findData("d"))
    assert cv._xaxis_item.tickStrings(ticks, 1, 10)[0] == "∞"

    # The cake and its axes are untouched throughout — labels only.
    assert cv._r_axis == pytest.approx(r_axis)
    assert cv._eta_axis == pytest.approx(eta_axis)
    assert np.array_equal(cv._cake, cake)


def test_x_unit_survives_a_display_state_round_trip(widgets):
    cv = widgets.CakeViewer()
    cv.set_cake(_stack(1)[0], *_axes())
    cv.set_axis_context(LSD, PX, WL)
    cv._xunit.setCurrentIndex(cv._xunit.findData("Q"))
    state = cv.display_state()
    assert state["xunit"] == "Q"

    other = widgets.CakeViewer()
    other.set_cake(_stack(1)[0], *_axes())
    other.set_axis_context(LSD, PX, WL)
    other.set_display_state(state)
    assert other.x_unit() == "Q"
    # Blocked signals mean set_display_state has to drive the relabel itself.
    assert other._xaxis_item._convert is not None


def test_ring_residual_viewer_keeps_its_own_axis(widgets):
    """RingResidualViewer's X is a ring index, not a radius. It inherits the
    selector but never calls set_axis_context, so the axis must stay plain."""
    rv = widgets.RingResidualViewer()
    assert rv._xunit.isHidden()
    assert rv._xaxis_item._convert is None
    assert rv._xaxis_item.tickStrings([0.0, 1.0, 2.0], 1, 1) == ["0", "1", "2"]


# ── CakeStackViewer ──────────────────────────────────────────────────

def test_set_cakes_shows_the_first_frame_and_arms_the_scrubber(widgets):
    sv = widgets.CakeStackViewer()
    stack = _stack(5)
    sv.set_cakes(stack, *_axes(), frame_ids=[f"f{i}" for i in range(5)])
    assert sv.frame_count() == 5
    assert sv.frame_index() == 0
    assert np.array_equal(sv._cake, stack[0])
    assert not sv._scrub_bar.isHidden()
    assert "1/5" in sv._frame_lbl.text() and "f0" in sv._frame_lbl.text()


def test_stepping_and_the_slider_both_change_frame(widgets):
    sv = widgets.CakeStackViewer()
    stack = _stack(4)
    sv.set_cakes(stack, *_axes(), frame_ids=["a", "b", "c", "d"])

    sv._step(1)
    assert sv.frame_index() == 1 and np.array_equal(sv._cake, stack[1])
    sv._frame_slider.setValue(3)
    assert sv.frame_index() == 3 and np.array_equal(sv._cake, stack[3])
    sv._step(1)                       # clamped at the last frame
    assert sv.frame_index() == 3
    sv._step(-99)                     # and at the first
    assert sv.frame_index() == 0


def test_scrubbing_preserves_the_users_zoom(widgets):
    sv = widgets.CakeStackViewer()
    sv.set_cakes(_stack(3), *_axes())
    vb = sv._iv.getView().getViewBox()
    vb.setXRange(5.0, 15.0, padding=0)
    zoomed = vb.viewRange()[0]

    sv._step(1)
    assert vb.viewRange()[0] == pytest.approx(zoomed, abs=1e-6)
    # A brand-new stack does reframe.
    sv.set_cakes(_stack(3, seed=1), *_axes())
    assert vb.viewRange()[0] != pytest.approx(zoomed, abs=1e-6)


def test_single_frame_stack_hides_the_scrubber(widgets):
    sv = widgets.CakeStackViewer()
    sv.set_cakes(_stack(1), *_axes())
    assert sv.frame_count() == 1
    assert sv._scrub_bar.isHidden()


def test_clear_and_empty_stacks(widgets):
    sv = widgets.CakeStackViewer()
    sv.set_cakes(_stack(3), *_axes())
    sv.clear()
    assert sv.frame_count() == 0 and sv._cake is None
    assert sv._scrub_bar.isHidden()
    # A 2-D array or an empty stack clears rather than raising.
    sv.set_cakes(_stack(3), *_axes())
    sv.set_cakes(np.zeros((0, N_ETA, N_R)), *_axes())
    assert sv.frame_count() == 0
    sv.set_cakes(_stack(3), *_axes())
    sv.set_cakes(_stack(1)[0], *_axes())
    assert sv.frame_count() == 0


def test_frame_ids_fall_back_to_indices_when_mismatched(widgets):
    sv = widgets.CakeStackViewer()
    sv.set_cakes(_stack(3), *_axes(), frame_ids=["only-one"])
    assert sv._frame_ids == ["0", "1", "2"]


# ── BatchTab wiring ──────────────────────────────────────────────────

@pytest.fixture
def tab(app):
    from midas_gui.tab_batch import BatchTab
    return BatchTab()


def _multi_azimuth_payload(n_frames=4):
    r_axis, eta_axis = _axes()
    return {"n": n_frames, "out_paths": [], "aborted": False,
            "r_axis_px": r_axis, "eta_axis": eta_axis,
            "profiles": _stack(n_frames),
            "frame_ids": [f"f{i}" for i in range(n_frames)],
            "multi_azimuth": True}


def test_tab_has_the_eta_r_cakes_view(tab):
    titles = [tab._view_tabs.tabText(i) for i in range(tab._view_tabs.count())]
    assert "Eta-R cakes" in titles


def test_multi_azimuth_run_fills_the_cake_tab(tab):
    tab._update_cake_stack(_multi_azimuth_payload())
    assert tab._cake_stack_view.frame_count() == 4
    assert tab._cake_stack_view._frame_ids == ["f0", "f1", "f2", "f3"]


def test_a_collapsed_run_clears_the_cake_tab(tab):
    tab._update_cake_stack(_multi_azimuth_payload())
    tab._update_cake_stack({"n": 3, "r_axis_px": np.arange(N_R, dtype=float),
                            "profiles": np.zeros((3, N_R), dtype=np.float32),
                            "eta_axis": None})
    assert tab._cake_stack_view.frame_count() == 0


def test_collapse_cakes_averages_only_filled_eta_bins(tab):
    from midas_gui.tab_batch import BatchTab
    cakes = np.zeros((1, 4, 3), dtype=np.float64)
    cakes[0, 0] = [2.0, 4.0, 6.0]
    cakes[0, 1] = [4.0, 8.0, 12.0]
    # rows 2 and 3 are unfilled coverage, not measured zeros
    out = BatchTab._collapse_cakes(cakes)
    assert out.shape == (1, 3)
    assert out[0] == pytest.approx([3.0, 6.0, 9.0])


def test_project_attempt_round_trip_restores_the_cakes(tab, tmp_path):
    """A multi-azimuth attempt written by _log_to_project comes back into the
    cake tab — and the 1-D views survive it, which they did not before."""
    from midas_gui import project

    proj_path = str(tmp_path / "proj.h5")
    project.create_project(proj_path)
    ctx = project.ProjectContext()
    ctx.path = proj_path
    tab.set_project_context(ctx)
    tab._last_run_inputs = {"src_cfg": {}, "kernel": "subpixel2",
                            "multi_azimuth": True}
    tab._last_run_fields = {"mask": None, "mask_is_file_backed": False}

    payload = _multi_azimuth_payload()
    tab._log_to_project(payload)

    ref = "/analysis/integrate/single/attempt_0001"
    meta = project.read_attempt(proj_path, ref)
    meta["_results_arrays"] = project.read_attempt_results(proj_path, ref)
    assert meta["_results_arrays"]["profiles"].shape == (4, N_ETA, N_R)
    assert meta["n_eta_bins"] == N_ETA

    fresh_tab = type(tab)()
    fresh_tab._populate_plots_from_attempt(meta)
    assert fresh_tab._cake_stack_view.frame_count() == 4
    assert fresh_tab._cake_stack_view._eta_axis == pytest.approx(_axes()[1])
    # ...and the 1-D views got an η-collapse, not the raw cakes.
    assert fresh_tab._waterfall._nrows == 4
    assert fresh_tab._waterfall._buf.shape[1] == N_R


def test_restoring_a_1d_attempt_leaves_the_cake_tab_empty(tab, tmp_path):
    from midas_gui import project

    proj_path = str(tmp_path / "proj.h5")
    project.create_project(proj_path)
    ref = project.append_integration_attempt(
        proj_path, "single", inputs={},
        finished_payload={"n": 3, "aborted": False,
                          "profiles": np.random.rand(3, N_R).astype(np.float32),
                          "r_axis_px": np.arange(N_R, dtype=float),
                          "frame_ids": ["a", "b", "c"]})
    meta = project.read_attempt(proj_path, ref)
    meta["_results_arrays"] = project.read_attempt_results(proj_path, ref)

    tab._cake_stack_view.set_cakes(_stack(2), *_axes())   # stale cakes present
    tab._populate_plots_from_attempt(meta)
    assert tab._cake_stack_view.frame_count() == 0
    assert tab._waterfall._nrows == 3


def test_attempt_without_a_recorded_eta_axis_falls_back_to_full_turn(tab):
    """Attempts logged before eta_axis_deg was recorded still render — the η
    axis is reconstructed as evenly spaced bins over a full turn."""
    meta = {"_results_arrays": {"profiles": _stack(2),
                                "r_axis_px": _axes()[0],
                                "frame_ids": ["a", "b"]}}
    tab._populate_plots_from_attempt(meta)
    assert tab._cake_stack_view.frame_count() == 2
    eta = tab._cake_stack_view._eta_axis
    assert eta.size == N_ETA
    step = 360.0 / N_ETA
    assert eta[0] == pytest.approx(-180.0 + step / 2)
    assert eta[-1] == pytest.approx(180.0 - step / 2)


def test_cake_display_state_is_persisted_by_the_tab(tab):
    tab._cake_stack_view.set_cake(_stack(1)[0], *_axes())
    tab._cake_stack_view.set_axis_context(LSD, PX, WL)
    tab._cake_stack_view._xunit.setCurrentIndex(
        tab._cake_stack_view._xunit.findData("d"))
    state = tab.get_state()
    assert state["cake_stack"]["xunit"] == "d"

    fresh = type(tab)()
    fresh.set_state(state)
    assert fresh._cake_stack_view.x_unit() == "d"


# ── Q-uniform: the axis the views are told they have ────────────────────

def test_q_uniform_profiles_carry_an_R_axis_not_a_Q_one():
    """Reported as "2theta is 180 deg" and a waterfall x-axis of -2.264e-11.

    Q-uniform is a REBIN of an R-uniform integration, and the axis that
    travels with the rebinned profile is rebin_grid_and_r's second return
    value — the R in PIXELS of each Q bin — not the Q grid. Telling
    the views the native unit was "Q" made _convert_radial read those pixel
    values as inverse angstroms: x·λ/4π exceeds 1 for any real radius, so
    it clips and every bin lands on 2θ = 180° exactly, which converts back
    to R = Lsd·tan(180°)/px ≈ -2.26e-11.
    """
    from midas_gui.widgets import _convert_radial
    from midas_gui.workers import rebin_grid_and_r

    lsd, px, wl = 13866027.0, 75.0, 0.13060
    qgrid, r_of_q = rebin_grid_and_r(
        {"QMin": 0.5, "QMax": 8.0, "QBinSize": 0.01}, lsd, px, wl)

    # The bug, reproduced exactly.
    wrong = _convert_radial(r_of_q, lsd, px, wl, "Q", "2th")
    assert np.allclose(wrong, 180.0), "the reported symptom no longer reproduces"
    assert _convert_radial(r_of_q, lsd, px, wl, "Q", "R")[0] == \
        pytest.approx(-2.2641e-11, rel=1e-3)

    # Read as what it is, every unit comes out physical and Q round-trips.
    tth = _convert_radial(r_of_q, lsd, px, wl, "R", "2th")
    assert 0.0 < tth.min() < tth.max() < 20.0
    assert np.allclose(_convert_radial(r_of_q, lsd, px, wl, "R", "Q"), qgrid,
                       rtol=1e-9)


@pytest.mark.parametrize("module", ["tab_batch.py", "hydra_batch_page.py"])
def test_no_tab_declares_Q_as_the_native_axis_unit(module):
    """The fix at its source. Both tabs that can run a Q-uniform batch had
    the same expression; hydra_batch_page kept it for a while after
    tab_batch was fixed (4cb7aaa), so the guard scans both."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "midas_gui" / module
    assert '"Q" if q_cfg else "R"' not in src.read_text(), \
        f"{module}: a set_axis_context site still labels the R-px axis as Q"


def test_the_batch_tab_has_exactly_the_two_known_axis_context_sites():
    """A third would be a site nobody has checked declares "R"."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "midas_gui" / "tab_batch.py"
    assert src.read_text().count("_axctx = ") == 2


def test_2theta_uniform_profiles_also_carry_an_R_axis():
    """The 180° symptom must not reappear in the unit added after the fix:
    a 2θ rebin carries R in pixels too, and reading those pixels as degrees
    would put every bin past the detector."""
    from midas_gui.widgets import _convert_radial
    from midas_gui.workers import rebin_grid_and_r

    lsd, px, wl = 13866027.0, 75.0, 0.13060
    grid, r_of_grid = rebin_grid_and_r(
        {"unit": "2th", "min": 0.5, "max": 9.0, "step": 0.01}, lsd, px, wl)

    # Read as what it is, the axis round-trips to the 2theta grid exactly.
    assert np.allclose(
        _convert_radial(r_of_grid, lsd, px, wl, "R", "2th"), grid, rtol=1e-12)
    # Read as degrees (the analogous mistake), it stops being a radial axis
    # at all: tan() wraps every 180 deg, so 1630..29266 "degrees" scatters
    # into huge values of both signs instead of rising monotonically.
    wrong = _convert_radial(r_of_grid, lsd, px, wl, "2th", "R")
    assert np.any(np.diff(wrong) < 0) and wrong.min() < 0, \
        "misreading the axis as degrees should not yield a plausible radius"

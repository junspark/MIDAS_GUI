"""``widgets.RadialThresholdEditor`` — the interactive drag-point curve used
by the Calibrate tab's and Hydra page's "radially adaptive threshold" card
(replacing both the earlier Gaussian and power-law parametric versions; see
.context/DECISIONS.md).

Drives the widget through its public methods and by calling
``pg.TargetItem.setPos()`` directly (which fires the real
``sigPositionChanged`` handler the same way a mouse drag would) — this
codebase has no ``QTest``-based mouse-event simulation anywhere, so every
interactive widget's tests touch internals directly instead. The
double-click-to-add/remove scene-hit-test plumbing (``_on_plot_clicked``)
is therefore NOT covered here (it would need real/simulated Qt mouse
events) — it's checked only via an offscreen screenshot, the same tier of
coverage already accepted for ``PickableImageViewer``'s own click-to-pick
wiring.

Its own module (and ``forked``) following ``tests/test_calib_radial_threshold.py``'s
convention — building Qt widgets with pyqtgraph plots in the same process
as other such tests risks the documented ViewBox teardown crash.
"""
import math

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.forked
def test_set_points_and_points_roundtrip_sorts_by_radius(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([50.0, 0.0, 120.0], [10.0, 100.0, 0.0])

    radii, values = w.points()
    assert list(radii) == [0.0, 50.0, 120.0]
    assert values == pytest.approx([100.0, 10.0, 1.0])   # 0.0 floors to 1.0 (log-scale "zero")


@pytest.mark.forked
def test_points_are_floored_at_one_not_zero(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(100.0)
    w.set_points([0.0, 50.0], [10.0, -5.0])   # a negative input is floored too

    assert w.points()[1] == pytest.approx([10.0, RadialThresholdEditor._Y_FLOOR])
    assert RadialThresholdEditor._Y_FLOOR == 1.0


@pytest.mark.forked
def test_y_values_round_trip_through_log_display(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(1000.0)
    w.set_points([0.0, 20.0, 500.0], [12345.0, 500.0, 3.7])

    radii, values = w.points()
    assert values == pytest.approx([12345.0, 500.0, 3.7])


@pytest.mark.forked
def test_x_axis_mouse_disabled_y_axis_mouse_enabled(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    vb = w._plot.getPlotItem().getViewBox()
    assert vb.state["mouseEnabled"] == [False, True]


@pytest.mark.forked
def test_set_domain_locks_the_x_range_limits(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(250.0)
    vb = w._plot.getPlotItem().getViewBox()
    assert vb.state["limits"]["xLimits"] == [0.0, 250.0]


@pytest.mark.forked
def test_drag_clamps_at_neighbors_so_points_cannot_cross(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 50.0, 120.0], [100.0, 50.0, 0.0])

    middle = w._targets[1]
    middle.setPos(300.0, 50.0)   # try to drag the middle point past the right one

    radii, values = w.points()
    assert list(radii) == sorted(radii)
    assert radii[1] < radii[2]   # still strictly between its neighbours


@pytest.mark.forked
def test_drag_clamps_y_to_the_floor_but_allows_a_bump_above_a_neighbor(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 50.0, 120.0], [10.0, 10.0, 10.0])

    middle = w._targets[1]
    middle.setPos(50.0, -5.0)   # a target's y is stored in log10 space
    assert w.points()[1][1] == pytest.approx(RadialThresholdEditor._Y_FLOOR)

    middle.setPos(50.0, math.log10(500.0))   # a "bump" above both neighbours is allowed
    assert w.points()[1][1] == pytest.approx(500.0)


@pytest.mark.forked
def test_drag_clamps_r_to_the_set_domain(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(100.0)
    w.set_points([0.0, 100.0], [10.0, 0.0])

    last = w._targets[-1]
    last.setPos(500.0, 0.0)
    assert w.points()[0][-1] == pytest.approx(100.0)


@pytest.mark.forked
def test_add_point_increases_count_and_emits_pointsChanged(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 100.0], [10.0, 0.0])
    seen = []
    w.pointsChanged.connect(lambda: seen.append(True))

    w._add_point(50.0, 5.0)
    radii, _ = w.points()
    assert len(radii) == 3
    assert seen


@pytest.mark.forked
def test_add_point_is_a_noop_beyond_max_points(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(1000.0)
    radii = list(np.linspace(0, 900, RadialThresholdEditor.MAX_POINTS))
    values = [0.0] * len(radii)
    w.set_points(radii, values)

    w._add_point(950.0, 1.0)
    assert len(w.points()[0]) == RadialThresholdEditor.MAX_POINTS


@pytest.mark.forked
def test_remove_point_decreases_count_and_emits_pointsChanged(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 50.0, 100.0], [10.0, 5.0, 0.0])
    seen = []
    w.pointsChanged.connect(lambda: seen.append(True))

    w._remove_point(w._targets[1])
    radii, _ = w.points()
    assert len(radii) == 2
    assert seen


@pytest.mark.forked
def test_remove_point_is_a_noop_at_min_points(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 100.0], [10.0, 0.0])

    w._remove_point(w._targets[0])
    assert len(w.points()[0]) == RadialThresholdEditor.MIN_POINTS


@pytest.mark.forked
def test_set_editable_toggles_widget_enabled_and_target_movable(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 50.0, 100.0], [10.0, 5.0, 0.0])

    w.set_editable(False)
    assert w.isEnabled() is False
    assert all(t.movable is False for t in w._targets)

    w.set_editable(True)
    assert w.isEnabled() is True
    assert all(t.movable is True for t in w._targets)


@pytest.mark.forked
def test_pick_state_roundtrip(app):
    from midas_gui.widgets import RadialThresholdEditor

    w = RadialThresholdEditor()
    w.set_domain(200.0)
    w.set_points([0.0, 50.0, 100.0], [10.0, 5.0, 0.0])
    state = w.pick_state()

    w2 = RadialThresholdEditor()
    w2.set_domain(200.0)
    w2.set_pick_state(state)
    radii, values = w2.points()
    assert list(radii) == [0.0, 50.0, 100.0]
    assert values == pytest.approx([10.0, 5.0, 1.0])   # 0.0 floors to 1.0 (log-scale "zero")

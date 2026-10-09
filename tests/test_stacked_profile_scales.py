"""Stacked profiles can be read on log-x, log-y and sqrt-y.

Asked for at 1-ID-E: "It will be great to have option to show this plot with
log-y and log-x scales. sqrt(y) is also a good option."

SAXS is what forces it. An AgBeh profile on the pixirad runs ~12000 counts
down to the noise inside the first 0.05 A^-1 of a 1.3 A^-1 axis, so linear-
linear is one spike against a flat line — the screenshot that prompted this
showed exactly that.

Three things here are easy to get wrong and are each pinned below:

* the transform has to be applied BEFORE the stack offset. log10(I + offset)
  compresses each successive frame more than the last, so the stack fans
  closed towards the top instead of staying evenly spaced;
* "spacing" means something different per scale. 500 is a sensible linear
  offset and an absurd number of decades, so each scale keeps its own;
* values with no image under the transform are dropped, not clamped.
  Background-subtracted SAXS goes negative routinely.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked

LOG = "log₁₀(I)"
SQRT = "√I"


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _saxs():
    """A profile shaped like the one that prompted this: steep decay, then a
    background-subtracted tail that goes negative."""
    q = np.linspace(0.01, 1.3, 200)
    prof = 12000.0 * np.exp(-q / 0.01) + 5.0
    prof[-40:] = -2.0
    return q, prof


@pytest.fixture
def view(app):
    from midas_gui.widgets import StackedProfileViewer
    v = StackedProfileViewer()
    q, prof = _saxs()
    for i in range(3):
        v.add_profile(q, prof * (1.0 + 0.1 * i), f"frame{i}")
    return v


def test_all_three_scales_are_offered(view):
    got = [view._yscale_combo.itemText(i) for i in range(view._yscale_combo.count())]
    assert got == ["Linear", LOG, SQRT]


def test_log_and_sqrt_drop_what_they_cannot_represent(view):
    """Not clamped to a floor: a plateau the detector never saw reads as
    data. pyqtgraph breaks the line on NaN, which reads as absence."""
    _q, prof = _saxs()
    view._yscale_combo.setCurrentText(LOG)
    y = view._y_display(prof)
    assert np.isnan(y[-1]), "a non-positive intensity survived log10"
    assert np.isfinite(y[0])
    view._yscale_combo.setCurrentText(SQRT)
    y = view._y_display(prof)
    assert np.isnan(y[-1]), "a negative intensity survived sqrt"


def test_linear_is_untouched(view):
    _q, prof = _saxs()
    view._yscale_combo.setCurrentText("Linear")
    assert np.array_equal(view._y_display(prof), prof)


def test_the_transform_is_applied_before_the_stack_offset(app):
    """The one that matters. With the offset added first, successive frames
    compress towards the top and the stack fans closed; applied first, every
    frame sits exactly `spacing` apart whatever the scale.

    IDENTICAL profiles on purpose. The shared `view` fixture scales each
    frame by (1 + 0.1*i), and under log that contributes its own
    log10(k_{i+1}/k_i) to every gap -- 0.0414 between the first two -- which
    is correct behaviour but confounds the thing being measured here.
    """
    from midas_gui.widgets import StackedProfileViewer
    v = StackedProfileViewer()
    q, prof = _saxs()
    for i in range(3):
        v.add_profile(q, prof.copy(), f"frame{i}")
    v._yscale_combo.setCurrentText(LOG)
    v._spacing.setValue(1.0)
    v._restack()
    firsts = []
    for curve in v._curves:
        _xd, yd = curve.getData()
        firsts.append(yd[np.isfinite(yd)][0])
    gaps = np.diff(firsts)
    assert np.allclose(gaps, 1.0, atol=1e-9), f"stack is not evenly spaced: {gaps}"


def test_offsetting_before_the_transform_would_be_visibly_wrong(app):
    """Shows what the previous test is protecting against, so the atol in it
    is not mistaken for fussiness: log10(I + offset) leaves gaps that shrink
    with every frame."""
    _q, prof = _saxs()
    spacing = 1.0
    wrong = [np.log10(prof[0] + i * spacing) for i in range(3)]
    gaps = np.diff(wrong)
    assert not np.allclose(gaps, spacing, atol=0.5)
    assert gaps[1] < gaps[0], "expected the fan to close towards the top"


def test_each_scale_keeps_its_own_spacing(view):
    def hop(name):
        view._yscale_combo.setCurrentText(name)
        return view._spacing.value()

    assert (hop("Linear"), hop(LOG), hop(SQRT)) == (500.0, 0.5, 10.0)
    hop(LOG); view._spacing.setValue(1.25)
    hop("Linear"); view._spacing.setValue(750.0)
    assert hop(LOG) == 1.25
    assert hop("Linear") == 750.0
    assert hop(SQRT) == 10.0, "an untouched scale lost its default"


def test_the_y_label_says_which_scale_is_showing(view):
    for name, want in (("Linear", "Intensity + offset"),
                       (LOG, "log₁₀(Intensity) + offset"),
                       (SQRT, "√Intensity + offset")):
        view._yscale_combo.setCurrentText(name)
        assert view._plot.getPlotItem().getAxis("left").labelText == want


def test_log_x_does_not_clamp_the_view_to_zero(view):
    """The linear path floors xMin at 0. In log x the axis is legitimately
    negative (Q = 0.01 is -2), so that floor would pin the view to the wrong
    end of the data."""
    view._logx_chk.setChecked(True)
    view._restack()
    vb = view._plot.getPlotItem().getViewBox()
    assert vb.state["limits"]["xLimits"][0] < 0.0


def test_toggling_scales_and_log_x_never_raises(view):
    for name in ("Linear", LOG, SQRT, LOG, "Linear"):
        view._yscale_combo.setCurrentText(name)
        for on in (True, False, True):
            view._logx_chk.setChecked(on)
    assert True

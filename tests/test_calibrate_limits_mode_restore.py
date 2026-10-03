"""Reopening a project must bring back the Limits column its calibrant needs.

``_set_state`` used to pin ``_limits_mode_is_dsp`` to the saved calibrant's
kind before calling ``_on_calibrant_changed``, to stop ``_sync_limits_mode``
overwriting the rows that had just been restored from "fields". But that sync
*shapes* the card as well as filling it — which rows exist, the header
wording, whether the tilt window spans ty+tz — so the pin made it early-return
and a project saved on AgBH came back wearing the crystalline card the tab was
built with: no tx/tz/BC_z windows to edit, a Distortion row the manual fit
ignores, and a footer quoting backend tolerances that fit never sees. The run
button, set after that guard, correctly read "Fit Geometry (manual)", so the
card and the button disagreed about which fit was about to run.

Own module + ``forked`` per .context/DECISIONS.md: each test builds two
CalibrationTabs for the round trip, and enough accumulated pyqtgraph instances
in one process segfaults during teardown.
"""
import pytest

pytestmark = pytest.mark.forked

_XTAL_HDR = "the MIDAS backend always bounds"
_DSP_HDR = "bound a refined parameter"


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _agbh(tab):
    return next(tab._cal.itemText(i) for i in range(tab._cal.count())
                if "AgBH" in tab._cal.itemText(i))


def test_dspacing_project_reopens_with_the_manual_limits_card(app):
    """A project saved on AgBH must reopen on the manual card — and keep the
    windows the user had set, which is what the pin was protecting."""
    from midas_gui.tab_calibrate import CalibrationTab

    src = CalibrationTab()
    src._cal.setCurrentText(_agbh(src))
    assert _DSP_HDR in src._limits_hdr.text()
    # Two rows the crystalline card has no control for at all.
    for slot, val in (("tz", 2.5), ("tx", 1.25)):
        src._limit_widgets[slot][0].setChecked(True)
        src._limit_widgets[slot][1].setValue(val)
    state = src.get_state()

    # A fresh tab cold-starts on a crystalline calibrant, so the restore has
    # to cross modes — the case the pin broke.
    dst = CalibrationTab()
    assert _XTAL_HDR in dst._limits_hdr.text()
    dst.set_state(state)

    assert dst._limits_mode_is_dsp is True
    assert dst._run_btn.text() == "Fit Geometry (manual)"
    assert _DSP_HDR in dst._limits_hdr.text()
    assert "Always applied" not in dst._limits_note.text()
    assert dst._limits["tz"] == {"on": True, "value": 2.5, "unit": "°"}
    assert dst._limits["tx"] == {"on": True, "value": 1.25, "unit": "°"}


def test_crystalline_project_reopens_with_the_crystalline_limits_card(app):
    """The other direction, so the fix is not just "always use the dsp card":
    restoring into a tab left on AgBH must shape the card back."""
    from midas_gui.tab_calibrate import CalibrationTab

    src = CalibrationTab()
    src._cal.setCurrentText("CeO2")
    src._limit_widgets["Lsd"][1].setValue(22.0)
    state = src.get_state()

    dst = CalibrationTab()
    dst._cal.setCurrentText(_agbh(dst))
    assert _DSP_HDR in dst._limits_hdr.text()
    dst.set_state(state)

    assert dst._limits_mode_is_dsp is False
    assert dst._run_btn.text() == "Run Calibration"
    assert _XTAL_HDR in dst._limits_hdr.text()
    assert dst._limits_note.text().startswith("Always applied")
    assert dst._limits["Lsd"]["value"] == pytest.approx(22.0)
    # Crystalline windows always apply, so the row comes back on whatever the
    # saved tick said.
    assert dst._limits["Lsd"]["on"] is True

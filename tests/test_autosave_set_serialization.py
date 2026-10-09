"""A set in a calibration result must not take session autosave down.

Observed at 1-ID-E: ``~/midas_gui_error.log`` had **1724** copies of

    Session autosave failed:
      File ".../midas_gui/app.py", in _autosave_tick
    TypeError: Object of type set is not JSON serializable

one every few minutes across a run of experiments. Each one is a
crash-recovery draft that was never written — the failure is caught and
logged, so the GUI looks fine and the safety net is simply absent.

The set is ``result.fit_at_limit``, which ``workers.py`` builds with
``set(fit["at_limit"])``. ``project.sanitize_result_dict`` passes every
non-tensor attribute straight through, so it reached
``state["tabs"]["Calibrate"]["result"]["fit_at_limit"]`` — three dict
levels down, which is exactly where the traceback's three
``_iterencode_dict`` frames ended.

Fixed at the serialization boundary rather than at that one field: the
result object comes from the backend and gains attributes between
releases, so the next set-valued one would reintroduce this.
"""
import json
from types import SimpleNamespace

import pytest

from midas_gui import project


def test_a_set_valued_result_field_survives_json():
    d = project.sanitize_result_dict(
        SimpleNamespace(Lsd=2.4e6, fit_at_limit={"tx", "Lsd", "BC"}))
    assert d["fit_at_limit"] == ["BC", "Lsd", "tx"], "not sorted, or not a list"
    json.dumps(d)


def test_sets_are_converted_wherever_they_sit():
    """Not just the one known field — nested dicts and lists too, since the
    point is to stop the NEXT set-valued attribute doing this again."""
    d = project.sanitize_result_dict(SimpleNamespace(
        distortion={"chosen": {"p1", "p0"}},
        history=[{"flagged": {"ty"}}]))
    assert d["distortion"]["chosen"] == ["p0", "p1"]
    assert d["history"][0]["flagged"] == ["ty"]
    json.dumps(d)


def test_the_ordering_is_stable_so_a_saved_project_does_not_churn():
    a = project.sanitize_result_dict(SimpleNamespace(f={"b", "a", "c"}))
    b = project.sanitize_result_dict(SimpleNamespace(f={"c", "a", "b"}))
    assert a["f"] == b["f"] == ["a", "b", "c"]


def test_everything_else_is_left_alone():
    """The sanitizer's existing contract: keep every non-tensor field, drop
    torch tensors (duck-typed on .numpy)."""
    tensor = SimpleNamespace(numpy=lambda: None)
    d = project.sanitize_result_dict(SimpleNamespace(
        Lsd=2.4e6, name="ceo2", flags=[1, 2], nested={"a": 1}, big=tensor))
    assert d["Lsd"] == 2.4e6 and d["name"] == "ceo2"
    assert d["flags"] == [1, 2] and d["nested"] == {"a": 1}
    assert "big" not in d


def test_none_is_still_none():
    assert project.sanitize_result_dict(None) is None


@pytest.mark.forked
def test_the_whole_autosave_payload_serialises(tmp_path):
    """End to end on the real path: a Calibrate tab holding a fitted result
    whose fit_at_limit is a set, serialised the way _autosave_tick does."""
    pytest.importorskip("PyQt5.QtWidgets")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from midas_gui.tab_calibrate import CalibrationTab
    tab = CalibrationTab()
    tab._result = SimpleNamespace(
        Lsd=2.39e6, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.0, tz=0.0,
        pxY=200.0, pxZ=200.0, NrPixelsY=2048, NrPixelsZ=2048,
        wavelength_A=0.1536, distortion={}, fit_at_limit={"tx", "Lsd"})
    state = tab.get_state()
    assert state["result"]["fit_at_limit"] == ["Lsd", "tx"]
    json.dumps({"tabs": {"Calibrate": state}})
    app.processEvents()

"""paramstest.txt must report the calibrant that was actually used.

Found at 1-ID-E: a calibration run on silver behenate reported
``SpaceGroup: 225`` and ``LatticeConstant: 5.4116 …`` in the Results panel —
ceria's structure, for a run that never saw any ceria. The cause was
``_SG.get(cal, 225)`` / ``_LC.get(cal, _LC["CeO2"])`` in
``write_standalone_paramstest``: AgBH is a d-spacing calibrant with no entry
in either table, so both fell through to the CeO2 default.

A d-spacing calibrant has no cell to report, so the fields are left unset and
the d-spacings the fit actually used are reported instead.
"""
from types import SimpleNamespace

import pytest

from midas_gui.helpers import paramstest_pairs

AGBH = "AgBH (silver behenate)"
CEO2_A = 5.4116


def _result(name, **kw):
    r = SimpleNamespace(
        Lsd=6453967.0, BC_y=91.923, BC_z=78.496, tx=0.0, ty=-0.11, tz=-0.29,
        distortion={}, pxY=62.0, pxZ=62.0, NrPixelsY=1024, NrPixelsZ=402,
        wavelength_A=0.153590, post_residual_strain_uE=None,
        _calibrant_name=name)
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def _pairs(name, **kw):
    return dict(paramstest_pairs(_result(name, **kw)))


# ── the bug ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cal", [AGBH, "Custom d-spacings…"])
def test_a_dspacing_calibrant_never_reports_ceria(cal):
    d = _pairs(cal, _d_used=[58.380, 29.190, 19.460])
    assert d["SpaceGroup"] == "0", "a non-crystalline calibrant has no space group"
    assert str(CEO2_A) not in d["LatticeConstant"], (
        f"ceria's cell leaked into a {cal} calibration: {d['LatticeConstant']}")
    assert float(d["LatticeConstant"].split()[0]) == 0.0


@pytest.mark.parametrize("cal", [AGBH, "Custom d-spacings…"])
def test_it_reports_the_calibrant_and_its_dspacings_instead(cal):
    d = _pairs(cal, _d_used=[58.380, 29.190, 19.460])
    assert d["Calibrant"] == cal
    assert [float(x) for x in d["DSpacings"].split()] == [58.380, 29.190, 19.460]


# ── which d-spacings ─────────────────────────────────────────────────────
def test_it_reports_the_rings_fitted_not_the_whole_material_table():
    """AgBH ships 10 d-spacings; a fit from 3 picked rings reports 3."""
    d = _pairs(AGBH, _d_list=[58.380 / n for n in range(1, 11)],
               _d_used=[58.380, 29.190, 19.460])
    assert len(d["DSpacings"].split()) == 3


def test_it_falls_back_to_the_material_table_when_picks_are_unrecorded():
    """Older results carry only ``_d_list``; reporting those beats nothing."""
    d = _pairs(AGBH, _d_list=[58.380, 29.190])
    assert [float(x) for x in d["DSpacings"].split()] == [58.380, 29.190]


def test_no_dspacings_line_when_there_is_nothing_to_report():
    d = _pairs(AGBH)
    assert "DSpacings" not in d
    assert d["Calibrant"] == AGBH      # the name is still worth recording


# ── crystalline calibrants must be untouched ─────────────────────────────
@pytest.mark.parametrize("cal,sg,a", [
    ("CeO2", "225", 5.4116), ("LaB6", "221", 4.15692),
    ("Si", "227", 5.43102), ("Al2O3", "167", 4.7589)])
def test_crystalline_calibrants_still_report_their_own_cell(cal, sg, a):
    d = _pairs(cal)
    assert d["SpaceGroup"] == sg
    assert float(d["LatticeConstant"].split()[0]) == pytest.approx(a)
    # Their structure already says what they are; no redundant provenance.
    assert "Calibrant" not in d

"""The ``.dat`` writer must hand GSAS-II the wavelength it was taken at.

GSAS-II's SAXS reader (``GSASII/imports/G2sad_xye.py``,
``txt_XRayReaderClass``) opens with ``wave = 1.5428  #Cuka default`` and only
ever overrides it from a header line, so a ``.dat`` with no wavelength line
loads as Cu Ka no matter the beamline energy — silently, with no warning in
the GUI. Found at 1-ID-E: an 80.725 keV (0.1536 A) Pixirad SAXS pattern
loaded into GSAS-II showing ``Lam (A): 1.542800``.

These tests re-implement GSAS-II's parse loop verbatim rather than asserting
on our own header string, so they fail if the file stops being readable *by
that reader* — which is the actual contract.
"""
import numpy as np
import pytest

from midas_gui.helpers import load_profile_file
from midas_gui.workers import write_profile

GSAS2_DEFAULT_WAVE = 1.5428      # G2sad_xye.py: "#Cuka default"


def _gsas2_wave(path):
    """GSAS-II's own wavelength extraction, transcribed from
    ``txt_XRayReaderClass.Reader``."""
    wave = GSAS2_DEFAULT_WAVE
    with open(path) as fh:
        for S in fh:
            if '=' in S:
                if 'wave' in S.split('=')[0].lower():
                    try:
                        wave = float(S.split('=')[1])
                    except Exception:
                        pass
    return wave


def _write(tmp_path, wl):
    r = np.linspace(1.0, 200.0, 40)
    prof = np.exp(-((r - 100.0) ** 2) / 500.0) * 1000.0
    base = str(tmp_path / "profile")
    write_profile(base, "dat", r, prof, np.sqrt(prof),
                  lsd=200_000.0, px=62.0, wl=wl)
    return base + ".dat", r, prof


@pytest.mark.parametrize("wl", [0.153588, 0.1729, 1.5428, 0.02])
def test_gsas2_reads_back_the_wavelength_we_wrote(tmp_path, wl):
    path, _, _ = _write(tmp_path, wl)
    assert _gsas2_wave(path) == pytest.approx(wl, rel=1e-7)


def test_the_header_line_survives_gsas2s_float_parse(tmp_path):
    """``float(S.split('=')[1])`` is unforgiving: a second ``=`` on the line
    or a trailing unit makes it raise, and GSAS-II swallows that and keeps
    Cu Ka. Pin both properties of the line we emit."""
    path, _, _ = _write(tmp_path, 0.153588)
    hdr = [l for l in open(path) if '=' in l and 'wave' in l.split('=')[0].lower()]
    assert len(hdr) == 1, f"expected exactly one wavelength line, got {hdr}"
    assert hdr[0].count('=') == 1, "a second '=' breaks split('=')[1]"
    float(hdr[0].split('=')[1])          # must not raise, and no units allowed


# ``None`` is deliberately absent: without a wavelength there is no Q axis
# at all, so ``axis_conversions`` raises long before the header is built.
# A ``.dat`` is impossible in that case and failing loudly is correct.
@pytest.mark.parametrize("wl", [0.0, -1.0, float("nan"), float("inf")])
def test_a_meaningless_wavelength_is_omitted_not_written(tmp_path, wl):
    """Writing ``wavelength = 0`` would read back as an assertion that the
    wavelength really is zero. Better to emit nothing and let GSAS-II fall
    back to its own default, which is at least visibly wrong."""
    path, _, _ = _write(tmp_path, wl)
    assert _gsas2_wave(path) == GSAS2_DEFAULT_WAVE


def test_header_does_not_disturb_our_own_round_trip(tmp_path):
    """``load_profile_file`` skips ``#`` lines; the data must be untouched."""
    path, r, prof = _write(tmp_path, 0.153588)
    x, y, sig = load_profile_file(path)
    assert y.shape == r.shape
    np.testing.assert_allclose(y, prof, rtol=1e-4)
    assert sig is not None

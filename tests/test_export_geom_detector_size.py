"""The exported detector size must describe the data, not a stale calibration.

``DetectorGeometryCard._export_geom`` builds its result from the loaded
calibration and then overrides the fields the user has edited. ``NrPixelsY``/
``NrPixelsZ`` have no live widget, so they were never in that override set and
were always inherited from the loaded file -- even with a completely different
detector on screen.

Measured at 1-ID-E on 2026-10-08: a CeO2 GE calibration (2048x2048, 200 um)
was loaded, Pixirad SAXS frames (1024x402, 62 um) were opened, Lsd/BC/px were
edited, and "save instrument parameters" wrote::

    px          62.000000     <- the edit landed
    NrPixelsY   2048          <- the GE's, not the Pixirad's
    NrPixelsZ   2048
    RhoD        2774.198856   <- derived from the wrong size

Batch Integrate then died inside the backend with "image shape (402, 1024)
does not match geometry (2048, 2048)". RhoD matters as much as the shape: it
normalises the distortion polynomial, and the correct value here is ~986 px.

``forked`` per .context/DECISIONS.md -- builds real pyqtgraph widgets.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.forked

# The GE CeO2 calibration that was loaded on the card.
GE_GEOM = {
    "wavelength_A": 0.1536, "Lsd": 2392223.0, "BC_y": 1024.0, "BC_z": 1024.0,
    "tx": 300.0, "ty": 0.0826, "tz": 0.372, "pxY": 200.0, "pxZ": 200.0,
    "NrPixelsY": 2048, "NrPixelsZ": 2048,
    "distortion": {"iso_R2": -0.0017338}, "im_trans": [],
}
PIXIRAD = (402, 1024)        # (rows, cols) = (NrPixelsZ, NrPixelsY)


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _card(app, frame):
    from midas_gui.hydra_geometry_card import DetectorGeometryCard
    c = DetectorGeometryCard()
    c.set_image_source(lambda: frame, None)
    return c


def test_pixirad_frames_export_pixirad_size_not_the_loaded_ge_size(app):
    card = _card(app, np.zeros(PIXIRAD, dtype=np.float32))
    card.set_geometry(GE_GEOM)          # GE calibration loaded on the card
    geom = card._export_geom()
    assert (geom["NrPixelsY"], geom["NrPixelsZ"]) == (1024, 402)


def test_the_rest_of_the_loaded_calibration_still_survives(app):
    """Only the size follows the data -- distortion and untouched boxes are
    still the loaded calibration's, or the fix would be throwing away the
    thing the user loaded."""
    card = _card(app, np.zeros(PIXIRAD, dtype=np.float32))
    card.set_geometry(GE_GEOM)
    geom = card._export_geom()
    assert geom["distortion"]["iso_R2"] == pytest.approx(-0.0017338)
    assert geom["tx"] == pytest.approx(300.0, abs=0.01)


def test_rho_d_follows_the_corrected_size(app, tmp_path):
    """RhoD normalises the distortion polynomial and is derived from the
    detector size, so the wrong size silently rescales every coefficient."""
    from types import SimpleNamespace
    from midas_gui.helpers import write_standalone_paramstest
    card = _card(app, np.zeros(PIXIRAD, dtype=np.float32))
    card.set_geometry(GE_GEOM)
    g = card._export_geom()
    ns = SimpleNamespace(
        NrPixelsY=g["NrPixelsY"], NrPixelsZ=g["NrPixelsZ"],
        pxY=62.0, pxZ=62.0, Lsd=6300000.0, BC_y=90.7, BC_z=82.0,
        tx=210.0, ty=0.0, tz=0.0, wavelength_A=0.153583,
        distortion=g["distortion"], im_trans=[2], _calibrant_name="CeO2")
    out = tmp_path / "p.txt"
    write_standalone_paramstest(ns, out)
    text = out.read_text()
    assert "NrPixelsY 1024" in text and "NrPixelsZ 402" in text
    rho = float([l for l in text.splitlines() if l.startswith("RhoD")][0].split()[1])
    assert rho == pytest.approx(986.5, abs=1.0)      # was 2774.2


def test_square_detector_matching_its_calibration_is_untouched(app):
    """The common case must not move: a GE calibration with GE data on screen
    exports exactly what was loaded."""
    card = _card(app, np.zeros((2048, 2048), dtype=np.float32))
    card.set_geometry(GE_GEOM)
    geom = card._export_geom()
    assert (geom["NrPixelsY"], geom["NrPixelsZ"]) == (2048, 2048)


def test_transpose_swaps_the_pair_back_to_raw_frame_order(app):
    """image_provider hands back the DISPLAY frame. With a transpose active a
    1024x402 Pixirad displays as 402 rows of 1024 -> the raw-frame counts the
    geometry dict wants are the swap of the displayed shape."""
    card = _card(app, np.zeros((1024, 402), dtype=np.float32))   # display, transposed
    card.set_geometry({**GE_GEOM, "im_trans": [3]})
    geom = card._export_geom()
    assert (geom["NrPixelsY"], geom["NrPixelsZ"]) == (1024, 402)


# ── Pixel size: the other half of "describe the detector on screen" ──────────

def test_pixi_files_are_recognised_as_pixirad():
    """1-ID-E writes Pixirad frames as ".pixi"; the detector table only had
    ".pxrd", so loading one auto-populated nothing and the px box kept the
    previously-loaded calibration's value (200 um, from a GE)."""
    from midas_gui.helpers import detect_detector_from_filename
    assert detect_detector_from_filename(
        "AgBeh_80p725keV_3s_003510.pixi.h5") == "pxrd"


def test_pixirad_pixel_size_auto_populates():
    from midas_gui.helpers import detect_geometry_from_path
    got = detect_geometry_from_path(
        "AgBeh_80p725keV_3s_003510.pixi.h5", profile="1-ID-E")
    assert got.get("pxY") == 62.0


def test_other_detectors_keep_their_pixel_sizes():
    from midas_gui.helpers import detect_geometry_from_path
    for name, px in (("x.ge3.h5", 200.0), ("x.vrx.h5", 150.0), ("x.pmg.h5", 55.0)):
        assert detect_geometry_from_path(name, profile="1-ID-E").get("pxY") == px


def test_detection_still_gated_to_the_profiles_that_use_the_convention():
    from midas_gui.helpers import detect_geometry_from_path
    assert detect_geometry_from_path("x.pixi.h5", profile="other") == {}


def test_card_remembers_the_pixel_size_the_data_implies(app):
    """A calibration loaded after the data overwrites the px box, so the
    detected value has to be kept separately for the save-time check."""
    card = _card(app, np.zeros(PIXIRAD, dtype=np.float32))
    card.apply_shared_fields({"pxY": 62.0})
    card.set_geometry(GE_GEOM)                  # GE calibration loaded on top
    assert card._px.value() == pytest.approx(200.0)   # box follows the load
    assert card._detected_px == pytest.approx(62.0)   # data's value survives

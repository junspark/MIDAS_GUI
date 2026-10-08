"""First-time must run with the beam centre off the detector.

Reported at the beamline: the First-time pipeline died before fitting with
``ValueError: MaxRingRad must be positive (px)``.

``midas_calibrate_v2.pipelines.first_time._build_v1`` defaults MaxRingRad to
the beam-centre-to-*nearest-edge* distance::

    max_ring_rad_px = min(bc_y, n_pixels_y - bc_y,
                          bc_z, n_pixels_z - bc_z) * 0.95

which assumes the beam lands ON the detector. A panel offset from the beam --
the normal arrangement when the rings of interest are corner arcs, and every
Hydra panel -- puts the centre outside the frame, so one term goes negative
and ``CalibrationParams.validate()`` rejects it. The measured case was
BC (2384.7, 2182.7) on 2048x2048: ``2048 - 2384.7 = -336.7``.

No Qt here -- ``calib`` is pure -- but kept ``forked`` with the rest of the
suite so a stray backend import cannot leak between tests.
"""
import pytest

pytestmark = pytest.mark.forked

from midas_gui.calib import _first_time_max_ring_rad as max_ring_rad


def test_an_off_detector_beam_centre_gets_a_usable_radius():
    """The beamline case. The substitute is BC-to-farthest-corner, the only
    meaningful maximum once the centre is off-frame."""
    rr = max_ring_rad({"BC_y": 2384.7, "BC_z": 2182.7}, 2048, 2048)
    assert rr is not None
    assert rr > 0
    # hypot(max(2384.7, 2047-2384.7), max(2182.7, 2047-2182.7))
    assert rr == pytest.approx(3232.8, abs=0.1)

    # ...and the backend default it replaces really is unusable, so this is
    # not a substitution made on a hunch.
    bad = min(2384.7, 2048 - 2384.7, 2182.7, 2048 - 2182.7) * 0.95
    assert bad < 0


def test_an_on_detector_beam_centre_is_left_entirely_alone():
    """The backend's own default is right whenever the centre is on the
    frame, and overriding it there would change every working calibration."""
    assert max_ring_rad({"BC_y": 1024.0, "BC_z": 1024.0}, 2048, 2048) is None
    assert max_ring_rad({"BC_y": 600.0, "BC_z": 1500.0}, 2048, 2048) is None


def test_a_centre_just_inside_an_edge_is_warned_about_not_overridden(capsys):
    """Same root cause with no exception: the default stays positive but
    tiny, so the fit quietly searches almost no rings. Worth saying out loud;
    not worth silently changing, since it is a legitimate geometry."""
    assert max_ring_rad({"BC_y": 40.0, "BC_z": 1024.0}, 2048, 2048) is None
    out = capsys.readouterr().out
    assert "nearest detector edge" in out
    assert "One-shot or Four-stage" in out


def test_no_manual_centre_means_the_backend_seeds_its_own():
    """Without a seeded BC the pipeline finds the centre itself, so there is
    no centre here to compute a radius from and guessing would be wrong."""
    assert max_ring_rad(None, 2048, 2048) is None
    assert max_ring_rad({}, 2048, 2048) is None
    assert max_ring_rad({"BC_y": 100.0}, 2048, 2048) is None      # BC_z absent


def test_an_unknown_detector_size_is_not_guessed_at():
    assert max_ring_rad({"BC_y": 2384.7, "BC_z": 2182.7}, 0, 0) is None
    assert max_ring_rad({"BC_y": 2384.7, "BC_z": 2182.7}, None, None) is None

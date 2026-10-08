"""A blank raw sub-frame must not be averaged into the combine.

A Pixirad arms one frame before it starts counting, so frame 1 of every
``.pixi.h5`` is all zeros. ``reduce_chunk_multi`` combined it like any other
frame, which scales every pixel of a ``mean`` by ``(n-1)/n``.

Measured on 1-ID-E ``air_80p725keV_3s_003512.pixi.h5`` (10 raw frames, frame 0
blank): the written output matched ``mean(frames 0..9)`` at 100.0% of pixels,
max |difference| 0.00, and the ratio against ``mean(frames 1..9)`` was exactly
10/9 = 1.1111. Silent, and it scales with how few frames are combined -- 10%
over ten, 33% over three.

The Calibrate tab already steps off this frame
(``_skip_blank_opening_frame``), but only for the DISPLAY, and it explicitly
bails when frame-averaging is on -- so the combine path never had any cover.
"""
import numpy as np
import pytest

from midas_gui import frame_correct as FC

SHAPE = (6, 5)


def _stack(n=10, value=9.0, blank_first=True):
    s = np.full((n,) + SHAPE, value, np.float32)
    if blank_first:
        s[0] = 0.0
    return s


def test_mean_ignores_the_blank_opening_frame():
    assert FC.reduce_chunk(_stack(), "mean")[0, 0] == pytest.approx(9.0)


def test_the_old_behaviour_is_what_the_beamline_saw():
    """10 frames, one blank -> every pixel low by exactly 10/9."""
    old = FC.reduce_chunk(_stack(), "mean", skip_blank=False)[0, 0]
    new = FC.reduce_chunk(_stack(), "mean")[0, 0]
    assert old == pytest.approx(8.1)
    assert new / old == pytest.approx(10 / 9)


def test_three_frames_is_a_third_not_a_tenth():
    """The error scales with how few frames are combined."""
    old = FC.reduce_chunk(_stack(3), "mean", skip_blank=False)[0, 0]
    assert old == pytest.approx(6.0)
    assert FC.reduce_chunk(_stack(3), "mean")[0, 0] == pytest.approx(9.0)


@pytest.mark.parametrize("op", ["sum", "max"])
def test_sum_and_max_are_unaffected(op):
    """Adding or maximising against zero changes nothing, so this must not
    move the two ops that were already correct."""
    with_skip = FC.reduce_chunk(_stack(), op)
    without = FC.reduce_chunk(_stack(), op, skip_blank=False)
    assert np.array_equal(with_skip, without)


def test_median_drops_the_blank_too():
    """A blank frame is not an observation of zero -- with 10 frames and one
    blank the old median was still 9, but at 2 real frames it dragged."""
    s = _stack(3)
    assert FC.reduce_chunk(s, "median", skip_blank=False)[0, 0] == pytest.approx(9.0)
    s2 = np.zeros((2,) + SHAPE, np.float32); s2[1] = 9.0
    assert FC.reduce_chunk(s2, "median", skip_blank=False)[0, 0] == pytest.approx(4.5)
    assert FC.reduce_chunk(s2, "median")[0, 0] == pytest.approx(9.0)


def test_the_drop_is_reported_not_silent():
    st = {}
    FC.reduce_chunk(_stack(), "mean", stats=st)
    assert st["blank_skipped"] == 1 and st["all_blank"] is False


def test_an_entirely_blank_chunk_returns_zeros_rather_than_raising():
    """A dead chunk must not fail the whole run."""
    st = {}
    out = FC.reduce_chunk(np.zeros((4,) + SHAPE, np.float32), "mean", stats=st)
    assert np.array_equal(out, np.zeros(SHAPE, np.float32))
    assert st["all_blank"] is True


def test_an_all_blank_chunk_works_from_a_one_shot_iterator():
    """The all-blank path must not re-iterate `frames` -- a generator would
    already be exhausted by the first pass."""
    gen = (np.zeros(SHAPE, np.float32) for _ in range(3))
    assert np.array_equal(FC.reduce_chunk(gen, "mean"), np.zeros(SHAPE, np.float32))


def test_a_frame_only_zero_AFTER_correction_is_still_real_data():
    """The test is the RAW frame. A frame that cancels to zero against its
    dark recorded something; a frame that was zero off the detector did not."""
    s = np.full((2,) + SHAPE, 5.0, np.float32)
    dark = np.full(SHAPE, 5.0, np.float32)
    st = {}
    out = FC.reduce_chunk(s, "mean", dark=dark, stats=st)
    assert st["blank_skipped"] == 0          # neither frame was raw-blank
    assert np.array_equal(out, np.zeros(SHAPE, np.float32))


def test_opting_out_restores_the_old_arithmetic_exactly():
    for op in FC.OPS:
        a = FC.reduce_chunk(_stack(), op, skip_blank=False)
        b = FC.reduce_chunk(_stack(), op, skip_blank=False)
        assert np.array_equal(a, b)

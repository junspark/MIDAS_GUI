"""Chunked reduction + field correction — ``midas_gui.frame_correct``.

The centrepiece is the order-of-operations group: correcting each raw
sub-frame BEFORE combining is the only order that is right for all four ops,
and ``sum`` is the one that proves it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_gui import frame_correct as FC
from midas_gui.helpers import (_COMBINE_OPS, _stack_chunk_bounds,
                               apply_field_corrections)

SHAPE = (4, 5)


def _stack(n=6, shape=SHAPE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 500.0, size=(n,) + shape).astype(np.float32)


# ── chunking ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,size", [(10, 3), (10, 5), (10, 10), (10, 20),
                                    (1, 1), (7, 1), (100, 7)])
def test_chunk_ranges_match_the_shared_bounds_function(n, size):
    """``chunk_ranges`` must stay a pure loop over the function Batch
    Integrate chunks with — if the two ever disagree, a reduced frame and
    the integration of "the same" frames cover different exposures."""
    got = FC.chunk_ranges(n, chunk_size=size)
    expect = []
    k = 0
    while (b := _stack_chunk_bounds(n, k, chunk_size=size,
                                    raw_start=None, raw_end=None)) is not None:
        expect.append(b)
        k += 1
    assert got == expect


def test_chunk_ranges_cover_every_frame_exactly_once():
    ranges = FC.chunk_ranges(10, chunk_size=3)
    covered = [i for lo, hi in ranges for i in range(lo, hi + 1)]
    assert covered == list(range(10))


def test_chunk_ranges_keeps_a_short_final_chunk():
    assert FC.chunk_ranges(10, chunk_size=4) == [(0, 3), (4, 7), (8, 9)]


def test_falsy_chunk_size_is_the_whole_file():
    assert FC.chunk_ranges(10, chunk_size=None) == [(0, 9)]
    assert FC.chunk_ranges(10, chunk_size=0) == [(0, 9)]


def test_chunk_ranges_respects_a_raw_sub_frame_window():
    assert FC.chunk_ranges(10, chunk_size=3, raw_start=2, raw_end=8) == \
        [(2, 4), (5, 7), (8, 8)]


def test_chunk_ranges_empty_when_the_window_excludes_everything():
    assert FC.chunk_ranges(10, chunk_size=3, raw_start=8, raw_end=2) == []


# ── reduction agrees with the shared op table ────────────────────────────────

@pytest.mark.parametrize("op", FC.OPS)
def test_uncorrected_reduction_matches_the_shared_combine_table(op):
    """``reduce_chunk`` streams mean/sum/max rather than calling
    ``_COMBINE_OPS``; it must still produce exactly what that table would."""
    stack = _stack()
    got = FC.reduce_chunk(stack, op, clip_negatives=False)
    expect = _COMBINE_OPS[op](stack)
    assert np.allclose(got, expect, rtol=1e-6, atol=1e-4)


def test_average_is_accepted_as_an_alias_for_mean():
    stack = _stack()
    assert np.allclose(FC.reduce_chunk(stack, "average", clip_negatives=False),
                       FC.reduce_chunk(stack, "mean", clip_negatives=False))


def test_unknown_op_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown combine op"):
        FC.reduce_chunk(_stack(), "mode")


def test_empty_chunk_is_rejected_rather_than_returning_zeros():
    for op in FC.OPS:
        with pytest.raises(ValueError, match="no frames"):
            FC.reduce_chunk(np.empty((0,) + SHAPE, np.float32), op)


# ── order of operations: the whole point ─────────────────────────────────────

def test_sum_subtracts_one_dark_per_frame_not_one_per_chunk():
    """The bug this feature exists to avoid. An n-frame sum must lose
    ``n × dark``; correcting the COMBINED frame would lose only one."""
    n = 6
    stack = _stack(n)
    dark = np.full(SHAPE, 20.0, np.float32)

    got = FC.reduce_chunk(stack, "sum", dark=dark, clip_negatives=False)
    assert np.allclose(got, stack.sum(axis=0) - n * dark, rtol=1e-5, atol=1e-3)

    correct_after = stack.sum(axis=0) - dark
    assert not np.allclose(got, correct_after), \
        "sum was corrected after combining — one dark instead of n"


def test_mean_subtracts_exactly_one_dark():
    stack = _stack()
    dark = np.full(SHAPE, 20.0, np.float32)
    got = FC.reduce_chunk(stack, "mean", dark=dark, clip_negatives=False)
    assert np.allclose(got, stack.mean(axis=0) - dark, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("op", ["mean", "max", "median"])
def test_correct_before_equals_correct_after_for_the_monotone_ops(op):
    """mean/max/median commute with a per-pixel dark subtraction and a
    flat-field divide, so the single code path is right for them too — this
    is why there is no per-op special-casing in reduce_chunk."""
    stack = _stack()
    dark = np.full(SHAPE, 20.0, np.float32)
    bright = np.full(SHAPE, 120.0, np.float32)

    before = FC.reduce_chunk(stack, op, dark=dark, bright=bright,
                             clip_negatives=False)
    after = apply_field_corrections(_COMBINE_OPS[op](stack), dark=dark,
                                    bright=bright, clip_negative=False)
    assert np.allclose(before, after, rtol=1e-4, atol=1e-2)


def test_sum_is_the_op_where_the_two_orders_genuinely_differ():
    """Guard the claim above from the other side: if correcting after
    combining ever agreed for sum too, the whole ordering argument would be
    vacuous and these tests would be asserting nothing."""
    stack = _stack()
    dark = np.full(SHAPE, 20.0, np.float32)
    before = FC.reduce_chunk(stack, "sum", dark=dark, clip_negatives=False)
    after = apply_field_corrections(stack.sum(axis=0), dark=dark,
                                    clip_negative=False)
    assert not np.allclose(before, after)


def test_clip_is_applied_once_at_the_end_not_per_sub_frame():
    """Clipping is not linear. Per-sub-frame clipping discards the negative
    half of the noise and biases a sum upward; clipping the combined frame
    once does not."""
    stack = np.array([[[10.0]], [[-10.0]], [[10.0]], [[-10.0]]], np.float32)
    dark = np.zeros((1, 1), np.float32)
    got = FC.reduce_chunk(stack, "sum", dark=dark, clip_negatives=True)
    assert got[0, 0] == pytest.approx(0.0), "per-sub-frame clipping leaked in"

    per_frame_clipped = np.clip(stack, 0, None).sum(axis=0)
    assert per_frame_clipped[0, 0] == pytest.approx(20.0)


def test_clipping_can_be_turned_off():
    stack = np.full((2, 1, 1), 5.0, np.float32)
    dark = np.full((1, 1), 50.0, np.float32)
    assert FC.reduce_chunk(stack, "mean", dark=dark, clip_negatives=False)[0, 0] < 0
    assert FC.reduce_chunk(stack, "mean", dark=dark, clip_negatives=True)[0, 0] == 0


@pytest.mark.parametrize("op", FC.OPS)
def test_reduction_is_float32_whatever_the_input(op):
    out = FC.reduce_chunk(_stack().astype(np.uint16), op)
    assert out.dtype == np.float32 and out.shape == SHAPE


def test_a_mismatched_field_is_skipped_not_fatal():
    """Inherited from apply_field_corrections' skip-and-warn contract — a
    dark left over from a different detector must not kill a long run."""
    stack = _stack()
    with pytest.warns(RuntimeWarning, match="dark shape"):
        out = FC.reduce_chunk(stack, "mean", dark=np.zeros((9, 9), np.float32),
                              clip_negatives=False)
    assert np.allclose(out, stack.mean(axis=0), rtol=1e-5, atol=1e-3)


# ── compression options ──────────────────────────────────────────────────────

def test_compression_kwargs_for_each_offered_choice():
    assert FC.compression_kwargs(None) == {}
    assert FC.compression_kwargs("none") == {}
    assert FC.compression_kwargs("gzip", 6, True) == {
        "compression": "gzip", "compression_opts": 6, "shuffle": True}
    assert FC.compression_kwargs("lzf")["compression"] == "lzf"


def test_gzip_level_is_clamped_to_the_legal_range():
    assert FC.compression_kwargs("gzip", 99)["compression_opts"] == 9
    assert FC.compression_kwargs("gzip", -3)["compression_opts"] == 0


def test_unknown_compression_is_rejected():
    with pytest.raises(ValueError, match="Unknown compression"):
        FC.compression_kwargs("brotli")


# ── output file ──────────────────────────────────────────────────────────────

def _read(path, dataset="exchange/data"):
    import h5py
    with h5py.File(str(path), "r") as f:
        return np.asarray(f[dataset])


def test_written_file_holds_the_frames_as_one_float32_stack(tmp_path):
    frames = [np.full(SHAPE, i, np.float32) for i in range(3)]
    out = FC.write_corrected_h5(tmp_path / "o.h5", frames)
    arr = _read(out)
    assert arr.shape == (3,) + SHAPE and arr.dtype == np.float32
    assert np.allclose(arr[2], 2.0)


def test_written_file_records_which_raw_frames_each_output_came_from(tmp_path):
    ranges = [(0, 2), (3, 5)]
    out = FC.write_corrected_h5(tmp_path / "o.h5",
                                [np.zeros(SHAPE, np.float32)] * 2,
                                frame_ranges=ranges)
    assert _read(out, "frame_ranges").tolist() == [[0, 2], [3, 5]]


def test_metadata_tree_is_written_at_its_original_paths(tmp_path):
    out = FC.write_corrected_h5(
        tmp_path / "o.h5", [np.zeros(SHAPE, np.float32)],
        metadata={"instrument/Scalers/E/US_IC": np.array([1.5])})
    assert _read(out, "instrument/Scalers/E/US_IC").tolist() == [1.5]


def test_attrs_and_provenance_land_on_the_root(tmp_path):
    import h5py
    out = FC.write_corrected_h5(tmp_path / "o.h5", [np.zeros(SHAPE, np.float32)],
                                attrs={"midas_gui_combine_op": "median"})
    with h5py.File(out, "r") as f:
        assert f.attrs["midas_gui_combine_op"] == "median"


@pytest.mark.parametrize("comp", [None, "gzip", "lzf"])
def test_every_compression_choice_round_trips(tmp_path, comp):
    frames = [np.full(SHAPE, 7.0, np.float32)]
    out = FC.write_corrected_h5(tmp_path / f"o_{comp}.h5", frames,
                                compression=comp, shuffle=True)
    assert np.allclose(_read(out)[0], 7.0)


def test_gzip_actually_shrinks_a_compressible_stack(tmp_path):
    # Constant planes — compressible by construction, so a size difference
    # here means the kwargs genuinely reached h5py rather than being dropped.
    frames = [np.full((256, 256), 3.0, np.float32) for _ in range(8)]
    plain = FC.write_corrected_h5(tmp_path / "plain.h5", frames)
    gz = FC.write_corrected_h5(tmp_path / "gz.h5", frames,
                               compression="gzip", shuffle=True)
    assert Path(gz).stat().st_size < Path(plain).stat().st_size


def test_compressed_dataset_is_chunked_per_frame(tmp_path):
    """A compressed dataset must be chunked, and a whole-stack chunk would
    force a full decompress to read one frame."""
    import h5py
    out = FC.write_corrected_h5(tmp_path / "o.h5",
                                [np.zeros(SHAPE, np.float32)] * 4,
                                compression="gzip")
    with h5py.File(out, "r") as f:
        assert f["exchange/data"].chunks == (1,) + SHAPE


def test_output_directory_is_created_if_missing(tmp_path):
    out = FC.write_corrected_h5(tmp_path / "deep" / "er" / "o.h5",
                                [np.zeros(SHAPE, np.float32)])
    assert Path(out).is_file()

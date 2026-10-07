"""The Hydra composite is rebuilt on the GUI thread; keep it cheap.

Reported at the beamline: "i am running the gui on haydn and the response
is very very slow." haydn is a 32-core/187 GB box with an RTX A6000 at load
~1, so it was never resources. ``perf`` on the live process put ~63% of all
cycles in ``scipy.ndimage.map_coordinates`` on the MAIN thread, with the
other three threads idle -- the windmill composite, rebuilt synchronously.

The numbers that made it unusable, on a real four-panel GE array (canvas
6656x6656 = 44.3 Mpx):

    compute_inv_coords x4   11.42 s      <- every beam-centre change
    remap x4                 2.09 s
    composite max            1.61 s
    autolevel percentiles    0.49 s

The 11.42 s was pure waste. ``bc_y``/``bc_z`` enter ``compute_inv_coords``
as a plain additive offset on the last two lines, and the old cache key
included them -- so moving the beam centre always missed, and rebuilt two
44-megapixel meshgrids per panel to get an answer differing by a constant.
Pick BC then did it TWICE, because it set the two spins separately.

These tests pin both, and the exactness requirement that makes the caching
legitimate: this is a reorganisation of where the arithmetic happens, not a
change to the arithmetic. Hydra's composite coordinate math has a history
of subtle sign/mirror regressions (two DECISIONS.md entries), so the test
compares against the untouched reference implementation bit-for-bit rather
than to a tolerance.
"""
import numpy as np
import pytest


# -- the caching must not change a single pixel -------------------------

@pytest.mark.parametrize("panel", [1, 2, 3, 4])
def test_the_cached_grid_is_bit_identical_to_recomputing_it(panel):
    """``compute_inv_coords`` is left in place as the reference; the cached
    path must agree with it exactly, for every panel's own tx."""
    from midas_gui import hydra
    st = hydra.DetectorState()
    st.load_default(panel)
    st.px = 200.0
    for bc_y, bc_z in ((1024.0, 1024.0), (900.5, 1100.25), (-37.0, 2300.0)):
        st.bc_y, st.bc_z = bc_y, bc_z
        got_z, got_y = st.get_inv_coords(512)
        ref_z, ref_y = hydra.compute_inv_coords(bc_y, bc_z, st.tx, 512, st.px)
        assert np.array_equal(got_z, ref_z), f"ge{panel} @ {bc_y},{bc_z}: z grid moved"
        assert np.array_equal(got_y, ref_y), f"ge{panel} @ {bc_y},{bc_z}: y grid moved"
        assert got_z.dtype == np.float32 and got_y.dtype == np.float32


def test_the_base_grid_carries_no_beam_centre():
    """The split is only valid because the cached half is BC-free."""
    from midas_gui.hydra import inv_coord_base
    a = inv_coord_base(27.3, 256, 200.0)
    b = inv_coord_base(27.3, 256, 200.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# -- what actually gets recomputed --------------------------------------

def _count_rebuilds(monkeypatch):
    from midas_gui import hydra
    calls = []
    real = hydra.inv_coord_base
    monkeypatch.setattr(hydra, "inv_coord_base",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    return calls


def test_moving_the_beam_centre_does_not_rebuild_the_grid(monkeypatch):
    """The whole fix: 11.4 s of meshgrid per BC change, down to an add."""
    from midas_gui import hydra
    st = hydra.DetectorState(); st.load_default(1)
    st.get_inv_coords(256)
    calls = _count_rebuilds(monkeypatch)
    for dy in range(1, 6):
        st.bc_y += dy; st.bc_z -= dy
        st.get_inv_coords(256)
    assert calls == [], "a beam-centre change still rebuilt the coordinate grid"


def test_a_real_geometry_change_still_rebuilds(monkeypatch):
    """tx, canvas size and pixel size genuinely change the grid -- caching
    through one of those would be a correctness bug, not a speedup."""
    from midas_gui import hydra
    st = hydra.DetectorState(); st.load_default(1)
    st.get_inv_coords(256)
    calls = _count_rebuilds(monkeypatch)
    st.tx += 90.0;  st.get_inv_coords(256)
    st.px += 1.0;   st.get_inv_coords(256)
    st.get_inv_coords(512)
    assert len(calls) == 3, f"expected a rebuild for each of tx/px/size, got {len(calls)}"


# -- one pick is one change ---------------------------------------------

@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def card(app):
    from midas_gui.hydra_geometry_card import DetectorGeometryCard
    return DetectorGeometryCard()


@pytest.mark.forked
def test_picking_a_beam_centre_emits_one_geometry_change(card):
    """Two setValue calls meant two composite rebuilds per click, the first
    of them at new-bc_y-with-old-bc_z -- a geometry nobody asked for."""
    seen = []
    card.geometryChanged.connect(lambda: seen.append(
        (card._bcy.value(), card._bcz.value())))
    card._on_bc_picked(812.0, 923.0)
    assert len(seen) == 1, f"one pick produced {len(seen)} geometry changes"
    assert seen[0] == (812.0, 923.0)


@pytest.mark.forked
def test_a_ring_fit_beam_centre_does_the_same(card):
    seen = []
    card.geometryChanged.connect(lambda: seen.append(1))
    card._on_ring_fit_bc(700.0, 650.0, 310.0)
    assert len(seen) == 1
    assert (card._bcy.value(), card._bcz.value()) == (700.0, 650.0)


@pytest.mark.forked
def test_the_spins_still_work_on_their_own(card):
    """blockSignals is scoped to the coalesced write; a later manual edit of
    either spin must still drive the refresh."""
    seen = []
    card.geometryChanged.connect(lambda: seen.append(1))
    card._on_bc_picked(500.0, 500.0)
    card._bcy.setValue(640.0)
    card._bcz.setValue(660.0)
    assert len(seen) == 3

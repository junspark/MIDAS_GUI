# STATE — current snapshot

_Keep this under ~1 page. Permanent history lives in DECISIONS.md, not here._
_Last updated: 2026-10-05 (new Batch Correction tab — all local, nothing pushed)_

## Now working on

**Batch Correction landed (2026-10-05); AgBeh calibration still open.**
New optional tab: chunked frame reduction (mean/median/sum/max over N
sub-frames per file) with field correction, HDF5 out. See "Recently
completed" and DECISIONS 2026-10-05.

**Clearing the decks before the next push.** PR #11 is behind us — upstream
merged all 48 of our commits on 2026-09-30 as eight staged checkpoints ending
at our `7ccf0fb`, then added follow-ups, all merged back here (`3bcba7f`,
then `1c5c9af` for their ring-coverage commit). Since then: three Calibrate
fixes reported from the beamline, and a housekeeping pass (ROADMAP reconciled,
the PR#7 pyflakes nit fixed, scratch dirs pruned).

`main` is **24 commits ahead of `origin/main` and nothing is pushed.** That is
the next action and it needs your word, since it is also the next PR upstream.

Needing you, in the order it matters:
- **Eyes on a real Batch Correction run.** Verified only against synthetic
  two-file stacks plus a name-level replay of the real share. What needs a
  live run: that the Log names a sensible dark for each file of a multi-file
  VAREX scan, that the reduced frames look right loaded back into the Data
  Viewer, and that the output HDF5 opens downstream.
- **Push, then open the next PR upstream.** Upstream's staged-checkpoint
  method worked well on 48 commits; this one is small enough not to need it.
- **Eyes on a real zarr-grouping run.** The `Fe9Cr_KGT6038_after_failure_ff`
  folder with Dark set to both bracketing dark files, grouping *per source
  file*, should give 3 archives rather than 435. Verified only against
  synthetic two-file stacks, and the combo has never been seen rendered.
- **Eyes on the cake HDF5.** Still never opened in a viewer here — the one
  remaining eyes-on item now that the zarr's `/Omegas` are confirmed.
- **`PoleFigureWorker`** is still single-frame and takes χ/φ from its cfg.
  Making it ω-aware across a series is the piece the whole omega arc exists to
  enable, and it now has a correct, verified angle to stand on.
- **Metadata provenance**, which you said you'd keep testing against.

Open follow-ups, none blocking:
- Frozen-point (high-tilt) does not forward the Refine card's ± tolerance
  window: `calib.py`'s `_seed_and_v1` call in that branch is missing
  `tols=tols`, unlike the identical bayesian/joint call above it. Upstream
  found this reading our checkpoint 1 and disclosed it with a console warning
  (`f279030`, now merged here) rather than fixing it. **Checked 2026-10-01 and
  deliberately left as a warning**: the v1→spec mapping is pipeline-independent
  (the installed 0.17.0's `compat/from_v1.py` reads `v1.tolLsd`/`tolBC`/
  `tolTilts` with no branching), so the one-liner is almost certainly right —
  but whether `iterate_frozen_point_until_stable` honours those bounds cannot
  be tested here, because 0.17.0 ships **no frozen-point pipeline at all**.
  That is the same gap `tests/test_frozen_point_vendor.py` already skips on.
  Fix it when a backend that has the pipeline is installed, not before.
  See DECISIONS 2026-09-30.
- `documentation/calibration_unification_plan.md` — the three Calibrate UI
  surfaces (`tab_calibrate.py`, `hydra_calib_page.py`,
  `hydra_geometry_card.py`) have drifted; Hydra has none of the d-spacing
  work. Phases 0–1 are the cheap half. Phase 5 (technique presets) would
  replace the AgBH-shaped special case with a conditioning advisory driven
  by the actual seed geometry, which is the more honest axis.
- Test suite needs two runs to cover: `--forked` races with `--basetemp`,
  unforked segfaults on multiple `CalibrationTab`s. See DECISIONS 2026-09-09.
- **Counting the suite:** a `--forked` run prints no `N passed` summary line
  at all — `test_apply_project_calibration_single_detector`'s SIGABRT emits a
  `Fatal Python error: Aborted` dump that swallows it, and counting the `.`
  progress characters undercounts for the same reason (the dump breaks the
  line format). Get the total from
  `pytest tests/ --collect-only -q | awk -F': ' '/^tests\/.*: [0-9]+$/{n+=$2} END{print n}'`
  and subtract the failures/skips the short summary does print. Suite counts
  recorded before 2026-10-03 came from the progress-character method and are
  therefore too low.
- `test_apply_project_calibration_single_detector` hits the known pyqtgraph
  teardown SIGABRT (reproduces on clean HEAD; not ours). **It is intermittent,
  not constant** — the full 2026-10-03 run passed it. Treat a single green run
  as weak evidence either way.

- Still untested (ROADMAP.md): `job_queue.py`, `peak_fit_panel.py`.
  `batch_cli.py` now has `tests/test_batch_cli_omega.py` and
  `tests/test_batch_cli_zarr_grouping.py`, which cover the omega flags, the
  zarr-grouping flag and the tab→argv→cfg round trip for both — but nothing
  else in the file.
- `test_pva_live_source_roundtrip`'s symptom has changed: as of 2026-09-30 it
  fails on an image-content mismatch rather than only the `libstdc++` CXXABI
  import error recorded below (the import error still appears at HEAD). Same
  test, same pre-existing status — it fails identically on clean HEAD — but
  the recorded cause is no longer the whole story.
- Branch cleanup done 2026-09-10: `pr-7-strain-cake`, `test-fork-imports` and
  the four fetched `refs/remotes/origin/pr/*` refs are gone; only `main` and
  `origin/main` remain. Re-fetch any PR head with
  `git fetch origin 'refs/pull/*/head:refs/remotes/origin/pr/*'`.

## Recently completed

**2026-10-05 — Batch Correction.** Asked for at the beamline: average /
median / sum / max over a designated number of frames per file, with
appropriate background subtraction, written out as HDF5. New optional tab
(ships hidden), `midas_gui/frame_correct.py`, and
`workers.BatchCorrectionWorker`.

Most of it is reuse: `helpers._COMBINE_OPS` already had all four ops
including median, `helpers._stack_chunk_bounds` already owned the per-file
chunk arithmetic, `DataLoaderPanel(unify_combine=True)` already surfaces the
"Combine sub-frames: N / op:" row, and `h5_metadata.align` already averages
per-frame metadata over each chunk. Genuinely new: the correction ORDER, the
dark ladder, and an HDF5 writer (`h5_metadata.write_into_hdf5`).

- **Correct per sub-frame, combine, clip once.** Only `sum` makes it
  visible — correcting the combined frame subtracts one dark from an
  N-times-larger signal. mean/max/median agree either way (monotone
  per-pixel maps commute with max/median), so one path serves all four. The
  clip is deferred because it is not linear and would bias a sum upward.
- **The dark is resolved per FILE, as the nearest preceding `dark_before`.**
  The first design was folder-wide ("first data number − 1"), which
  surveying the real share proved wrong: darks are re-measured mid-scan, so
  a folder is a series of bracketed segments. 936/936 real data files have a
  preceding `dark_before` (median 8 files away, max 121); a folder-wide rule
  would have been right only for the first segment.
- **Both corpora are replayed as tests**, skipping when the share isn't
  mounted: 936 real files on `/home/beams/S20IDUSER/mnt/s20a` and 191 of 192
  historical runs from `~/midas_runs/midas_screen_logs`. The 192nd is named
  in `MANUAL_OVERRIDE_LOGS` — an operator `-P` override pointing at another
  scan's folder, unknowable from filenames — rather than hidden behind a
  loosened assertion.
- Output carries `frame_ranges`, the chunk-averaged `instrument/` tree, the
  dark that was used, and a provenance stamp. Compression is optional
  (none/gzip/lzf + shuffle), chunked one frame per HDF5 chunk.
- HDF5 input only; a TIFF/GE selection says why rather than silently
  grouping consecutive files.

Same day, from beamline feedback: methods are now **checkboxes, not a
dropdown** — any combination of Mean/Median/Sum/Max in one run, each into
its own `dark_subtracted_<op>/` leaf of Batch Integrate's output folder, and
all computed from **one pass** over the data. A **Suggest** button fills the
convention. Found and fixed a shared-widget trap while doing it: `end(0=all)`
is only true for a non-unify panel, so `end = 0` in Batch Integrate / Batch
Correction silently clamped the window to sub-frame 0 — relabelled, and the
tab now prints "N raw → M output frame(s)".

Cross-checked against the beamline's own `BatchCorrection.m`, which
independently confirms the `N × dark` ordering. Four deliberate differences
from it are recorded in DECISIONS (no zero-clipping there, `CorrectBadPixels`,
`FramesToIgnore`, and the SAXS variant's per-second normalisation).

Suite: **1,626 collected, 1,620 passed**, 3 failures, 3 known skips.
`test_app_builds_offscreen` is the third failure and is **config-driven, not
code-driven**: it asserts `count == len(ALWAYS_TABS) + len(DEFAULT_VISIBLE_TABS)`,
but a saved `ui.visible_tabs` overlays `DEFAULT_VISIBLE_TABS` with a list that
*includes* the four ALWAYS tabs, so the sum double-counts them. It passes with
a clean `HOME`, and breaks for any user who has ever toggled a tab in
Preferences. Pre-existing (CLAUDE.md already lists it); the fix is to compare
against `shipped_defaults()` or against the tab set rather than a count.

**2026-10-03 — the Mask Builder shows the detector the same way up as
everything else.** Reported from the beamline: the displayed image should
apply the Data Viewer's flips. It was the one image surface still painting
the raw array, so the detector appeared mirrored relative to every other
tab while you picked which pixels to throw away.

**Display only, and that distinction is the whole design.** The mask this
tab computes, saves and emits stays raw-frame, because
Calibrate/Batch/Integrate pre-flip it themselves (DECISIONS 2026-08-25);
a mask that arrived pre-flipped would be flipped twice. Pinned directly:
`maskReady`'s payload is asserted byte-identical across all 8 code
combinations.

- **Propagation** — `DataViewerTab.imTransChanged(list)` → `app.py`'s
  existing `_connect` broadcast, the same mechanism as `maskReady` and
  `calibrationDone`. Plus a one-shot push after project restore, because
  tabs restore in dict order and the Mask tab can be rebuilt before the
  Data Viewer knows its transforms.
- **Drawn shapes stay on the same detector pixels.** Each ROI is re-placed
  from the old painted frame into the new one by new
  `helpers.map_roi_state`, rather than left at the same screen position
  (which would silently re-aim a mask you had already built). Picked points
  move with it; a half-drawn freeform polygon is cancelled, having no
  anchor yet.
- **`helpers.map_point_xy`** is the continuous-coordinate sibling of
  `im_trans_map_point`. A flip of *geometry* is `x -> W - x`, not
  `W - 1 - x`: index `c` covers `[c, c+1]`, so using the index form on an
  edge coordinate shifts every shape half a pixel.
- **`_apply_shapes`** rasterises in display space then inverse-maps with
  `_apply_im_trans(drawn, reversed(codes))` — the idiom `workers.py`
  already uses to bring an azimuthal-clip mask back to raw.
- **The pixel readout now composes both legs** — undo the display
  transform to raw, then apply `result.im_trans` — instead of assuming the
  tab's codes and the calibration's agree.

**Worth recording, because it nearly went the other way.** The first
version of the ROI test compared rasters and failed on a 45°-rotated rect
by *exactly* 0.5 px. That was not a placement bug: a reflection reverses
the polygon's winding and Qt's scanline fill breaks ties differently.
Loosening the tolerance would have made it pass and would equally have
hidden a real half-pixel shift — the one bug worth catching here. Replaced
with an exact geometric assertion (the mapped ROI's corners must equal the
point-mapped corners, 1e-6, immune to rasterisation) plus a structural
interior check. `map_roi_state`'s handedness rule — anchor at the mapped
`(0, h)` corner for an odd number of reflections — is what that test
actually proves.

**Verified:** 1,501 collected, **2 failed, 3 skipped** (1,496 passed) — the
known set exactly. 257 new tests in `tests/test_mask_display_transform.py`.
An offscreen whole-app check confirms toggling Flip Y / Transpose on the
Data Viewer moves the Mask tab's codes and display shape live.
**Not verified with eyes on it:** nothing here has been seen rendered —
whether the red overlay still lands on the module gaps after a flip, and
whether a pre-drawn ROI visibly stays on its pixels, both want a look.

**2026-10-03 — the pixel readout says where you are in reciprocal space.**
Asked for at the beamline as "re-add the q/2θ readout". Checked first: it
was never there. `ImageViewer._coord_text` has carried x/y/intensity and
nothing else since `9640a3b`; the R/2θ/d/Q display that does exist is the
**cake** viewer's (`widgets.py`) and the profile viewer's x-unit selector.
So this is new, built to the same end.

- **2θ, Q, d and η** now follow x/y/intensity in the bar under every image
  viewer that has geometry — Calibrate, Data Viewer, Batch's Detector view
  and Mask Builder.
- **2θ is tilt-aware**, via the existing `helpers._pixel_to_two_theta_deg`
  (exact closed-form inverse of `_tilt_project_YZ`) — no new maths. Pinned
  by a round-trip test: forward-project a ring at a known 2θ through
  `tilted_ring_xy` at four tilt settings, read every pixel back, assert it
  returns that 2θ. So the bar and the ring overlay cannot disagree.
- **η matches the backend's `pixel_to_REta`** — `atan2(-Yc, Zc)`, η = 0
  straight up (+Z). New `helpers.pixel_eta_deg`, with the four cardinal
  directions pinned, because the sin/cos-swap failure mode puts every value
  90° out and still looks plausible.
- **`ImageViewer.set_radial_readout_fn(fn)`** keeps `widgets.py`
  geometry-free, exactly as `DataLoaderPanel.set_omega_hint_fn` does for ω:
  the tab hands in a callable. Unset, the bar is byte-identical to before —
  asserted, since four existing readout tests depend on it. A raising `fn`
  degrades to the plain readout rather than taking the status bar down.
- **Calibrate works before a fit**, off the live seed boxes, tagged
  `(seed)` so the number never quietly changes meaning when a result lands.
  That is the case the feature exists for: knowing 2θ *while* picking
  d-spacing points. Seed edits, Pick BC, "Send →" and the result all
  refresh it — the cursor does not move when any of them fire, and
  `sigMouseMoved` only emits on real motion (the same staleness that bit
  the Limits note in `86c1b27`).
- **Mask Builder needed a frame hop.** It displays the raw detector image
  on purpose while the calibration's beam centre lives in the transformed
  frame, so a hovered pixel is carried across by new
  `helpers.im_trans_map_point` (the point-wise counterpart to
  `_apply_im_trans`, which only ever handled arrays) before any geometry
  touches it. Proven against `_apply_im_trans` for all 13 code
  permutations on a *non-square* array, which is what catches a transpose
  that fails to swap the bounds. If the loaded image does not match the
  detector the calibration was fit on, the reciprocal-space fields are
  dropped rather than guessed — same guard as the mask overlay in
  `bc1c370`.
- Batch caches its geometry and drops the cache in
  `_refresh_detector_preview`: its "From file" source parses a calibration
  file, which must not happen at cursor-move rate.

**Verified:** 1,244 collected, **2 failed, 3 skipped** (1,239 passed) — the
known set exactly. 49 new tests (42 in `tests/test_pixel_readout.py`, 7
appended to `tests/test_viewer_origin_and_readout.py`). Offscreen
whole-app check confirms all four viewers have the callable bound and
Calibrate renders a seeded clause.
**Not verified with eyes on it:** the bar is now ~150 characters with all
four quantities, and this toolbar is already noted as eliding at the app's
default window width. Wants a look at real width; dropping 2θ to 3 decimals
is the cheap trim if it reads badly.

**2026-10-03 — upstream's ring-coverage commit merged back, plus a
housekeeping pass.** `1c5c9af` (merge) and the cleanup commit on top of it,
both local. Upstream's `6030371` (predicted rings bounded by true detector
coverage rather than a fixed 30°) was the only commit of theirs we did not
have. It touches `helpers.py`/`tab_calibrate.py`/`hydra_calib_widgets.py` —
the same `tab_calibrate.py` the three Calibrate fixes below had just
rewritten — but git auto-merged everything except `STATE.md`, and the
auto-merge was checked by hand rather than trusted: the `rmax_corner_px`
import, the `_draw_rings` bound, and this fork's mask-overlay work all
landed intact. `max_two_theta_deg` was also spot-checked numerically (a
centred 2048² detector at Lsd 200 mm gives 55°, off-centre 69°, short-Lsd
71° — all far past the old hardcoded 30°, which is the bug it fixes).

Housekeeping in the same pass:
- `tab_batch.py`'s unused local `spec` in `_run_as_job` (the one pyflakes
  warning PR #7 introduced, tracked in ROADMAP) is gone. The *call* stays —
  it is the pre-flight that reports a bad calibration before a detached job
  launches — with a comment saying so, since the bare call now looks
  pointless. An AST sweep for other unused locals turned up only
  tuple-unpack/loop targets (which pyflakes does not flag) and
  `batch_cli.py`'s deliberate `app = QApplication(...)` GC anchor; nothing
  else was touched.
- ROADMAP reconciled: the pyflakes item removed, and the PR#7 coverage item
  corrected from three uncovered modules to two — `batch_cli.py` picked up
  `test_batch_cli_omega.py` and `test_batch_cli_zarr_grouping.py` along the
  way and nobody had updated the entry.
- Pruned `.scratch/`, `.pytest_cache/`, every `__pycache__/` and five stale
  `/tmp/mg_suite_*` basetemps. All gitignored; nothing tracked was touched.

**Verified:** full suite on the merged tree — 1,195 collected, **2 failed, 3
skipped** (so 1,190 passed), the known set exactly (`test_pva_live_source_roundtrip`,
`test_apply_project_calibration_single_detector`, and the three
frozen-point skips). Worth recording: the *pre-merge* baseline run passed
`test_apply_project_calibration_single_detector` and this one failed it, on
the same tree-modulo-cleanup — that failure is **intermittent**, so a single
green run is not evidence it is fixed.
**Not verified with eyes on it:** nothing new on screen — the merge changes
which rings get predicted on a wide/short-Lsd geometry, and that has not
been seen rendered.

**2026-10-01 — Calibrate: predicted rings now bounded by true detector
coverage, not a fixed 30°.** `helpers._predict_ring_radii` generated
candidate rings from the calibrant's wavelength/d-spacings with a hardcoded
`two_theta_max_deg=30.0`, decided before any detector geometry was
consulted — so on a short-Lsd/wide-detector/off-centre-beam geometry whose
real coverage exceeds 30°, real rings beyond it were never generated at all
(the single-detector image overlay, its radial-profile ring markers, and the
Hydra multi-panel overlay all fed from this one function). New
`helpers.max_two_theta_deg()` computes the true max 2θ from the farthest
detector corner (reusing the same beam-centre/corner-distance reasoning as
the existing `rmax_corner_px`), with a safe fallback to 30° only when
detector dimensions aren't known yet. Also fixed the redundant post-hoc
pixel filter in `tab_calibrate._draw_rings`/`hydra_calib_widgets._redraw_rings`
(`max(NrPixelsY, NrPixelsZ)` → `rmax_corner_px(...)`), which was an
axis-aligned approximation, not the true corner distance, and could have
clipped a few farther rings even after the generation-side fix. New test
`tests/test_helpers.py::test_predict_ring_radii_uses_detector_coverage_not_fixed_30deg`.
**Verified:** the 5 touched/related test files green per-file on a clean
`HOME` (helpers/manual-dspacing-ui/manual-fit-conditioning/batch-queue-ui/
hydra-calib-ui), plus `test_smoke.py`; `pyflakes` unchanged (same
pre-existing warnings only).

**2026-09-30 — one zarr per rotation, not one per frame.** `4ec0174`,
committed 2026-10-01, not pushed. The
`zarr` format wrote one archive per combined output frame; the folder that
prompted this (three 1442-sub-frame VAREX files at `OME_SUM 10`) would have
produced 435 single-cake archives. A **Zarr grouping** combo in the Output
card now offers *per output frame* (unchanged default), *per source file* and
*per run*.

- **A group is one rotation — the same unit ω is measured from.**
  `_HDF5StackGlobSource.zarr_group_key` is defined as
  `omega_channel_window(idx)[0]`, exactly as `raw_window_for_index` is
  `omega_channel_window(idx)[1:]`, so an archive cannot disagree with the
  angles inside it. TIFF/`.ge*` returns a constant: that selection is the
  rotation.
- **Streams, does not buffer.** Uses the backend's `GSASZarrWriter`
  (`add_frame`/`close`) through a new `_ZarrGroupWriter`, so peak memory is
  one frame — buffering a 1442-frame group would have cost ~830 MB.
- **Batch Parallel splits on group boundaries** (`_split_into_chunks_on_groups`),
  since two workers would otherwise open the same `.zarr.zip`. Caps workers at
  the group count; "run" grouping falls through to sequential.
- **Both run paths carry it** — `--zarr-grouping` on the `batch_cli` argv, with
  a round-trip test, which is the direct lesson of the 2026-09-29 ω bug below.
- Provenance moved from per frame to per group (fewer repack passes);
  `h5_metadata.align` already took a sequence of frame ranges and needed no
  change. A "run" group over several files skips the `instrument/` copy and
  logs it rather than guessing.
- Touched: `workers.py`, `tab_batch.py`, `widgets.py` (new
  `OutputFormatSelector.changed`), `batch_cli.py`, `project.py`,
  `tests/test_batch_zarr_output.py`, new
  `tests/test_batch_cli_zarr_grouping.py`, `documentation/gui_documentation.md`
  §7, `.context/DECISIONS.md`.
- Suite: **1178 collected** (was 1149 — 29 new), same two known failures
  (`test_apply_project_calibration_single_detector`,
  `test_pva_live_source_roundtrip`) and three known skips. Both failures
  reproduce on clean HEAD. The new tests were confirmed to fail against HEAD
  in a throwaway worktree before being called done.

**2026-09-30 — PR #11 (junspark) merged into `main`: 48 commits, 8 staged
checkpoints, 3 real bugs found and fixed along the way.** Full rationale,
per-checkpoint GUI-risk table and verification detail in DECISIONS. Headline
additions: new Zarr Viewer tab (visible by default), a cake-parameters editor
dialog, omega (rotation-angle) tracking end-to-end through Batch Integrate,
✕-to-close on every optional tab, the horizontal polarization-plane fix, and
assorted Calibrate/Corrections polish. `main` was never touched mid-flight —
everything happened on a disposable `merge/pr11-staged` branch, checkpoint by
checkpoint, with a test sweep + offscreen screenshot diff after each before
advancing. Full 75-file per-file sweep is green except the one known
pre-existing `test_apply_project_calibration_single_detector` SIGABRT.

**2026-09-29 — ω verified live, then made to say what it means; three bugs
out.** Six commits, `4c7b776`…`dc24f12`, pushed. You confirmed a real run's
zarr `/Omegas` are correct — that closes the live-verification item the omega
work had been carrying, and everything below was written on top of it.

- **ω restarts at every file** (`236ffa0`). The global ramp meant the computed
  `OME_START`/`OME_STEP` ramp and a *measured* ω channel indexed two different
  axes for the same frame, and a file-number filter silently rebased the ramp
  anyway (dropped files were gone before the source saw them). One rule now:
  ω is measured from raw sub-frame 0 of the rotation the frame came from — an
  HDF5 sub-frame stack is one rotation, a one-frame-per-file series is one.
  `raw_window_for_index` is now literally `omega_channel_window(idx)[1:]`, so
  there is one place that decides what the axis is.
- **The GUI says frame numbers and angles are the same axis** (`1bb4341`).
  The cake summary line and the loader's range hint both map the sub-frame
  range to the angle range, spelling out Δω/sub-frame **vs** Δω/frame — the
  `OME_SUM` multiplication nobody should have to do in their head. A loaded
  rotation with no angles set now says so rather than quietly producing
  all-zero `/Omegas`. `DataLoaderPanel` still knows nothing about ω; it takes
  a callable (`set_omega_hint_fn`), so every other tab's hint is unchanged.
- **Background jobs wrote ω = 0** (`dc24f12`). The `batch_cli` argv carried no
  omega flags at all, so "Run as background job" recorded zeros while Start
  Integration recorded the right angles, and nothing could report it. Fixed by
  serialising the config rather than re-deriving it; start/step go out even at
  0/0 so the command line in the Logs tab always states what the job will
  record.
- **2D CSV wrote nothing with multi-azimuth off** (`6ed405d`), and reported a
  file it had not written, under the wrong name. `cake_2d is not None` had
  come to mean both "a cake exists" and "fan out per η"; those are now
  separate, and `write_profile` raises rather than no-op'ing.
- **One frame no longer costs a whole-file decode** (`520ae44`). Previewing
  one frame of the 1442-sub-frame VAREX file (23.9 GB) read all of it: ~230 s
  and ~1.9 GB to draw 33 MB, paid again by every parallel worker. Now ~4 s /
  415 MB. This is the second half of the fix whose first half (counting
  without decoding) landed earlier.
- **`kill -USR1` dumps every thread's stack** to `~/midas_gui_hang.log`
  (`4c7b776`) — a freeze leaves no traceback, and that is exactly when the app
  can no longer be asked anything.

**Verified:** full suite 1,149 collected, 2 failed / 3 skipped — the same
known pair (`test_pva_live_source_roundtrip`, a system `libstdc++` CXXABI
mismatch, and `test_apply_project_calibration_single_detector`, the pyqtgraph
teardown SIGABRT) and the same three skips. Four new test files, ~890 lines.
**Not verified with eyes on it:** the two new readouts have not been seen
rendered — they are pinned by `tests/test_omega_readout.py` at the text level
only, and the exact wording in a narrow panel wants a look.

**2026-09-29 — Upstream's cake HDF5, merged.** `cc1045d` (merge) and
`43c672c` (docs), pushed. Upstream's two 2026-09-28 Batch Integrate commits —
`31e904c` (multi-azimuth HDF5 output, new `midas_gui/cake_hdf5.py`) and
`61feeb3` (flatter layout, 2θ/d/Q axes, wider provenance) — land on exactly
the code the omega work had rewritten. Only `workers.py` and `tab_batch.py`
conflicted; DECISIONS 2026-09-29 records which side won where. Three things
worth knowing without opening that entry:

- Upstream's `all_omegas` is frame *indices* (the combined-HDF5 `<lo>_<hi>`
  stem); this fork had already split that list in two, so upstream's became
  `all_frame_idx` and its append condition widened to the union of both
  guards. No behaviour on either side changed.
- Upstream hoisted the BinArea count out of the `want_zarr` branch to share
  it with the cake writer, but hoisted the version that hands `geom` straight
  to `count_cake` — `None` on the corrections path, the crash this fork had
  already fixed for zarr. Resolved to upstream's structure with this fork's
  geometry fallback.
- `write_cake_h5` gained an `omegas` dataset, so the fork's angle reaches the
  cake file too — that file is what a pole figure over a rotation series
  would read.

**Verified:** full suite 1089 passed / 2 failed (the same known pair) /
3 skipped, with upstream's own `tests/test_batch_cake_h5.py` (+12) green and
two new tests of ours pinning the cake file's omegas and its survival of the
corrections path.
**Not verified with eyes on it:** the cake HDF5 has never been opened in a
viewer here, and no live run has produced one.

**2026-09-29 — Omega: a real rotation angle on every frame, and in the
zarr.** `1117cc1`, pushed to `origin/main` along with the eight commits that
had been sitting local (the 2026-09-28 entry below said "nothing pushed"; that
is no longer true).

- The cake CSV's `OME_START`/`OME_STEP` stop being carried-and-ignored. One
  formula covers every case — `ω = OME_START + mean(raw sub-frame indices of
  the frame) × OME_STEP` — which reduces exactly to mpe_wf's own
  `ome_start + (idx*ome_sum + (ome_sum−1)/2)*ome_step`, and to the mean of the
  collapsed window when the loader combined everything into one frame.
  `cake_params.omega_for_window`/`omega_series` own it.
- The raw indices come from new `raw_window_for_index` methods on
  `_HDF5StackGlobSource` and `_ChunkCombinedFileSource` (sources without one
  fall back to `(i, i)`). **Superseded the same day** — they were global
  across the run; `236ffa0` makes the HDF5 one restart at every file. See the
  entry above.
- Two new inputs in the Cake parameters dialog, outside the nine CSV columns:
  an editable **omega channel** combo (blank = the computed ramp; populated
  from the loaded HDF5 by new `helpers.list_h5_1d_datasets`) and an
  **averaged/summed** override that gives every frame the one run-wide mean.
- Where it lands: the zarr's `/Omegas` (both writers), the combined HDF5's
  `omegas` dataset, the attempt record (`results/omegas`), and the Save button's
  `integrated.h5`. The export path *stores* rather than recomputes, and tags
  the provenance entry with `omega_source` = recorded / recomputed from
  `omega_cfg` / unavailable.
- **Behaviour change:** `/Omegas` used to hold the frame index labelled as
  degrees. An unconfigured run now writes `[0.0, 0.0, …]` — a stationary sample
  really is at ω = 0, and that is a better wrong answer than an index.
  Deliberate; see DECISIONS 2026-09-29.
- `PoleFigureWorker` was left alone — making it ω-aware across a series is the
  next piece, and now has a correct angle to stand on.

**Verified:** full suite 1073 passed / 2 failed, both the known pre-existing
pair (`test_pva_live_source_roundtrip`, `test_apply_project_calibration_single_detector`);
48 new tests across five files, one new (`tests/test_omega_windows.py`), and a
re-run of the ten files touched after that suite started (190 passed). Offscreen
check confirmed the summary line, the dialog round trip, and `SPEC` still being
exactly `CAKE_KEYS`. Note `test_app_builds_offscreen`, which CLAUDE.md lists as
a third known failure, passed both times here — its tab-count assertion depends
on the active profile's tab set, so CLAUDE.md was left as-is rather than
rewritten off two green runs.
**Verified live 2026-09-29:** a real run's zarr `/Omegas` are the angles
expected. Not walked through: the averaged/summed override (every frame
reporting the one run-wide angle) and a measured ω channel picked from the
combo — both still only pinned by tests.

**2026-09-28 — The source HDF5's `instrument/` tree reaches the zarr; an ✕
closes any optional tab.** Three commits, all local, nothing pushed.

- `7414502` — every optional tab gets an ✕ that is the same act as unchecking
  it in Preferences ▸ Tabs (widget kept, choice persisted to the active
  profile); the four pinned tabs have theirs stripped. Same commit wrote down
  what GSAS-II's importer *actually* reads from a MIDAS zarr and corrected a
  factually wrong claim in `gsas_export.py`'s docstring about the sidecar
  convention.
- `35a7b8b` — new `midas_gui/h5_metadata.py` copies `instrument/` +
  `active_instrument/` out of the source HDF5 and into the finished
  `.zarr.zip`, wholesale, from both writers (Batch Integrate and the GSAS-II
  export, the latter rebuilding the source from the attempt's recorded
  `src_cfg`). `provenance.rewrite_zip`/`stamp_extracted` factored out so the
  copy and the provenance stamp share one extract/repack pass.
  **Costs ~0.15 s and ~130 KiB per output frame** — see DECISIONS for why
  there is deliberately no opt-out, and what the cheapest one would be if the
  cost turns out to bite on a long scan.

**Verified:** full suite green apart from the two known pre-existing failures
(`test_pva_live_source_roundtrip`, the system libstdc++ CXXABI mismatch; and
`test_apply_project_calibration_single_detector`, the pyqtgraph teardown
SIGABRT). New tests: `tests/test_h5_metadata_copy.py` (+7),
`tests/test_zarr_layout_parity.py` (+1 cross-writer instrument-tree test),
`tests/test_smoke.py` (+3 tab-close tests).
**Not verified with eyes on it:** no live X11 session — the ✕ and the
enriched Zarr Viewer tree still want a look on a real run.

**2026-09-26 — Batch Integrate: "stride" replaced by unified "Combine
sub-frames" (HDF5 + TIFF alike).** `DataLoaderPanel`'s start/end/stride +
HDF5-only "Combine sub-frames" consolidated into one control: stride is gone
(the panel gains a new opt-in `unify_combine=True`, used only by
`tab_batch.py`; Pump Probe's own loader instance keeps stride untouched —
see DECISIONS for why the flag exists at all). Combine sub-frames now applies
to a TIFF/`.ge*` folder too, via new `workers._ChunkCombinedFileSource`
(groups consecutive FILES, mirroring `_HDF5StackGlobSource`'s within-file
chunking). Start/end filtering moved from a post-hoc index range into
`source_cfg()` itself (`frame_start`/`frame_end`, new
`workers._filter_paths_by_frame_number`), applied *before* chunking so a
narrowed range always restarts chunk-counting at its own start.
Opportunistic fixes in the same touched code: background "Run as background
job" (`batch_cli.py`) previously had no `--chunk-size`/`--combine-op` at all
(silently ignored); `tab_batch.py`'s `_run_as_job` mis-routed multi-file HDF5
through `--source-type tiff_list` (now has its own `hdf5_stack_glob` branch).
MONITOR now also refuses when combine/filtering is active (live-combining
isn't supported). New tests: `test_batch_data_source.py` (+6),
`test_frame_naming.py` (+9), `test_project.py` (+1) — all existing HDF5
multi-file combine tests needed zero changes.
**Verified:** 13 touched/related test files green per-file on a clean `HOME`
(one pre-existing unrelated SIGABRT); `pyflakes` unchanged at 37; offscreen
screenshot confirmed the new layout. Full detail in DECISIONS.md.

**2026-09-25 — Data Viewer: folder format filter, under-viewer frame
scrubber, profile-file lineout; app-wide frame-nav slider/button styling.**
- Image folder loads can be filtered to one detected format via a new
  "Format:" combo (`helpers._folder_format_groups`, opt-in
  `DataLoaderPanel(folder_format_filter=True)`, Data Viewer only).
- Frame scrubber moved from the left loader column to directly under the
  image viewer (`tab_view._build_frame_scrub_bar`, copies `tab_calibrate.py`
  /`CakeStackViewer`'s pattern); `hide_frame_field` now also hides
  `DataLoaderPanel`'s `mode="stack"` nav row (previously "single"-only).
- Radial Profile tab gained "Source: Detector frame / Profile file…" to load
  an existing `.csv/.xye/.dat/.fxye` integration output directly
  (`helpers.load_profile_file`, generalized off the PDF tab's reader).
  `helpers.native_axis_to_r_px` converts a 2θ/Q-native file to r_px once a
  calibration is attached; without one it plots in its native unit with the
  R/2θ/Q toggle locked (new `ProfileViewer.set_profile(...,
  native_unit=...)`). Image viewer stays blank in this mode.
- Found + fixed along the way: `DetectorGeometryCard._simulate()` (ring-radius
  computation) hard-required an image it never actually used the pixels of —
  new `simulate_rings_without_image()` (shares `_compute_material_rings()`
  with `_simulate`) lets ring overlays work with no image loaded.
- Separately requested: every frame-nav ◀/▶ button + slider app-wide (Data
  Viewer, Calibrate, Mask Builder, Hydra `mode="nav"` loader,
  `CakeStackViewer`, `DataLoaderPanel`'s own stack-mode row) now carries
  `objectName` `frameNavBtn`/`frameNavSlider`, styled in `style.py` — several
  were plain `QToolButton`s with no default border/background, nearly
  invisible on the dark theme.
- New `tests/test_dataviewer_format_filter.py` (7),
  `test_dataviewer_frame_scrub.py` (5), `test_dataviewer_profile_file.py`
  (12) — **not** fork-isolated, same reason as `test_view_tab_controls.py`
  (a forked `DataViewerTab`-building test SIGSEGVs on this machine).
**Verified:** 11 touched/related test files green per-file on a clean
`HOME`; `pyflakes midas_gui/*.py` 38→37 (only change: `Path` in
`tab_view.py` went from unused to used); offscreen screenshot confirmed the
new button/slider colors. Full detail in DECISIONS.md.

**2026-09-24 (`d224c97`) — Mask Builder: folder/multi-frame Image loading +
threshold-mask projection.** The Image field now accepts a folder of
single-frame files, a multi-page TIFF, a 3-D HDF5 dataset, or a multi-frame
`.geN` file (new `_detect_multiframe`, peeks shape/page-count/file-size
only, never loads pixel data), with a Frame navigator (◀/▶ + spinbox) to
step through them — same pattern as the existing section-2 Stack folder
loader. New Projection combo (Current frame / Average / Sum / Max) lets the
threshold step (section 1, `_compute`/`_threshold_source_image`) build its
mask from a reduction across every frame instead of just the frame shown,
since a per-frame threshold on one noisy frame is often unreliable; disabled
and locked to "Current frame" for a plain single image. Frame index
persists in both mask-attempt provenance and sidecar state. New
`tests/test_mask_folder_frames.py` (10 tests, forked per the pyqtgraph
teardown-crash pattern).

**2026-09-11 (later) — One honest ring overlay, Batch's cakes made visible,
and the whole calibration in the provenance record.** Two commits.
- **`54cdd48` Calibrate.** The predicted-ring overlay is *always* the full
  forward model — fitted tilt **and** refined distortion — and the "Corrected"
  tick is gone from both the single-detector tab and the Hydra calib page
  (with its saved state). It was off by default and applied tilt only, so on a
  distortion-refined detector both of its states drew rings off the measured
  ones, which reads as a bad calibration rather than a bad overlay. New
  `helpers.ring_xy_corrected` inverts the backend's own
  `R_corrected = D(ρ,η)·R` by fixed-point iteration, calling `midas_distortion`
  rather than copying the model; `helpers.distortion_rho_d_um` reproduces
  `spec_from_calibration_result`'s normalisation radius exactly. Reduces
  bit-exactly to `tilted_ring_xy` (and so to a circle) with nothing to apply.
  The status line now names what was applied and says when the empirical
  `residual_corr_map` is *not* drawn. Also: the seed overlay carries fed-back
  distortion (fresh fit **and** project restore — it silently drew tilt-only
  rings before); **"Use seed as calibration (no fit)" removed** (superseded by
  the Data Viewer's Ring simulation card + `Geometry: [← Get]`); d-spacing pick
  controls hidden for crystalline calibrants, and leaving a d-spacing calibrant
  cancels an active pick mode; Refine card compacted to three rows with Limits
  as its own block, Seed/Advanced three-per-row, `S.Form.row` stretching every
  field column; `IntegrationWorker` float64 end-to-end (it narrowed to float32
  and widened back, so the Calibrate preview differed from the Batch run it
  previews). New `tests/test_ring_projection.py` (11, every drawn point fed
  back through `pixel_to_REta`) and `tests/test_calibrate_integration_accuracy.py`
  (4, pinning the Calibrate profile equal to the Data Viewer's accurate path).
- **`f44314d` Batch Integrate.** A multi-azimuth run's per-frame `(η, R)` cakes
  were computed, written to disk and embedded in the project — and shown
  nowhere; reopening such a project *raised* (2-D cake rows fed to the 1-D
  waterfall buffer). New `CakeStackViewer` + an "Eta-R cakes" view tab, frame
  scrubber, zoom preserved across steps; the 1-D views get an η-collapse over
  filled bins; a non-multi-azimuth run clears the tab rather than showing stale
  cakes. `CakeViewer`'s x-axis can be labelled R / 2θ / d / Q — tick strings
  only, no resampling, d → "∞" at R = 0 — and the selector stays hidden until a
  caller supplies geometry, which is what leaves Calibrate, Hydra and
  `RingResidualViewer` untouched. Separately, an attempt's
  `calibration_snapshot` is now the whole calibration via new
  `helpers.full_calibration_snapshot` (a strict superset of the display
  fields, so every existing reader is unchanged). New
  `tests/test_batch_cake_stack.py` (21) + 2 in `test_project.py`.
**Verified:** the 5 touched/new test files pass per-file on a clean `HOME`
(11/4/21/23, and `test_project.py` 42 pass + the known
`test_apply_project_calibration_single_detector` SIGABRT); `pyflakes
midas_gui/*.py` 35 warnings before and after, line numbers only.

**2026-09-11 — MIDAS backends bumped to current PyPI latest; the tilt/im_trans
issue this repo filed came back fixed.** `midas-calibrate-v2` 0.13.0→**0.17.0**,
`midas-hkls` 0.10.0→0.11.0, `midas-stress` 0.13.0→0.14.0 (the rest of the set was
already latest; nothing else moves — numpy/torch/numba/zarr untouched). The
calibrate-v2 jump is upstream implementing
`.context/issue_draft_calibrate_v2_tilt_imtrans.md` nearly whole: `initial_tx/ty/tz`
on `calibrate()`, `CalibrationSpec.im_trans` reaching every pipeline via one shared
`io/transforms.apply_im_trans`, `im_trans` recorded on the result and carried into
`IntegrationSpec.TransOpt`, and `first_time_calibrate` gaining both. **ROADMAP P3-1
and P3-4 closed**; P3-2/P3-3 re-checked, still open.
- **`calib._prep_transformed()` deliberately kept.** 0.15.0 introduces a silent
  double-transform for callers that pre-transform, but the GUI is not exposed: its
  specs come from `build_v1_params`, which sets no `ImTransOpt`, so `spec.im_trans`
  is `()` and every pipeline's `if spec.im_trans:` guard is false. Verified, plus
  `apply_im_trans` proven bit-identical to `helpers._apply_im_trans` on all 8 opcode
  combos. Removing the workaround is optional cleanup and must be done whole (delete
  `_prep_transformed` **and** set `v1.extra["ImTransOpt"]` in the same change) — see
  DECISIONS 2026-09-11.
- **Behaviour change to know about:** `initial_tx/ty/tz`, which the one_shot branch
  has always passed speculatively and `_supported_kwargs` silently dropped, now take
  effect. Also inherited: RhoD µm unit fixes, `use_diplib` defaulting False,
  distortion phase bounds ±90→±180.
- **`requirements.txt` was two bumps stale** (the 2026-09-08 bump never reached it),
  so it and `pip install .` installed different backends. Regenerated in sync; all
  three pin files now cross-checked against the installed env.
- **`first_time` now gets the transform (same session).** That branch had never
  passed one — a first_time calibration on a flipped detector ran in the wrong
  frame, silently. It now forwards the codes and hands over the RAW frame (the
  backend flips image/dark/panel_mask and re-derives `n_pixels_y/z`, so it must not
  also pre-flip). Fixing it exposed a wider bug: `workers.CalibrationWorker` read
  `NZ, NY = image.shape` off the **raw** image, so on a non-square detector with a
  transpose every non-plain-one_shot mode recorded `NrPixelsY/Z` for a detector it
  never fitted — now `calib.effective_pixel_counts()`. New
  `tests/test_first_time_im_trans.py` (19 tests, mutation-checked).
  **Still open:** first_time is not passed a *tilt* seed, which 0.17.0 now accepts
  (`initial_tx` + `tilt_prior_deg`→ty/tz); `test_calib_tilt_seed.py`'s "at any
  backend version" wording is stale.
- **Verified:** 43-file per-file sweep byte-identical before/after (same one
  pre-existing `test_smoke` local-config failure, 10/10 under a clean `HOME`); all 46
  modules import. One sweep run had `test_live_stream.py` exit 139 — the documented
  pyqtgraph teardown flake, not this work (5/5 green on re-run; that file imports
  neither changed module).

**2026-09-10 — Data Viewer: accurate integration on demand,
rings that stay put, and a two-way geometry hand-off.** Six requested changes:
- **"Accurate" tick** above the radial profile (off by default) switches it
  from the fast path (circle binning, or the engine's `hard` kernel once a
  calibration/tilt exists — the live-view-capable default, unchanged) to the
  **Batch-Integrate pipeline verbatim**: geometry synthesized from the live
  widgets even at zero tilt, `subpixel2` kernel.
- **Eta-vs-R Cake** gained its own `R bin` / `η bin` spinboxes next to
  `Calculate` (defaults 1.00 px / 5.00°, i.e. what it computed before) and now
  *always* runs the accurate pipeline. Because the two bin independently,
  `radial_integrate` no longer fills the cake as a by-product once
  `set_cake_controls` is bound (Hydra binds none, keeps the old behaviour).
  The single engine-context slot became a bounded 6-entry cache keyed on
  kernel + both bin sizes.
- **Simulate rings** is a plain button + `live` tick + `✕` again. A one-shot
  click leaves the button orange and **freezes** the overlay at the parameters
  it was simulated with (`_ring_draw_geom`) — fixes rings drifting on a BC/tilt
  edit while not live. Green only when live is armed; `✕` clears the simulated
  rings and disarms (leaves the click-picked magenta ring).
- **"Pick d-spacing pts" + "Ring #"** are hidden unless an *enabled* material
  is `kind == "dspacing"` (AgBH, custom lists), via new
  `PickableImageViewer.set_dspacing_picking_visible()`. Visible by default, so
  the Calibrate tab is untouched.
- **Transforms card** moved between Projection and Ring simulation (inside the
  card, so Hydra's panel cards match).
- **`Geometry: [Send →] [← Get]`** replaces the single Send button; `← Get`
  pulls via new `CalibrationTab.geometry_for_viewer()` (shared with Calibrate's
  own "→ Send to Data Viewer") and says so plainly when there is no result.
**Files:** `hydra_geometry_card.py`, `tab_view.py`, `widgets.py`,
`tab_calibrate.py`, `app.py`; new `tests/test_view_tab_controls.py` (31 tests).
**Verified:** 21-file per-file sweep green, zero new pyflakes warnings vs.
HEAD, offscreen screenshots of all three toolbars + the card column.
**2026-09-09 (later) — Parameter limits for crystalline calibrants; One-shot
refine flags made real.** The "limits are not available for this calibrant"
label shipped in the entry below was **wrong**: `CalibrationParams` carries
`tolLsd`/`tolBC`/`tolTilts`/`tolDistortion`/`tolWavelength` and
`param_vector.bounds()` makes them hard LM box constraints, so crystalline fits
were already bounded at invisible defaults (±15 mm / ±20 px / ±3°). The Refine
card now shows those windows, always-on and prefilled from the installed
backend, merged to the backend's coarser granularity (one BC window, one tilt
window, no tx). Plain One-shot is rerouted through
`build_v1_params` + `pipelines.single.autocalibrate` whenever `calibrate()`
cannot express what was asked — a custom window, a held Lsd/BC, or exactly one
of ty/tz — which also makes those checkboxes genuinely control the fit for the
first time. Seed arrow steps now follow the window (10 % of the full range).
**Files:** `calib.py` (`tol_defaults`/`tols_are_default`/`_resolve_seed`,
`build_v1_params(tols=)`, reroute), `tab_calibrate.py` (`_sync_limits_mode`,
`_crystalline_tols`, `_sync_seed_steps`), `dialogs.py` (distortion row; dead
`ParameterLimitsDialog` deleted), tests in `test_calibrate_panel_save.py` and
`test_manual_dspacing_calib_ui.py`. Docs: DECISIONS + `gui_documentation.md` §5.

_(Older entries — `eebce45` Batch Integrate run/restore crash from stale
views under a new axis context (reset stack/waterfall/cake views before
re-deriving axis context), 2026-09-09 manual d-spacing (AgBH/SAXS) fit trustworthiness
(BC-only default refinement, per-parameter 1σ, Limits… dialog), 2026-09-04
Jun-Sang Park's PR #7 (Strain Cake tab, job queue, peak-fit panel,
provenance, batch CLI; +108 tests), `fd7f67a` Workstation provenance + Hydra Overall-Cake
per-panel `tx` rotation fix + Batch Integrate Rmin/Rmax + Detector-view
preview, `18c9b77` crash-safe project saves (staging-group swap + rolling
`.bak`) + Save-As analysis-history choice + Open-Project unsaved-changes
guard, `d84c58e` Batch Multi-azimuth cake output + Export for
GSAS-II + MIDAS backend bump + `pytest-forked` isolation, `5954a57` Mask Builder raw-detector-space fix (removed
double-transform bug), `0332683` Batch-Parallel live-view frame-ordering fix,
`21faaf8` Project schema redesign (`gui_workspace` + `analysis`) + unified
Open Project dialog, `c67ad1b` multi-panel calibration refinement fix +
persistence, `e6f2e50` Output-format/Run-mode popups, `a27790a` Batch
Integrate cosmetic overhaul + Batch-Parallel workers, `ae3b665` merged
Workspace+Project into one `.h5`, `af8066f` Batch Browse… parity,
`a54f796`/`ac13797` Browse… popup (multi-file/folder/name-stem + polish),
`101558a` Calibrate Multi-panel→downstream-integration feed, `ccce056`
Flip-Z/Multi-panel fix — trimmed here; full detail in
`documentation/development_history.md`.)_

## Open questions / blockers

- **Windows user (`lheald`) calibration failure, unresolved.** Two
  different tracebacks seen so far, both breaking on a bare
  `from midas_calibrate_v2[.x] import y` statement (once in the plain
  single-detector branch, once via `_build_panel_layout` →
  `midas_calibrate_v2.forward.panels.PanelLayout`) — never inside real
  calibration math. Suggests either an outdated `midas_calibrate_v2`
  install (predating the 2026-08-25 upgrade) or a Windows DLL/native-ext
  load failure in that package. Asked the user to run `import
  midas_calibrate_v2` / `from midas_calibrate_v2.forward.panels import
  PanelLayout` / `pip show midas_calibrate_v2` directly in their env to get
  the untruncated traceback + version — response not yet received.
- **New follow-ups (tracked in ROADMAP.md "Package-side fixes" P3-2/P3-3
  and the Texture per-tab item):** (1) ~~several `midas_calibrate_v2`
  pipelines have no native `im_trans` param~~ — fixed upstream in 0.15.0,
  see 2026-09-11 above;
  (2) `*BinGeometry.from_spec()` has no `apply_trans_opt` hook for masks —
  GUI must keep pre-flipping masks in Python; (3) Texture tab's
  `PoleFigureWorker` has a pre-existing, unrelated mask/ImTransOpt bug;
  (4) `spec_from_calibration_result` has no panel-layout support — GUI
  already works around it (see P3-3).
- **Known-failing baseline (2026-09-10, per-file runs): none.** Every test
  file is green on a clean config. The one failure you will see on this
  machine, `test_smoke` 1 (`test_app_builds_offscreen`), is a local-config
  artifact, not code: `constants._apply` replaces `MATERIALS`/`CALIBRANTS`
  wholesale from the saved config, so a stale block changes what the
  Calibrate combo offers. `HOME=$(mktemp -d) pytest tests/test_smoke.py`
  gives 10/10. **Re-check any suspicious failure that way before calling it
  a regression**, and still capture a per-file baseline before reviewing an
  incoming change (this is how PR #7's and PR #8's real regressions were
  isolated; see the github-skill project memory for the review recipe).
- **The 29 forked SIGSEGVs are fixed (2026-09-10).** They were never the
  pyqtgraph teardown crash — the forked children died before the test bodies
  ran. Cause: pytest imports test modules during collection in the *parent*,
  and importing PyQt5 there (directly, or transitively via any `midas_gui`
  GUI module) initialises macOS CoreFoundation, which a forked child may not
  use. Proved causal by adding one `from PyQt5 import QtWidgets` line to
  `test_set_raw_frame.py`: 12 passed → 12 failed, restored on removal.
  `test_hydra_ui` (8), `test_manual_dspacing_calib_ui` (17),
  `test_hydra_batch_ui` (2) and `test_hydra_calib_ui` (2) now defer every
  Qt-pulling import into a `_load_qt()` called from their `app` fixture,
  which publishes the names (and the `QtCore.QObject` fake workers, which
  cannot be defined at module scope for the same reason) into module
  globals. All 29 pass. **Rule for new Qt test files: import PyQt5 and
  `midas_gui` GUI modules inside a fixture, never at module level** —
  `tests/test_set_raw_frame.py` is the reference.
- **Pre-existing interpreter-teardown crash risk**, especially around
  `CakeViewer`'s ViewBox (`tests/test_hydra_calib_ui.py`,
  `tests/test_hydra_ui.py`) and any module-scoped-fixture MainWindow
  (`tests/test_workspace_ux.py`, `test_smoke.py` run as a whole file).
  Trust per-file isolated runs, not a combined `tests/` run; do not reach
  for `gc.collect()` (confirmed to make it worse). Out of scope, see
  DECISIONS.md for the 2026-08-26 bisection. **2026-08-30: `pytest-forked`
  now isolates this for the trusted per-file workflow** — `test_hydra_
  calib_ui.py`, `test_hydra_ui.py`, `test_smoke.py`, `test_project.py` (and,
  from 2026-09-03, `test_set_raw_frame.py`, which builds one `ImageViewer`)
  all carry `pytestmark = pytest.mark.forked`, so a crash inside one of them
  run alone is a clean `FAILED ... CRASHED with signal N` instead of an
  interpreter abort. Does NOT fix a combined `tests/` run — see DECISIONS.md
  2026-08-30: `os.fork()` itself becomes unsafe once torch/numba/Qt/HDF5
  have spun up background threads earlier in the session, so forked tests
  late in a combined run can crash regardless of their own content (even
  `test_helpers.py`, pure logic). Keep trusting per-file runs only.
  **Widened 2026-08-29:** also
  seen with a `Fatal Python error: Aborted` in a leaked `workers.py`
  `build_geom`-running `QThread` at pytest teardown, reproduced even
  running `tests/test_project.py` (pure-logic, no Qt) alone; confirmed
  present on HEAD *before* the `c67ad1b` panel-refinement commit too — not
  introduced by it. **Widened again 2026-08-29 (schema-redesign session):**
  `tests/test_smoke.py` run alone is *non-deterministic* even on
  unmodified HEAD — 3 consecutive runs gave 10/10 pass, then a Bus error at
  4 dots, then a Segfault at 4 dots (each MainWindow-constructing test adds
  more pyqtgraph widgets to the same process; teardown corruption seems to
  accumulate randomly rather than at a fixed test). Don't trust a single
  green/red `test_smoke.py` run as signal either way — rerun a few times
  before concluding a change broke or fixed it.
- `test_smoke.py::test_app_builds_offscreen` has a pre-existing, unrelated
  local-config flake (stale `visible_tabs` count) — hit again this session,
  confirmed unrelated to the truncation fix.

## Standing rules (from memory)

- Commit history is the record of recent work — see `git log`. Docs
  (`development_history.md`, `gui_documentation.md`) are updated only when
  explicitly asked, not automatically per commit.

# DECISIONS — append-only, newest first

Each entry: what was decided and *why* (the reasoning that would be expensive
to reconstruct later). Never rewrite history; add a new entry to supersede.

## 2026-10-05 — Batch Correction: correct per sub-frame, and pick the dark per file

New feature (asked for at the beamline): reduce each HDF5 file's sub-frame
stack to one frame per group of N sub-frames — mean/median/sum/max — with
dark/bright/background correction, written back out as HDF5, one output file
per input file. New `midas_gui/frame_correct.py` + `tab_batch_correct.py` +
`workers.BatchCorrectionWorker`. Two decisions are worth the ink.

### 1. Corrections are applied per raw sub-frame, and the clip is deferred

The order is: correct each sub-frame with `clip_negative=False`, combine,
then clip the combined frame once.

The obvious alternative — combine, then correct — is wrong for `sum` and
only for `sum`: an N-frame sum must lose `N × dark`, and correcting
afterwards loses one. This is the same bug `workers.StreamPreviewWorker`
already documents avoiding for its preview sum. For `mean`/`max`/`median`
the two orders agree exactly, because dark subtraction and a flat-field
divide are monotone per-pixel maps and `max`/`median` commute with any such
map. So the per-sub-frame rule is correct for all four and there is no
per-op branching in `reduce_chunk`.

The clip is separated out because it is *not* linear: clipping each
sub-frame at zero before summing discards the negative half of the read
noise and biases the sum upward, while clipping once at the end is unbiased
for `sum`/`mean` and (clip being monotone) identical for `max`/`median`.
Clipping once at the end is therefore uniformly right, and is also what
makes an "untick to see negative pixels" option meaningful for checking
whether a dark over-subtracts.

`mean`/`sum`/`max` accumulate one frame at a time rather than materialising
a corrected stack — a 10-frame chunk of 2880² in float64 is 660 MB, paid on
every chunk of a long run. `median` has no streaming form and does
materialise, in float32.

### 2. The dark is resolved PER FILE, as the nearest preceding `dark_before`

The first design (from the screenlog corpus alone) was "the bracketing dark:
`<froot>_dark_before_<first data number − 1>`". Surveying the real data
proved that wrong, and the correction matters:

* The 198 caking screenlogs in `~/midas_runs/midas_screen_logs` gave 192
  resolved runs, and **192/192 read the dark from `exchange/data_dark`** —
  so there is no separate dark image format to support, only the question of
  which file holds it. Two patterns, ~50/50: a `<froot>_dark_before` sibling,
  or the data file's own `data_dark`.
* The 160 VAREX scan folders still on
  `/home/beams/S20IDUSER/mnt/s20a/*/varex*/` showed what the logs could not:
  **darks are re-measured throughout a scan.** A folder is a series of
  segments, each of the form `dark_before … data … dark_after`.
  `Fe9Cr_KGT6038_load1_waxs` alone holds 11 `dark_before` and 8 `dark_after`
  files among its data. Of the 936 data files in the 56 dark-bearing
  folders, **936 have a preceding `dark_before`**, at a median distance of 8
  files and a maximum of 121. The other 102 folders carry none at all (the
  self-dark convention).

So a folder-wide rule would be right only for a scan's first segment and
would hand a stale dark to the ~900 files after it. The per-file rule —
nearest preceding `dark_before`, then nearest following `dark_after`, then
the file's own `data_dark`, then the loader's Dark field — reproduces the
logged choice whenever a scan has one segment, so it is compatible with the
log corpus rather than a departure from it. `dark_after` never appears in
the logs but is common on disk; it is handled anyway.

Both corpora are replayed as tests (`tests/test_dark_autodetect.py`), which
skip when the share isn't mounted. They currently pass at 936/936 real files
and 191/192 logged runs. The one exception is named explicitly in
`MANUAL_OVERRIDE_LOGS` rather than hidden behind a loosened assertion: that
operator passed an explicit `-P` override pointing at a *different scan's*
folder, at a file with no dark marker in its name. No filesystem rule can
recover that, and none should guess — a deliberate cross-scan dark is what
the loader's own Dark field (`resolve_dark(auto=False)`) is for. A second
such case will fail the test rather than pass unnoticed.

The chosen dark is named in the Log per file and stored in the output's
`midas_gui_dark` attribute, so a run is auditable afterwards rather than
trusted.

### Scope

HDF5 input only. "Chunks restart at every file" has no meaning for TIFF/GE,
where a file holds exactly one frame and there is no sub-frame stack to
group; a TIFF selection says so rather than silently applying the
consecutive-*file* grouping `workers._ChunkCombinedFileSource` implements
for Batch Integrate. That remains available if it is ever wanted here.

---

## 2026-10-01 — Ring prediction bound by detector geometry, not a flat 30°

Reported symptom: after a Calibrate fit, the image overlay and the radial
profile's ring markers were missing rings that visibly lie within the frame.

Cause: `helpers._predict_ring_radii()` generated the candidate ring list from
the calibrant's wavelength/d-spacings with `two_theta_max_deg=30.0` hardcoded
inline for the lattice/space-group path, and silently defaulted to the same
30° by never passing `max_2theta_deg` to `simulate_rings_from_dspacings()`
for the explicit-d-spacing path. Both decisions happen purely in
wavelength/d-spacing space — no Lsd, pixel size, beam centre, or detector
size is consulted. A later pixel-radius filter in the two overlay call sites
(`tab_calibrate._draw_rings`, `hydra_calib_widgets._redraw_rings`) used
`max(NrPixelsY, NrPixelsZ)` as a stand-in for "does it fit on the detector,"
but that is an axis-aligned size, not the true distance to the farthest
corner from an (often off-centre) beam centre — and it runs only *after* the
30°-limited list was already built, so it cannot recover a ring the
generation step never produced. Net effect: any geometry whose real 2θ
coverage exceeds 30° (short Lsd, wide detector, off-centre beam — not an
unusual combination) silently lost real rings near the frame edges/corners.

Fix: added `helpers.max_two_theta_deg(bc_y, bc_z, ny, nz, lsd_um, pxY_um,
pxZ_um)` — the true max 2θ reached at the farthest detector corner, reusing
the same beam-centre/corner-distance reasoning as the existing
`rmax_corner_px` (used for the Batch/Queue Rmin/Rmax "Corner" preset), but in
physical units so it also works with non-square pixels. `_predict_ring_radii`
now computes this from the result's own geometry and passes it as
`max_2theta_deg` on both ring-generation paths, falling back to 30° only when
`NrPixelsY`/`NrPixelsZ` aren't known yet (a result built before an image was
loaded) or geometry fields are missing entirely (a minimal `SimpleNamespace`
in a test) — both existing cases pinned by
`tests/test_manual_dspacing_calib_ui.py::test_manual_fit_result_flows_through_on_done_with_correct_rings`
and
`tests/test_helpers.py::test_predict_ring_radii_uses_d_list_branch_not_crystalline_fallback`
respectively, both unchanged by this fix. The two overlay call sites' post-hoc
pixel filter was also corrected to `rmax_corner_px(...)` (was
`max(NrPixelsY, NrPixelsZ)`), since leaving the old axis-aligned filter in
place would have silently re-clipped exactly the farther, corner-region rings
the generation-side fix now produces. New
`tests/test_helpers.py::test_predict_ring_radii_uses_detector_coverage_not_fixed_30deg`
pins a short-Lsd/wide-detector geometry whose true coverage is ~70° and
asserts a 45°-2θ ring (inside the old 30° cutoff's blind spot) is predicted.

Deliberately out of scope: the Hydra multi-panel geometry card
(`hydra_geometry_card.py`) already exposes a user-editable max-2θ spinbox
(default 25°) on a separate code path from the single-detector Calibrate tab
and from `hydra_calib_widgets.py`'s per-panel overlay — it already lets a
user dial coverage in, so it wasn't touched. `simulate_rings`/
`simulate_rings_from_dspacings` keep their `max_2theta_deg=30.0` default
signature; only the caller that had no business picking a fixed number
(`_predict_ring_radii`) now supplies a computed one.

**Verified:** `tests/test_helpers.py`, `tests/test_manual_dspacing_calib_ui.py`,
`tests/test_manual_fit_conditioning.py`, `tests/test_batch_queue_ui.py`,
`tests/test_hydra_calib_ui.py`, `tests/test_smoke.py` all green per-file on a
clean `HOME`. `pyflakes` on the three touched files shows only the same
pre-existing unused-import warnings as before.

_Condensed 2026-08-31 (~2150 → ~500 lines): cut verification transcripts,
file-by-file implementation narrative, and duplicated/superseded content;
kept the durable "why" behind each decision. See git history before this
date for the full uncondensed entries if ever needed._

## 2026-09-30 — One zarr per rotation, not one per frame

Batch Integrate wrote the `zarr` format one archive per combined output
frame. That was deliberate — mpe_wf writes one zarr per scan point, and we
extended it so a file split by "Combine sub-frames" got one per chunk — but
it was parity with a convention, not a requirement, and it does not scale.
The folder that prompted this (three 1442-sub-frame VAREX files at
`OME_SUM 10`) would have produced **435 single-cake archives**. Asked for one
per source HDF5 instead.

Nothing downstream forced per-frame. GSAS-II's `readMidas` validates only
`InstrumentParameters` / `REtaMap` / `OmegaSumFrame` (see the 2026-09-26
entry), all multi-frame by construction; the Zarr Viewer slices any leading
axis; MIDAS's own native output is multi-frame. Per-frame was a default
nobody had chosen.

**Grouping reuses the rotation key rather than inventing a second one.**
A group is one rotation — the same unit ω is measured from — so
`_HDF5StackGlobSource.zarr_group_key` is defined as
`omega_channel_window(idx)[0]`, exactly as `raw_window_for_index` is defined
as `omega_channel_window(idx)[1:]`. One walk decides which rotation a frame
belongs to, and both the angle and the archive read it. Had grouping done
its own file walk, a future change to one would silently desynchronise an
archive from the angles inside it. `_ChunkCombinedFileSource` returns a
constant for the same reason its ω window counts across the selection:
one-frame-per-file data only becomes a rotation as a series.

**Streaming, not buffering.** The obvious implementation — collect a
rotation's cakes and hand `write_gsas_zarr_zip` a list — would have cost
~830 MB for a 1442-frame group at 72x1000 float64. The backend already
exposes `GSASZarrWriter` (context manager, `add_frame`/`close`), which
`write_gsas_zarr_zip` is itself a thin wrapper over and whose docstring
promises "a long scan never has to be held in memory". Holding one open
writer per group keeps peak memory identical to the per-frame path. This is
why the work is a `_ZarrGroupWriter` owning a writer rather than a list.

`omega_sum_frames` is deliberately left at 1. It chunks ALREADY-INTEGRATED
frames inside the archive; `OME_SUM` collapses raw sub-frames before
integration. They are different axes and feeding one from the other would
collapse the same data twice. Noted because the names invite exactly that
mistake.

**Provenance moved from per frame to per group**, which is strictly cheaper
(one extract/repack per archive, not per frame) and is what
`h5_metadata.align` was built for — it already reduced "one entry per output
frame" from a *sequence* of frame ranges and needed no change, only to be
handed the whole group's ranges. `storage_ring_current_mA`/`sample_motors`
stay scalars for a single-frame group so a "frame"-grouped archive is
byte-comparable with what shipped before, and become per-frame lists
otherwise. A "run" group spanning several source files has no single
`instrument/` tree to align, so the copy is skipped and logged rather than
guessing which file's tree to use.

**Batch Parallel splits on group boundaries** (`_split_into_chunks_on_groups`).
The count-based splitter would hand two workers halves of one source file and
they would open the same `.zarr.zip` path. Alignment caps parallelism at the
number of groups — a 3-file folder uses 3 workers, not 8 — which is a real
throughput cost, accepted because the alternatives were worse: forcing
sequential loses all cores, and refusing the combination is a dead end. Groups
are formed from CONSECUTIVE equal keys, not by gathering every index with a
given key: reordering frames to suit the writer would be a worse bug than
writing two archives for an interleaved source.

**Default stays "frame".** Every project file and script written so far ran
under it, and a silent change to what a re-run produces is not worth the
convenience. The combo is in `_state_widgets` so it persists with the rest of
the GUI state, and `project.integrate_attempt_gui_fields` falls back to
"frame" for attempts recorded before it existed — which is what those
attempts actually ran.

The CLI flag (`--zarr-grouping`) and the argv emission went in with the
feature, not after it. That is the direct lesson of the 2026-09-29 omega
entry below: a setting wired into only the in-process path is invisible, not
an error — the background job just quietly does something else.

## 2026-09-30 — PR #11 (junspark) merged into `main`: 8 staged checkpoints, not one big merge

**Why staged rather than a single merge.** PR #11 was titled "Major
documentation update" but was actually 48 commits / 67 files / +12405/-709
lines — a new Zarr Viewer tab, a cake-parameters editor, omega tracking
through Batch Integrate, several Calibrate/Corrections fixes, and a
tab-close-button feature. The explicit concern was GUI layout: with one
merge, a button/field/arrangement regression anywhere in that diff would
have been nearly impossible to isolate. Instead `main` was advanced through
the PR's own commit sequence in 8 checkpoints (grouped by the PR's already-
atomic commit boundaries), each on a disposable `merge/pr11-staged` branch,
with a targeted test run + `pyflakes` diff + stale-import grep + an offscreen
screenshot pass after every checkpoint, before advancing to the next.
`main` was never touched until all 8 were validated.

**Mechanics.** `main` was a full ancestor of `origin/pr/11`'s tip (junspark
periodically merges `main` back into his branch), so the *final* PR head was
a clean fast-forward — but every *intermediate* checkpoint commit was not
(his early commits predate our later `main` commits), so each checkpoint was
a real 3-way merge, not `--ff-only`. Conflicts were almost always in
`.context/STATE.md`/`DECISIONS.md` (both sides narrating their own recent
work) and resolved as a straight concatenation — newest dated entries first,
since both files are append-only/newest-first and the two branches' new
entries never actually overlapped in time. Two exceptions needed real
reading rather than mechanical concatenation, both in `workers.py`: see the
two bug entries below.

**Checkpoint table** (commit ranges into `origin/pr/11`, oldest to newest):

| CP | Ends at | Content |
|----|---------|---------|
| 1 | `8f578a6` | Calibrate: seed spinbox steps, real ± parameter windows for crystalline calibrants, full-width Run/Save, named working dir + `.midas_scratch/`, `.instr.*` suffix, Distortion-selection persistence |
| 2 | `1f25933` | Terminology: "average"/"avg"/"ave" → "mean" across 11 files (labels/tooltips/one method rename, no behavior change) |
| 3 | `8e1a7f1` | Batch "Show bin grid" no longer resets pan/zoom; Corrections polarization plane default 0°→90° (physically correct, horizontal ring plane) |
| 4 | `6f9c7c7` | Frozen-point backend guards (net-zero vs. our `main`, already had it via an earlier upstream merge); provenance `script`/`script_sha256`/`tag` fields |
| 5 | `3e0e6c2` | **New Zarr Viewer tab**, visible-by-default; Batch's stream-mode preview read moved off the GUI thread (`StreamPreviewWorker`) to fix a real HDF5-over-NFS freeze; `HDF5_USE_FILE_LOCKING=FALSE` |
| 6 | `35a7b8b` | Zarr Viewer moved after Batch Queue in the tab bar; ✕-to-close on every optional tab (with a legible custom icon); GSAS-II zarr contract + source HDF5 instrument-tree carry-through |
| 7 | `6ef8e75` | Cake-parameters editor dialog (all 9 `cake_parameters` CSV columns in one place) |
| 8 | `7ccf0fb` (PR head) | Omega (rotation-angle) tracking end-to-end: cake-summary/loader-hint readouts, background-job omega fix, 2D-CSV fix, chunk-read perf fix, freeze diagnostics (`kill -USR1`) |

**Three real bugs found during the merge, not present in either branch alone:**

1. **Checkpoint 1** — Frozen-point (high-tilt) silently ignores the Refine
   card's ± tolerance window: `calib.py`'s `_seed_and_v1` call in that branch
   is missing `tols=tols`, unlike the identical bayesian/joint call one block
   above. Not a backend limitation (`spec_from_v1_params` reads `v1.tol*` the
   same way regardless of pipeline) — a one-line omission. Disclosed with a
   console warning (matching `first_time`'s existing, genuine disclosure)
   rather than fixed inline, per instruction; the real fix is still open
   (see STATE.md).
2. **Checkpoint 3 merge conflict** — the PR's own zarr-crash fix
   (`count_cake(None, ...)` when a physics correction leaves `geom` deliberately
   `None`) only patched the zarr output path. Our own more-recent multi-azimuth
   HDF5 "cake" output (`want_h5_cake`) shared the exact same crash, unpatched —
   Multi-azimuth output + HDF5 + any physics correction together would have
   crashed Batch Integrate. Fixed by sharing one geometry-fallback computation
   between both consumers. Reproduced the crash on the unpatched code and
   confirmed the fix via a standalone smoke run (no existing test covered this
   combination). Checkpoint 8 independently re-derived the identical fix while
   merging upstream on their side — confirms it was the correct fix, not just
   a workaround.
3. **Checkpoint 8** — `tests/test_viewer_origin_and_readout.py` (pre-existing,
   untouched by the PR) went from 100% reliable to a ~50% reproducible
   SIGSEGV/SIGBUS. Root cause: checkpoint 5's async `StreamPreviewWorker`
   moved `BatchTab`'s default-preview read to a background `QThread`; that
   thread's lazy `midas_integrate_v2` import (which pulls in `torch`) can be
   the *first* torch import in the process when a test constructs a bare
   `BatchTab()` as the first heavy widget — racing the main thread's own
   import machinery inside torch's C-extension init. An `isRunning()`/`wait()`
   mitigation was tried first and didn't help (confirming the race is inside
   torch's init, not about thread lifetime). A real app never hits this,
   since some earlier-constructed tab (Calibrate, via `midas_calibrate_v2`)
   has always imported torch on the main thread first. Fixed by pre-warming
   the import in the test itself. Verified 10/10 clean runs after, vs. 5/10
   before.

**Verified overall:** full 75-file per-file `pytest` sweep on a clean `HOME`
green except the one known pre-existing `test_apply_project_calibration_
single_detector` SIGABRT (present on clean `main` too — see 2026-08-30 entry).
`pyflakes midas_gui/*.py` unchanged at 39 throughout (new warnings, when any
appeared, were always the same pre-existing "intentional side-effect import"
pattern applied to a new file). No stale imports or accidental deletions at
any checkpoint (checked via `git ls-tree` diffs, not just `git status`). Every
tab's content below the tab bar pixel-diffed byte-identical against a
pre-merge baseline screenshot except where a checkpoint's own commits said it
should differ.

## 2026-09-29 — A frozen GUI has to be able to tell us where it is stuck

Two "the GUI is hanging" reports in one day, and neither could be answered.
The first had a real cause (the Detector-view preview read the whole file);
the second could not be reproduced at all — the full file-selection path,
timed offscreen against the same 24 GB VAREX file, costs 0.27 s for
`list_h5_datasets` and milliseconds for everything after it, and the preview
that used to block now runs in a `StreamPreviewWorker`. Guessing from a
screenshot is not a diagnosis, and every guess costs a beamline round trip.

So `launch.py` now arms `faulthandler.register(SIGUSR1, all_threads=True,
chain=False)`. `kill -USR1 <pid>` appends every thread's stack to
`~/midas_gui_hang.log` **without** killing or interrupting the process, which
is the whole point: the fatal-error handlers `app.py` already installs only
fire on a crash, and a freeze is not a crash. Signalling twice shows whether
the stack is moving. It is inert until signalled, guarded on `hasattr(signal,
"SIGUSR1")` for Windows, and wrapped in a bare `except` because a debugging
aid must never be the reason startup fails.

`py-spy` would also have worked and needs no code, but it is not in the
`midas-gui` env and `environment.yml` is the source of truth for that env —
adding a dev tool to a beamline environment mid-session is a bigger change
than eleven lines in the launcher.

## 2026-09-29 — The ω readout must not put a filesystem walk on every signal

`_recompute_omega_span` derives its mapping line from the real source
(`_open_source_cfg` → `raw_window_for_index`) rather than from arithmetic on
the raw count, so the readout cannot drift from what a run computes. That is
the right call and it stays — but it means the readout opens an HDF5 header
per file, on the GUI thread, and it was wired directly to `dataChanged` and
to "Combine sub-frames" `valueChanged`. Both fire several times for one user
action, and holding a spin box fires once per step; a folder pick multiplies
each firing by the file count. Measured: ~1 ms for one file (plus a one-time
2.3 s lazy import inside `_open_source_cfg`, which the run would pay anyway),
but that scales linearly with the pick.

Two defences, both cheap. A 150 ms single-shot `QTimer` coalesces a burst
into one walk, and a cfg-keyed short circuit makes a repeat refresh for an
unchanged source free. Deliberately *not* moved to a background thread: the
readout must be correct before the user can press Start, and a walk that is
already ~1 ms for the common case does not justify the lifecycle of another
worker.

This was found while chasing the second hang report. It is not that hang —
the timings above rule it out — but it was a real per-signal cost on the path
the report pointed at, and it is the kind that only shows up on the big
folder picks the beamline actually uses.

## 2026-09-29 — ω restarts at every file, and both ω paths share one window

Reverses the origin rule set when per-frame ω landed a few days earlier. The
computed ramp counted raw sub-frames **globally** across a multi-file pick, so
file 2 continued file 1's rotation. The argument for that was continuity: a
scan split across files is one sweep, and restarting the count would make two
files look like two sweeps at the same angles.

Two things were wrong with it.

*It disagreed with the measured channel.* `_omega_resolver`'s `_measured`
branch reads a 1-D omega dataset **stored inside each file**, indexed from 0
in each, so it has no choice but file-local indices. The ramp therefore ran in
a different coordinate system from the channel it is supposed to be the
fallback for — documented on both sides as correct, which is how it survived.

*And the global count was not even reliable.* `_filter_paths_by_frame_number`
drops files **before** `_HDF5StackGlobSource` is constructed, so the cumulative
offset could only count the files that survived the filter. Dropping leading
files silently rebased the whole ramp — the exact behaviour the documented
rule ("a filter shifts the angles rather than rebasing them") promised would
not happen. The global rule could not be made true without pushing the filter
down into the source, which is a much larger change for a worse answer.

So: **ω is measured from raw sub-frame 0 of the rotation the frame came
from.** An HDF5 sub-frame stack *is* one rotation, so each file restarts at
`OME_START`. One-frame-per-file data (TIFF, `.ge*`) is not — a single such
file is one exposure and only the series is a rotation — so
`_ChunkCombinedFileSource` still counts across the selection. The two classes
now differ on purpose, and each says why in its docstring.

Mechanically the point is that there is exactly **one** window function left:
`raw_window_for_index` returns `omega_channel_window(idx)[1:]` rather than
repeating the walk. Two implementations that must index the same axis is the
defect that was just removed; leaving two copies of the walk would invite it
back. `tests/test_omega_windows.py` pins that identity for every frame across
six chunking/filter configurations, because the failure it prevents (the two
drifting apart) reads off the end of a later file's array instead of raising.

Still correct under Batch-Parallel: the window is a pure function of the
absolute frame index and the per-file header counts, both identical in every
worker, so two workers cannot disagree about a frame's angle.

The user chose this reading when the inconsistency was put to them. The
GUI now states the resulting mapping rather than leaving it implicit — see
the next entry.

## 2026-09-29 — Saying that frame numbers and angles are one axis

`start`/`end`/**Combine sub-frames** count raw sub-frames; `OME_START`/
`OME_STEP`/`OME_SUM` put degrees on the same sub-frames. The app never said
so, and the user asked for the two to be made consistent *and visible*.

Consistency was largely already there and worth recording as such: `OME_SUM`
is not mirrored into the cake dialog, it **is** the loader's
`_combine_chunk` widget, read and written in place — so it cannot drift. What
was missing was the statement.

Decided (with the user): labels stay as they are — on a multi-file pick
`start`/`end` really are file numbers, so renaming them "sub-frame" would be
wrong — and the correspondence goes into readouts and tooltips instead, in
**both** places it is operated: the cake summary line and the loader's range
hint.

Two judgements inside that:

*The summary states both rates.* `Δω 0.25°/sub-frame = 6.25°/frame` is the
whole point — combining 25 sub-frames makes consecutive output frames
25×`OME_STEP` apart, and that multiplication is what a user gets wrong.

*The "not set" state is called out.* A loaded rotation with `OME_START`/
`OME_STEP` both 0 gets a genuine 0.0 per frame (deliberately — a stationary
sample really is at 0°), so nothing downstream looks wrong. That is exactly
the state a real run was in when its `/Omegas` came out all zero. The line
says `ω 0° on all 58 frames (OME_START/OME_STEP not set)` rather than
silently rendering a flat ramp.

`DataLoaderPanel` is shared with tabs that have no cake parameters, so it does
not learn about ω: `set_omega_hint_fn(fn)` takes a callable from whoever owns
the angles (the pattern `set_preview_sum`/`set_tab1_mask` already use), and
unset the hint text is byte-identical. The tail is re-rendered separately from
the hint body so an `OME_START` edit costs no file access — the windows are
cached when the *source* changes, and the angles are arithmetic over them.

## 2026-09-29 — 2D CSV wrote nothing, and reported that it had

`want_cake` includes `"2d_csv" in self._fmts`, so the cake **was** computed;
the write site then threw it away with
`cake_2d=(cake_2d if self._multi_azimuth else None)`, `write_profile`'s
`elif fmt == "2d_csv" and cake_2d is not None` fell through to nothing, and
the caller appended `<base>.2d_csv` to the reported paths — a file that was
never written, under a name that does not exist in either mode (the real one
is `<base>_cake.csv`). A run with the format checked and multi-azimuth off
produced an empty `2d_csv/` folder and a success message.

Root cause was an overloaded flag: `cake_2d is not None` meant both "a cake is
available" and "fan out one lineout per η bin". The Batch write site wanted
the second gated on the checkbox and had no way to say so except by
withholding the cake. Split into an explicit `per_eta` parameter, defaulting
to the old `cake_2d is not None and cake_sigma is not None` so every other
caller is untouched.

`write_profile` now **raises** for `2d_csv` with no cake rather than returning
quietly. The one legitimate cake-less caller — `write_all_profiles`, the Save
button's in-memory path after a 1-D run — excludes the format itself and both
call sites already log a note explaining why, so nothing reaches the raise by
accident. A silent no-op in a writer is worth converting to a loud failure
precisely because this one cost a real run its output without anyone noticing
until the folder was opened.

## 2026-09-29 — Reading one frame must cost one chunk: finishing the _HDF5StackGlobSource fix

`_HDF5StackGlobSource` was fixed once already, for *counting*: `n_frames`
used to decode every selected file, so a run looked hung before it started
(that fix is in the class docstring). Reading was left on the old shape —
`get(idx)` called `read_hdf5_stack_combined`, which decodes every chunk of
the owning file and returns the list, then kept one element.

Measured on the user's own data (`…_029531.vrx.h5`, `exchange/data`
(1442, 2880, 2880) uint16 uncompressed, 23.9 GB, NFS): **~230 s and ~1.9 GB
resident to return one 33 MB frame**. Now 415 MB and ~4 s. Two callers paid
it: the Detector-view preview, which asks for exactly one frame (the user's
"takes a really long time to show the pattern"), and each parallel
`BatchWorker` chunk — four workers start at four different offsets, so a
cache keyed on "the current file" helps none of them, and the run reads the
file about four times over.

**Decided: one definition of the chunking, used by both readers.**
`helpers._stack_chunk_bounds(n, k, …)` is now the only place that says where
chunk *k* starts and ends; `read_hdf5_stack_combined` (all chunks) and the
new `read_hdf5_stack_chunk` (exactly one) both go through it. This is not
tidiness — `_stat` derives the chunk *count* from the dataset shape alone,
and that only stays honest while the chunking rule has a single definition.
With bounds shared, `get`/`__iter__` can take `n_chunks` from the header
counts instead of `len(frames)`, which is what made the whole-file decode
structurally necessary before.

**Kept `read_hdf5_stack_combined` rather than deleting it.** It is a correct,
well-tested, generally useful helper; the bug was a caller using a
whole-file read for random access, not the function. Its remaining
production use is none, and that is fine — the source class no longer has a
whole-file path to fall back into.

**Decided: the cache holds one chunk, not one file.** Chunk-at-a-time would
be worse than the bug if a sequential pass re-read anything, so
`test_iterating_reads_each_chunk_exactly_once` pins that `__iter__` still
costs the file exactly once. Both `__iter__` and a chunked `BatchWorker`
walk frames in order, so one chunk of lookahead is all either needs.

**Tests assert slices, not seconds.** Wall-clock and byte counts are not
assertable, so `tests/test_stack_chunk_reads.py` proxies `h5py.File` and
records every `dset[a:b]` taken — "reads one chunk" becomes an exact set
comparison. Four of the eleven fail against the old implementation; the rest
pin frame-for-frame equivalence with it, because a faster wrong answer is
not the deliverable.

## 2026-09-29 — Omega on the background-job path: serialise the config, don't re-derive it

The omega feature shipped wired into `BatchTab._run` only. `_run_as_job`
builds an argv for a detached `python -m midas_gui.batch_cli`, and that argv
carried no omega flags, so every background job wrote ω = 0 on every frame
regardless of the Cake parameters. Found from a real 20-ID run whose log said
`OME_START=0, OME_STEP=0` while the tab's summary read `Δω 1°/sub-frame`.

**Why it was invisible.** Nothing could raise. From inside `BatchWorker` an
unpassed `omega_cfg` and a genuine stationary 0°/0° are the same thing — which
is a *deliberate* earlier decision (see the "0/0 is a real angle, not a
sentinel" entry), and the right one: a sentinel would have made a stationary
sample unrepresentable. The cost is that the config must be proven to arrive,
because its absence is indistinguishable from a valid value. That is a
testing obligation, not a reason to reintroduce a sentinel.

**Decided: one shape, serialised, never recomputed.** `batch_cli._omega_cfg`
mirrors `BatchTab._omega_cfg` key-for-key and the tab emits from that same
method, rather than the CLI re-deriving angles from a cake CSV it would have
to be handed separately. Two derivations of the same quantity is how the
paths drifted apart in the first place, and the measured-channel case cannot
be re-derived from a CSV at all.

**Decided: `--ome-start`/`--ome-step` go out unconditionally**, including at
0/0 where they are no-ops. The launched command line is echoed into the Logs
tab and is the only window a user has into a detached job's configuration; an
omitted flag is precisely what made this bug unobservable. Channel and
collapse stay conditional — they have an unambiguous "off" spelling.

**The regression guard is a round trip, not two half-assertions.**
`tests/test_batch_cli_omega.py` drives the real argv builder and feeds its
output to the real CLI parser, asserting `batch_cli._omega_cfg(parsed) ==
tab._omega_cfg()`. Asserting "the tab emits a flag" and "the CLI parses a
flag" separately is exactly the pair of green tests that would have coexisted
with this bug. `batch_cli.py` had no tests of its own beyond `_source_cfg`,
which is why this path was the one that rotted.

## 2026-09-29 — Merging upstream's cake HDF5: which side won, and why

Upstream (`d-beniwal/MIDAS_GUI`) added two commits on 2026-09-28 —
`31e904c` (multi-azimuth HDF5 output, new `midas_gui/cake_hdf5.py`) and
`61feeb3` (flatter cake layout, 2θ/d/Q axes, wider provenance). They land on
the same code the omega work had just rewritten. Merged on
`merge/upstream-2026-09-29`; only `workers.py` and `tab_batch.py` conflicted.

**Taken from upstream wholesale**: `cake_hdf5.write_cake_h5` and everything
that feeds it, the `calibration_snapshot` widening of both provenance
entries, `provenance_history` as a dataset as well as attrs (invisible in a
tree view otherwise), `json.dumps(..., indent=2)`, `project.json_default` as
a public name, and `helpers.collapse_cake_eta` — this fork had the identical
implementation inline in `tab_batch._collapse_cakes`, so the shared helper
simply wins and the local copy delegates.

**Taken from this fork**: the whole omega feature. Upstream's own
`all_omegas` is frame *indices*, feeding the combined-HDF5 stem's `<lo>_<hi>`
token; this fork had already split that list in two for exactly the reason
upstream's rename would have re-hidden. So upstream's list became
`all_frame_idx` here, and its append condition was widened to the union
(`(want_zarr or multi_azimuth) and cake_2d is not None`) — upstream needs it
in cake mode for the stem, this fork needed it in zarr mode, and both are
now satisfied without either behaviour changing.

**A real bug avoided in the merge.** Upstream hoisted the BinArea count out
of the `want_zarr` branch so the cake HDF5 could share it — but hoisted the
version that passes `geom` straight to `count_cake`. On the corrections path
`ctx["geom"]` is deliberately `None`, and this fork had already fixed that
crash for the zarr branch (build a plain-kernel geometry purely for the
count; see the 2026-09-13 entry on why `corr_counts` is not a substitute).
Resolving to upstream's structure with this fork's geometry fallback keeps
both: one shared `cake_bin_area`, computed safely.
`test_the_cake_hdf5_survives_physics_corrections` pins it, mirroring the zarr
test that exists for the same reason.

**One thing added rather than merged.** `write_cake_h5` gained an `omegas`
argument and writes a root-level `omegas` dataset (degrees, `units` attr),
dropped rather than padded on a length mismatch. Upstream's cake file is a
GUI-native archive and explicitly the thing a pole figure over a rotation
series would read, so leaving the angle out of it would have made the fork's
own invariant — the angle reaches every output — false in exactly the mode
that needs it most. Threaded from `BatchWorker`'s `all_omegas`, from
`write_all_profiles`' new `omegas=` on the cake branch, and from
`BatchRunCoordinator`'s merged write.

## 2026-09-29 — Omega becomes a real angle, and `/Omegas` stops being an index

**What was actually wrong.** Both zarr writers filled the per-frame omega axis
with `float(frame_index)`, and the backend stores that as `/Omegas` with
`attrs {"Units": "Degrees"}`. So every zarr this app had ever written recorded
frame counts as angles. Nothing downstream read it (`gsas_ii_refine.py`
doesn't), which is why it survived — but the next piece of work is peak fits
and pole figures, where omega is the independent variable, and those cannot be
built on an axis that is silently an index. Meanwhile the cake CSV's
`OME_START`/`OME_STEP` — the real rotation, fetched at the beamline off
`20idaSoft:userTran9.H`/`.I` — were parsed, round-tripped, and applied to
nothing (see the 2026-09-28 entry, which deliberately left them inert).

**Why one formula instead of mpe_wf's three cases.** `gui_data_explorer.py`
handles no-combining, `OME_SUM = n`, and collapse-everything as separate
expressions. All three are the same statement:

    omega(frame) = OME_START + mean(raw sub-frame indices of that frame) * OME_STEP

written as `start + 0.5*(lo+hi)*step` over the frame's inclusive raw window.
For `OME_SUM = n` this is algebraically `ome_start + (idx*ome_sum +
(ome_sum-1)/2)*ome_step`, mpe_wf verbatim — pinned by a test that runs both
expressions over several `n`. The user's "averaged or summed" requirement then
isn't a branch at all: it falls out of the window spanning the run. Three
cases would have been three places for the sign of a reverse scan or an
off-by-one on a short last chunk to go wrong independently.

**Why the raw indices are global, not per-file.** A rotation scan split over
147 VAREX files keeps rotating; file 2's frames are at higher angles, so its
raw indices have to continue rather than restart. The same reasoning makes
them *absolute* rather than per-worker: in Batch-Parallel mode each chunk sees
only its slice, and `abs_i` is already the absolute output-frame index in both
`_iter_frames` branches, so the window — and therefore the angle — is
reproducible regardless of which chunk computed it. The collapse override
deliberately takes its run-wide window from `source.n_frames` for the same
reason; taking it from the frames a worker happens to iterate would make two
chunks disagree about the single collapsed angle. A measured omega channel
wants the opposite (a per-file 1-D dataset is indexed file-locally), hence
`omega_channel_window` as a separate method rather than a flag on the first.

**Why `OME_START = OME_STEP = 0` is a genuine 0.0 and not a sentinel.** A
stationary sample really is at ω = 0, and the alternative — treating the
default as "unset" and falling back — is exactly how the frame index got in
there. 0.0 is a far better wrong answer than an index if the user forgot to
set the keys. The cost is a visible behaviour change for anyone who never
opens the cake editor, so it is called out in the docs and a `[batch] omega:`
line names the source on every run.

**Why omegas are stored in the logged attempt rather than recomputed at
export.** A measured omega channel is read off the raw frames, and the
GSAS-II export path never reopens them — recomputing would silently downgrade
a measured angle to the computed ramp. So `results/omegas` rides along with
`frame_ids`. Legacy attempts that predate it fall back to recomputing from a
recorded `omega_cfg`, and failing that to zeros — deliberately *not* to
`range(n_frames)`, which would reintroduce the bug on the one path that still
had it. Which of the three happened is recorded as `omega_source` in the
provenance `extra`, because a stored 0.0 and an unrecorded 0.0 are the same
number and a reader has to be able to tell them apart. A stored list whose
length doesn't match the frames is discarded rather than padded: angles are
matched positionally, and a misfiled angle is worse than a missing one.

**Why the combined HDF5 gets `extra_datasets={"omegas": …}` rather than a
`ProfileMetadata` field.** It is per-frame data, not a per-run scalar, and
`m.write_h5` already accepts exactly that. Same length rule as above — the
branch skips rather than pads.

**What was left alone.** `PoleFigureWorker` is still single-frame and takes
χ/φ from its own cfg; making it omega-aware across a series is the work this
entry exists to make possible, not part of it.

## 2026-09-28 — Cake parameters get one editor that mirrors, not one that hosts

**Question that started it.** "It will be good to have the ability to change
the integration / caking parameters and save (like the mpe_wf gui)." Batch
Integrate could already *read* an mpe_wf `cake_parameters` CSV, but could not
write one, and the nine values it carries were scattered across three places:
R behind `R bins…`, η behind `Azimuthal bins…`, `OME_SUM` over on the loader
card, and `OME_START`/`OME_STEP` parsed then discarded.

**Why `_CakeParamsDialog` copies values instead of hosting the tab's widgets.**
`_RadialBinsDialog` and `_AzimuthalBinsDialog` each *contain* the real
spinboxes — `_r_min` genuinely lives inside the radial dialog's layout. That
works only because each of those widgets has exactly one home; a widget has
one parent, so adding `_r_min` to a third dialog's layout would silently
reparent it out of `_RadialBinsDialog`, and the R bins popup would come up
empty. So the new dialog keeps its own spinboxes and copies in on `showEvent`
and out on Apply. That constraint turned out to be a feature: Apply now means
something, and you can open the editor, try numbers, and close without having
touched the next run. The mirrors copy their target's range, decimals and
suffix so the dialog can't accept a value the tab would silently clamp.

**Why `OME_START`/`OME_STEP` are invisible widgets rather than plain floats.**
They drive mpe_wf's integration backend and have no counterpart in
`midas_integrate_v2`'s `IntegrationSpec`, so they are applied to nothing —
but mpe_wf's reader rejects a missing or empty column, so a CSV we write has
to carry them. Holding them in two `_fspin`s that are in no layout means
`_state_widgets()` round-trips them through Save/Load GUI State for free, and
gives the dialog a real range to mirror, with no new persistence code. They
are the sort of thing a later reader deletes as dead widgets, so the comment
where they're built says why they exist.

**Why Save does not reproduce mpe_wf's silent write.** mpe_wf's editor writes
straight to `/home/beams/S20IDUSER/mnt/s20a/<expid>_bc/cake_parameters.<bl>.<det>.csv`
with no prompt. It can, because it runs as `S20IDUSER`; this GUI runs as
whoever launched it, into the same shared beamline tree. So the same path is
*suggested* in a Save-As dialog instead — and only when that directory already
exists. `bc_path_parts`'s positional derivation is right for the full four-deep
mpe_wf layout and invents a plausible-looking sibling for anything shallower
(the failure `helpers.suggest_working_dir` documents); seeding a save dialog on
a path nobody has made, inside a shared tree, invites creating it by accident.
The ladder falls back through the Output directory field and the source folder
to `Path.home()`, all existence-checked.

**Scope.** Deliberately the nine cake keys and nothing else — kernel,
corrections and output formats already round-trip through Save/Load GUI State.
Keeping the file to mpe_wf's exact nine columns is the whole point:
`tests/test_provenance.py` asserts a plain `csv.DictReader` (what their tools
use) sees exactly `CAKE_KEYS` in order, because our own reader is lenient
enough that reading our own file back would prove nothing.

## 2026-09-28 — The source HDF5's `instrument/` tree is copied into the zarr wholesale

**Question that started it.** "How come other instrument parameters from the
HDF5 file are not available?" — comparing the Zarr Viewer's tree against the
source file's much richer one in VS Code.

**Why they were missing.** Not a bug, an architectural consequence. MIDAS's
own pipeline gets these PVs into its output via an intermediate step:
`ffGenerateZipRefactor._copy_hdf5_group_to_zarr` builds an "input zarr"
holding the whole HDF5 tree, and `integrator.py:_enrich_zarr_with_metadata`
re-copies the parts it wants into the final file. MIDAS_GUI skips that
intermediate entirely — it reads frames plus six named scalars straight into
memory and calls `write_gsas_zarr_zip`. That writer has no slot for arbitrary
metadata: only `instrument_params` (a flat `str→float` that becomes
`InstrumentParameters/`) and the per-frame `temperatures`/`pressures`/
`currents`/`currents_i0` lists. So everything else stopped at the HDF5.

**Decided: copy all of `instrument/`, not a curated subset** (user's choice
from an explicit either/or). Mirrors what `_enrich_zarr_with_metadata` does.
Measured against the user's real file: `instrument/` is 295 datasets and
**23.5 KiB**, `active_instrument/` another 5 and 1.2 KiB — negligible against
a 332 MB source, and it never needs revisiting when the DAQ adds a PV.
GSAS-II ignores unknown groups (see the entry below), so it cannot affect an
import. `misc/` is deliberately not copied: it is empty in practice, and its
one real dataset (`NDArrayTimeStamp`) is *consumed* here rather than
forwarded. `exchange/`/`NDArray` are the image data, already in the file as
cakes.

**The alignment rule is ours, not MIDAS's, and this is the part to remember.**
`_enrich_zarr_with_metadata` decides "is this a per-frame array?" by testing
`len(arr) == total_frames`. On real 20-ID files **that matches nothing at
all**: the DAQ records one metadata sample per *acquisition*, lights and darks
together in one flat array, so a 10-frame scan carries length-20 metadata.
Copying at raw length would be wrong and averaging on a length match would
never fire. So the copy takes an explicit `n_aligned` from
`workers._HDF5StackGlobSource._metadata_frame_count` — the light-block length
recovered from the timestamp gap, already in use for Temperature/Pressure —
and matches `size >= n_aligned` against that instead. Anyone porting this
back upstream, or comparing outputs against a MIDAS-produced file, needs to
know the two pipelines answer this question differently.

**Decided: the export path reconstructs the same tree from the recorded
source path** (also the user's explicit choice). `tests/test_zarr_layout_parity.py`
asserts both writers produce the same layout; a copy on the Batch Integrate
path alone would have broken that the first time someone exported an
HDF5-backed attempt. It turned out `project.append_integration_attempt`
already records `inputs.src_cfg` — paths included — so no new field was
needed. `gsas_export._source_metadata_snapshot` rebuilds the source through
`workers._open_source_cfg` rather than reimplementing chunk arithmetic, which
is the only way to be sure its notion of "which raw sub-frames are behind
output frame *i*" matches what Batch Integrate already wrote. It returns
`None` — export proceeds without the tree — for a TIFF attempt, a source
that has since moved, or frames spanning several HDF5 files, since a single
snapshot cannot honestly represent two files' PVs and picking one would
mislabel the rest.

**Measured cost, since it scales with frame count and nobody asked about it
up front:** about **+0.15 s and +130 KiB per output frame** on a 300-PV tree.
The bytes are overwhelmingly zarr's per-array bookkeeping (a `.zarray` JSON
plus a chunk file for each of ~300 length-1 arrays), not the 23.5 KiB of
actual readings; the time is the repack, since a zip can't be edited in
place. Negligible on a 10-frame scan, roughly +9 min and +0.5 GB on a
3600-frame one. Deliberately left with **no opt-out**: the decision was to
copy the tree, and a switch nobody asked for is a setting to maintain and a
second behaviour to reason about. If the cost does bite, the cheapest fix
that keeps the chosen layout is a config key (default on) gating the copy —
not a different representation, because a single JSON blob in the root attrs
would be ~1 KiB and instant but would no longer match what MIDAS writes,
which is the whole point of the choice.

**`provenance.rewrite_zip` was factored out of `append_to_zip`.** A zip-backed
zarr can't be edited in place, so every after-the-fact change costs a full
extract/repack. The metadata copy and the provenance stamp are two such
changes to the same file, per output frame. `rewrite_zip(path, mutate)` hands
the extracted directory to a callback and `stamp_extracted` is the
in-directory half of the old function, so the two edits share one pass.
`append_to_zip` is now a two-line wrapper and its behaviour is unchanged.

**Known limits, recorded so they aren't rediscovered:** `active_instrument` is
five empty strings in every real file checked (a documented upstream DAQ gap —
the same one that forces hutch detection to key off `varexE`/`varexD` in the
source path); `misc` is empty; `Encoders` has no datasets. The copy is
therefore mostly `instrument/` in practice, and `active_instrument` is
included on the bet that it starts carrying meaning for free once the DAQ
fills it in.

**Not done, and worth not forgetting:** mpe_wf's `qa/test_zarr_metadata.py`
asserts top-level `misc`/`Detector`/`StorageRing` groups in a MIDAS output.
`_enrich_zarr_with_metadata` does not write those, and `PROVENANCE.md:186`
concedes that test never runs against a real MIDAS-produced file. Do not
treat it as a contract this repo has to satisfy.

## 2026-09-28 — What GSAS-II actually reads from a MIDAS zarr (and why `GSAS2_PVS` is a red herring)

Asked to make our zarr carry full provenance *and* whatever metadata GSAS-II
needs, with mpe_wf's `instrument/GSAS2_PVS/*` container cited as the reference
for the latter. Investigated before changing anything, because this is a file
format an external tool consumes.

**Read the real importer.** Fetched `GSASII/imports/G2pwd_MIDAS.py` from
AdvancedPhotonSource/GSAS-II `main` rather than relying on the reconstruction
in `tests/test_gsas_export.py`. Its `ContentsValidator` is one line —
`midassections = ('InstrumentParameters', 'REtaMap', 'OmegaSumFrame')` — and
`readMidas` touches only:

- `REtaMap` rows 1/2/3 (2theta, eta, bin area; row 3 == 0 is the mask)
- `OmegaSumFrame/<k>` arrays, and their attrs `Number Of Frames Summed`,
  `FirstOme`, `LastOme`, `Temperature`, `Pressure`
- `InstrumentParameters/<key>[0]`, renaming `Polariz`→`Polariz.` and
  `SH_L`→`SH/L`, and reading `Distance` as Gonio. radius (microns → mm)
- optional text sidecars `<stem>.samprm` / `<stem>.instprm`, which OVERRIDE
  the zip

**We already satisfy all of it.** Every group, array and attr in that list is
written by the shared backend writer plus our ion-chamber/temperature stopgap.
No gap on the GSAS-II side.

**`GSAS2_PVS` is named for GSAS-II but GSAS-II never opens it.** It is not a
MIDAS construct at all: mpe_wf's own `PROVENANCE.md` records `/instrument/*`,
`/misc/*`, `/StorageRing/*`, `/Detector/*` (~230 scalar arrays) as an EPICS PV
snapshot written into the **source detector HDF5** by the areaDetector plugin
at acquisition time, copied HDF5→zarr by `ffGenerateZipRefactor.py` and
re-copied by `integrator.py:_enrich_zarr_with_metadata()` (in the local MIDAS
checkout at `FF_HEDM/workflows/integrator.py:595`). Verbatim, not computed.
Replicating that tree would buy richer provenance and *nothing* for GSAS-II.

Two further caveats, both worth recording because they are invisible from the
outside and would otherwise be rediscovered the hard way:

- `readMidas` does read `Temperature`/`Pressure` off the per-frame attrs, then
  does `sampleprmList['Temperature'] = ...` — but `sampleprmList` is a *list of
  tuples*, so that is a `TypeError`, swallowed by a bare `except`. The values
  we write are dropped upstream. `.samprm` is today the only route by which
  sample metadata reaches a GSAS-II histogram.
- GSAS-II's default `InstrName` is `'APS 1-ID'`. An un-annotated 20-ID
  histogram silently claims the wrong beamline.

**What this means for the ask.** The real GSAS-II-facing gap is not a zarr
container — it is the `.samprm`/`.instprm` pair we do not write. Our
`gsas_export.py` docstring claimed the `.provenance.json` sidecar "mirrors the
`.samprm`/`.instprm` convention"; it does not (wrong extension *and* wrong
stem — GSAS-II wants `os.path.splitext(filename)[0] + '.samprm'`, i.e.
`<stem>.zarr.samprm`). Corrected that docstring in place rather than leaving a
claim that reads as "we write these".

Deliberately **not** implemented yet, pending a decision: writing `.samprm`
(InstrName, Temperature, Pressure) and `.instprm`, and/or copying the raw
HDF5's `instrument/` group forward the way `_enrich_zarr_with_metadata` does.
The second is feasible here despite having no intermediate zarr — MIDAS_GUI
reads the same source HDF5 and already chunk-averages per-frame metadata in
`_HDF5StackGlobSource`, which is the same rule that function applies. Both add
files or groups to every export, so they are the user's call, not a quiet
change.

One caution on treating mpe_wf's `qa/test_zarr_metadata.py` as the schema of
record: it asserts top-level `misc`/`Detector`/`StorageRing`, but
`_enrich_zarr_with_metadata` copies only `instrument/` and
`measurement/process/scan_parameters`. mpe_wf's own `PROVENANCE.md` says that
QA file runs against a synthetic fixture and "has no actual round-trip coverage
of [the MIDAS pipeline's] output, only of its own assumptions about what it
produces". Parts of that contract may be aspirational.

## 2026-09-28 — A zarr records the calibration that made it, and both writers agree on layout

Asked where calibration parameters live in a `.zarr.zip`. They did not, in any
readable form. What a file actually carried:

- `InstrumentParameters/` — `Distance` (Lsd) and `Lam`, and that is all of the
  fit. Its other entries (`Polariz`, `SH_L`, `U`/`V`/`W`, `X`/`Y`/`Z`) are
  GSAS-II peak-profile defaults the backend writer emits regardless; on a real
  file they read `U/V/W = 1.163/-0.126/0.063`, the stock GSAS values. They look
  like calibration output and are not.
- `REtaMap` — the full geometry, but *applied*: beam centre, tilts, distortion
  and pixel size exist only as their effect on each bin's Radius/2θ/Eta/Q.
  Recovering `tx` means inverting the map.
- root `provenance_history` — cake params, backend versions, input sha256. No
  geometry.

So `BC_y`/`BC_z`, `tx`/`ty`/`tz`, `pxY`/`pxZ`, `NrPixelsY`/`NrPixelsZ` and the
fifteen distortion harmonics were unrecoverable from the file. You could ask
"what did this calibration do to each bin" but not "what calibration was this".

Fixed by `provenance.instrument_params_from_spec(spec)`, passed to
`build_entry(instrument_params=...)` at every zarr-writing call site. Three
choices worth recording:

- **Read off the live spec, not a paramstest file.** `read_instrument_params`
  already parsed a geometry snapshot from disk, but a file says what someone
  wrote earlier; the spec is what the integration is about to run with. The
  new function is its companion, not its replacement.
- **All fifteen distortion harmonics, always, even at zero.** Fifteen zeros is
  noise, but an absent key cannot distinguish "no distortion was applied" from
  "this writer did not record distortion". Zeros are a positive statement.
  Panel fields and the residual-correction map stay conditional — those really
  are absent from a plain single-panel geometry rather than zero.
- **Coerce on the way out.** Geometry comes off `IntegrationSpec` as 0-dim
  torch tensors, which no JSON attr accepts. `_plain()` handles tensor/array/
  list and degrades anything unrecognised to `str` rather than raising — a
  provenance stamp is best-effort and must never fail the write it describes.

### Layout parity is now a tested contract

Follow-on ask: whichever path writes a zarr, the layout must be identical. The
two paths — `workers.BatchWorker` and `gsas_export.export_gsas_zarr` — call the
same backend writer, so arrays and groups always matched by construction. The
provenance did not:

| | Batch Integrate | GSAS export (before) |
|---|---|---|
| provenance | `provenance_history` in the root attrs | sidecar `.provenance.json` only |
| entry shape | real `build_entry()` | bare `dict(meta)` from the attempt |

A file's history therefore lived in a different *place* and a different *shape*
depending on who wrote it, and the export path's zip was anonymous from the
inside. `export_gsas_zarr` now stamps a matching `build_entry()` into the zip.

**The sidecar stays.** It was tempting to drop it as redundant, but it carries
the attempt's own metadata — stored params, calibration snapshot, frame ids —
which the Batch Integrate path has no equivalent for, and it mirrors the
`.samprm`/`.instprm` convention `G2pwd_MIDAS.py` already expects. It is an
addition to the in-zip entry, not the place geometry lives any more.

`tests/test_zarr_layout_parity.py` drives both real writers over one geometry
and diffs the result: same groups/arrays, same provenance entry key set, same
`instrument_params`, same per-frame attr keys, differing only in the `tool`
field that names the writer. Confirmed to fail 7/10 against the pre-change tree
— a parity test that only ever passed would be worth nothing.

Per-frame `I`/`I0`/`Temperature`/`Pressure` on the Batch path are deliberately
*not* required of the export path: that is data an HDF5 source has and a logged
attempt does not, which is a data difference, not a layout one.

## 2026-09-26 — Batch Integrate: "stride" replaced by "Combine sub-frames" (now applies to HDF5 and TIFF alike)

Consolidated two frame-selection controls that had drifted: `start`/`end`
(file/scan numbers) plus a `stride` spinbox always shown, and a separate
"Combine sub-frames" (chunk size + mean/sum/max/median op) shown only for
HDF5 sources. Requested change: keep start/end, drop stride, and make
Combine sub-frames apply the same way to a TIFF/`.ge*` folder as to HDF5.

**Scoped via a new `DataLoaderPanel(unify_combine=False)` flag, not a global
change to the shared widget.** `DataLoaderPanel` (`mode="stream"`) is also
embedded by Pump Probe (`tab_pumpprobe.py`), which genuinely depends on
today's `stride` to subsample its own pooled per-delay frame list
(`self._frames[start:end:stride]`), unrelated to HDF5/combine. Only
`tab_batch.py` passes `unify_combine=True`; every other consumer (Pump Probe,
bare-constructed test panels) is byte-for-byte unchanged — the `stride:` row
stays, the combine row stays HDF5-only-gated. Confirmed with the user before
implementing (the alternative — dropping stride everywhere — would have
silently regressed Pump Probe).

**TIFF "combine" groups consecutive FILES, since each file already is one
raw frame with nothing smaller inside it to group** — unlike HDF5, where
chunking groups raw sub-frames *within* one file and never crosses a file
boundary (that stays intact for `hdf5`/`hdf5_stack_glob`, unchanged). New
`workers._ChunkCombinedFileSource` mirrors `_HDF5StackGlobSource`: chunk_size
falsy → combine everything into one frame, `1` (default) → no combining
(identical output/ids to before this existed). A multi-file chunk's fid is
`"<first_stem>.frame_<start>_<end>"` — reusing `_HDF5StackGlobSource`'s exact
`.frame_<start>_<end>` suffix convention so `froot_and_frame_num` recovers
the first file's real scan number instead of inventing a new naming scheme.

**Start/end filtering moved from a post-hoc index range into `source_cfg()`
itself.** For a `unify_combine` panel, `source_cfg()` now embeds the raw
file-number bounds as `frame_start`/`frame_end` (alongside `chunk_size`/
`combine_op`, now also on `tiff_glob`/`tiff_list`, not just HDF5 types), and
`frame_range()` always returns `(0, None, 1)` — filtering is applied by a new
`workers._filter_paths_by_frame_number` inside `_open_source_cfg`, *before*
chunking, so a narrowed range always restarts chunk-counting at its own
start (e.g. start mid-list → chunk 0 begins there, not at global file 0).
This sidesteps needing a HDF5-style "file-index → combined-frame-index"
translation for TIFF (where chunks cross file boundaries, unlike HDF5's
per-file-only chunks) — filter first, then chunk what's left, always from
index 0. `_hdf5_multi_file_counts`'s existing translation logic is untouched
and still used for any non-`unify_combine` HDF5 multi-file source.

**Fixed two adjacent bugs while touching this code, both flagged rather than
silently rolled in:** (1) a background "Run as background job" invocation
(`batch_cli.py`) had no `--chunk-size`/`--combine-op` flags at all — Combine
sub-frames was silently ignored for any background run; (2) `tab_batch.py`'s
`_run_as_job` mis-routed a multi-file HDF5 pick (`hdf5_stack_glob`) through
`--source-type tiff_list` (no `hdf5_stack_glob` branch existed) — background
jobs for a multi-file HDF5 scan were already wrong before this change.
`batch_cli.py` gained `hdf5_stack_glob` support and `--frame-start`/
`--frame-end` are now file/scan-number bounds (matching the GUI), replacing
`--frame-stride`.

**MONITOR gains one more disqualifying condition**: it already refused
non-`tiff_glob` sources; now also refuses when Combine sub-frames ≠ 1 or a
start/end filter is active, since live-combining newly-arrived frames isn't
supported (would need buffering logic out of scope for this change).

**Project restore** (`project.integrate_attempt_loader_state`) now reads the
new `frame_start`/`frame_end` keys from a saved attempt's `src_cfg` when
present (a `unify_combine` attempt's own `frame_range` is always
`(0, None, 1)`, so it carries no real information), falling back to parsing
the legacy `frame_range` tuple for projects saved before this change.

New tests: `test_batch_data_source.py` (6, unify_combine on/off parity +
TIFF/HDF5 filter+chunk source_cfg shapes), `test_frame_naming.py` (9,
`_ChunkCombinedFileSource`/`_filter_paths_by_frame_number`), `test_project.py`
(1, new vs. legacy restore path). All existing HDF5 multi-file combine tests
(`test_frame_range_multi_file_hdf5_spans_all_files_with_combine_chunk` etc.)
needed zero changes — confirms the non-`unify_combine` path is untouched.
**Verified:** 13 touched/related test files green per-file on a clean `HOME`
(the one pre-existing failure, `test_project.py`'s
`test_apply_project_calibration_single_detector` pyqtgraph-teardown SIGABRT,
reproduces on clean HEAD too); `pyflakes midas_gui/*.py` unchanged at 37;
offscreen screenshot of the Batch Integrate loader card confirmed the stride
row is gone and Combine sub-frames shows for a plain TIFF folder.

## 2026-09-25 — Data Viewer: folder format filter, under-viewer frame scrubber, profile-file lineout; app-wide frame-nav slider/button visibility

Three Data Viewer requests plus a visibility fix applied everywhere a
frame-navigation slider exists.

**Folder format filter is a `DataLoaderPanel` opt-in (`folder_format_filter`),
not a Data-Viewer-only special case.** `helpers._collect_frame_paths`'s
folder-glob loop was factored into `_folder_format_groups(folder) ->
{label: [paths]}` (TIFF/HDF5/GE/CBF/EDF), reused by both the unfiltered
default path and a new `ext_group` filter param. The panel builds a
"Format:" combo (shown only when more than one group is actually present)
gated behind the new constructor flag so Calibrate/Mask Builder/Batch/Refine
— every other `DataLoaderPanel` consumer — are unaffected; only
`tab_view.py` passes `folder_format_filter=True`.

**Frame scrubber relocation reused an existing pattern instead of
inventing one.** `tab_calibrate.py` already solved "scrubber under the
viewer, loader's own nav hidden" for its `mode="single"` loader
(`_build_frame_scrub_bar` + `hide_frame_field=True`), itself modeled on
`widgets.CakeStackViewer`'s scrub bar. Copied verbatim for `tab_view.py`'s
`mode="stack"` loader — required one small `widgets.py` fix:
`hide_frame_field` only ever gated `mode=="single"`'s frame row, never
`mode=="stack"`'s `_nav_row`, so a stack-mode consumer had no way to hide
its loader's own nav row before this. Extended to cover both modes; no
other stack-mode consumer exists yet, so no behavior change elsewhere.

**Profile-file loading is a separate control in the Radial Profile tab,
not an overload of the Data/Image field** — confirmed with the user before
building. Mirrors `tab_pdf.py`'s existing "I(Q) source" combo. Keeps the
shared `DataLoaderPanel` untouched for this feature (lower risk than
teaching it to recognize non-image extensions) at the cost of one
tab-local `_profile_file_mode` flag that `_on_loader_data`/`_on_fields_changed`
must check before touching the image viewer.

**Axis-unit handling for a loaded 2θ/Q-native file (`.xye`/`.fxye`/`.dat`)
without a calibration attached.** `ProfileViewer` always stored/plotted
`r_px` and converted to 2θ/Q via `_r_to_x` using live lsd/px/wl — feeding it
a native 2θ/Q axis directly (relabeling only) would have silently
mislabeled the plot as "R (px)" whenever `_lsd` is `None`. Rather than
gate the whole file-lineout feature on requiring a calibration up front (the
ask was to plot immediately), `set_profile` gained an optional `native_unit`
param: when set, `_replot` skips the r_px conversion, plots the stored axis
as-is, and locks the R/2θ/Q combo onto the matching entry (still reused for
ring-marker placement, since the combo's `currentIndex()` is what that code
already reads). Once a calibration supplies real lsd/px/wl,
`helpers.native_axis_to_r_px` (algebraic inverse of `_r_to_x`) converts the
loaded axis back to genuine r_px and the profile is re-plotted through the
normal (non-native) path — full unit toggle and ring overlay "for free".
Extension → native unit follows `workers.write_profile`'s own dispatch
(`.csv`→r_px, `.xye`/`.fxye`→2θ, `.dat`→Q) so a file this app wrote round-trips
correctly.

**Real gap found while wiring ring simulation to the no-image case:**
`DetectorGeometryCard._simulate()` — the only place a material's ring radii
(`m["_rings"]`) ever get computed — hard-required an image
(`if img is None: QMessageBox.warning(...); return`), even though the ring
math (`simulate_rings`/`simulate_rings_from_dspacings`) only needs
wavelength/Lsd/pixel-size/d-spacing, never pixel data; the image check
exists only to gate the *separate* on-image overlay (`_redraw_rings`, which
already has its own independent `img is None` guard). Split the computation
loop into `_compute_material_rings()` (reused, verbatim behavior, by
`_simulate`) and added `simulate_rings_without_image()`, which computes rings
and calls `_refresh_profile_markers()` directly — `tab_view.py` calls it
whenever a profile file is loaded or `geometryChanged` fires while in
file-lineout mode, since the normal image-gated call chain
(`_on_sim_param_changed` → `_simulate`) never reaches it with no image
loaded. `_refresh_profile_markers()` itself already needed no image (drew
straight from cached `_rings` + lsd/px/wl) — confirmed by reading before
building on it, not assumed.

**App-wide frame-nav slider/button visibility.** Separately requested:
every "iterate over frames" slider (Data Viewer, Calibrate, Mask Builder,
Hydra's `mode="nav"` loader, `CakeStackViewer`, and `DataLoaderPanel`'s own
`mode="stack"` nav row) plus its ◀/▶ buttons. Several of the ◀/▶ buttons are
plain `QToolButton`s, which have no default background/border at all
(native/flat) — nearly invisible against the dark theme — and the global
`QSlider::groove` is a dark `#2a2a2d` that barely contrasts with the panel
background it sits on. Fixed via one pair of `objectName`s
(`frameNavBtn`/`frameNavSlider`) applied at each of the ~9 construction
sites, with one new ID-scoped QSS block in `style.py` (accent-gradient
button, brighter `#707070` groove) — deliberately scoped to those names
rather than a blanket `QToolButton {}`/`QSlider {}` rule, so unrelated
QToolButtons (help "?", browse "⋯", ROI ribbon) and threshold sliders
(`_thr_slider` in Mask Builder/Calibrate) are untouched.

**Verified:** new `tests/test_dataviewer_format_filter.py` (7),
`test_dataviewer_frame_scrub.py` (5), `test_dataviewer_profile_file.py` (12)
— not fork-isolated (same SIGSEGV-on-fork reason as `test_view_tab_controls.py`).
11 touched/related test files green per-file on a clean `HOME`
(`test_hydra_ui/_calib_ui/_batch_ui`, `test_manual_dspacing_calib_ui`,
`test_mask_folder_frames`, `test_ring_projection`, `test_hydra_geometry`,
`test_helpers`, `test_viewer_*`, `test_batch_data_source` included). `pyflakes
midas_gui/*.py` 38→37 (the one change: `Path` in `tab_view.py` went from
unused to used). Offscreen screenshot confirmed the new button/slider colors.

## 2026-09-24 — Mask Builder: multi-frame Image detection peeks metadata only; threshold projection defaults to "current frame"

Commit `d224c97`.

**Frame-count detection never reads pixel data.** `_detect_multiframe()`
answers "how many frames does this path have" from `tifffile`'s
`series[0].shape` (not `len(tf.pages)` — a small `(N,H,W)` stack can pack
into one TIFF page, so page count under-reports), an HDF5 dataset's
`.shape[0]` via a plain `h5py.File` open (no read), or a `.geN` file's byte
size minus the 8192-byte header divided by candidate square-detector sizes
(2048/4096/1024/512 px). All are O(1) metadata reads so opening a large
Image path to just *check* frame count stays cheap.

**Threshold defaults to the displayed frame, not a full-frame reduction.**
"Current frame" is the default and the only choice for a plain single image
(Projection combo disabled + forced). Requested this way because reducing
across every frame by default silently changes what section-1 thresholds
against without the user asking — the Projection combo makes the behaviour
explicit and opt-in. `Average`/`Sum` accumulate in float64 to avoid overflow
across many frames; `Max` uses `np.maximum(..., out=acc)` in float32 since
there's no summation to overflow.

**Frame source dict (`self._img_frames`) has four `kind`s** (`files`, `array`,
`h5`, `ge`) rather than eagerly loading every frame into memory — each
`_get_frame_array(idx)` call reads (or slices, for the in-memory `array`
case from a small multi-page TIFF) exactly one frame on demand, so a folder
of many large frames or a big HDF5 stack doesn't blow up memory just because
the Frame navigator is open.
## 2026-09-25 — Batch Integrate froze completely: HDF5-over-NFS locking hang, plus backgrounding the preview read

Live report: picking a 17-file HDF5 source (10-frame "Combine sub-frames",
network-mounted) for Data, then checking Dark, made the Dark checkbox itself
stop responding to clicks — not slow, genuinely stuck, confirmed by trying
uncheck/recheck and getting nothing.

**Root cause 1 — the actual freeze.** Checking Dark fires
`FieldSelector.toggled` → `fieldReady` → `DataLoaderPanel.fieldsChanged` →
`BatchTab._refresh_detector_preview()` → `current_frame()`, which for
"stream" mode used to run `_peek_stream_frame()` **synchronously on the GUI
thread** — real file I/O, no QThread. Its own docstring already admitted
this could take "sometimes multi-second"; a 17-file network-mounted HDF5
source measured a lot longer than that. But the user's report ("completely
frozen", not just slow) pointed at something worse than slow I/O: HDF5
`flock()`s every file it opens, and on many NFS servers/clients that lock is
never granted — the call hangs *indefinitely*, not just slowly. Fixed at
the source: `midas_gui/_paths.py` now sets `HDF5_USE_FILE_LOCKING=FALSE`
(via `setdefault`, so a user needing locking left on can still override it)
— the standard, documented workaround, safe here since this is a
read-only/single-writer workflow where the corruption risk locking exists
to prevent doesn't apply.

**Found along the way:** `midas_gui/batch_cli.py` (the headless "Run as
background job" runner) never imported `midas_gui._paths` at all — a
standalone entry point that never goes through `app.py`'s import chain, so
a long-running background job reading the exact same NFS-mounted HDF5 data
got *neither* this fix *nor* the existing `KMP_DUPLICATE_LIB_OK` protection.
Fixed the same way `app.py` does it — one import, first thing.

**Root cause 1 alone doesn't make freezes impossible** — even with locking
disabled, a large multi-frame combine over merely-slow (not broken) network
storage can still block the GUI for a real, user-visible stretch. So also:
backgrounded the preview read itself. `workers.StreamPreviewWorker` does the
file-reading + per-frame correction (dark/bright/background, applied before
summing — matching the real batch run's per-frame correction, same as
before) off the GUI thread; `DataLoaderPanel.current_frame()` now returns
immediately (cached/stale/`None`) and kicks off the worker rather than
blocking, with a new `previewFrameReady` signal firing once the real result
lands (`BatchTab` connects it straight to `_refresh_detector_preview`). Only
one worker runs at a time — a second dirty trigger arriving mid-flight just
flags a restart rather than piling up concurrent reads against the same
storage.

**A second, more dangerous bug found while fixing the first.** The obvious
first cut parented `StreamPreviewWorker` to the panel (`parent=self`).
That's exactly wrong for a QThread: Qt's parent-owns-children cascade
destroys a QThread the instant its parent widget is — including while
`run()` is still executing, which is a fatal "QThread: Destroyed while
thread is still running" abort, not a graceful stop. Every `BatchTab`
construction starts a preview read of the nickel-standard default path
(`self._loader.set_path(DEFAULT_NICKEL_DIR)` in `__init__`) — previously
synchronous and finished before `__init__` even returned, so this had never
been a real hazard before. Backgrounding it turned "constructing a
`BatchTab`" into "starts a real background thread," and the full suite
caught it immediately: ~35 unrelated tests (`test_geom_cache_key`,
`test_batch_output_dir`, `test_batch_cake_stack`, `test_batch_job_results`,
…) that merely build a `BatchTab` for other purposes started crashing with
SIGABRT/SIGSEGV — whichever test's teardown raced past the thread finishing.
Fix: construct the worker unparented. PyQt keeps a *running* QThread's
wrapper alive on its own with no parent and no remaining Python reference,
specifically to prevent this — the explicit `self._preview_worker`
reference plus the `finished`/`failed` slots dropping it are enough for
correct cleanup once it's actually done, regardless of what happens to the
panel/its owning tab in the meantime. Full suite confirmed back to the
3-failure baseline twice in a row after the fix.

Tests: `tests/test_paths_env.py` (new) pins the env-var default, a user
override, and the `batch_cli.py` regression specifically, each via a fresh
subprocess (env vars set at import time can't be re-tested in an
already-running interpreter). `tests/test_batch_stream_preview.py` (new)
pins `StreamPreviewWorker`'s correct-before-summing behavior and error
handling standalone, plus the full async contract through
`DataLoaderPanel`: `current_frame()` never blocks, `previewFrameReady`
fires once real, dark correction is baked in, and a second dirty trigger
mid-flight doesn't spawn a second worker.

## 2026-09-25 — Real ion-chamber + sample-motor metadata in zarr output (stopgap)

While checking the (separately-branched) Zarr Viewer tab against a real
`.ave.zarr.zip`, the user asked why the file had no ion-chamber or
sample-manipulation-system metadata. Traced the actual bug and gathered the
real per-station facts directly from the user plus the beamline's own
HDF5-layout docs (`~/mnt/s1b/bluesky_dev/mpe_xml/docs/hdf5_layout_overview.md`)
and attribute-translation XML (`20ide_instr_attributes_trans.xml`).

**The bug.** `GSASZarrWriter` (the shared `midas_integrate_v2` writer) has two
real per-frame beam-monitor slots, `"I"`/`"I0"` (GSAS-II's ion-chamber-
intensity convention). MIDAS_GUI never read an actual ion chamber — it read
**storage-ring current** (`instrument/StorageRing/SRCurrent`) and wrote that
into `"I"` instead; `"I0"` was never populated at all. What looked like a
beam-monitor reading in the file (`"I": 200.025`) was APS ring current in mA.
Sample-stage motor positions had no code path anywhere — not read, not
written, not attempted. (`GSAS2_PVS/Temperature`/`Pressure` reading NaN,
separately, is *not* a MIDAS_GUI bug: the raw source file has NaN there too,
confirmed against the same layout docs — "Placeholder PVs... until
repointed.")

**Real per-station mapping** (I0 = incident, I = transmitted):

| Hutch | I0 | I |
|---|---|---|
| D | `instrument/Scalers/D/IC2` (confirmed: `IC2D` = `20dT1:TM:Current1`, the first TetrAMM channel) | doesn't exist yet |
| E | `instrument/Scalers/E/US_IC` | setup-dependent: `instrument/Scalers/E/D2PD` (pin diode) when present, else none |
| A | n/a | n/a — no sample in station A's beam path; its `IC4_foil_I0`/`IC5_foil_I1` names are misleading for this purpose, not a per-sample monitor pair |

Sample-stage motors (`instrument/SMS/<hutch>/...`): D has one config (`HR`);
E has two coexisting ones (`HL`, `HR`) with no reliable signal for which is
physically in use — on the one real file checked, `HL`'s channels held real
values and `HR`'s were all NaN, i.e. the data itself already shows which was
active. Captured both rather than guessing, for exactly that reason.

**Why hutch detection is path-based, not file-based.** The file's own
`active_instrument` field (meant to say which hutch/station produced it) is
empty in every real file checked, and is independently documented as a known,
open gap in the same beamline docs repo ("`active_instrument` is currently
empty. Populating it from Bluesky would let downstream tooling select an
analysis pipeline automatically.") — so it can't be read from the file today.
Resolved instead from the source path containing `varexE`/`varexD`
(case-insensitive), one level in `_HDF5StackGlobSource._resolve_hutch()`.

**Explicitly a stopgap** (the user's own framing) — three simplifications,
deliberately not built out further:
- No Preferences UI for any of this; no configurable station/monitor mapping.
- E hutch's `I` is auto-detected (present only when `D2PD` exists in the
  file) rather than made user-configurable, even though which channel is
  "the" transmission monitor is genuinely setup-dependent.
- E's `HL`/`HR` ambiguity is resolved by capturing both, not by picking one.

**Where each field landed.** Real ion-chamber I/I0 go into the writer's
existing `currents`/`currents_i0` slots (`write_gsas_zarr_zip` already
accepted `currents_i0`; nothing in MIDAS_GUI ever passed it before). Storage
current relocates to the provenance entry's `extra['storage_ring_current_mA']`
instead of the `"I"` slot it was squatting in. Sample motors have no writer
slot at all (`GSASZarrWriter` only knows temperature/pressure/current/
current_i0), so they go into `extra['sample_motors']` — `provenance.py`
needed no code change, since `build_entry(..., extra=...)` already accepts an
arbitrary dict verbatim. `zarr_prov_entry` used to be built once and reused
verbatim for every frame in a run; since ring current and sample motors are
per-frame quantities, each frame's `append_to_zip` call now uses a shallow
copy with a per-frame `extra`, not the shared object.

## 2026-09-24 — Zarr Viewer: live-verified, promoted out of "work in progress"

Follow-up to the two entries immediately below. Once visible-by-default (see
the entry right below this one) and enabled in Preferences ▸ Tabs on a real
X11 session, opened a real `.vrx.ave.zarr.zip` from a finished Batch
Integrate run at 20-ID-E (`PUP_AML_stubbins_sep26_bc/...`): tree populated
correctly (`InstrumentParameters`, `OmegaSumFrame`, `REtaMap`, `SumFrames`),
selected `OmegaSumFrame/LastFrameNumber_0` and it plotted (R bin vs. Eta,
viridis, 2-D map), and the metadata/attributes panel showed that array's real
attrs (`FirstOme`, `LastFrameNumber`, `Pressure`/`Temperature` as `NaN`,
etc.) correctly.

Promoted out of "work in progress" in `README.md` (tabs 0–5 now read as
verified, 6–10 as WIP) and dropped the `*(work in progress)*` marker from
`documentation/gui_documentation.md`'s §8. Still an `OPTIONAL_TAB`, not moved
into `ALWAYS_TABS` — verified-and-visible-by-default is exactly the tier
Calib. Refinement/Batch Queue/Pump Probe already occupy, and there's no
reason to promote it further than that.

## 2026-09-24 — Zarr Viewer: promoted to visible-by-default, moved next to Batch Integrate

Follow-up to the entry immediately below: after landing hidden-by-default,
asked to make it "a permanent tab next to the batch integration tab." Two
changes, both narrower than they could have been:

- **Visible, not pinned.** Added to `DEFAULT_VISIBLE_TABS` alongside Calib.
  Refinement/Batch Queue/Pump Probe — shown out of the box, no Preferences
  trip required. Deliberately *not* moved into `ALWAYS_TABS` (the hard-pinned,
  can't-hide-it tier reserved for Data Viewer/Mask Builder/Calibrate/Batch
  Integrate — tabs the app can't function without): it's still an
  `OPTIONAL_TAB`, so it can still be hidden from Preferences ▸ Tabs like any
  of the other three default-visible optional tabs, and it still isn't
  claiming to be "verified" — Pump Probe is proof that default-visible and
  work-in-progress aren't mutually exclusive in this codebase already.
- **Position:** moved in `app.py`'s `_tab_specs` (and its construction line)
  to sit immediately after Batch Integrate, before Batch Queue.

**Real finding surfaced while doing this:** `DEFAULT_VISIBLE_TABS` is
overlaid from the active profile's saved config at import time
(`constants.reload_from_config()`), and on this machine the `20-ID-E` and
`Default` profiles both have an explicit `ui.visible_tabs` saved from before
even **Batch Queue** existed as a default-visible tab — so neither Batch
Queue nor (now) Zarr Viewer will actually appear for this user until they
re-save Preferences ▸ Tabs once, regardless of what ships in code. Left the
live profile JSON files alone (editing a real, in-use per-user config file
outside the repo isn't this branch's call to make); told the user directly
instead.

Knock-on effect on testing: this means `constants.DEFAULT_VISIBLE_TABS` is
*not* a reliable thing to assert against in a test that runs on a real
machine with a real saved profile — it reads whatever that profile last
saved, not what the code ships. `constants.shipped_defaults()` is the
existing, already-provided escape hatch (a pristine pre-overlay snapshot),
so `tests/test_tab_zarrviewer.py`'s registration tests read
`shipped_defaults()["ui"]["visible_tabs"]` instead of the live global. This
also explains — more precisely than the existing "tab-count assertion vs.
WIP-tab gating" note — *why* `test_app_builds_offscreen` is a known
pre-existing failure on this machine: the same stale saved profile makes the
live tab count disagree with what `ALWAYS_TABS`/`DEFAULT_VISIBLE_TABS` claim
at assertion time. Not fixed here (that test's own hermeticity is a broader,
separate concern — `tests/conftest.py` isolates nothing about
`~/.config/midas_gui/`), just diagnosed precisely and worked around locally.

## 2026-09-24 — Zarr Viewer: a standalone top-level tab, matplotlib, hidden pending a live check

Ported `mpe_wf_saxs_waxs/gui_view_zarr.py` (a zarr tree browser + plot canvas
for MIDAS `.zarr.zip` files) in as `tab_zarrviewer.ZarrViewerTab`, at the
user's request, once the terminology/lab-frame PR was out the door.

**Placement: asked, didn't assume.** The screenshot that prompted this showed
Batch Integrate's own results tab bar (Detector view / Waterfall / Stacked
profiles / Eta-R cakes / Logs — `tab_batch.py`'s `_view_tabs`), which reads as
a plausible home for "a new tab in ___". Asked directly rather than guessing;
the answer was a **standalone top-level app tab**, not a Batch Integrate
sub-tab — a general-purpose `.zarr.zip` browser, not tied to any one run.
Toggled from Preferences ▸ Tabs like Corrections/PDF/Texture, added to
`constants.OPTIONAL_TABS` and `app.py`'s `_tab_specs` right after Batch
Integrate/Batch Queue (it's a consumer of Batch Integrate's output) and
before Corrections.

**Matplotlib, not pyqtgraph.** Same call `peak_fit_panel.py` already made and
documented for the same reason: no existing pyqtgraph-based zarr
tree/attribute browser to build on, and matplotlib is already an environment
dependency. This is MIDAS_GUI's second embedded matplotlib canvas.

**Ships hidden.** Not added to `DEFAULT_VISIBLE_TABS` — the underlying
browsing/plotting logic has been in daily use as a standalone tool for a
while, but its integration as a tab *here* hasn't had eyes on a live
rendering yet (this repo's standing constraint: the Qt GUI can't be verified
beyond an offscreen import/build check without an X11/VNC session). Same
treatment as every other WIP tab — flip it on in Preferences once confirmed
live, or ask for `DEFAULT_VISIBLE_TABS` to be updated.

**Kept vs. dropped from the source** (see the new file's own docstring for
the full list): kept every control and all plotting/axis-conversion logic
method-for-method. Dropped the standalone `QMainWindow` shell (window title,
font-size combo, Exit button — the app's own tab chrome and
`constants.DEFAULT_UI_SCALE` already cover this), the `PySide6`/`QT_BACKEND`
fallback (PyQt5 only, like everywhere else in this app), and any
`closeEvent`-driven store cleanup (a tab widget embedded in the main window's
`QTabWidget` never reliably receives its own `closeEvent` — only top-level
windows do — so that would have been dead code; `_load_file` already closes
the previous `zarr.ZipStore` before opening the next one, which is the part
that actually matters). No cross-tab wiring and no saved-project state: it's
opened via its own file dialog, and none of its plot/display state is
meaningful to persist into a Project file.

**Tests build a real fixture rather than a fake store.** `test_tab_zarrviewer.py`
reuses `test_batch_zarr_output.py`'s `BatchWorker` fixture-building pattern to
produce one real `.ave.zarr.zip`, so the tests exercise the actual production
schema (real `REtaMap`, real group layout) instead of an invented one.

**While in there: provenance field parity with mpe_wf.** Before trusting the
viewer against real files, diffed `midas_gui/provenance.py` field-by-field
against its source, mpe_wf_saxs_waxs's own `provenance.py` — see the entry
immediately below for that finding. The zarr array/group schema itself needed
no reconciliation (same shared backend writer, verified empirically).

## 2026-09-24 — Provenance: script/script_sha256/tag brought into parity with mpe_wf

Asked to make sure the zarr writer's "metadata and provenance structure and
content are identical to the development in mpe_wf_saxs_waxs" (prompted by
building the Zarr Viewer above against real output). Two things to check,
kept separate since they have very different answers:

**The zarr array/group schema** (`REtaMap`, `InstrumentParameters/<key>`,
`Omegas`, `provenance_history`) needed no reconciliation at all — both
projects' single-panel `.zarr.zip` files go through the same shared backend
writer, `midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip` (`gsas_export.py`
calls it directly, and Batch Integrate's "zarr" output format goes through
it via `workers.py`), so it's identical by construction. Verified empirically
rather than trusted: built a real fixture via `BatchWorker` and confirmed its
tree matches what mpe_wf's own `combine_hydra_zarr.py` expects from every
panel it merges (that script's docstring spells out the exact schema it
requires — a strong independent check).

**The `provenance_history` entry schema** (`midas_gui/provenance.py`,
originally ported from mpe_wf's own `provenance.py`) did have two real,
unintentional gaps, found by a field-by-field diff of the two `build_entry()`/
`_git_rev()` implementations:
- `script`/`script_sha256` — the running entry-point's resolved path and
  content hash, letting a reader tell a locally-modified/uncommitted script
  apart from the git commit recorded alongside it. Added.
- `tag` on `_git_rev()` — the nearest reachable annotated git tag, separate
  from `describe`'s "N commits past a tag" form. Added.

Everything else that differs between the two files — MIDAS_GUI's
`midas_gui`/`backends` fields replacing mpe_wf's `git`/`mpe_wf`/`midas`
git-repo trio, and no standalone `git` field — is the *already-documented*,
deliberate one-repo/PyPI-backend adaptation from when `provenance.py` was
first ported (MIDAS_GUI doesn't vendor a MIDAS git checkout, so backend
identity is PyPI package versions instead of a second repo's git info; a
separate `git` field would be redundant with `midas_gui` here anyway, since
there's only ever the one repo). Not a gap, so left alone. Also didn't port
mpe_wf's `read_cake_csv()`: MIDAS_GUI already has the equivalent
(`cake_params.parse_cake_csv`) in its own module — an existing deliberate
refactor, not a missing function.

## 2026-09-24 — Upstream's frozen-point native-pipeline switch outruns the pinned backend; guarded, not reverted

Merging `upstream/main` brought in `1893e97 Drop vendored frozen_point_calib now
that midas-calibrate-v2 ships it natively`, which deletes the vendored
`midas_gui/_vendor/frozen_point_calib` and calls
`midas_calibrate_v2.pipelines.iterate_frozen_point_until_stable` directly. That
function does not exist in **0.17.0**, the version `environment.yml` currently
pins (PyPI's latest at merge time is **0.22.0** — the pipeline shipped somewhere
in between; not yet bisected). Unguarded, this breaks collection of
`tests/test_frozen_point_vendor.py` outright (`ImportError` at import time) and
makes `calib.py`'s `frozen_point` branch raise a bare `ImportError` with no
guidance if a user ever picks that pipeline from the GUI.

**Guarded, not reverted.** This is a maintainer's own commit to their own repo,
made against a newer backend than what's pinned here — reverting it inside a PR
back to that repo would be presumptuous, and the right fix (bumping
`midas-calibrate-v2`) is `environment.yml`'s call, not this branch's. So:

- `midas_gui/calib.py`'s `frozen_point` branch now imports
  `iterate_frozen_point_until_stable` in a `try/except ImportError`, raising a
  `RuntimeError` that names the installed version (via
  `importlib.metadata.version`) and says to upgrade the backend or choose a
  different pipeline, instead of surfacing a bare `ImportError` from a
  now-deleted vendor path.
- `tests/test_frozen_point_vendor.py` uses `pytest.importorskip` plus a
  `getattr(..., None)` check on both `autocalibrate_frozen_point` and
  `iterate_frozen_point_until_stable`, skipping the whole module rather than
  aborting collection when either is absent.
- `tests/test_calib_frozen_point.py`'s two tests that monkeypatch
  `midas_calibrate_v2.pipelines.iterate_frozen_point_until_stable` directly
  (`test_frozen_point_subtracts_dark_and_dispatches`,
  `test_frozen_point_logs_note_for_non_cpu_device`) get the same
  `skipif(getattr(...) is None)` treatment — they were failing with
  `AttributeError` from `monkeypatch.setattr`, not from anything this branch's
  own changes touched; confirmed by reproducing the same failure against the
  merge commit before any guard was added.

Net effect on the pinned 0.17.0 environment: Frozen-point (high-tilt) is
selectable in the GUI but errors with a clear message rather than a traceback;
the three tests above skip with a stated reason instead of failing red. Nothing
here silently disables the feature or changes its behavior once the backend
catches up — the guards fall away on their own the day `environment.yml` bumps
past whichever release added the native pipeline.

## 2026-09-24 — The polarization plane is η = 90° (horizontal), and the whole lab-frame chain is pinned by tests

The user, looking at a CeO2 pattern, asked that the polarization correction be
"consistent with the lab frame view where X-Z plane is the ring plane", and that
images be "interpreted as how they are plotted and also plotted consistent with
the MIDAS lab coordinate system".

**MIDAS η is measured from vertical.** `midas_calibrate_v2/forward/geometry.py:209`
computes `eta = atan2(-XYZ_y, XYZ_z)`, and `lattice.py:87` builds the detector
coordinate as `Yc = (-Y_pix + BC_y) * pxY`. So η = 0 is straight up (+Z_MIDAS =
+Y_Lab), and η = ±90 is horizontal (∓Y_MIDAS = ±X_Lab). The storage ring's
X_Lab–Z_Lab plane is horizontal and the beam is polarized in it, so **the
polarization plane is η = 90, not η = 0.**

**The GUI shipped 0.0.** Three sites — the Corrections widget, the Corrections
tab, and `batch_cli --pol-plane` — all defaulted to a vertical polarization
plane. The functional form was right; it was applied a quarter turn away. Since
the factor goes as `cos(2(η − plane))`, plane = 0 does not merely fail to remove
a ring's azimuthal modulation — it ADDS it. `midas_integrate_v2` had already
fixed its own default to 90 on 2026-08-29 and measured it on 1-ID CeO2: plane =
90 takes a ring's cos(2η) modulation from 2.813 % to 0.744 %, while plane = 0
makes it 1.84× worse. All three sites now read
`constants.POL_PLANE_HORIZONTAL_ETA_DEG = 90.0`; ±90 are equivalent (period 180).

**Saved projects keep their own stored value.** A project written before today
restores `pol_plane = 0.0`, and it is deliberately *not* rewritten — silently
changing the physics of a reopened project would make old and new runs of the
same project incomparable with no record of why. Instead `BatchWorker.run()`
logs the plane and fraction at every run start, and appends "← NOT horizontal;
η is measured from vertical, so the storage-ring plane is 90°" whenever the
value is off-plane. The user sees it and decides.

**The rest of the orientation chain was already correct — and is now pinned.**
Audited end to end against the user's diagram (+Y_Lab up = η 0, +X_Lab left =
η −90, +Z_Lab = beam into the page): the backend's η, the viewer's screen
mapping (`disp = d.T` so the array's column axis is pyqtgraph's x, with
`invertY(False)` putting row 0 at the bottom), the lab-frame compass overlay
(`build_lab_frame_axes_items`, `x_screen_sign = -1.0`) and the bin-grid spokes
(`draw_polar_bin_overlay`, `bc + r·(sin η, cos η)`) all agree with each other
and with the backend. Nothing needed changing; `tests/test_lab_frame_conventions.py`
now locks it down so nothing can drift.

**Note on writing those tests — two symmetries make the obvious test toothless.**
Both mistakes worth catching map the spoke set onto itself for natural parameter
choices: swapping sin for cos sends η → 90° − η, so ANY bin size dividing 90
(45°, 30°, the four cardinals) draws an invariant set; flipping the sign of the
Y term sends η → −η, so any η range symmetric about 0 — including the obvious
−180…180 — is invariant too. The first version of the spoke test used 30° bins
over −180…180 and passed happily with the axes exchanged. It now uses 20° bins
over −10…170, and both mutations were confirmed to fail it. Verified by mutation
testing, not by reading.

## 2026-09-24 — A named working directory for calibration, and `.midas_scratch/`

Triggered by a live integration failure:

```
FileNotFoundError: ResidualCorrectionMap
'/net/s20iddata/export/s20a/PUP_AML_stubbins_sep26_bc/residual_corr.bin'
is set but cannot be read
```

Three separate problems behind one traceback, fixed separately.

**1. A returned path is a claim, not a fact.** `calibrate()` mints
`residual_corr_bin_path` from `output_dir` alone
(`midas_calibrate_v2/pipelines/auto.py:680-683`) and returns it at `:976`
without checking the file exists — but it only *builds* the map when
`build_residual_corr` is on, and disables map building outright for any
multi-panel fit (`:702-703`). Untick "Build residual map", or run Hydra, and
the result names a file nothing wrote, which `midas_integrate_v2` treats as
fatal. `calib._confirm_residual_bin` is now a single choke point on
`normalize_result`'s way out: a path naming no file becomes `None`. Downstream,
`helpers._drop_missing_residual_map` degrades an unreadable map to "no map"
rather than raising. The earlier reroute-only fix did not cover this path.

**2. Scratch had nowhere to live.** `residual_corr.bin` and
`panel_shifts.txt` landed either in the user's *data* directory or, with the
Output field blank, in a `tempfile.mkstemp` file they could never find again.
Neither is tidy, and the user asked for neither. There is now one working
directory per calibration, and every intermediate goes in
`<workdir>/.midas_scratch/<run-id>/` — deletable wholesale, because everything
in it is re-derivable and every deliberate save still goes through a file
dialog. The GUI never deletes it: an analysis folder that empties itself
between sessions is its own kind of surprise.

The Calibrate tab's existing `Output:` field was *repurposed* rather than
joined by a second one. It had no deliberate outputs — every real save went
through a dialog — so it was already a working directory in all but name and
default. The state key stays `out_ed` so pre-change projects still restore.

Per-run and per-panel leaves are not decoration: everything the backend writes
there is generically named (`residual_corr.bin`, `calibration.json`,
`panel_shifts.txt`), so two fits sharing a folder overwrote each other and four
parallel Hydra panels raced. The fit-time `panel_shifts.txt` also picked up the
`<stem>_panelshifts.txt` naming the *save* path had used for this reason since
`test_calibrate_panel_save.py` was written; only the fit path had missed it.

**3. `_bc` recognition must precede the positional derivation.** Batch's
`_suggest_output_dir` reads expid/detector/outroot *positionally*, assuming
mpe_wf's four-deep `<outroot>/<expid>/<detector>/<froot>/<files>`. That is
correct for what Batch is pointed at, and wrong here. The user's `.h5` sits
*directly inside* an already-`_bc` directory, three levels below the mount, so
the positional read calls `export` the expid and proposes
`/net/s20iddata/export_bc` — a sibling of the mount root nobody can create.
So `suggest_working_dir` checks for a `_bc` ancestor **first** (which also
happens to be demonstrably writable — the data is sitting in it), and only then
falls back to the positional read and the header Exp ID. It deliberately has no
fallback into the data tree: a working directory inside the raw data is exactly
the littering this feature exists to stop, and an empty field that makes the
user choose is the better answer.

Batch's behaviour is unchanged. The shared parse was extracted to
`helpers.bc_path_parts`, with `suggest_integration_output_dir` (Batch's full
`/<froot>/<detector>` tail) and `suggest_working_dir` (the bare `_bc` root) as
the two callers. `tests/test_batch_output_dir.py` was written as
characterization *before* the refactor — the function had zero coverage — so
the extraction is provably behaviour-preserving, including the case that pins
Batch still producing `/net/s20iddata/export_bc` for the user's real path.

**An unwritable candidate is never pre-filled.** Autofill runs
`check_output_dir_writable` and, on failure, leaves the field empty and logs
the reason once — a path that looks accepted and then fails at Run time is
worse than no path. The Suggest button *does* fill it, with the warning: there
the user asked, so they get the answer and the reason it won't work. Restoring
a project logs a warning for a stored directory that has gone stale (a host
without that mount) rather than silently rewriting it — it is the user's choice
to correct. At Run time an unwritable working directory blocks the fit rather
than warning and carrying on: a fit that runs for minutes and only then finds
it cannot record its residual map has wasted the user's time and left them a
result silently missing the refinement they asked for.

`scratch_dir()` **raises** `OSError` carrying that reason rather than returning
`None`, for the same reason — a silent fallback is how the original bug got
this far. With no working directory set it falls back to one `mkdtemp` per
process, removed at exit, so a blank field is never fatal and never litters.

**Folder designation elsewhere.** `tab_corrections.py` was the one tab the user
named that had no folder field at all (only a save dialog); it got one on the
established idiom. `tab_batch`, `tab_export` and `tab_queue` already had theirs.
`tab_pdf.py` and `tab_texture.py` also lack one and were deliberately left out —
both are work-in-progress per the README.

**Out of scope, deliberately:** `app.py`'s `~/midas_gui_error.log` and
`job_queue.py`'s `~/.midas_gui/jobs/` stay in `$HOME`. Both are app-lifetime
rather than per-calibration state, and the job store must stay at a stable path
to be adoptable across GUI instances.

## 2026-09-23 — The Calibrate Run/Save block keeps this fork's full-width layout

Upstream's `6104310` did two things at once: pinned Output/Run/Save into a
non-scrolling footer, and re-centred all four buttons at fixed pixel widths
(`189`/`132`). The footer is kept; the centring is reverted to full-width Run +
Abort and an `S.button_grid(..., 2)` pair of Saves.

Fixed widths cannot follow the splitter or a different font scale — the same
class of bug `0ecc80a` had just fixed elsewhere — and centring detaches the
block from the Output field it acts on.

This is a deliberate fork-local divergence, kept as its own commit and placed
*last* so the branch sent upstream is simply `main~1`: asking the maintainer to
undo their own design choice does not belong in a PR about filenames. Expect it
to re-conflict on the next upstream merge; the footer widget itself is not in
dispute, only the geometry inside it.

## 2026-09-23 — Saved calibrations are named `<expid>_<image>.instr.*`

Both save paths hardcoded their suggestion — `"calibration.json"` in
`_save_json`, `"paramstest.txt"` in `_SaveParamstestDialog` — as a *bare*
filename, which makes the dialog open on the process CWD, i.e. wherever the app
was launched from. Every calibration a user saved therefore had to be renamed and
moved by hand. `CalibrationTab._default_save_stem()` now suggests
`<expid>_<calibration image stem>`, and `_default_save_path(suffix)` puts it under
the Output dir (falling back to the image's own folder).

`.instr.txt` / `.instr.json` is the user's own convention; the string appears
nowhere else in the codebase. It is only a default — the dialogs stay editable.

**Amended 2026-09-24:** the suffix was `.instru.*` as first shipped; renamed to
`.instr.*` at the user's request. Nothing reads the suffix back — no glob, no
dialog filter keys on it — so the rename is confined to the two default strings
and their tests. Files already saved under the old name still open normally.

Both halves of the stem are optional and independently droppable. The Exp ID
header is free-form and often blank, and the Data path may not be set yet;
joining unconditionally would offer `_.instr.txt`, which is worse than either
half alone. With neither, the suggestion is `calibration`.

Exp ID reaches the tab through the same `set_expid_provider` callback Batch
Integrate already uses (`app.py` `_build_ui`) rather than a copy of the header
text, so it tracks edits. `DataLoaderPanel.data_path()` was added as the public
read side of the existing `set_path()`. `_SaveParamstestDialog`'s new
`default_out` is keyword-only *and optional*: `hydra_calib_widgets.py` builds the
same dialog with no meaningful suggestion to offer, and keeps the blank field.

## 2026-09-23 — Merging upstream 44a0aa1..b25d7e0: which side wins where the two branches both touched the Refine card

Branch `merge/upstream-2026-09-23`, merge base `4a0e8ba`. Upstream (`d-beniwal`)
and this fork both worked the Calibrate tab's Refine card after PR #8 landed, so
the conflicts were semantic, not textual. Recording which side won and why,
because in three of the four cases the losing side is the *newer* code.

**Upstream's `_resolve_seed()` supersedes this fork's `_seed_for_v1()`.** Both
existed to stop the "manual seed or `make_seed_safe()`" block from being copied a
fourth time in `calib.py`. Upstream's is strictly better: it handles a *sparse*
manual seed (BC pair alone, or Lsd alone, or tilts alone) rather than requiring
the full set, and it returns `None` on auto-seed failure so each caller can raise
an error naming its own pipeline. Adopted at all four call sites. `_seed_for_v1`
was deleted rather than kept alongside — it called `_manual_seed_dict`, which
upstream removed outright, so the helper was already a latent `NameError` in
anything that reached it.

**This fork's interleaved limits grid supersedes upstream's `_limits_host` /
`_limits_na_lbl`.** Upstream reflowed the refine checkboxes into a compact 2x3
grid with the limits in a block underneath. Two reasons that block loses:

1. It still carries the label *"the MIDAS calibrate backend takes no bounds
   arguments, so there is nothing to pass them to."* That claim is false and this
   fork disproved it (see 2026-09-09): `CalibrationParams.tolLsd/tolBC/tolTilts/
   tolWavelength/tolDistortion` become hard `(lo, hi)` box constraints in
   `midas_calibrate/param_vector.py:bounds()`. Crystalline fits have always been
   bounded, invisibly, at defaults nobody chose.
2. A compact grid has nowhere to hang a per-parameter window. The whole point of
   the layout here is one row per parameter — "refine this?" in column 0 and the
   +/- window bounding that same parameter on the rest of the line — so the two
   decisions about one parameter are read together.

Upstream's `rfl_bottom` row is kept as-is: Distortion needs its "..." button for
the per-coefficient dialog, and Residual map is an output rather than a fit
parameter, so neither belongs on a parameter row. Upstream's test
`test_refine_card_lays_out_as_three_rows` was rewritten to
`test_refine_card_interleaves_each_flag_with_its_window` against the surviving
layout, keeping upstream's bottom-row assertion verbatim.

**`refine_distortion` accepts a sequence, so the distortion-subset reroute is
gone.** This fork rerouted plain One-shot through `pipelines.single.autocalibrate`
whenever only a *subset* of distortion coefficients was ticked, on the belief that
`calibrate()`'s `refine_distortion` was an all-or-nothing bool. Upstream found
otherwise, and the installed signature confirms it:
`refine_distortion: Union[bool, str, Sequence[str]] = True`. A partial selection
reaches `calibrate()` exactly as ticked. That clause was therefore not just
redundant but harmful — rerouting skips `calibrate()`'s STAGE-1 multi-hypothesis
Lsd search, so it was paying a real accuracy cost for nothing. Removed; the other
three reroute conditions (only one of ty/tz refined, Lsd or BC held fixed,
non-default parameter limits) stand, because `calibrate()` genuinely has no
`refine_lsd`/`refine_bc` kwarg and its `refine_tilts` is a single bool covering
both tilts.

**A disabled checkbox is not a label — fixed structurally and in the
stylesheet.** The user reported two check marks against Distortion and no way to
untick either. Cause: the limits column named each row with the *text of its own
opt-in checkbox*, and crystalline rows hide the opt-in (their window always
applies) — so the row was kept visible-but-disabled purely to caption itself.
`QCheckBox::indicator:checked` paints the accent fill with no `:disabled`
variant, so a disabled ticked box is pixel-identical to a live one. It looked
ticked, it looked clickable, it did nothing.

Fixed on both axes, deliberately:

- *Structural:* the row name is now its own `QLabel` beside the box
  (`_limit_name_lbls`), the box carries no text, and crystalline mode hides the
  box outright. Labels are mode-aware via `_XTAL_ROW_LABEL` — only `distortion`
  needs one, since every other crystalline row is named by its refine checkbox in
  column 0, and reusing the manual fit's `"BC_y"` would misname a window
  (`tolBC`) that covers both centre coordinates.
- *Defensive:* `style.py` gained `:disabled` states for `QCheckBox` /
  `QRadioButton` / `QGroupBox` indicators. The structural fix handles this one
  card; the stylesheet gap would have produced the same illusion anywhere else a
  box is disabled rather than hidden.

## 2026-09-22 — Batch Integrate: clear views before re-deriving axis context, not after

Commit `eebce45`.

**The bug:** `_run()` and `_restore_run()` (`tab_batch.py`) called
`set_axis_context()` on `_waterfall`/`_stack_view` (and `set_axis_context`/
`clear()` on `_cake_stack_view`) while those widgets still held curves from
the *previous* run or attempt. `set_axis_context()`'s internal `_restack()`
re-plots whatever is currently held under the new context — so a leftover
curve from a prior run got re-plotted under the new run's geometry. If that
leftover curve had no finite data (an empty or fully-masked profile),
`StackedProfileViewer`'s `autoRange()` call received a `[nan, nan]` bounding
range from pyqtgraph and crashed.

**The fix:** `reset()`/`clear()` the three views *before* setting axis
context, in both `_run()` and `_restore_run()`, so `_restack()` only ever
operates on data that belongs to the thing being drawn now. Order matters
here — clear-then-context, not context-then-clear — because the crash lives
inside `set_axis_context()`'s own `_restack()` call, not in anything after it.

**Also:** `StackedProfileViewer.autoRange()` was unconditionally called
whenever `self._curves` was non-empty, regardless of whether any curve
actually had finite data. Moved the `autoRange()` call inside the `if
xmins:` branch so it only fires when there's real data to range over — a
second, independent guard against the same nan-range crash, for curves that
survive `reset()` but are individually all-nan/all-masked.

**Why this matters beyond this one fix:** any future view that layers
"set new context" on top of "may still hold stale content" needs the clear
to happen first. This is the same shape of bug as the ring-overlay staleness
fixed 2026-09-11, and the `CakeStackViewer`/`RingResidualViewer` family
mentioned there — check them if a similar nan/stale-plot crash turns up.

## 2026-09-11 (later) — One honest ring overlay; Batch's cakes made visible; the whole calibration in the provenance record

Commits `54cdd48` (Calibrate) and `f44314d` (Batch Integrate).

### The predicted-ring overlay is always the full forward model — no toggle

**The decision:** delete the "Corrected" checkbox (single-detector tab *and*
Hydra calib page, plus its saved project state) and draw every ring through
tilt **and** refined distortion, always.

**Why, and why it is not merely a default change.** The tick was off by
default, so the overlay a user actually read was a plain circle about the beam
centre. Worse, ticked it applied *tilt only* — so on a detector whose
calibration refined distortion harmonics, both states were wrong, one just
less so. A ring overlay that sits off the measured rings is the most expensive
kind of wrong: it still looks like rings, so it reads as a bad calibration
rather than a bad drawing, and the user goes off refitting a geometry that was
fine. Correctness of a *prediction* is not a display preference the way a
colormap is. The reduction property is what makes removing the choice safe:
`ring_xy_corrected` is bit-identical to `tilted_ring_xy` when there is no
distortion, and that in turn is a circle at zero tilt, so the untilted
undistorted case draws exactly what it always drew.

**Why invert the backend's relation instead of writing a ring equation.** The
backend reports a pixel's radius as `R_corrected = D(ρ, η) · R_projected`
(`midas_calibrate_v2.forward.geometry.pixel_to_REta`). A ring is therefore the
locus of pixels whose *corrected* radius equals the Bragg radius — not the
locus where the undistorted ray lands. So per η we solve
`D(rad/ρ_d, η) · rad = Lsd·tan(2θ)` for `rad` by fixed-point iteration (D is
within a few percent of 1 for any physical calibration, so the map is a strong
contraction; ~5 passes reach float64 noise, 20 is the cap) and hand the
recovered radius back to the tilt projector as a per-point effective 2θ. The
factor comes from `midas_distortion`, the shared leaf both `midas_calibrate_v2`
and `midas_integrate_v2` evaluate, so there is one definition of the model
rather than a second copy in the GUI that can drift from it.

**ρ_d has to match `spec_from_calibration_result`'s definition exactly** —
beam-centre-to-farthest-corner in px measured to `N-1`, times the mean pixel
pitch. The harmonic basis is evaluated at `ρ = R_µm / ρ_d`, so a ρ_d off by even
one pixel pitch rescales every term and the polynomial stops describing the
detector it was fitted on. `helpers.distortion_rho_d_um` reproduces it and
returns `None` (→ "cannot evaluate distortion here") rather than substituting a
guess when the detector size is unknown.

**What is deliberately *not* drawn:** the empirical `residual_corr_map`, a
smooth per-pixel ΔR the backend adds after the harmonics converge. It is
sub-pixel and would need the map tensor available in a redraw path. Rather than
let it go silently missing, the status line beside the toolbar says
"residual map not drawn" when a result carries one — the same line that now
reports "(tilt + distortion applied)", so "these rings are bent" reads as the
geometry rather than as a bug.

Verified against the backend rather than against a hand-derived expectation:
`tests/test_ring_projection.py` feeds every point the projector draws back
through `pixel_to_REta` and requires it to come back at the ring's radius.

### "Use seed as calibration (no fit)" removed (added 2026-09-09, superseded)

It existed because every export was gated on a fit result, leaving a
hand-matched geometry with nowhere to go. Two things since have made it a
duplicate rather than a feature: the Data Viewer's Ring simulation card is
purpose-built for dialling a geometry in by eye, and `Geometry: [← Get]`
(2026-09-10) seeds it from this tab's calibrated values. Keeping a second,
weaker no-fit publishing path inside Calibrate meant two places to hand-build a
geometry and one of them silently produced a "result" with no uncertainties
sitting in the same grid as fitted ones. The docs now point at the Data Viewer.

### Calibrate's panel column compacted

The Refine card is three rows (Lsd/BC/Wavelength · the tilts · Distortion +
Residual map) with **Limits** as a block below it, instead of one control per
line woven through a ± column. The ± column was the reason for the one-per-line
shape, and it applies to the manual d-spacing fit only — the crystalline
backend takes no bounds arguments — so for most calibrants the tallest card in
a six-card column was tall for a feature that was hidden. Seed and Advanced pack
three controls per row. `S.Form.row` now stretches every field column rather
than columns 1 and 3, which is what a third pair needs to not sit pinned at its
minimum width.

### `IntegrationWorker` stays in float64

It narrowed to float32 and widened back at the torch call. `BatchWorker` does
not, so the Calibrate tab's profile differed from the Batch run it is meant to
preview — 4e-8 relative, harmless in magnitude, but a difference with no reason
to exist in the one plot a user reads a peak position off.
`tests/test_calibrate_integration_accuracy.py` pins the Calibrate profile equal
to the Data Viewer's accurate path, so the cheaper kernel cannot be substituted
back in for speed without a failing test.

### Batch Integrate: the cakes it already computed are now visible

A "Multi-azimuth output" run keeps each frame's `(η, R)` cake instead of the
η-collapsed profile. The tab computed those, wrote them to disk and embedded
them in the project — and displayed none of them. Reopening such a project
*raised*, because the restore path fed 2-D cake rows to the 1-D waterfall
buffer, and that took the whole restore down with it.

- `CakeStackViewer` (a `CakeViewer` over an `(n_frames, n_eta, n_r)` stack with
  a scrubber) backs a new "Eta-R cakes" view tab. Frame-stepping suppresses the
  view reframe: the axes are identical across the stack by construction, so
  re-fitting the range on every step only discards the zoom the user is
  scrubbing *with*. Levels still auto-scale per frame, as every other viewer does.
- The two 1-D views get an η-collapse of the stack (mean over filled bins).
  Exact-zero bins are unfilled coverage, not measured zeros — the same
  convention `CakeViewer`'s auto-levelling uses — so they are excluded rather
  than dragging the mean down. It is an approximation of the engine's
  count-weighted collapse and is documented as one; the run's own collapsed
  profile does not exist to restore, because multi-azimuth mode keeps the cake
  instead of computing it.
- A non-multi-azimuth run **clears** the tab. Leaving the previous run's cakes
  sitting there is worse than an empty tab, because they look current.

### Cake x-axis in R / 2θ / d / Q — labels only, and opt-in

Only the tick *strings* convert; the image keeps its native R-pixel coordinates
(setRect, view limits, the readout's bin lookup). So the axis is exact for a
nonlinear unit and switching costs nothing — no resampling, and what you see
stays the integrated bins. d diverges at R = 0 and renders as "∞", because
"inf" on an axis reads as a bug. The selector stays **hidden until a caller
supplies geometry** via `set_axis_context`, which is what leaves the Calibrate
and Hydra cakes — and `RingResidualViewer`, whose X is a ring index and not a
radius at all — behaving exactly as before. `_convert_radial` moved up beside
the cake viewers (it was defined near the bottom, next to the 1-D waterfall)
and gained the d case.

### An integration attempt records the *whole* calibration

`calibration_snapshot` was the 13 display fields from
`resolve_calibration_fields`. An attempt has to be able to reconstruct the
geometry it ran under, and those cannot: they drop `_calibrant_name`,
`_panel_unpacked`, `panel_layout`, and the refined-parameter σ / at-limit flags.
`helpers.full_calibration_snapshot` overlays the display fields *on top of* the
sanitized full result — not merged under it — so no value can drift (both are
read off the same object) while the display fields still supply defaults a bare
result may not carry (`tx/ty/tz` → 0.0, `distortion` → {}, `im_trans` → []).
Keeping the output a strict superset is what lets `project.calibration_namespace`,
`render_calib_value_grid` and `gsas_export` consume it unchanged; the test
asserts that superset property by name rather than checking a sample of keys.

## 2026-09-11 — MIDAS backend bump to current PyPI latest; the tilt/im_trans issue came back fixed

**What moved:** `midas-calibrate-v2` 0.13.0→**0.17.0**, `midas-hkls`
0.10.0→0.11.0, `midas-stress` 0.13.0→0.14.0. Everything else in the MIDAS set
was already at PyPI latest. pip confirmed those three are the only packages
that move — numpy 1.26.4 / torch 2.4.0 / numba 0.59.1 / zarr<3 all unchanged.

**Why it matters more than a version bump:** the four calibrate-v2 releases
(0.14–0.17, all 2026-09-10) are upstream implementing
`.context/issue_draft_calibrate_v2_tilt_imtrans.md` — the issue this repo filed
on 2026-09-08 — nearly in full: `initial_tx/ty/tz` on `calibrate()` (A),
`CalibrationSpec.im_trans` applied by every pipeline through one shared
`io/transforms.apply_im_trans` (B1), `im_trans` recorded on the result and
carried into `IntegrationSpec.TransOpt` (C), and `first_time_calibrate` gaining
both (D). `refine_tx` was declined on gauge-degeneracy grounds. ROADMAP P3-1
and P3-4 are closed; P3-2 and P3-3 were re-checked and still stand.

**The decision that needed making — leave `calib._prep_transformed()` alone.**
Upstream flags a real behaviour change in 0.15.0: a caller that pre-transforms
its image *and* lets the pipeline see an `im_trans` now applies the transform
twice, and a double flip is silent. It would have been easy to read that as
"the GUI must stop pre-transforming" and rip the workaround out as part of the
bump. That would have been wrong on both counts:

* The GUI is *not* exposed. Every pipeline gates on `if spec.im_trans:`; the
  spec comes from `spec_from_v1_params`, which reads `ImTransOpt` off
  `v1.extra`; and `calib.build_v1_params` never sets it. So `spec.im_trans` is
  `()`, the guard is false, and the pre-flip lands exactly once. Verified
  directly, not inferred — and `apply_im_trans` was checked bit-identical to
  `helpers._apply_im_trans` on all 8 opcode combinations, so the two
  implementations can coexist safely.
* Removing the workaround is optional cleanup, not a fix, and it is the one
  change that must not be made half-way. Doing it means deleting
  `_prep_transformed` **and** teaching `build_v1_params` to set
  `v1.extra["ImTransOpt"]` in the same commit; either half alone is a silent
  double flip or a silent no flip. Deliberately left for its own change, with
  its own test, rather than smuggled into a dependency bump.

**What does change behaviourally:** `calib.run_pipeline`'s one_shot branch has
always passed `initial_tx/ty/tz` speculatively through `_supported_kwargs`,
which silently dropped them on ≤0.13.0 with a printed note. They now take
effect — which is what that code always intended, and why the issue was filed.
Also inherited: RhoD unit fixes (it is µm; three pipelines fell back to a
pixel-valued `MaxRingRad`), `make_seed(use_diplib=)` now defaulting to False
(the macOS segfault `midas_gui` already passed False for), and distortion phase
bounds widened ±90→±180 so a phase near the seam stops railing.

**Also fixed here: `requirements.txt` had silently fallen two bumps behind.**
The 2026-09-08 bump reached `pyproject.toml` and `environment.yml` but not
`requirements.txt`, so `pip install -r requirements.txt` and `pip install .`
installed *different* backend versions (calibrate-v2 0.11.0 vs 0.13.0, and the
whole midas-saxs→transforms→stress chain missing). Now regenerated in sync, and
a cross-check of all three files against the installed env is part of the
verification below. Worth a standing habit: all three files move together.

**Follow-up taken in the same session — `first_time` now gets the transform,
and a wider bug fell out of it.** `run_pipeline`'s `first_time` branch had never
passed a transform (no entry point accepted one before 0.15.0), so a first_time
calibration on a flipped detector ran in the wrong frame and returned a
confident wrong geometry with nothing in the output to say so. It now forwards
the codes and hands over the RAW frame, because the backend flips
image/dark/panel_mask itself and re-derives `n_pixels_y/z` — so this branch must
*not* also pre-flip. Passed unguarded rather than through `_supported_kwargs`
deliberately: on too-old a backend a loud `TypeError` is the right outcome,
since silently dropping the kwarg is precisely the bug being fixed.

Wiring that up exposed a second bug the first one was hiding.
`workers.CalibrationWorker` took `NZ, NY = image.shape` off the **raw** image and
handed it to `normalize_result`, which uses it for every mode except plain
one_shot (that one returns the package's own result untouched). A transpose
swaps Y and Z, so on a **non-square** detector first_time, four_stage, bayesian,
joint and the partial-distortion re-route all recorded `NrPixelsY`/`NrPixelsZ`
for a detector they never fitted — and every downstream consumer (integration
spec, paramstest export, ring overlays) inherited it. Fixed centrally with
`calib.effective_pixel_counts()` rather than only in the first_time branch:
fixing one caller of a shared helper and leaving the other four wrong would have
been arbitrary, and the correct value is the same computation in all five.

Covered by `tests/test_first_time_im_trans.py` (19 tests). The tests were
mutation-checked, not just observed green: reverting the forwarding fails
`test_im_trans_is_forwarded` + `test_multiple_codes_are_forwarded_in_order`, and
adding a pre-flip on top fails `test_image_is_handed_over_raw_not_pre_flipped`.
`effective_pixel_counts` is asserted against the backend's own `apply_im_trans`
across 11 opcode combinations rather than against a re-derivation.

**Verified:** per-file test sweep (43 files) byte-identical before and after the
upgrade — same single pre-existing failure (`test_smoke::test_app_builds_offscreen`,
the known stale-local-config artifact; 10/10 under `HOME=$(mktemp -d)`), no new
ones, plus the new file. All 46 `midas_gui` modules import. The issue draft's own
runnable repro re-run against 0.17.0, now asserting the gaps are closed rather
than open. (One sweep run showed `test_live_stream.py` exiting 139 — the
documented non-deterministic pyqtgraph teardown crash, not this change: that file
imports only `widgets`/`constants`, and it passed 5/5 on re-run.)

## 2026-09-10 — Data Viewer: accurate integration on demand, and rings that stay where you put them

Six Data Viewer changes landed together. Four are UI placement; two encode a
judgment worth recording.

**A second integration path rather than a faster accurate one.** The radial
profile has to serve two jobs that pull in opposite directions: keeping up with
a live PV stream, and being trustworthy enough to read a peak position off. The
existing profile is the first — circle binning with no calibration, MIDAS engine
with the `hard` kernel once a tilt/calibration exists. Rather than trying to
make one path do both, an **Accurate** tick (off by default) switches it to the
Batch-Integrate pipeline verbatim: geometry synthesized from the live widgets
even at zero tilt, `subpixel2` kernel — the same one `constants.DEFAULT_KERNEL`
gives Batch. So the numbers a user compares against a Batch run come from the
same code, and the live view costs nothing when they don't need that. Rejected:
making accurate the default and throttling it, which trades a correctness
property the user can see for a latency property they can't reason about.

**The cake is always accurate, and no longer a by-product.** It is an explicit
Calculate, not a per-frame refresh, so there is no live budget to protect and no
reason to offer the fast path at all. Giving it its own R/η bin spinboxes then
forces a second decision: `radial_integrate` used to fill the cake for free,
since both ran through one `_midas_radial` call. With independent bin sizes that
by-product would silently overwrite a cake the user had just binned differently,
so it is now suppressed whenever the cake controls are bound (`set_cake_controls`).
Hydra binds none and keeps the old free cake. The single `_calib_ctx`/`_calib_ctx_sig`
slot became a 6-entry dict cache keyed on kernel and both bin sizes — with three
callers wanting three different contexts, one slot would thrash on every switch.

**Simulated rings are frozen at the parameters they were simulated with.**
Reported as *"the rings on the image are still getting influenced by the
parameters when Simulate rings is not live"*. The cause was that `_redraw_rings`
always read the live BC/tilt/px/Lsd widgets while the ring *radii* only changed
on an actual `_simulate()` — so a BC nudge moved rings whose radii belonged to
the old geometry, which is a half-updated overlay, not a stale one. A checkable
"live mode" button made this worse by hiding the distinction: the button being
off did not mean the overlay was static. `_simulate()` now snapshots the
placement geometry into `_ring_draw_geom` and `_redraw_rings` draws from it,
except in live mode where it reads the widgets (a BC edit only *moves* rings, so
it never reaches `_simulate` and would otherwise pin them to the snapshot). The
control became a plain button + a `live` tick + a `✕`, so "one-shot" and "armed"
are two visibly different states — green only when armed. The beam-centre `+`
marker deliberately still tracks the live BC: it reports where the beam centre
*is*, not where the rings were drawn from.

`✕` clears the simulated overlay and disarms live (otherwise the next edit
redraws it immediately), but leaves the click-picked magenta radius ring — that
one comes from clicking the profile, not from a simulation.

**d-spacing picking follows the enabled materials.** "Pick d-spacing pts" +
"Ring #" only mean something for a non-crystalline calibrant, so
`PickableImageViewer.set_dspacing_picking_visible()` hides them and the geometry
card drives it off `kind == "dspacing"` among *enabled* materials — the general
rule AgBH is the shipped instance of, rather than a name check. Visible by
default, so the Calibrate tab (which has its own calibrant combo) is untouched.

**Transforms moved between Projection and Ring simulation**, in the card itself
rather than by re-parenting: the Data Viewer and Hydra both put this card in a
Projection-first column, so one ordering serves both. Transforms describes the
image; it never belonged inside the ring model.

**Files:** `hydra_geometry_card.py`, `tab_view.py`, `widgets.py`,
`tab_calibrate.py` (new public `geometry_for_viewer()`, shared by its own
"→ Send to Data Viewer" and the Viewer's new "← Get" pull), `app.py`; new
`tests/test_view_tab_controls.py` (27 tests).
## 2026-09-09 (later) — Crystalline calibrants DO have parameter bounds; the earlier claim was wrong

The entry below shipped a label saying "Parameter limits are not available for
this calibrant: the MIDAS calibrate backend takes no bounds arguments, so there
is nothing to pass them to." **That is false**, and it was written from an
incomplete check (`calibrate()`'s own signature) rather than from the params
object the solve actually uses.

`midas_calibrate.params.CalibrationParams` carries `tolLsd` / `tolBC` /
`tolTilts` / `tolDistortion` / `tolWavelength`, and
`midas_calibrate/param_vector.py:bounds()` turns each into a hard
`(value − tol, value + tol)` box constraint on the LM solve. So crystalline fits
were **already bounded all along** — at defaults nobody chose and nobody could
see: ±15 mm on Lsd, ±20 px on BC, ±3° on tilt, ±0.001 Å on λ. On the reported
13.5 m SAXS geometry that Lsd window is ±0.1 %.

Lesson worth keeping: "the backend does not support X" needs checking against
the object the solver consumes, not only the entry point's signature. The GUI
builds `CalibrationParams` itself in `build_v1_params`, so anything that
dataclass carries was always reachable.

**Presentation: always-on rows showing the effective values, not opt-in
checkboxes.** For the manual fit an unticked row means *unbounded*, which is
true there. For the crystalline backend "off" would mean "backend default", and
a user reading it as unbounded is exactly the misconception that produced the
wrong label in the first place. So crystalline rows are always active, have no
enable checkbox, and are prefilled from `tol_defaults()` — read off the
installed dataclass rather than hardcoded, so a backend release that retunes
them cannot leave the GUI displaying stale windows.

The backend's granularity is coarser than the manual fit's — one window for both
BC coordinates, one for both refined tilts, one for all fifteen distortion slots,
and `refine_mask` never refines tx — so the surplus rows are hidden rather than
left as controls that would silently do nothing. The tilt row spans ty/tz in the
grid so it does not read as bounding only ty.

**Rerouting One-shot rather than leaving its flags inert.** Plain One-shot calls
`calibrate()`, which builds its own `CalibrationParams` (no tol* kwarg) and
hardcodes `Refine={"Lsd": True, "BC": True, ...}` (`auto.py:619`). Three things
it therefore cannot express: a non-default window, a held Lsd/BC, and refining
exactly one of ty/tz (its single `refine_tilts` bool is computed as `ty or tz`).
Each now routes through `build_v1_params` + `pipelines.single.autocalibrate` —
the escape hatch this file already used for a distortion subset, for exactly the
same reason. Rejected the alternative of warning that the checkboxes do nothing:
the mechanism to make them work already existed and was one branch away.

The trade-off is real and is logged rather than hidden: the v1 route skips
`calibrate()`'s STAGE-1 multi-hypothesis Lsd search and uses the seed as given.
`first_time` still cannot be bounded at all (it takes neither a
`CalibrationParams` nor a tol* kwarg — it has its own `tilt_prior_deg` /
`half_window_px` on a different footing) and warns when limits are set.

**Seed arrow steps follow the window** at 10 % of the full range (±15 mm → 3 mm,
±2 mm → 0.4 mm). A window is a statement about how far a value can sensibly
move, which is a better step than a constant; rows with no window fall back to
the `DEFAULT_STEP_*` preferences.

**A visible window is not enough: report when the fit lands on one.** Showing
the bounds fixes half the problem; a fit that *stops* at its bound is reporting
the bound rather than a measurement, and looked identical to a converged one.
The manual fit already said this via `at_limit`; the crystalline path now does
too (`_crystalline_at_limit`), marked `(at limit)` in the results grid and named
in the Log. Computed GUI-side by comparing the fitted value against seed ± the
effective window, because the backend returns no active-set information.

Two deliberate silences there. Parameters held fixed are never flagged — they
never moved, so a window cannot have stopped them, and flagging them would put
noise on exactly the rows the user already knows are pinned. And an auto-seeded
run reports nothing at all: the window is centred on a seed the GUI never sees,
so there is no honest comparison to make. Both are cases where saying nothing
beats guessing.

The windows in force are also logged at the start of each run, next to the
refine summary — the card shows whatever is set *now*, which is not necessarily
what a given run used.

Not done: emitting `tolLsd`/`tolBC`/`tolTilts` into paramstest.txt, which
`midas_calibrate/params.py:214` does parse. `paramstest_pairs` is deliberately
generated by the backend's own writer so the readout matches the file byte for
byte; adding rows would break that invariant for a provenance gain the project
record already covers (the windows travel in the attempt's `cfg`).

Also fixed here: `_limit_bounds()` (which feeds the manual fit) was not
mode-filtered, so once crystalline rows became always-on it would have handed
their windows to a solver that never runs. `dialogs.ParameterLimitsDialog` was
dead code — superseded by the inline column, only its row table and
`limit_window()` were ever imported — and is deleted.

## 2026-09-09 — Manual d-spacing fit: BC-only default, parameter limits, and σ reporting

The manual ring-pick fit added in `bdd9b51` was wired to the tab's Refine
checkboxes, whose defaults (Lsd + BC + ty + tz) are correct for a crystalline
calibrant and badly wrong for AgBH at a SAXS geometry. Reported as *"calibration
runs away and the AgBH rings are significantly off"* on a real 13.5 m dataset
(λ = 0.1730 Å, 55 µm, 3072×512). Reproduced numerically: with 8 picks on the one
visible 42° arc of ring 1, refining Lsd+BC+tilt returns `success=True` with
Lsd = 12886 ± 3690 mm, BC_z = 157 ± 387 px, ty = −2.4 ± 406°, drawing the ring
4.8 px off; BC-only returns BC to ±1 px and draws it 0.95 px off.

This is identifiability, not a solver defect. At small 2θ, Lsd, beam centre and
tilt are nearly degenerate, so the residual surface is a long flat valley and
pick noise picks the point along it. Three decisions follow:

**Per-calibrant-kind refine defaults, not one global default.** The checkboxes
are shared UI, but the right defaults are not: a crystalline pattern fills the
detector and constrains Lsd and tilt well. So `_refine_state_xtal` /
`_refine_state_dsp` are remembered separately and swapped *on the calibrant-kind
transition only* (`_sync_refine_mode`) — a user who ticks Lsd while on AgBH keeps
it while they stay there. Rejected: globally defaulting tilt off, which would
have degraded CeO2 to fix AgBH.

**Report σ, don't block or auto-select.** `fit_geometry_from_ring_picks` now
returns a per-parameter 1σ from the linearised covariance (`inv(JᵀJ)·s²`,
`_fit_parameter_sigma`), `inf` when `JᵀJ` is singular — the honest answer for a
degenerate fit. `ManualDspacingCalibWorker._identifiability_lines()` turns that
into `value ± σ` log lines plus an explicit "not constrained by these picks —
largely fitted noise" warning past a *this-is-not-a-measurement* scale (1 % of
Lsd/λ, 5 px of BC, 0.5° of tilt). These are coarse usefulness thresholds, not
statistical tests. It warns rather than refusing, because loose data may be
deliberate, and a good default plus a visible σ beats a hard rule the user has
to fight. σ is the reason it is now safe for a user to re-enable Lsd or tilt:
the runaway is still possible, but no longer silent.

**Bounds switch solvers; unbounded stays bit-identical.** scipy's `lm` rejects
bounds outright, so `bounds` (a dict in fit units) selects `trf` and clamps `p0`
into the box first (`trf` raises on an outside `x0`), recording `clamped`. With
every bound infinite the call is unchanged — `method="lm"`, same numbers as
before — so existing callers and saved results are unaffected. A parameter
resting on an active bound is reported `at_limit` instead of with a σ, which the
covariance formula would misstate there.

The **Limits…** dialog defaults each row's unit by whether the quantity has a
non-zero scale: `%` for Lsd and λ, absolute for BC and the tilts. Tilts seed at
0°, where ±5 % pins the parameter exactly — the opposite of asking for a window
— so a degenerate percentage window widens to the row's absolute default rather
than pinning. All rows start off, so an untouched dialog changes nothing.

Also added **Use seed as calibration (no fit)**: nudging BC/Lsd until the
predicted overlay sits on the measured rings is a legitimate calibration that
had no way to reach Save/Send, which were all gated on a completed fit. Every
row is marked `(fixed)` since nothing was refined. Documented alongside it that
a good-looking overlay is weak evidence at long Lsd (313 mm of Lsd error moves
the first AgBH ring ~17 px at 13.5 m).

**d-spacing pick markers are haloed, not merely recoloured.** Picks were filled
dots in a per-ring colour; ring 1's `#e05656` is essentially the hot colormap's
own red, so the marker disappeared into the arc the user had just clicked.
Recolouring only moves the collision to another colormap, so each pick is now a
black halo with the ring colour over its middle and an open centre (the picked
pixel stays visible). `_dsp_pt_items` consequently holds a tuple of items per
pick rather than a flat list, so undo/clear remove a whole marker.

### Same-session fixes, unrelated to the above

**Lab-frame compass sized in screen pixels.** A wide/short SAXS strip auto-fits
at a much lower effective zoom than a square WAXS panel, so a compass sized as a
fraction of image dimensions shrank to nothing while the zoom-independent
`TextItem` labels stayed full size — guaranteed overlap. `L` is now driven by
`px_iso` (data units per screen pixel) with a data-unit clamp, label boxes are
placed from font metrics converted into data units, each box is anchored on the
edge facing the beam centre so it grows away from the compass, and the η=0° /
η=−90° labels are folded into the +Y_Lab / +X_Lab boxes they sat on top of.

**Ring labels anchor to the visible arc** (`_ring_label_pos`). Twelve o'clock is
wrong whenever the beam centre is near an edge: on a SAXS strip every ring's top
lies far below the frame and all labels pile up off-screen. Now the highest
on-image point of the arc, falling back to the plotted point nearest the image.

**Mismatched correction fields are skipped and flagged, not fatal.** A dark/
bright/background left over from a session saved against a different detector
used to raise inside `apply_field_corrections`. It now skips the mismatched
field, and `FieldSelector.note_frame_shape` marks it inline (`⚠ SKIPPED: data is
…`), mirroring `MaskSelector` — a silent skip would be worse than the crash.

**Batch Integrate persists the Tab-2 calibration result** (`sanitize_result_dict`
made public for it). `set_calibration()` was split so `_apply_calib_result()`
restores state without also forcing the "From Tab 2" radio, which would override
a session where the user had deliberately chosen "From file".

**Save Project As appends `.h5`** when the typed name has no suffix; the dialog's
own filter does not force one on every platform.

### Test-infrastructure findings

`pytest --forked` combined with `--basetemp` (added in `1e5cace`) races: each
forked child re-creates the basetemp directory, so a different test errors in
fixture setup on each run. Unforked runs are clean. Unforked runs of the Qt
suite, however, still segfault once several `CalibrationTab`s exist — so neither
mode runs everything, and the suite is split by which failure mode a file
triggers. Not fixed here; recorded so the next person does not re-diagnose it.

`test_apply_project_calibration_single_detector` asserted on `_ring_items`, but
with the manual seed card active `_draw_rings` delegates to the seed preview and
the items land in `_seed_ring_items` (verified: 166 of them). The assertion now
accepts either. The test still hits the known pyqtgraph teardown SIGABRT, which
reproduces identically on a clean HEAD worktree.

## 2026-09-04 — PR #7 is a live branch, not a snapshot: do not push-and-close `main` until it is re-merged

Attribution first, since it was implicit in earlier entries: PR #7 and its
whole Strain Cake body of work are **Jun-Sang Park's** (`junspark`,
jun.sang.park@outlook.com), an external contributor to what had been a
solo-owned repo. The 20 commits we reviewed (`c21ba35`..`078f216`, +3492/−295,
19 files, 6 new modules) are merged into local `main`; the two commits on top
(`092fbba` collision-safe frame naming, `46e0fec` +108 tests) are ours.

Going to push `main` and close the PR, we found PR #7 had **moved 16 commits
past the head we reviewed** (`078f216` → `bd32b62`, +1982/−285 over 23 files,
pushed 2026-09-02 and 2026-09-04). The PR head is therefore not an ancestor of
`main`: a push would not auto-close it, and a manual close would silently
discard those 16 commits. **Did not push and did not close** — recorded here
because the failure mode is quiet and easy to repeat.

The general rule this yields: **before pushing a branch to close somebody's
PR, re-fetch `refs/pull/N/head` and check containment**
(`git merge-base --is-ancestor <pr-head> main`). "We merged it locally two
days ago" is not the same as "the PR is merged", and this contributor pushes
daily.

Three specific collisions to resolve when re-merging:
- `66b25da` **retires `midas_gui/zarr_cake.py`**, rewiring Zarr output onto
  `write_gsas_zarr_zip` and adding `tests/test_batch_zarr_gsas.py` (186 lines).
  Our `tests/test_zarr_cake.py` (150 lines) imports the deleted module. Git
  auto-merges this cleanly into a broken state — no conflict marker, just an
  ImportError. The invariants that test pinned (REtaMap channel order, the
  (5, nR, nEta) orientation that GSAS-II's `G2pwd_MIDAS.py` reads) still
  matter and should be re-pointed at the new writer, not dropped.
- `midas_gui/workers.py` conflicts (their +409 vs our `frame_output_base`).
  Our fix must survive verbatim — it is the one guarding against silent frame
  loss (see the 2026-09-02 entry below).
- `.context/DECISIONS.md` conflicts: they appended their own 235-line entry to
  this file. Both sides are keepers; resolve by interleaving newest-first, not
  by taking a side.

Also confirmed this session: `gh` is not installed on this machine, so PR
comments and closes cannot be scripted — either use the GitHub web UI or
`brew install gh && gh auth login`.

## 2026-09-04 — Header Exp ID field; Batch Integrate output-folder "Suggest" + writability preflight

Added a header-level "Exp ID" field (`app.py`, next to the Profile
selector) in the `mpe_wf_saxs_waxs` style (e.g. `park_may26`) — shared
app-wide via `MainWindow.expid()`/`set_expid()` rather than owned by any
one tab, since more than one tab's output-path logic may eventually want
it (today just Batch Integrate). Persists as a plain last-used value
(`settings.get_last_expid`/`set_last_expid`, global like recent files —
an Exp ID tracks the current experiment, not the beamline Profile) until
a Project is open, at which point the Project's own saved Exp ID
(`/gui_workspace` meta) travels with it instead.

Batch Integrate's Output-folder row gained a "Suggest" button
(`_suggest_output_dir`/`_apply_suggested_output_dir`/
`_maybe_autofill_output_dir`, wired to fire automatically the first time
a source loads) that fills in `<outroot>/<expid>_bc/<file-root>/
<detector>/`, mirroring `mpe_wf_saxs_waxs`'s own
`outroot/<expid>_bc/<froot>/<detector>/` convention. `expid`/`detector`/
`outroot` are read *positionally* off the loaded source's own directory
depth (mpe_wf's fixed `<outroot>/<expid>/<detector>/<froot>/<files>`
layout) rather than requiring the Exp ID field to be typed first — the
field is only consulted as a fallback when the source path isn't deep
enough to read those segments off directly. Unlike mpe_wf's own GUIs
(placeholder-text hints only), this auto-fills live, since MIDAS_GUI has
no separate confirm-and-launch step to catch a wrong guess — manual
Browse still overrides it once the user types/picks their own path.
Output within a run is now split into per-format subfolders
(csv/xye/fxye/dat/2d_csv/h5/zarr) instead of one flat directory, so the
"Saved to" message after a run now names the chosen Output dir rather
than one file's own parent, and the Save button (previously always
enabled after a run) is now disabled whenever an Output folder was set —
the run already wrote everything there as it went, so Save would only
produce a second, flat, unsubfoldered duplicate of the text formats.

Added `helpers.check_output_dir_writable()` as a preflight for all three
places a chosen/suggested output dir gets used — `_run()`,
`_run_as_job()`, and the two auto-suggest paths — since a wrong guess or
a stale manual path landing on a directory someone else owns (real
production output folders are not necessarily world-writable) previously
surfaced only as an opaque write failure partway through a run. Also
checked in `batch_cli.py`'s `main()` (background-job entry point) before
any work starts, so a background job fails fast with a clear message
instead of partway through, and — being the async, unattended-error-log
path this exists for — logs it via `[batch] ERROR: ...` rather than a
dialog.

These three files landed as one commit rather than being split by
sub-feature (Exp ID vs. Suggest-button vs. writability check) because
`_apply_suggested_output_dir`/`_maybe_autofill_output_dir` call
`check_output_dir_writable` directly and `tab_batch.py`'s import line
pulls it in — splitting further would have meant an intermediate commit
with a broken import.

## 2026-09-03 — File menu: "Open Last Project" and "Quit" actions

Added "File ▸ Open Last Project" — a one-click shortcut for the top entry
of Recent Projects, going through the same `_open_project_path`
confirmation flow, for the common case of reopening whatever was worked
on last without opening the submenu first. Added a standard "File ▸ Quit"
action (Ctrl+Q) as well, since the File menu had no explicit quit entry.

## 2026-09-03 — Grey out the calibration-file field when "From Tab 2" is selected

Batch Integrate's calibration-file field/browse button stayed enabled even
while "From Tab 2" was the selected calibration source, where they have no
effect — misleading, since it looks editable but isn't consulted. Added
`_update_calib_src_enabled()`, wired to `_use_tab2_btn.toggled` and called
once at init, that enables the field/button only while "From file" is
selected.

## 2026-09-03 — `dialogs.show_error` always logs to `~/midas_gui_error.log`, even for worker-thread exceptions

`show_error` is shown for exceptions a worker thread already caught and
turned into a Qt signal — they never reach `app.py`'s global excepthook on
their own, so before this the full traceback was lost the moment the
dialog was dismissed, with no on-disk record. `show_error` now always
appends `full_text` to the same log file the excepthook writes to
(`app._log`/`app._LOG_FILE`), and the dialog's informative text names that
file path so the user knows where to find it after closing the dialog.

## 2026-09-03 — "Run as background job" pins the spawned screen session's cwd to the repo root

`JobQueuePanel.launch()` runs `python -m midas_gui.batch_cli` inside a
detached `screen` session. `-m` only resolves the `midas_gui` package when
the process's cwd is the repo root (that's what puts the repo root on
`sys.path[0]`) — but the spawned `screen`/`bash` session inherits whatever
cwd the GUI itself happened to have at launch time, which isn't guaranteed
to be the repo root. Fixed by prefixing the wrapped shell command with
`cd <repo_root> &&`, where `repo_root` is derived from `Path(__file__)`
rather than assumed from the caller's environment.

## 2026-09-03 — Detector-view overlay: fixed η-spoke placement and made "Show bin grid" actually hide everything when unchecked

Two bugs in `helpers.draw_polar_bin_overlay`, found while cross-checking
its geometry against `pixel_to_REta`'s convention (`η = atan2(-Yc, Zc)`,
so η=0 points straight up, +Z, not along +Y):

- The non-tilted η-spoke formula used `Y = bc_y + r·cos(θ)`,
  `Z = bc_z + r·sin(θ)` — swapped relative to that convention, so every
  spoke was drawn 90° off from where it actually is. Fixed to
  `Y = bc_y + r·sin(θ)`, `Z = bc_z + r·cos(θ)`.
- `if not show_grid: return` ran *after* the Rmin/Rmax circles were
  already drawn, so unchecking "Show bin grid" still left those circles
  on screen. Moved the early-return before them so unchecking the box
  hides the overlay entirely, and updated both tabs' tooltips
  (`tab_batch.py`, `hydra_batch_page.py`) to say so.

`widgets.build_lab_frame_axes_items`'s always-on compass overlay had the
same follow-on fix: its old 0°→45° arc-sweep+arrowhead η indicator is
replaced with four cardinal-angle (0°/+90°/−90°/180°) tick+label marks
using the same `(-y_sign)·sin(η)`/`cos(η)` convention, so the lab-frame
compass and the real caking overlay now agree on which way η points.

## 2026-09-03 — `tilted_ring_xy` closes its polyline, fixing a seam in tilted ring overlays

`tilted_ring_xy` sampled η over `np.linspace(0, 360, n, endpoint=False)`, so
the returned `(Y, Z)` arrays never repeated their first point. Fed straight
into `pg.PlotDataItem` (a plain polyline, which pyqtgraph never auto-closes
back to its start), a tilted ring drawn with this always showed a visible
gap between its last and first sampled point. Switched to
`endpoint=True` so the last point exactly equals the first and the
polyline closes with no seam.

## 2026-09-03 — "Files sharing a name stem" now searches subfolders recursively

`BrowseFilesDialog._stem_matches()`, `helpers._collect_frame_paths()`, and
`widgets.DataLoaderPanel._raw_source()`'s live filestem filter all glob'd
only the immediate directory (`<dir>/<stem>*`), so a stem match silently
missed any file whose scan point landed in a sibling subfolder rather than
directly under the picked folder. Changed all three to a recursive
`<dir>/**/<stem>*` pattern (`glob(..., recursive=True)`) so a stem search
finds a matching file anywhere below the selected folder, not just as a
direct child.

This was found while investigating a still-unconfirmed `FileNotFoundError`
report in Batch Integrate's stem-match mode when source files were split
across sibling subfolders — three attempts to reproduce it from a fresh
repro script did not trigger the error, so this is not a confirmed root
cause, but it's the strongest candidate found while reading through this
code path: the old non-recursive glob is exactly the kind of gap that
would produce a "file the GUI expected to find isn't there" style error
for that scenario. Recorded here as what changed, not as a confirmed fix —
if the original report resurfaces, revisit with the actual failing
directory layout in hand.

**Verified**: `test_stem_filter_becomes_glob_pattern_in_source_cfg` and
`test_stem_filter_roundtrips_through_get_set_state` updated to expect the
`**` pattern in `source_cfg()`'s output.

## 2026-09-03 — Single-file HDF5 sources now honor "Combine sub-frames" and frame-range bounds the same way multi-file sources do

A single bare HDF5 file picked in Batch Integrate's Data Loader silently
ignored the "Combine sub-frames" (chunk_size/combine_op) control — it was
routed through a plain `HDF5FrameSource` in `workers._open_source_cfg`,
while several separate HDF5 files (an explicit multi-select, or a folder/
stem-filter pick that resolves to several files) went through
`_HDF5StackGlobSource`, the only source that actually implements combining.
This also meant frame-range bounds for a chunked multi-file HDF5 pick were
wrong: the Data Loader's start/end spinboxes hold FILE numbers, but
`_HDF5StackGlobSource.n_frames` counts COMBINED-FRAME chunks, so a range
computed by file-index arithmetic silently stopped partway through an
early file whenever "Combine sub-frames" split it into more than one
chunk.

Fix: `_open_source_cfg`'s `"hdf5"` case now routes through
`_HDF5StackGlobSource` (a single-element path list) instead of a bare
`HDF5FrameSource`, so combining takes effect for single files too.
`widgets.DataLoaderPanel` follows suit: "Combine sub-frames" is now shown
for single bare HDF5 files (previously hidden), `source_cfg()` includes
`chunk_size`/`combine_op` for the single-`"hdf5"` case, a single-file
source's start/end spinboxes are reset-then-locked to its own scan number
(parsed via `froot_and_frame_num`) rather than left editable over
sub-frames that were never addressable that way, and `frame_range()` gains
a `_hdf5_multi_file_counts()` helper that expands FILE-index bounds
through each matched file's own chunk count before turning them into the
COMBINED-FRAME-index range `_HDF5StackGlobSource.n_frames` actually counts
over.

**Verified**: new test
`test_frame_range_multi_file_hdf5_spans_all_files_with_combine_chunk`
(3 synthetic 10-raw-frame HDF5 files, chunk size 3) asserts
`frame_range() == (0, 12, 1)` — 3 files × 4 combined frames each — rather
than the file count of 3 a naive file-index range would have produced.

## 2026-09-03 — Batch Integrate's Zarr checkbox rewired onto `write_gsas_zarr_zip`; retired the homegrown `zarr_cake` writer

A GSAS-II user hit "Read Error" on a `.zarr.zip` produced by Batch
Integrate's "Zarr (cake, REtaMap)" checkbox. Initial guess ("wrong button —
should've used the dedicated Export-for-GSAS-II card instead") was rejected:
in `mpe_wf_saxs_waxs`, ONE zarr output is read by both its own viewer and
GSAS-II with no separate export step, so this GUI having two divergent
writers producing the same-labeled output was itself the bug, not
by-design.

Root cause, confirmed by reading `midas_integrate_v2/io/zarr_gsas.py`
directly and by opening real production mpe_wf `.zarr.zip` files with
`zarr.open(..., mode='r')`: the old `zarr_cake.write_cake_zarr` wrote cake
data to `/IntegrationResult/FrameNr_<i>`, a group GSAS-II's importer never
reads; the schema it actually reads is `/OmegaSumFrame/LastFrameNumber_<i>`,
which `write_cake_zarr` never wrote at all. Its `/InstrumentParameters` also
only carried 2 of the 10 keys GSAS-II requires (`Lam`, `Distance` — missing
`Polariz, SH_L, U, V, W, X, Y, Z`), a compounding failure. This GUI already
had a *correct* writer for this exact schema —
`midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip`, used until now only
by the separate "Export for GSAS-II" button (`gsas_export.py`).

Fix: `BatchWorker.run()` (`workers.py`) now calls `write_gsas_zarr_zip`
directly instead of `zarr_cake.write_cake_zarr`, writing **one zarr per
combined output frame** (`<fid>.ave.zarr.zip`, into a `zarr/` subfolder)
rather than one bundled `integrated.zarr.zip` for the whole run — mirrors
mpe_wf's own one-zarr-per-scan-point convention, and sidesteps
`write_gsas_zarr_zip` having no multi-frame API of its own. Provenance is
stamped post-hoc via the existing `provenance.append_to_zip()` (the writer
has no attrs slot for it, same pattern as mpe_wf's own
`stamp_zarr_provenance.py`). `zarr_cake.py` deleted outright — no other call
site, no test imported it. The other lineout formats (csv/dat/xye/fxye/
2d_csv) and the combined HDF5 output moved into their own subfolders
(`<fmt>/`, `h5/`) alongside `zarr/` at the same pass, so per-frame zarr
files don't collide with per-format lineout files in one flat directory.
`_HDF5StackGlobSource` gained `metadata_for_index()` (Temperature/Pressure/
StorageRing-current, always the chunk mean regardless of pixel combine-op)
so the per-frame zarr can carry the same instrument metadata mpe_wf's own
output does, when the source is a VAREX-style HDF5 stack.

**Verified**: `tests/test_batch_zarr_gsas.py` (new) opens the written zarr
with `zarr.open` and asserts all 3 of GSAS-II's `ContentsValidator`-required
groups (`InstrumentParameters`/`REtaMap`/`OmegaSumFrame`) are present, plus
`provenance_history`; a second test confirms per-chunk metadata is always
the arithmetic mean of the raw sub-frames even when the pixel op is Sum/Max/
Median. Full suite re-run file-by-file (the standing isolation method, see
2026-08-30 below) shows no new failures beyond the already-documented
baseline. Independently, the user opened the real zarr files this fix
produced against the actual C611 dataset in GSAS-II and confirmed they load
correctly.

## 2026-09-02 — PR #7's `<froot>_<NNNNNN>` output naming kept, but made collision-safe rather than reverted

junspark's PR #7 changed Batch Integrate's per-frame profile filenames from
the frame id verbatim (`frame_000.csv`) to a zero-padded
`<froot>_<NNNNNN><tag>` (`frame_000000.csv`), matching mpe_wf_saxs_waxs's own
output convention. Nothing in the PR's 20 commit messages mentions this, and
it silently broke the two tests that asserted the old names.

Two ways to resolve it: revert the convention, or keep it and fix what it
broke. **Kept it** — the whole point is interoperating with mpe_wf_saxs_waxs's
tooling, and a padded, sortable name is genuinely better than a raw stem; the
tests were simply stale (confirmed: with only the expected filenames updated,
22/22 pass, so the writer itself was correct).

But the padding normalisation introduced a real defect worth recording,
because it is the kind that produces no error: `froot_and_frame_num` maps
`scan_1`, `scan_01` and `scan_001` all to frame 1, so all three wrote to
`scan_000001.csv` — three frames in, one file out, the first two silently
overwritten. Reachable through the Browse dialog's "Multiple files" /
"Files sharing a name stem" modes when a pick spans differently-padded files.

Fixed by separating *parsing* from *naming*: `froot_and_frame_num` stays a
pure (and deliberately non-injective) parser, and a new
`frame_output_base(out_dir, fid, fallback_idx, used)` owns filename
allocation, carrying a per-run `used` set. On a clash it falls back to the raw
frame id — unique by construction, since it is a file stem or a stem plus a
chunk suffix — and only then to an index suffix. Chose de-duplication over
"detect and abort" because a mixed-padding folder is a legitimate input, not
user error, and over "always use the raw id" because that throws away the
sortable-name benefit for every well-formed run. All three call sites
(`BatchWorker`, `FolderMonitorWorker`, `write_all_profiles`) go through it.

Also taught the parser the `_c<NN>` chunk suffix the PR's own
`_HDF5StackGlobSource` mints: those ids matched no numeric run, fell to the
fallback branch, and produced names like `run_009243.vrx_c00_000000.csv` that
no longer sort by frame. Split off before the numeric parse and re-attached
after, so a stem with its own frame number keeps it and a stem without one
uses the chunk index as the frame number.

**Open, not decided here:** junspark has not been told about the convention
change. It breaks any user script globbing Batch Integrate output, and
deserves either a note on the PR or a line in the release notes.

## 2026-09-02 — Reviewing an incoming PR: baseline first, and cover the new code rather than trusting a green run

PR #7 was 20 commits / +3492 lines / 6 new modules with **zero test changes**
— the largest external contribution this repo has taken. Two process
decisions came out of reviewing it, both worth repeating.

**Baseline the suite on `main` before reading the PR.** This repo has four
permanently-failing test files (see STATE.md for the current counts). Run
per-file on `main` first, capture every summary line, then run the identical
sweep on the PR branch and diff. Without that, the PR's two genuine
regressions are indistinguishable from the standing noise — and, worse, a
reviewer who knows "those four always fail" is primed to wave through a
*fifth* failing file. Diff per-file results, never a combined `pytest tests/`
run (see the 2026-08-30 entry below for why a combined run is untrustworthy
here regardless).

**A green suite is not coverage.** The only two failures PR #7 tripped were in
*old* tests, and they were tripped by an intentional rename — pure luck, not
verification. Nothing in the suite touched the strain-cake maths, the zarr
schema, the provenance stamping or the tilt-seed gating. So 108 tests were
added before merging, prioritising the failures that are *silent* over the
ones that crash:
- `test_zarr_cake.py` pins the REtaMap channel order and (5, nR, nEta)
  orientation. This layout is not ours — GSAS-II's `G2pwd_MIDAS.py` and
  mpe_wf_saxs_waxs read it — so a transposition yields a file that opens
  fine and is wrong.
- `test_strain_cake.py::test_subpixel_refinement_beats_bin_quantization`
  fails if `ring_azimuth_residual` ever regresses to a plain argmax, which
  produces a smooth, plausible-looking map carrying no azimuthal
  information at all.
- `test_set_raw_frame.py` pins that the centralized transform returns
  exactly what `_apply_im_trans` would, so the overlay-vs-rings
  misalignment the refactor fixed cannot silently return.
- `test_helpers.py` pins the documented invariant that `tilted_ring_xy`
  reduces to the plain circle at zero tilt, and that ring and spoke agree
  where they meet.
Deliberately **not** covered, and left open in ROADMAP.md: `job_queue.py`,
`peak_fit_panel.py`, `batch_cli.py`. They are subprocess/GUI-shell code whose
useful tests need either a live `screen` session or a full widget harness —
disproportionate to this session's scope, and honest to record as a gap
rather than paper over with import-only smoke tests.

Merged as a fast-forward (`092fbba`, `46e0fec` on top of the PR's own 20),
since the branch was already based on current `main`.

## 2026-08-31 (`fd7f67a`) — Hydra Overall Eta-R Cake rotates each panel by its own `tx` before summing

Fixes the Overall cake piling every panel onto the same wedge instead of
covering -180°..180°. Supersedes the 2026-08-26 "Hydra Overall Cake
verified, no code change" entry below — that verification only tested the
default `tx=0` case (all panels share one η axis, no rotation needed by
construction); it didn't cover a panel calibrated with its own true,
distinct installation `tx`, which needed a rotation the un-updated compose
function never applied. `_resample_rows_to_eta_grid` (new) handles the
general case; existing R-axis (2θ) resampling logic unchanged.

## 2026-08-30 — `pytest-forked` isolates the known teardown-crash test files; root cause is `fork()`-after-multithreading, not just pyqtgraph

Added `pytest-forked` + `pytestmark = pytest.mark.forked` to the four
known interpreter-teardown-crash-prone files. **Confirmed working** for a
single isolated file run: a pyqtgraph-teardown segfault that used to kill
the whole pytest process now surfaces as a clean `FAILED ... CRASHED with
signal N` instead. **Does NOT fix a combined `pytest tests/` run** — `os.fork()`
of an already-multithreaded process (torch/numba/Qt/HDF5 thread pools
accumulated earlier in the run) is itself unsafe and crashes regardless of
which test forked, including files with zero Qt content. Trust per-file
isolated runs only; a real fix would need spawn-based workers (e.g.
`pytest-xdist`), not attempted.

## 2026-08-29 — Export for GSAS-II: native MIDAS zarr, not a GUI-specific format; v1 scoped to single-detector/R-uniform/embedded-mask

`gsas_export.py` calls `midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip`
directly rather than inventing a GUI format — verified against GSAS-II's
own `G2pwd_MIDAS.py` import contract (required groups, `REtaMap` index
order, its >20-unmasked-point-per-azimuth filter) so the output needs no
adapter. Picks ONE attempt from history (dropdown) rather than "latest" —
GSAS-II imports one dataset at a time and silently picking latest would be
a hidden assumption. A `.provenance.json` sidecar carries full metadata,
kept *next to* the zip so the zip's internal structure stays exactly what
GSAS-II expects.

v1 scope, each enforced as a named `ValueError`: single-detector only
(Hydra composite = possible fast-follow); R-uniform binning only (a
Q-rebinned attempt's `r_axis_px` isn't a plain function of geometry, so
`REtaMap`'s bin area can't be reconstructed); embedded mask only (a
file-backed mask reference isn't guaranteed to still resolve at export
time). Works with either the legacy 2-D profile or the multi-azimuth cake
(below) — forces η bin to 360° for the 2-D case since that run's η bin was
only ever a collapse-weighting knob, not a real azimuthal-sector count.

## 2026-08-29 — Batch Integrate "Multi-azimuth output (cake)": opt-in, off by default, repurposes the existing η bin field

`midas_integrate_v2` always computes a full (η,R) cake internally — Batch
was collapsing it to one full-circle profile before anything downstream
ever saw it. The checkbox keeps every azimuthal sector as its own output
profile (`(n_frames, n_eta, n_r)`), needed for per-azimuth GSAS-II/texture
work. Deliberately reuses the existing η bin/range fields rather than
adding a parallel control — they already meant "how finely to slice
azimuth," so overloading their meaning when the checkbox is on is more
honest than a second control that could drift out of sync. Not combinable
with Q-uniform bins (`rebin_R_to_Q` only handles 1-D profiles) — blocked at
both UI and worker level. HDF5 output skipped in this mode
(`write_h5` expects one profile per frame); use text formats or GSAS-II
export instead.

## 2026-08-29 — Project `.h5` schema redesign (`21faaf8`): `gui_workspace` + `analysis`, clean cutover, global mask history, one combined Open Project dialog

Three explicit user decisions (via `AskUserQuestion`, each a real
trade-off):
1. **Mask attempts are a global history** (`/analysis/mask/attempt_NNNN`),
   not per-panel like calibrate/integrate — `/analysis/mask` only stores
   mask-*creation* attempts; calibrate/integrate records separately embed
   whichever mask they actually used, so every analysis attempt stays
   self-sufficient regardless.
2. **Clean cutover, no backward compatibility** with schema v1/2 — chosen
   over a dual-path reader (used at the 1→2 bump) because the old one-JSON-
   blob layout has no natural per-tab boundary to migrate into
   automatically. Opening a pre-3 file shows a clear version warning
   instead of a silent partial restore.
3. **One combined dialog** (file-tree + live checkbox-tree preview) rather
   than `QFileDialog` + a second custom dialog, reusing `BrowseFilesDialog`'s
   navigation building blocks.

Mask Builder's "Log to Project" is an explicit button (not piggybacked on
Save) since — unlike Calibrate/Batch's single "run finished" moment — a
mask can be finalized via Compute, Load, or hand-drawn shapes in any
combination; the button puts the user in control of when it's "final."

## 2026-08-27 — Browse… popup: 4-mode file picker, `list[str]` end-to-end (not glob-collapsed), Hydra registry wiring, plus later polish

`dialogs.BrowseFilesDialog`: one popup, up to 4 mutually-exclusive modes
(Single file / Multiple files / Full folder / Files sharing a name stem) —
which modes a field offers is passed in by the caller since different
pipelines can't take every shape (e.g. Hydra's main Data field has one
anchor file, so `modes=("file",)`). **A confirmed Multiple-files/Files-
sharing-a-stem pick is carried through as a plain `list[str]` end to end,
deliberately not collapsed to a glob** — a stem match or arbitrary
multi-select may not share one glob pattern (non-contiguous numbering,
mixed extensions), and collapsing would silently include/exclude files the
user didn't pick. `helpers.source_kind`/`_collect_frame_paths` special-case
`isinstance(raw, list)` as a "folder"-like kind, returned as-is. Hydra's
loader joined the same `DataSourceRegistry` its single-detector sibling
used, gated to Hydra-labeled fields only so anchor paths aren't offered
across modes that can't consume them. Follow-up polish: field text shows a
folder/file path instead of an "N files selected" count
(`helpers.display_text_for_paths`); default browse dir is the repo root,
not `$HOME`; file-tree name column doubled in width (Qt's stock 100px
default).

## 2026-08-27 — Feed Calibrate's Multi-panel results to downstream integration

Root cause (3 gaps, all GUI-side workarounds for upstream package gaps —
see `.context/ROADMAP.md` P3-3): `spec_from_calibration_result()` has no
panel awareness at all; `geometry_fields_from_file()` never parsed panel
keys out of a paramstest/calibration.json; `_save_paramstest()` wrote the
per-panel shifts file but never the panel *grid* keys, so even MIDAS's own
native reader would see `NPanelsY=0`. Fixed by having `calib.py` set two
new plain JSON-safe attrs on the result (`panel_layout`, `panel_shifts_path`)
and `helpers._apply_panel_fields()` apply them to any spec built from that
result — additive, so every non-panel caller (Hydra, PumpProbe, PDF,
Export) is an unaffected no-op.

## 2026-08-27 — Fix: Flip Z ignored when "Multi-panel detector" is checked in Calibrate

Root cause: `calib.py`'s manual pre-flip workaround (needed because several
calibration pipelines have no native `im_trans` param — ROADMAP P3-1)
computed the auto-seed from the *raw* image but ran the solve on the
*manually flipped* image, in four branches of `run_pipeline()` — with Flip
Z on, the seed landed ~`NrPixelsZ` away from the true position and these
pipelines only do local refinement from the seed, so they converged near
the wrong (unflipped) position. Fixed via `_prep_transformed(image, dark,
im_trans)`, applied once at the top of all four branches (also fixed dark
being left untransformed in the same branches). **Class-of-bug lesson**:
whenever an array is used for both a seed step and a solve step, check
both are computed from the *same* transformed frame — a silent mismatch
between two local variables named similarly (`image` vs `img`) is easy to
introduce and won't raise an error. `first_time` pipeline branch left with
its own separate, pre-existing gap (ignores `im_trans` entirely) — tracked
as a follow-up, not fixed here.

## 2026-08-27 (`5cf2e8c`) — Error dialogs/logs never truncate the underlying exception, app-wide

A `msg[:400]`/`traceback.format_exc()[:N]` pattern was copy-pasted across
~15 files' worker-failure handlers — any long enough traceback could get
cut mid-word. Fixed once via a shared `dialogs.show_error(parent, title,
full_text, log=, log_prefix=)` using Qt's `setDetailedText` (compact
one-line summary + scrollable "Show Details…"), not just deleting the
`[:N]` slices — a full multi-KB traceback dumped as plain `QMessageBox`
text resizes into an unreadable giant box. Asked the user explicitly
whether to fix app-wide vs. just the one reported dialog; chose app-wide.

## 2026-08-26 — Project records embed calibration results + skip embedding file-backed masks; Hydra Overall Cake full UI; Data Viewer Cake tab; Workspace/Project active-Profile persistence

- **Selective mask embedding**: a mask built purely from file/folder
  sources is already reconstructable from its path+hash, so embedding its
  array too was pure size bloat; a mask with anything hand-drawn/computed
  has no file to point back to and must still be embedded.
- **Calibration attempts embed their computed Radial Profile/Eta-R Cake**
  so Open Project's Populate step is instant and independent of the
  original data file's continued existence (FAIR self-containment).
- **Hydra Overall Cake got the button/checkbox UI** matching the
  already-existing compose function; a finished Overall run also logs as
  its own `hydra_composite` attempt for provenance, deliberately not wired
  into Populate (no tab widget to restore an Overall result into).
- **Workspace/Project restore the active beamline Profile on load** — a
  project made under one Profile (different calibrants/devices/tab
  visibility) would otherwise silently apply against whatever Profile
  happens to be active at load time. Silently skipped if the recorded
  Profile no longer exists locally.

All additive to the HDF5 schema — no field renamed/removed.

## 2026-08-26 — Hydra cake-plot independent-axis zoom fix

Cake plots forced every right-drag to η-only zoom. Root cause: `pg.ImageView.
__init__` unconditionally aspect-locks its ViewBox, and an aspect-locked
ViewBox recouples both axes on every range-change (not just on drag) — no
custom `mouseDragEvent` can work around it. Fix: `vb.setAspectLocked(False)`;
stock pyqtgraph's own right-drag handler then already computes independent
per-axis zoom for free. (See ARCHITECTURE.md's Qt gotchas list.) Note: this
session's separate verification that "Overall cake needs no η-axis
resampling" only held for the `tx=0` default case — see the 2026-08-31
entry above, which found and fixed the general case.

## 2026-08-26 — Workspace/Project UX rework: unify the mental model at the UI layer only, never touch the FAIR HDF5 schema (`6b1564b`)

Root cause of the "doesn't feel convenient" complaint: two functionally
unrelated concepts ("GUI State" — a mutable JSON snapshot — and "Project" —
an append-only FAIR log) both occupied the "project" mental-model space
with neither having recents/dirty-indicator/autosave/an in-app history
browser. **Guiding constraint, honored throughout: unify the UX without
ever touching `project.py`'s schema or its opt-in/append-only guarantees**
— every new feature reuses existing read-side API or lives in new additive
sidecar files.

Key decisions (confirmed via `AskUserQuestion`): renamed "GUI State" →
**Workspace** in all user-facing labels (Workspace = editable draft,
Project = permanent FAIR record); recent-files list is global, not scoped
per beamline Profile (a recent project shouldn't stop being recent just
because the active Profile changed). Dirty-state tracking reuses each
tab's existing `get_state()` on a ~7s timer (hashed and diffed) rather than
wiring per-widget change signals across 10 tabs — can't drift from what
Save actually persists. **Autosave/crash-recovery's restore-prompt is
wired only from `main()`, never `MainWindow.__init__`** — every test
constructs `MainWindow()` directly, and a leftover autosave draft would
otherwise pop a blocking modal under the offscreen QPA platform with no
one to click it, hanging tests/CI. Project History viewer shows raw JSON
metadata rather than parsing into table columns, since the schema varies
by attempt kind and has evolved before — a generic viewer can't drift out
of sync with it.

## 2026-08-25 — ImTransOpt fix: the MIDAS backend does the pixel flip everywhere it can; GUI only flips masks (never images) and only for on-screen display (`2358ae4`)

Root cause of a Batch Integrate lineout bug: `spec_from_calibration_result()`
never copied `im_trans` onto the built spec, so raw untransformed frames
were integrated against a geometry fit on a *transformed* image — a
coordinate-frame mismatch. **First pass (pre-flip the array in GUI Python)
was explicitly rejected by the user**: MIDAS's own packages already accept
`ImTransOpt`/`TransOpt` and apply it internally — the GUI's job is to pass
the parameter, not duplicate the transform logic. **Final architecture,
one rule for the whole app: the backend performs every pixel-array flip
used for an actual calibration/integration computation; `midas_gui` never
does. The only exception is a viewer's on-screen preview array**, which may
be flipped locally for display (explicitly confirmed fine by the user —
"ok to transform temporarily to view the effect of the parameter").

**Masks are the one genuine, still-open exception**: `*BinGeometry.
from_spec(spec, mask=mask)` has no `apply_trans_opt` hook at all (unlike
every `integrate_*` function), so a mask must still be pre-flipped in
Python before reaching `from_spec`/`build_geom` — see ROADMAP P3-2 for the
upstream ask. `azimuthal_sigma_clip()` has the same gap, so Mask Builder's
azimuthal-clip branch is the one call site that deliberately keeps feeding
it an already-transformed array. **Self-inflicted bug caught and fixed
before landing**: a mask used by two different consumers in the same
worker needs to be checked per-consumer for which coordinate space it
expects — they aren't always the same (raw for pointwise zeroing, flipped
for `build_geom`). `PoleFigureWorker` (Texture) has the same untransformed-
mask gap and was left unfixed (out of scope) — tracked in ROADMAP.

## 2026-08-25 — MIDAS backend package upgrade procedure: clone env first, hold numpy/torch/numba fixed, static-diff the API before installing

Reusable procedure for the next backend version bump: (1) `conda create
--clone` rather than a fresh env from `environment.yml` — guarantees
byte-identical Qt/PyQt5 bindings, since a "faithful" rebuild from the
recipe can still solve to a different conda-forge build. (2) Hold
numpy/torch/numba fixed deliberately, not because the resolver forces it —
a numpy 2.x move is a much larger, separately-scoped change. (3) **Static
API-diff every symbol `midas_gui` imports (old vs. new) before touching any
environment** — this is why a 0.3.x→0.6.x-scale jump needed zero
`midas_gui` source changes; don't assume semver discipline instead. (4)
Verify via per-file isolated pytest runs in both old and new env, run
several times each — a single run can't distinguish "always crashes now"
from "sometimes crashed before" given the known teardown-crash flakiness.

## 2026-08-25 — Open Project's auto-plots reuse the exact live-fit code paths; Batch's replay is new code because no such path existed

Calibrate/Hydra Calibrate needed **no new "replay" logic** — the same
methods a live fit calls (`_draw_rings`, `_run_integration`) are already
pure functions of a duck-typed result object, so a stored project attempt's
reconstructed result works identically to a fresh `CalibrationWorker`
result. **Batch/Hydra Batch needed genuinely new code** — the only existing
data path into the waterfall/stacked-profile widgets was `_on_frame`'s
incremental per-frame `add_profile()`; there was no "given a complete
profiles array, populate the whole plot" entry point, and
`project.read_attempt` never read integration `results/*` arrays at all
(only JSON metadata) until a new `read_attempt_results()` was added.
**Verification finding, not a fix**: the pre-existing pyqtgraph
interpreter-teardown segfault was confirmed (via `git stash` A/B) to
reproduce on unmodified `main` too — not a regression from this session.
Per-file test isolation is the only trustworthy verification method going
forward, not just for the two previously-named heavy Hydra UI files.

## 2026-08-25 — Profile switch refreshes option *lists* live, never re-seeds already-built field *values*

Two related bugs: profile switching was buried in Preferences (fixed by
adding a header combo alongside it, both routing through one
`on_profile_changed()`), and several dropdowns/menus (device list,
calibrant list, pixel-size/K-edge popups) silently kept showing the
*previous* profile's choices until restart. **Deliberate asymmetry**:
option *lists* refresh live (a finite, profile-owned vocabulary — the
widget's current selection is preserved if it still exists in the new
list), but numeric/path *default values* a profile seeds into a field are
NOT re-pushed into already-built fields on a switch — there's no way to
tell a field's current value is the seeded default vs. the user's own
in-progress edit, so re-seeding on every switch risks silently clobbering
real work. Hydra mode gated to the 1-ID-E profile (only that beamline has
the 4-panel GE detector).

## 2026-08-24 — Open Project's "populate the GUI" reuses GUI-State's widget-key vocabulary; Batch Integrate gets a real calibration, not just display fields (`e693316`)

Opening a project previously only marked it active for *future* logging —
fields were never actually populated. The attempt→GUI mapping reuses
`_state_widgets()`'s existing widget-key vocabulary via
`apply_dict_to_widgets()`, rather than a bespoke schema, because the
single-detector tab's state dict is literally the union of the two Hydra
per-panel/shared dicts — one pure function
(`project.calib_attempt_gui_fields`) serves all three call sites. **Batch
Integrate's populate step calls `set_calibration()` directly** rather than
going through `set_state()` (which deliberately never restores a live
calibration result — no live object to restore from plain GUI State) —
a project attempt is different: it already recorded the exact calibration
values used, so `project.calibration_namespace()` turns that stored dict
into a duck-typed object the existing "From Tab 2" path already accepts.
Explicitly out of scope: restoring dark/bright/background/mask sources
that were embedded directly with no owning file (pre-existing gap, not a
regression).

## 2026-08-24 — Hydra seed-mode linking is signal-fanout across per-panel checkboxes; cake data reuses an already-computed, previously-discarded array (`162fef1`)

**Seed-mode linking kept the checkboxes on each per-panel card** and
synced their state via signals, rather than hoisting one shared checkbox
to the toolbar — several existing per-panel methods already toggle the
checkbox as a side effect scoped to *that* card; moving it would have
meant rewriting all those call sites for a purely cosmetic win. **The
Eta-vs-R cake array was already being computed by the backend and silently
discarded** — `integrate_frame(return_cake=True)` existed before this
session (used by Batch's `2d_csv` export), but `IntegrationWorker`'s
post-fit auto-integration call site never passed the flag. New `CakeViewer`
widget rather than retrofitting `ImageViewer`, because `ImageViewer`
assumes displayed array indices *are* physical detector-pixel coordinates
starting at (0,0) — a cake's axes are R(px)/η(°) bin centres, neither
starting at the origin nor 1-unit-per-pixel.

## 2026-08-24 — File ▸ Project: FAIR provenance is separate from GUI State, always best-effort, and links Batch → Calibrate attempts (`e8dea6b`)

**Deliberately a separate concept from GUI State**, not an extension of
it: GUI State is a point-in-time snapshot meant to be overwritten (Ctrl+S
semantics); a provenance record must never be overwritten — it accumulates
a full history across many separate GUI launches over an experiment's
lifetime. **Raw scan data is referenced by path + checksum, never
duplicated** (a Hydra Batch run can touch thousands of frames) — hashed
fully under 200MB, head+tail fingerprint above. Masks/dark/bright/
background *are* embedded directly (small, and a hand-drawn mask has no
file to reference). **Logging is always best-effort and must never affect
the run's own outcome** — wrapped in a bare `try/except` that logs failure
to the tab's own Log panel; a provenance-write problem must never make a
successful run look failed. **Batch→Calibrate linking uses a ref string
bolted onto the result object** (`result._project_attempt_ref`), cheaper
than re-deriving "which calibration attempt matches these values" by
scanning the HDF5 file — correct as long as the same GUI session did both
runs.

## 2026-08-24 — Batch Integrate Hydra split: automatic hand-off, per-panel masks (diverges from Calibrate's scope-cut), lazy page construction

**Hand-off is automatic push, not a manual pull button** — unlike the Data
Viewer↔Calibrate hand-off (manual, because the Data Viewer has no
"finished" event), a Hydra panel's fit genuinely finishes, so Batch
Integrate's calibration source populates itself the moment that panel's
fit completes, mirroring the single-detector tab's existing auto-wiring.
**Masks ARE wired per panel for Hydra Integrate — a deliberate departure
from Hydra Calibrate's `mask=None` scope-cut** (confirmed with the user):
bad-pixel/beamstop masking matters more for integration-profile quality
than for ring-centroid fitting; each panel gets its own independent
`MaskSelector` with no cross-panel auto-discovery (mask files are
physically panel-specific, unlike data files). Drift correction and live
MONITOR mode deferred for Hydra v1 (user confirmed). `HydraBatchPage` is
built lazily (unlike its eagerly-built siblings) since it owns 8
pyqtgraph widgets and most sessions never open it.

## 2026-08-24 — Calibrate tab Hydra split: shared-vs-per-panel field boundary; `CalibrationWorker`'s stdout capture made optional for Parallel mode

**Field-sharing boundary** (confirmed with the user): wavelength, pixel
size, calibrant, and refine-parameter selection are shared across all 4
panels (one beam, one choice of what to refine); Transforms and the
initial seed (BC/Lsd/tilts) are independent per panel (each GE module is a
physically separate detector). **`CalibrationWorker`'s stdout capture had
to become optional** for the new Parallel run mode — the worker redirects
the *process-global* `sys.stdout` for the pipeline call's duration, safe
with one worker at a time but a data race with several racing in parallel.
Added `capture_stdout: bool` (Sequential=True unchanged, Parallel=False —
each worker's fine `print()` output goes to the real console instead of
the Log tab, but the Qt signals that don't touch global state are
unaffected).

## 2026-08-23/24 — Hydra composite windmill orientation: final state is CCW rotation by `tx` plus a separate vertical-axis mirror

Took several iterations to land on the correct orientation for
`hydra.py::compute_inv_coords`'s composite placement math:
**counterclockwise rotation by each panel's own `tx`** (not clockwise — an
earlier same-day fix flipped the sign based on a plausible-looking nominal-
value match, without actually rendering and comparing against a known-
correct reference; that fix was itself wrong and was reverted), **plus** a
separate left-right mirror of the whole finished composite canvas (`Y_lab =
(half - Yo) * px`, not `(Yo - half) * px`) needed to put the correct panels
on the correct sides. No `invertX` on the underlying viewer is needed
beyond that — verified via real Debye-Scherrer ring continuity across all
4 panel boundaries plus an actual rendered comparison.

**Methodological lesson (why this took several wrong turns)**: ring-
continuity / self-consistency checks are **blind to a global orientation
error** — a mirror combined with a compensating rotation-direction error
produces locally plausible, continuous-looking arcs just as easily as the
correct answer, especially when each panel's own tilt was fit
independently. Only a real windowed comparison against a known-correct
reference image (someone who knows what the physical detector actually
looks like) can catch this class of bug — don't trust a sign-convention
conclusion for a global rotation/mirror without that check, no matter how
good the self-consistency argument looks on paper.

This same 2026-08-23 pass also added: shared λ/max-2θ/px mirrored live
across ge1-4 + Composite cards; dark/bright/background correction for
Hydra (`HydraFieldSelector`, closing a scope-cut in the original module
docstring); a stale-radial-integration-geometry fix (`_effective_calib_geom()`
was returning a frozen snapshot instead of live BC/λ/Lsd/tilt widget
values); vmin% percentile now excludes exact-zero pixels (was washing out
auto-level on the Hydra composite's mostly-empty canvas — single fix point
in the `ImageViewer` base class, benefits every viewer).

## 2026-08-23 — Rare pyqtgraph teardown crash under a large test suite; mitigated, not fixed

A large `tests/test_hydra_ui.py` (many `pg.ViewBox`/`pg.ImageView`
instances per test) intermittently crashed the interpreter outright
(segfault/bus error) during the full suite — pyqtgraph's own `ViewBox`/
`WidgetGroup` global-registry teardown fragility, not a bug in this
codebase. **`gc.collect()` in test teardown made it WORSE** (forces
Python-level GC into pyqtgraph's half-torn-down object graph more often).
Just pumping the event loop (`app.processEvents()`) after each test, with
no forced GC, measurably reduced the crash rate (not a guaranteed fix).
See the 2026-08-30 `pytest-forked` entry above for the deeper, related
fork()-after-multithreading problem this doesn't solve.

## 2026-08-23 — Detector-image origin flipped to bottom-left (MIDAS convention)

Standing design decision: every image viewer places pixel (0,0) at
bottom-left, not top-left, because MIDAS assumes this origin and the
Flip Y/Flip Z/Transpose controls only make sense as "align raw readout to
the world view looking downstream along the beam" if the baseline
rendering already matches that world view. Root cause/fix: `pg.ImageView.
__init__` unconditionally calls `invertY()` — one `vb.invertY(False)` in
`ImageViewer.__init__` fixes every viewer built on it (Data Viewer, Mask
Builder, Calibrate). Confirmed a pure visual flip needing no data/geometry
changes: all click-to-pixel code goes through pyqtgraph's invert-aware
`mapSceneToView`, and ring-geometry math computes in pixel-index space with
no reference to `invertY`. The other three 2D displays in the app (pump-
probe heatmap, pole figure, waterfall) already default to bottom-left via
plain `PlotWidget`/`ImageItem` and were left untouched.

## 2026-08-12 — Mask tab: multi-file stack picker + configurable bad-pixel dilation

Explicit `self._stack_files` list (not a delimiter-packed string) keeps
multi-select fully separate from existing folder/file/glob parsing;
cleared automatically whenever the path field is edited directly, so
switching input modes can't leave stale multi-select state behind.
Dilation applied once, in `_set_mask()`, to the *computed* mask only —
never hand-drawn shapes, which are combined in afterward and must stay
exactly what the user placed. User later corrected the dilation semantics
from 4-connected to **8-neighbor (full-block)** growth — an explicit 3×3
structuring element with `iterations=n` gives the requested `(2n+1)×(2n+1)`
square per bad pixel, no custom BFS needed.

## 2026-08-12 — PDF tab rebuilt for full Stage 2-3 workflow

Rebuilt from Stage-1-only to the full workflow (absorption, detector-
efficiency, absolute normalization, multiple scattering, fluorescence
diagnostic, CIF-driven structure fit, Δ-PDF) — deliberately excluding
Bayesian SVI/NUTS, RMC, SAXS/SANS joint refinement, multi-phase, anisotropic
ADP, directional strain-PDF (kept out of `pdf_backend.py`'s re-exports on
purpose; see ROADMAP.md). Three gotchas worth remembering if touched again:
**`QCheckBox`, not a checkable `QGroupBox`, for optional-stage toggles** —
`widgets_to_dict`/`apply_dict_to_widgets` only persist `QAbstractButton`
subclasses, and a checkable `QGroupBox` isn't one, so its checked state
would silently fail to save/restore. **`pdf.delta_pdf(...)` requires
`torch.Tensor` inputs**, not numpy arrays. **`refine_structure`'s `fitted`
dict can contain a `"bg_coef"` value that's a `list`** (when `bg_order` is
set) rather than a scalar — code formatting `fitted.items()` must branch on
type. Also: the ~30x detector-efficiency amplification seen for thin Si at
hard X-ray energies is physically correct, not a bug (confirmed by direct
computation).

## 2026-08-10 — Upgrade all MIDAS backend pins; retire vendored `midas_pdf`

**`midas_pdf` is now the real public PyPI package**, not vendored — deleted
`midas_gui/_vendor/` (33 files), rewrote `pdf_backend.py` to a plain
`import midas_pdf` + re-export, no more `sys.path` tricks or the
`midas_hkls.absorption` compatibility shim (safe now that installed
`midas-hkls>=0.5.0` ships `absorption` natively). Newly pinned for
completeness several transitive MIDAS deps that were already being pulled
in implicitly but never declared (`midas-integrate`, `midas-peakfit`,
`midas-zipper`, `hdf5plugin`, `psutil`) plus `scikit-image` (a soft/
try-except optional dep of `midas-calibrate-v2`'s better ring seeding that
no MIDAS package declares) — goal was a `pip install .` that reproduces the
verified-working set on its own. NumPy-1.x pin chain re-confirmed
unchanged (numba/torch/pvapy compatibility).

## 2026-07-16/17 — Adopted the two-layer `.context` system; migrated legacy `claude/` knowledge into it

STATE (disposable-but-current) + DECISIONS (permanent) split so returning
to this project after a gap is cheap — only STATE.md auto-loads, detail is
read on demand. Folded the old per-session `claude/` folder into this
structure and deleted it; the build-critical `midas_pdf` reference stack
was *moved* (not summarized) to `.context/reference/midas_pdf/` — too
detailed to lose. Discarded as stale: `CLAUDE_original_scratch.md` and
`claude/gui_documentation.md` (a strict subset of the shipped doc).

## 2026-09-24 — The Distortion selection is project state, and the workspace outranks an attempt

Reported from the beamline: untick Distortion, save the project, reopen — and it
comes back ticked at 15/15. Three independent faults, all of which had to go:

1. **`_dist_coeffs` was never serialized.** `ref_dist` (the tick) was in
   `_state_widgets()`, but the set of harmonics it gates has no widget, so
   `widgets_to_dict` could not see it and nothing else wrote it out. A reopened
   project silently refined all fifteen whatever the user had picked. Fixed the
   way `seed_dist` already was: a top-level key in `get_state()`. Restored as
   `None`-means-absent, so a pre-existing project keeps the constructor default
   rather than being narrowed to "refine nothing".

2. **The caption went stale.** `apply_dict_to_widgets` restores with signals
   blocked, and `_update_dist_label()` is the only writer of the
   `Distortion (n/15)` text — so a freshly built tab kept its `(15/15)` caption
   next to a checkbox that had just been restored to unticked. `_set_state` now
   calls it unconditionally, not only when the new key is present: the caption
   is wrong after *any* restore, including of an old project.

3. **The attempt replay clobbered the workspace.** `_open_project_selection`
   restores the GUI Workspace first and then, if the user also ticked a
   calibration attempt in the picker, replays that attempt's fields over the
   top. Those fields are strictly staler — the workspace is written at Ctrl+S,
   the attempt when the fit ran — so the replay reverted every input the user
   had touched since their last run, which is what actually put the tick back.

The precedence rule is the part worth arguing. `apply_project_calibration` grew
`restore_fields`, and Open Project passes `False` when the Calibrate tab's
workspace was restored in the same action. The workspace is a superset of the
attempt's fields *and* newer, so the field replay was pure loss; what the attempt
uniquely carries — the embedded cake/profile arrays and the materialized panel
shifts — is in the result half, which still runs either way. Opening an attempt
without its workspace is unchanged and now restores the coefficient subset too,
via `project.calib_attempt_dist_coeffs()`; the set was already in the stored
metadata (`_json_default` sorts sets to lists), just never read back.

Not done: the same precedence question applies to Batch Integrate's
`apply_project_integration`, which has the identical shape. Left alone — no
report against it, and the fix belongs with evidence of the symptom.

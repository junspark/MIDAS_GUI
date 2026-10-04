# ROADMAP — open / pending items

_On-demand: outstanding work worth tracking. Not auto-loaded._
_Migrated from `claude/gui_plan.md` + `analyze_workflows/` on 2026-07-17;_
_verify against current code/commits before assuming still-open._

Phases 1–3 (pipeline dropdown, refine flags, kernels, variance, Q-uniform,
extra formats, learnable mask, refinement tab, corrections preview,
multi-panel, Bayesian UQ, PDF Stage 1, texture, export hub, joint-cake) are
**complete**. **PDF Stage 2–3 is also complete** (2026-08-12: CIF structure
fit, Δ-PDF, multiple scattering, absorption, fluorescence — confirmed live
in `midas_gui/tab_pdf.py`). Deliberately left out of scope, not currently
planned: Bayesian SVI/NUTS, RMC/DISCUS export, SAXS/SANS joint refinement,
multi-phase/core-shell, anisotropic ADP, directional strain-PDF.
Build-critical reference for maintaining the PDF tab:
`.context/reference/midas_pdf/` (esp. `01_core_api.md`, `05_gui_integration.md`).

## Per-tab open items

- **Calibrate:** multi-distance (`autocalibrate_multi`), doublet calibrants,
  NN-residual augmenter, per-ring δr_k JSON sidecar export, full custom-calibrant UI.
- **Batch:** per-frame outlier rejection (cosmic-ray / azimuthal σ-clip in
  batch), Compton/empty/absorption in batch, Zarr/GE/EDF sources, soft
  (autograd) kernel.
- **Corrections:** empty-scale LBFGS refine, absorption-param refine.
- **Refinement:** ProfileMSE / PeakPosition losses, multi-distance, in-tab
  Laplace UQ, energy-sweep drift.
- **Data Viewer:** tilt/distortion in ring **overlay** (radial integration
  already handled by df544d2), multi-detector.
- **Mask:** DAC gasket / angular wedge exclusion.
- **Texture:** multi-frame (χ,φ) stacks, ODF/WIMV. **Known bug (found
  2026-08-25, not fixed — out of scope for the ImTransOpt session that found
  it):** `PoleFigureWorker` (`workers.py`) passes its mask straight to
  `build_geom(spec, "subpixel2", mask_t)` with **no transform handling at
  all** — never did, before or after the 2026-08-25 ImTransOpt fix. If the
  active calibration has a non-zero ImTransOpt, the mask will be misaligned
  with the geometry (same class of bug as the one fixed everywhere else this
  session, but this one call site was outside the reported scope). Fix the
  same way as `BatchWorker`/`RefinementWorker`: pre-flip the mask (not the
  image — `spec.TransOpt` already handles the image) before it reaches
  `build_geom`.
- **Cross-cutting:** multi-detector merge, energy-sweep calibration.

## Inherited from PR #7 (merged 2026-09-03, `092fbba`/`46e0fec`)

- **No test coverage for two of the PR's six new modules** —
  `job_queue.py` (background batch-integrate job queue, `screen`-backed) and
  `peak_fit_panel.py` (GSAS-2 peak-fit view). Skipped deliberately: useful
  tests need a live `screen` session or a full widget harness, which was out
  of scope for the review session. Import-only smoke tests were rejected as
  false assurance. `provenance.py`, `zarr_cake.py` and `cake_params.py` are
  covered — see `tests/test_provenance.py`, `tests/test_zarr_cake.py`.
  **Updated 2026-10-03:** the third, `batch_cli.py`, is now partly covered by
  `tests/test_batch_cli_omega.py` and `tests/test_batch_cli_zarr_grouping.py`
  (the omega flags, the zarr-grouping flag, and the tab→argv→cfg round trip
  for both) — but nothing else in that file is exercised.
- **The per-frame output filename change is undeclared.** PR #7 moved Batch
  Integrate / Folder Monitor output from the verbatim frame id to
  `<froot>_<NNNNNN><tag>` (see DECISIONS 2026-09-02). No commit message in
  the PR mentions it. Anyone with a script globbing Batch Integrate output
  is broken by it. Needs a note to junspark, release notes, or both.

## Package-side fixes (for MIDAS maintainers — NOT done in GUI)

P0-1 normalize corrections cake · P0-2 wire Q-uniform into kernels · P1-1 finite
autograd geometry grads · P1-2 robuster tilt in one_shot/bayesian · P2-1 smooth
absorption at μR=1.5 · P2-3 fold `analyze_workflows/` round-trips into package CI.
(GUI already works around P0-1/P0-2/P1-1/P1-2 — see DECISIONS.)

**P3-1 — ~~`im_trans`/`ImTransOpt` not accepted by most calibration pipeline
entry points~~ RESOLVED UPSTREAM in `midas-calibrate-v2` 0.15.0, verified on
0.17.0 (2026-09-11).** Upstream took fix B1 from the P3-4 issue draft:
`CalibrationSpec.im_trans`, populated by `spec_from_v1_params()` from
`v1.extra["ImTransOpt"]`, applied at the entry point of every pipeline through
one shared `io/transforms.apply_im_trans()` (image + dark + mask together, with
`NrPixelsY/Z` re-derived from the transformed shape). `calib._prep_transformed()`
is therefore no longer *required* — but it is still correct and still in use:
the GUI's specs come from `calib.build_v1_params`, which sets no `ImTransOpt`,
so `spec.im_trans` is `()`, the pipelines' `if spec.im_trans:` guard is false,
and the pre-flip is applied exactly once. Removing the workaround is optional
cleanup, NOT a bug fix, and it is the one change that must not be made
half-way: pre-transforming *and* letting the spec carry the transform applies
it twice, silently (upstream flags this as the 0.15.0 behaviour change).
The remaining panel-layout half of this item is unresolved — see below.
Historical detail follows.

Only `midas_calibrate_v2.calibrate()` accepts
`im_trans` as a native kwarg (and flips `image`/`dark` internally, then
derives NrPixelsY/Z from the transformed shape, then seeds — everything
downstream in one consistent frame). `autocalibrate_four_stage`,
`autocalibrate_bayesian`, `autocalibrate_joint`, `first_time_calibrate`, and
`pipelines.single.autocalibrate` (the panel-layout / partial-distortion-
refinement routes `calib.py` uses) have **no such parameter** — confirmed
via `inspect.signature()` against the installed package.

Additionally (found 2026-08-27): `calibrate()`'s own `panel_layout` support
doesn't help either — it runs panel refinement internally
(`pipelines/auto.py:587-638`, computes `panel_delta_*` into `cr.unpacked`)
but the final `return AutoCalibrationResult(...)` never copies those keys
out, and the dataclass has no fields for them. So even `calibrate()` can't
be used for panel-layout calibration without losing the per-panel
shift/rotation output the GUI needs (its panel_shifts.txt export). Upstream
fix needs **either** `im_trans` added to the four entry points above, **or**
`panel_delta_*` exposed on `AutoCalibrationResult` (either one would let
`calib.py` stop manually pre-flipping pixels for panel-layout calibration).
**Status 2026-09-11:** the `im_trans` half landed (above); the `panel_delta_*`
half did NOT — `AutoCalibrationResult` still has no panel fields (verified on
0.17.0), so the `four_stage` re-route in `calib.run_pipeline`'s one_shot branch
stays necessary. Tracked with P3-3, which is the same underlying gap.

**Correction (2026-08-27):** the previous note here ("`midas_gui`'s
`calib.py` already works around this correctly") was wrong. The manual
pre-flip workaround had a real bug: it flipped the image for the *solve*
call but computed the auto-seed from the *unflipped* image in the same
branch, so seed and solve ran in two different frames whenever a transform
was active — exactly the failure a user hit with Flip Z + Multi-panel
detector. Fixed in `calib.py` via `_prep_transformed()`, used consistently
for seed + solve + dark in every affected branch. See DECISIONS 2026-08-27.
This upstream ask stood until 0.15.0; the workaround is now belt-and-braces
rather than load-bearing (see the RESOLVED note at the top of P3-1).

**P3-3 — `midas_calibrate_v2.compat.to_integrate.spec_from_calibration_result()`
has no panel-layout support (found 2026-08-27; still true in 0.17.0,
re-checked 2026-09-11 — all 7 panel fields still come back 0/[]/'' and
`AutoCalibrationResult` still has no panel fields).** It sets none of
`IntegrationSpec`'s 7 panel fields (`NPanelsY/NPanelsZ/PanelSizeY/
PanelSizeZ/PanelGapsY/PanelGapsZ/PanelShiftsFile`) from an
`AutoCalibrationResult` — the same gap `TransOpt` already had (confirmed:
`_build_spec` already manually patches `spec.TransOpt` after calling this
function, for the same reason). `midas_gui` now works around it by
stashing `result.panel_layout`/`result.panel_shifts_path` (plain,
JSON-safe attributes; see DECISIONS 2026-08-27 "Feed Calibrate's
Multi-panel results to downstream integration") and patching the spec
itself via `helpers._apply_panel_fields()` everywhere a spec is built from
a calibration result. Upstream fix: teach `spec_from_calibration_result`
to read panel fields off the result the same way it already will need to
once P3-1's `panel_delta_*` exposure on `AutoCalibrationResult` lands —
these two upstream asks are related (both are "the panel_layout config +
refined shifts don't survive on `AutoCalibrationResult`").

**P3-2 — no `apply_trans_opt` hook on `*BinGeometry.from_spec(spec,
mask=mask)` (found 2026-08-25; still absent in `midas-integrate-v2` 0.7.1,
re-checked 2026-09-11 — `midas-integrate-v2` is unchanged at 0.7.1 and `azimuthal_sigma_clip()` still lacks it too).** Every `midas_integrate_v2.integrate_*`
function accepts `apply_trans_opt=True` (default) and flips the *image*
internally via `spec.TransOpt`. Geometry construction itself
(`HardBinGeometry`/`SubpixelBinGeometry`/`PolygonBinGeometry.from_spec`) has
no equivalent — a `mask=` array passed to `from_spec()` is evaluated
directly against the untransformed pixel-index grid, with no way to ask it
to honor `spec.TransOpt`. `midas_gui` therefore must keep manually
pre-flipping every mask in Python (once, before `from_spec`/`build_geom`)
even though it never flips images anymore — see DECISIONS 2026-08-25
("backend does the flip"). Upstream fix would be an `apply_trans_opt` param
on `from_spec()` itself, mirroring the `integrate_*` functions.

**P3-4 — ~~`im_trans` and the `tx/ty/tz` seed are honoured by *disjoint* sets of
calibration entry points~~ RESOLVED UPSTREAM in `midas-calibrate-v2` 0.14.0-0.17.0
(filed 2026-09-08 against 0.13.0; fixed upstream 2026-09-10, verified here
2026-09-11).** Upstream implemented fixes A, B1, C and D from the draft
essentially as written:
* **A** — `calibrate()` gained `initial_tx/ty/tz`. Note the GUI has always passed
  these speculatively through `calib._supported_kwargs`, which silently dropped
  them on <=0.13.0; they now take effect. That is the behaviour change this bump
  introduces, and it is what `calib.run_pipeline` always intended.
* **B1** — `CalibrationSpec.im_trans` + one shared `io/transforms.apply_im_trans`,
  reaching every pipeline (see P3-1).
* **C** — `AutoCalibrationResult.im_trans`, carried into `IntegrationSpec.TransOpt`
  by `spec_from_calibration_result`. `helpers._build_spec` still assigns
  `spec.TransOpt` itself; that is now redundant but harmless — both read the same
  `result.im_trans`, and `TransOpt`/`NrTransOpt` were verified consistent.
* **D** — `first_time_calibrate()` gained `im_trans` and `tx/ty/tz`. **`im_trans` is
  now wired up (2026-09-11):** `calib.run_pipeline`'s `first_time` branch forwards
  the codes and hands over the RAW frame (the backend flips image/dark/panel_mask
  itself and re-derives `n_pixels_y/z`), so it must NOT also pre-flip. Fixing that
  exposed a second, wider bug: `workers.CalibrationWorker` took `NZ, NY =
  image.shape` off the **raw** image and passed it to `normalize_result`, so on a
  non-square detector with a transpose active every non-plain-one_shot mode
  (first_time, four_stage, bayesian, joint, partial-distortion) recorded
  `NrPixelsY/Z` for a detector it never fitted. Now `calib.effective_pixel_counts()`.
  Covered by `tests/test_first_time_im_trans.py` (19 tests, mutation-checked
  against both the dropped-codes and double-flip failure modes).
  **Still open:** the GUI does not pass first_time a *tilt* seed — 0.17.0 accepts
  `initial_tx` and feeds `tilt_prior_deg` into `ty/tz`, so `tilt_seed_effective(
  "first_time") is False` is now a GUI limitation rather than a backend one, and
  `test_calib_tilt_seed.py`'s "at any backend version" wording is stale.
* **Declined:** `refine_tx`. Upstream's reasoning is now documented in
  `forward/geometry.py`: `tx` reaches a ring radius only through the azimuthal
  harmonics, so with them free `(tx, phi_k) -> (tx + d, phi_k + k*d)` is an exact
  gauge orbit and refining `tx` corrupts all six phases with no residual
  signature. Seed it from grain spots or Friedel pairs, don't fit it from powder.

Historical statement of the problem follows.
The generalisation of P3-1/1b: `im_trans` reaches only `pipelines.auto.calibrate()`
(and `ff_calibrate.calibrate_ff_from_files()`, which delegates to it); a `tx/ty/tz`
seed reaches only the eight pipelines taking `v1_params` (carried as `init` by
`spec_from_v1_params`). **No entry point accepts both**, so a flipped detector with
a known non-zero tilt cannot be described to any single pipeline —
`calib.py:_prep_transformed()` exists precisely to work around this.
`first_time_calibrate()` honours neither (`tilt_prior_deg` only steers the
cone-aware BC seed; `_build_v1` never sets tilts). `calibrate()` further hardcodes
`tx=ty=tz=0.0` (`pipelines/auto.py:612`) and takes no `spec=` override, and
`compat/from_v1.py:49` freezes `tx` at `refined=False`, so `tx` is structurally
unreachable through `calibrate()` even though `forward/geometry.py` models it.
`AutoCalibrationResult` also does not record the applied `im_trans`.
**Upstream issue drafted** (verified matrix + runnable repro + proposed fixes A–D):
`.context/issue_draft_calibrate_v2_tilt_imtrans.md`. Ask is
additive/backward-compatible: `initial_tx/ty/tz` on `calibrate()`, an `im_trans`
field on `CalibrationSpec` (or the kwarg on all pipelines), one shared transform
helper factored out of `auto.py:418-439`, and `im_trans` recorded on the result.

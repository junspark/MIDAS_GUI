# Questions for colleagues

Running log of things to raise about MIDAS_GUI / MIDAS backend behavior.

---

## 1. Why are the tx/ty/tz initial guesses ignored during calibration? — **RESOLVED**

**Resolved 2026-09-24 by the backend.** The installed `midas-calibrate-v2` 0.17.0
exposes `initial_tx`/`initial_ty`/`initial_tz` on `calibrate()`, defaulting to
`0.0` — exactly the Tier 2 ask below, signature and default included. Nothing in
the GUI had to change: `calib.tilt_seed_effective()` was deliberately written to
introspect the installed signature at call time rather than hardcode today's
answer, so `_supported_kwargs()` now passes the seed through on the plain
One-shot path and the pre-run "your tilt seed will be dropped" warning stopped
firing by itself.

Two caveats before treating this as closed for good:

* **Not yet confirmed against a real fit.** That the kwargs are accepted is
  verified; that a seeded tilt measurably moves the solution is not. Worth one
  deliberate run with an obviously-wrong tilt seed to confirm the answer shifts.
* **First-time is still unseeded.** `first_time_calibrate()` never gets a tilt
  seed from the GUI at all, independent of this fix, and
  `tilt_seed_effective("first_time")` still returns False. If that pipeline
  should honour a seed, it is a separate ask.

The original question and its investigation are kept below, unedited, because
the per-pipeline branch map is still accurate and still the reason
`tilt_seed_effective()` is shaped the way it is.

### Original question


While running a one-shot calibration from the GUI, the log printed:

```
[calib] note: calibrate() does not accept initial_tx, initial_ty, initial_tz in this backend version — ignoring.
[calibrate] STAGE 1: seeding from CeO2 rings...
[calibrate]   user-provided BC=(1427.100, 1342.400)  Lsd=1049.836 mm — auto-seed bypassed
[calibrate]   BC=(1427.100, 1342.400)  Lsd=1049.836 mm  (0 arcs, 0 ring matches, 0.0s)
[calibrate] STAGE 2: autocalibrate + residual map...
[autocalibrate] RhoD resolved to 317105.6 µm (×px (was pixels))
```

The GUI accepts user-entered `initial_tx`/`initial_ty`/`initial_tz` values, but the installed
`midas-calibrate-v2` backend (0.10.0, from PyPI) doesn't accept those kwargs in its `calibrate()`
signature, so they're silently dropped (only a log note, no UI warning).

- Is this a known/expected limitation of this backend version, or a regression vs. an older API?
- Is there a different parameter name/mechanism for seeding tilt in this version?
- If it's expected to stay unsupported, should the GUI disable/grey out those input fields
  instead of accepting values it can't use?

**Version check (2026-08-27):** diffed `calibrate()`'s full signature between the installed
`midas-calibrate-v2` 0.10.0 and the latest PyPI release, 0.11.0 — `initial_tx`/`initial_ty`/
`initial_tz` are absent from both. Upgrading does not fix this; still an open question for
colleagues. (0.11.0 does add `mask`, `eta_bin_size`, `r_bin_size`, `peak_width_um`,
`weight_by_radius`, `doublet_separation_px`, `outlier_factor` — none related to tilt seeding.)

**Not fixed by the `fix/calibrate-load-file-tilts` branch:** that branch fixes a related but
separate bug — `_load_calib_file()`/`apply_geometry()` in
[tab_calibrate.py](midas_gui/tab_calibrate.py) weren't populating the tx/ty/tz *seed spinboxes*
from a loaded file, even though the file's values were parsed correctly. That's now fixed, so
the seed fields honestly show what's in the file. But this item is about a different, later
step: even with the seed fields correctly populated, clicking "Run" still calls `calibrate()`
with `initial_tx/ty/tz` filtered out by `_supported_kwargs()` before the backend ever sees them
— the values still can't influence an actual calibration run. This item stays open.

**Narrowed scope (2026-08-31), Tier 1 fix applied:** traced every pipeline branch in
[calib.py](midas_gui/calib.py) `run_pipeline()` — tilt seeding is **not** universally broken.
`four_stage`/`bayesian`/`joint` (and `one_shot` when routed through the lower-level
`build_v1_params()`/`CalibrationParams` path — i.e. Multi-panel enabled, or Distortion
refinement restricted to a coefficient subset) all correctly seed tx/ty/tz today, because
`CalibrationParams` takes them directly. The gap is specific to the **plain One-shot** path
(direct `midas_calibrate_v2.calibrate()` call) and to **First-time**, which never attempts to
pass a tilt seed at all. Added `calib.tilt_seed_effective(mode, panel_layout, refine)` (mirrors
`run_pipeline`'s branching, checks the installed `calibrate()` signature at call time rather
than hardcoding today's answer) plus a pre-run warning in `tab_calibrate.py` that fires only
when a non-zero tilt seed is set AND the current pipeline/settings would silently drop it —
so users get a clear, specific warning instead of a buried log note, and it stops firing on its
own once/if Tier 2 below lands upstream.

**Tier 2 proposal for colleagues:** since `CalibrationParams`/`build_v1_params()` already
accepts and uses tx/ty/tz as a genuine LM seed (used today by four_stage/bayesian/joint/panel/
partial-distortion), this isn't a new capability request — it's asking `calibrate()`'s plain
one_shot path to plumb through to the same internal mechanism its sibling pipelines already
use. Concrete ask: add `initial_tx`/`initial_ty`/`initial_tz` to `calibrate()`'s signature
(mirroring the existing `initial_BC_y`/`initial_BC_z`/`initial_Lsd` convention already there),
defaulting to `0.0` to preserve current behavior when omitted.

---

## 2. Can we get pseudo-strain resolved over both ring *and* azimuth?

The "Ring Residuals" tab plots Δr = r_obs − r_pred (px) as one bar per ring
(30 rings, RMS Δr = 1.387 px in the example run), with one clear outlier ring
sitting at ~−6.5 px (screenshot on file, not embedded here).

Looking at [widgets.py](midas_gui/widgets.py) (`ResidualBarChart.set_data`), this Δr is
computed from a single **azimuthally-integrated** radial profile — it finds one peak per
predicted ring in the collapsed 1D profile, so it can't distinguish "this ring is
off in Lsd/BC" from "this ring is fine on average but distorted anisotropically
by tilt/distortion" (a pseudo-strain that varies with azimuth η).

The refinement backend already bins in both r and η internally for the
`eta_uniformity` loss (`r_bin`, `eta_bin` in
[workers.py](midas_gui/workers.py) `AutoCalibrationWorker`) — so an (r, η)-resolved
signal exists somewhere in the pipeline, just not surfaced.

- Can we extract per-(ring, azimuth-bin) Δr — or pseudo-strain ε = Δr / r_pred —
  as a 2D map, the same way the "Eta vs R Cake" tab already renders intensity
  vs (R, η)?
- Would this reuse the existing r_bin/eta_bin machinery from the autocalibrate
  stage, or does it need a separate azimuthally-binned peak-fit pass?
- Is a 2D heatmap (ring index × azimuth) the right visualization, or would
  colleagues rather see it as an overlay on the existing cake plot?

**Version check (2026-08-27):** the latest `midas-calibrate-v2` (0.11.0) adds no new
azimuth-resolved output — only two new *scalar* fields, `post_residual_strain_median_uE`
and `post_residual_strain_trim_uE` (still single numbers, not per-ring/azimuth). However,
`AutoCalibrationResult.residual_corr_map` — a full per-pixel `[NrPixelsZ, NrPixelsY]`
residual correction map — already exists in the *currently installed* 0.10.0 and is already
captured by the GUI in [calib.py](midas_gui/calib.py) (today only used to write out
`ResidualCorrectionMap` files). This map could likely be re-binned into (r, η) polar
coordinates with the same geometry transform the "Eta vs R Cake" tab already uses —
meaning this might be buildable in the GUI now, without a backend change or colleague
input. Worth prototyping before asking.

**Implemented (2026-08-31), no longer an open question — see PR
[feature/strain-cake-lab-axes](https://github.com/d-beniwal/MIDAS_GUI/compare/main...junspark:MIDAS_GUI:feature/strain-cake-lab-axes):**
new "Strain Cake" tab in Calibrate with two selectable sources:

- **Model** — the `residual_corr_map` prototype above, rebinned onto the cake's (R, η)
  grid via `IntegrationWorker` (`apply_trans_opt=False` — the map's already in the
  im_trans-applied frame). Dense but interpolated/smoothed.
- **Ring (η-resolved)** — turned out to be the better match for "what is strain here":
  the exact same local-peak-near-predicted-radius measurement `ResidualBarChart`
  already makes per ring, just run independently per azimuth row instead of once on
  the collapsed profile (`ring_azimuth_residual()` in `widgets.py`). This is the direct
  deviation from the calibrant's actual known ring position (tied to its real
  d-spacings) — no smoothing/interpolation, and needs no backend data at all, just
  `cake_2d` + predicted ring radii already available. X-axis is ring index (rings
  aren't evenly spaced in R).

Both display as ΔR (px) or pseudo-strain (µε, matching `post_residual_strain_uE`'s
units), diverging colormap centered on zero, NaN (not 0) for bins with no data.

**Correctness fix (2026-08-31), two rounds.** First pass: the strain (µε) mode used
the small-angle approximation ε ≈ Δr/r_pred (Δr = r_obs−r_pred). Replaced with an
exact Bragg-law formula, ε = sin(θ_pred)/sin(θ_obs) − 1 — reasoning at the time was
that the small-angle ratio has the wrong *sign*, not just imprecise magnitude.
Verified against a manual per-sample calculation on real data (exact match), after
fixing a bug the rewrite introduced (Ring mode's plot x-axis is the ring *index*, not
the real radius — the physics needs `self._ring_radii`, not that index).

Second pass, after checking what the calibration's own LM fit actually minimizes
(`midas_calibrate_v2/loss/pseudo_strain.py`: *"Per-spot pseudo-strain residual:
1 - R_obs / R_pred. This is the v1 calibrant cost."*) — the "wrong sign" conclusion
above was an overgeneralization from one specific sign convention (Δr = r_obs−r_pred),
which really does flip sign vs. the exact Bragg relation, to *any* radius-ratio strain.
The optimizer's own convention, `1 − R_obs/R_pred`, is the exact negative of that
and comes out sign-**correct** (verified by Taylor expansion: exact strain ≈
+cos²θ·cos(2θ) · (1−R_obs/R_pred), → 1 at small angles). Since R_obs already carries
the full geometry+distortion+parallax model and R_pred is the exact ideal Bragg
radius, this ratio is self-consistent and — critically — it's the exact same quantity
`post_residual_strain_uE` already reports elsewhere in the app, so there's one
"strain" definition in MIDAS_GUI, not two computed differently. Final implementation:
`radius_ratio_strain_ue() = 1e6·(1 − R_obs/R_pred)`, no exact-Bragg trig, no
Lsd/pixel-size inputs needed (the ratio is dimensionless).

**Still open, not yet fixed:** Model mode (the `residual_corr_map` source) can
double-count that map's correction whenever a calibration was run with an output
folder specified (`build_residual_corr=True`, the default) — the cake's own R-axis
already has the map baked in via `pixel_to_REta`'s geometry (verified: shifts R by up
to ~0.9px, mean |shift| ≈ 0.26px on real data), and Model mode's strain calc adds the
same map on top a second time. Ring mode is unaffected (self-contained peak-fit
measurement, doesn't depend on whether the map was baked into the geometry) and is
the default source for this reason.

---

## 3. What's the actual difference between Tab 2 "Calibrate" and Tab 3 "Calib. Refinement"?

From reading [tab_calibrate.py](midas_gui/tab_calibrate.py) and
[tab_refine.py](midas_gui/tab_refine.py) / [workers.py](midas_gui/workers.py), the two
tabs use genuinely different algorithms, not just "coarse pass then fine pass" of the same method:

- **Tab 2 (`CalibrationWorker`, calls into `midas-calibrate-v2`)** — seeds geometry from
  detected CeO2 **Bragg rings/arcs** (peak positions in the raw image), then fits
  Lsd/BC/tilts/distortion to match observed vs. predicted ring radii. This is the
  ring-detection-based pipeline discussed in items 1–2 above (one-shot / first-time /
  four-stage).
- **Tab 3 (`RefinementTab` → `RefinementWorker`)** — takes Tab 2's result as a starting
  point and refines geometry by gradient descent (Adam/L-BFGS, via `midas_integrate_v2`'s
  differentiable integration path), minimizing an **η-uniformity loss** — i.e. it
  doesn't look at ring peak positions at all, it minimizes azimuthal intensity variation
  along each ring in the *integrated* profile. Docstring calls this "closing the gap
  between Bragg-spot calibration and profile-level accuracy."

So: Tab 2 = discrete peak/geometry fit; Tab 3 = continuous, profile-level gradient
refinement seeded from Tab 2's answer. Worth confirming with colleagues:

- Is Tab 3 meant to be run on *every* calibration, or only when Tab 2's ring residuals
  (see item 2) show it's needed?
- Since Tab 3's loss is azimuthal-uniformity, does it correct the same kind of error as
  Tab 2 (Lsd/BC/tilt), or does it fix a different failure mode (e.g. distortion Tab 2
  can't see because it only looks at ring radii, not azimuthal shape)?
- Should the GUI say this more explicitly in the tab itself (e.g. a one-line subtitle),
  since "Calibrate" vs. "Calib. Refinement" reads like the same step twice?

---

## 4. Can the Batch Integrate output-format picker be checkboxes instead of a dropdown?

The "Format:" dropdown in Batch Integrate (`OUTPUT_FORMATS` in
[constants.py](midas_gui/constants.py)) only allows one of CSV / XYE / FXYE / DAT / HDF5 /
2D CSV at a time. Often we want more than one representation of the same integration
run (e.g. CSV for our own plotting *and* FXYE for GSAS-II) without re-running the batch job
twice.

Note this isn't purely a widget swap — `fmt` is threaded through as a single string
into the workers (`FolderMonitorWorker`/batch worker in
[workers.py](midas_gui/workers.py), down to `write_profile(base, fmt, ...)`), and HDF5
is special-cased separately (writes the full stack, not a per-frame profile file). So
supporting multi-select means the worker loops over the checked formats and calls
`write_profile` once per format per frame, not just changing `QComboBox` → checkboxes.

- Is multi-format output actually wanted often enough to justify this, or is
  re-running Batch Integrate with a different format acceptable (it reuses the
  cached geometry, so it's presumably cheap)?
- If we do this, should HDF5 be allowed to combine with the others, or does its
  "full stack" nature make it mutually exclusive?

**Resolved (2026-08-31) — already implemented upstream, no longer open.** Confirmed
in the current code: `OutputFormatSelector` (`tab_batch.py`) is checkboxes, not a
dropdown, `self._fmt.checked_keys()` returns a set, and `write_frame_profiles()`
(`workers.py:262`) writes one file per selected format per frame. HDF5 *is* allowed
to combine with the others (answering the second question above) — except in
Multi-azimuth mode, where it's silently skipped; see item 7.

---

## 5. Batch Integrate fails on a folder of per-frame HDF5 files ("no TIFF files matched")

Reproduced with a folder of per-frame `.vrx.h5` files (e.g.
`.../C611_017Fe_1_load3/C611_017Fe_1_load3_009243.vrx.h5`, dataset `exchange/data`,
shape `(10, 2880, 2880)`). Batch Integrate raised:

```
Traceback (most recent call last):
  File ".../midas_gui/workers.py", line 847, in run
    source = self._open_source()
  File ".../midas_gui/workers.py", line 788, in _open_source
    return TIFFGlobSource(c["path"])
  File ".../midas_integrate_v2/streaming/frame_source.py", ...
    raise FileNotFoundError(f"no TIFF files matched glob pattern {glob_pattern!r}")
```

**Root cause (traced in the GUI code, not the backend):**
`DataLoaderPanel.source_cfg()` ([widgets.py:2869](midas_gui/widgets.py)) only special-cases
a *single file* with an HDF5 extension (`is_h5(raw)` → `{"type": "hdf5", ...}`); anything
else — including a directory — falls through to `{"type": "tiff_glob", "path": raw}`.
`TIFFGlobSource` then only globs `*.tif`/`*.tiff`, so a folder that holds only `.h5` files
matches nothing and fails immediately, before any frame is touched.

This is inconsistent with the GUI's *own* preview/browse loader —
`_collect_frame_paths` in [helpers.py:268](midas_gui/helpers.py) already globs
`*.h5`/`*.hdf5`/`*.ge*`/`*.cbf`/`*.edf` (not just `*.tif`/`*.tiff`) when scanning a
folder — so browsing this same folder and previewing frames in the Data panel works
fine; only the actual Batch Integrate run breaks.

- Is "folder of per-frame HDF5 files" (as opposed to one big multi-frame HDF5, or a
  folder of TIFFs) an intended, supported input for Batch Integrate at all? If so,
  `midas_integrate_v2.streaming` presumably needs a source class for it (or the GUI
  needs to iterate file-by-file itself and feed frames some other way).
- Separately: should `source_cfg()`/`_open_source()` validate the resolved file list
  *before* starting the worker thread and show a clear inline message ("no matching
  files"/"unsupported source shape"), instead of a raw Python traceback dialog?
- Still need to confirm: does pointing "File…" directly at one `.vrx.h5` (rather than
  the folder) also fail, or does that route correctly to `HDF5FrameSource`? The code
  suggests it should work — if it still fails the same way, that's a second, separate
  bug worth its own repro.

**Version check (2026-08-27):** the latest `midas_integrate_v2` (0.7.0) has the identical
set of streaming source classes as 0.6.0 (`TIFFGlobSource`, `HDF5FrameSource`,
`GEBinaryFrameSource`, `EDFFrameSource`, `ZarrFrameSource`, `NumpyArraySource`,
`TriggerTaggedFrameSource`) — no new "folder of per-frame HDF5" source was added.
Confirms this needs a GUI-side fix in `source_cfg()`/`_open_source()`, not a version bump.

---

## 6. "Show rings" checkbox does nothing at the Pick BC / Pick Ring seeding stage

On Tab 2 (Calibrate), before running a calibration — while still at the "Pick BC" /
"Pick Ring" seeding stage on the raw image — toggling "Show rings" has no visible
effect. Traced in [tab_calibrate.py](midas_gui/tab_calibrate.py):

- `_draw_rings(result)` is the function that actually creates the predicted-ring
  overlay (green circles + BC marker) and is only called from `_on_done()`, i.e.
  **after a calibration run completes**.
- `_on_show_rings_toggled()` — what the checkbox itself triggers — only sets
  `.setVisible()` on whatever's already in `self._ring_items`. Before any
  calibration has run, that list is empty, so there's nothing to reveal.

So today "Show rings" means "show predicted rings from the last completed
calibration result" (a post-run QC overlay), not "preview predicted rings for my
current seed guess before running." The bright arcs visible in a raw calibrant image
are the actual diffraction data, not a drawn overlay.

The checkbox's placement right next to "Pick BC"/"Pick Ring" — tools used *before*
running calibration — makes it easy to expect it to help with seeding itself (e.g.
overlay predicted CeO2 ring positions from the current BC_y/BC_z/Lsd seed spinboxes,
to sanity-check a pick before committing to a run).

- Is a live pre-run preview (predicted rings from current seed values, updating as
  BC/Lsd/tilt seed fields change) a wanted feature, or is "Show rings" intentionally
  scoped to post-calibration QC only?
- If wanted, would it reuse `_predict_ring_radii()` (already used by `_draw_rings`)
  fed from the seed spinboxes instead of a `result` object?

---

## 7. Batch Integrate silently skips HDF5 output in Multi-azimuth mode, with no upfront warning

Reproduced: ran Batch Integrate with **Multi-azimuth output (cake)** checked and
Output format = **CSV, HDF5**, 10 frames, η bin 5° (→ 72 sectors). The run completed
normally and the output folder had ~720 CSV files (`<frame>_etaNNN.csv`, one per
frame per azimuthal sector) — but **no `.h5` file at all**, with nothing in the UI
before or after the run indicating HDF5 hadn't been written.

**Traced the cause:** `workers.py:1073-1080` — HDF5 is unconditionally skipped
whenever Multi-azimuth is on (`midas_integrate_v2.write_h5` expects one profile per
frame, not one per sector), logging only `"[batch] Note: HDF5 output isn't written
in multi-azimuth mode..."` to the Log tab.

**The inconsistency:** this is the *second* known incompatibility between Multi-azimuth
and another option — the first (Q-uniform bins) is handled completely differently.
`tab_batch.py:583-588` actively **blocks** Q-uniform + Multi-azimuth with a
`QMessageBox.warning()` dialog *before* the run starts ("Multi-azimuth output isn't
supported together with Q-uniform bins yet. Uncheck one of them.") and refuses to
proceed. HDF5 + Multi-azimuth gets no such dialog — the run just proceeds and quietly
produces less than what was asked for.

- Should HDF5 + Multi-azimuth get the same eager `QMessageBox` treatment as
  Q-uniform + Multi-azimuth, for consistency? (Simplest fix, and matches an
  already-established pattern in this exact file.)
- Alternative: greyed-out/auto-unchecked HDF5 checkbox in the Output-format picker
  whenever Multi-azimuth is toggled on, so the incompatibility is visible before
  you even hit Run rather than discovered by a missing file afterward.

---

## 8. Calibration `tol*` windows bound each E-M iteration, not the answer — can we get a true cap?

Reported at the beamline on a 1-ID CeO2 frame: *"the range limits are not respected
when optimizing for instrument parameter. Lsd was set to 2390mm with +/-2 mm but the
value converged to 2382 mm."*

**The limits are applied; they just don't mean what the number suggests.**
`midas_calibrate_v2/pipelines/single.py` (installed 0.17.0, lines 181-193) re-centres
every refined parameter's bounds on that iteration's result before the next LM call:

```python
# Bounds move with the value: they were built as init +- tol, so
# leaving them anchored to the original seed would let the trust
# region drift off-centre as the fit walks.
half = 0.5 * (float(prm.bounds[1]) - float(prm.bounds[0]))
prm.bounds = (new_init - half, new_init + half)
prm.init = new_init
```

So `tolLsd` is a per-iteration trust region. A fit that rails against it moves one
full window per E-M iteration, and the run's total excursion from the seed is
`n_iter × tol`. Measured on the beamline runs, exact to the micron:

| run | seed (µm) | tolLsd | n_iter | predicted floor | observed |
|---|---|---|---|---|---|
| attempt_0016 | 2390051.00 | ±2 mm | 4 | 2382051.00 | **2382051.00** |
| attempt_0025 | 2390351.79 | ±1 mm | 4 | 2386351.79 | **2386351.79** |

In both, every iteration stepped by exactly `-tol`, i.e. the bound was binding each
round and the fit genuinely wanted to go much further.

**Why this matters beyond the labelling.** For Hydra, the four GE panels are mounted
on a single frame at a fixed sample distance, so a per-panel Lsd cannot physically
deviate by more than a few mm — 5 mm is already generous. Calibrating panels
individually therefore needs `tolLsd` to be a *constraint on the answer*, not a step
size. With the current semantics, a ±5 mm window at the default 4 iterations permits
±20 mm, which is far outside what the hardware allows.

**What the GUI does for now.** `CalibrationTab` has an opt-in "± values are a hard cap
on the whole run" tick that divides each `tol*` by `n_iter` before building the config,
so `n` re-centred iterations cannot carry the answer past the number the user typed.
That is exact, but it is arithmetic around the backend rather than a real constraint,
and it has a cost: the per-iteration trust region shrinks by `n`, which slows
convergence for a parameter that legitimately needs to travel (a bad BC seed, say).

Questions:

- Is the re-centring intended to be unbounded, or should the walk be clamped to the
  original seed ± tol as a backstop? The comment argues for a trust region that
  follows the fit, which is reasonable for conditioning — but a trust region and a
  physical constraint are different things and `tol*` is currently serving as both.
- Could `CalibrationSpec`/`Parameter` carry a separate hard bound alongside the
  moving one — e.g. `Parameter.hard_bounds`, clamped after each re-centre — so a
  caller can say "step at most 1 mm per iteration, but never leave ±5 mm of the
  seed"? That is what a mechanically constrained geometry actually needs, and it
  would let the GUI drop the division hack.
- Failing that: would you accept the per-iteration meaning being documented on
  `autocalibrate`'s docstring? Nothing currently says the window moves, and the
  natural reading of "tolerance" is a bound on the result.

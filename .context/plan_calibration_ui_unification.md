# Plan — unify the single-detector and Hydra calibration interfaces

Written 2026-10-07 from a beamline session. Not started; nothing in this
file is implemented. Agreed scope, references and findings, recorded so it
can be executed without reconstructing the reasoning.

## Resolved first: feedback was erasing tx (fixed, not yet pushed)

Found while chasing the degrading calibrations of 2026-10-07 and fixed
ahead of this plan, because it was actively corrupting runs. "Feed result
back to seed" is on by default and wrote `result.tx` into the seed spin
after every Run. No powder pipeline refines tx -- it is frozen at
`midas_calibrate_v2/compat/from_v1.py` -- and `first_time` does not even
carry the seed, so `result.tx` came back 0 for a panel physically at
296.885 deg. Each Run therefore replaced the measured azimuth with 0, and
the tx tick then read as "not set", so the next fit ran at the wrong
azimuth without saying so. Observed across attempt_0003 -> attempt_0005:
Lsd 2768.895 -> 2384.979 mm, post-refine strain 364 -> ~1400 microstrain.

Fix: `helpers.result_refined_tx(result)` -- one predicate, used by both
surfaces, that asks whether the fit reported a 1-sigma for tx (only the
manual d-spacing fit ever does). Feedback writes tx only then.

Worth noting for this plan's purposes: the bug existed identically in
`tab_calibrate._seed_from_result` and
`hydra_calib_widgets.seed_from_result`, two copies of the same six lines,
and had to be fixed in both. That is a fourth divergence-by-duplication in
one day, and the argument for the shared per-detector row below.

Separately ruled out, so it is not re-chased: the ring overlay does NOT
mishandle tx. Measured through `helpers.ring_xy_corrected`, the drawn ring
is the same curve for tx = 0 / 90 / 296.885 -- 0.000 px difference at
matched azimuth, with and without ty/tz tilts. `DetectorGeometryCard`
hardcoding `"tx": 0.0` is therefore harmless for ring rendering. A
green/white ring mismatch is a poor fit, not a rendering bug; the backend
reports `azimuth_coverage 80 deg (22 %)` for a single GE panel, which is
the real cause and needs a joint four-panel fit via `panel_layout`.

## The invariant

In the user's words: *"a lot of things we do per detector we might want to
do for detector array as well for each detector that constitute the
detector array."*

Everything that exists per detector should exist for every detector of an
array. The single-detector Calibrate tab and the Hydra Calibrate page are
currently independent implementations of the same thing, so each feature
is built twice and drifts. Three divergences surfaced in ONE session, all
found by the user at the beamline rather than by a test:

| | single | Hydra |
|---|---|---|
| `set_refine_boxes` on the Manual seed dialog | yes | missing (fixed, 4b4d725) |
| Manual ring-picking card | yes | still missing |
| `tx` seed range | was ±180 | was ±180 (both fixed, 55876dc) |
| Mean of frames | plain `QGroupBox` | `make_card` |
| Card names, Run presentation | differ | differ |
| `panel_layout` joint fit | yes | not wired |

Fixing these one at a time is treating symptoms. **Nothing enforces
parity** — that is the defect.

## Design: one row per detector, colour-coded

Take the layout from mpe_wf_saxs_waxs (`~/opt/mpe_wf_saxs_waxs`). All
three of these are agreed references:

- `gui_caking_launcher.py`
- `gui_bc_launcher.py`
- the cake-parameters editor (nine-column `cake_parameters` CSV, one row
  per detector — Batch Integrate already reads this format)

**One colour-coded row per GE panel, all four visible at once**, carrying
that panel's caking and calibration parameters.

This replaces the `QStackedWidget` that shows one panel at a time. That
single decision is the root of three separate reports in one session:
"unclear what parameters are dedicated to each GE panel", "unclear which
panel is being fitted", "I do not know what Tx they are getting". A
summary line was added as a stopgap (`_update_panels_overview`); the table
is the real fix, because it puts the per-panel/shared split in the layout
instead of in label text that has now been reworded three times and still
had to be explained twice.

It also *is* the refactor: a per-detector row widget used once by the
single-detector tab and N times by Hydra makes parity structural rather
than something a test polices after the fact.

### Steps

1. **Per-detector row widget** — owns transforms, seed, ring-picking,
   result, residuals. Single tab renders one; Hydra renders one per
   selected panel. Colour-coded per panel.
2. **Shared recipe cards** — Pipeline, Detector & Calibrant, Threshold,
   Mean of frames, Refine. One implementation; Hydra adds only its
   "(all 4 panels)" scope suffixes. Note `Mean of frames` is a bare
   `QGroupBox` in the single tab and a card in Hydra — it is checkable,
   and `_avg_check` IS the groupbox, so converting it must keep
   `isChecked`/`setChecked`/`toggled` (state save/restore depends on it).
3. **Parity test** — assert both views expose the same per-detector
   controls, so the next divergence fails CI instead of surfacing
   mid-experiment. This is what makes the refactor durable.
4. Hydra then adds only genuinely array-level things: panel selection, the
   composite, the Overall cake, and the joint fit below.

## Two findings this plan does not fix

**The calibration is wedge-limited, not seed-limited.** The backend's own
check: `azimuth_coverage 80° (22 %)`. One GE panel sees a narrow arc, so
beam centre and tilts trade against Lsd and are weakly determined.
Measured: ticking BC/ty/tz changed the result by 0.6 µε (364.3 -> 364.9)
and moved BC by 0.4 px. Four panels at 80° each is ~320° — pooled, they
would constrain the geometry. `panel_layout` (joint refinement of
`panel_delta_yz` / `panel_delta_theta` / `panel_delta_lsd`) exists in
`autocalibrate_four_stage` and `first_time_calibrate` and is NOT wired
into the Hydra page. **This is the credible route below a few parts in
1e-4**, which is the mpe_wf baseline; more Refine ticks demonstrably is
not.

**Two pipelines silently ignore the Refine card.** `first_time_calibrate`
takes no refine argument at all (`calib.py` warns that parameter *limits*
are ignored there but says nothing about the refine flags), and
`four_stage.py:159` unconditionally re-thaws all 15 distortion harmonics,
which also catches One-shot whenever Multi-panel is ticked. Fix or surface
before anyone tunes pseudo-strain against either.

Also wanted: a **pseudo-strain vs η plot**. Would have shown the wedge
problem at a glance (two-lobed in η = beam-centre error, four-lobed =
tilt). Today only the scalar and a per-ring Δr bar chart exist.

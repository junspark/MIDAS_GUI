# MIDAS GUI — User Documentation

**Version:** 1.0.0
**Application:** `midas-gui` (or `python -m midas_gui`)
**Backends:** `midas_calibrate_v2`, `midas_integrate_v2`, `midas_calibrate`, `midas_hkls`, `midas_distortion`
**Last updated:** 2026-08-31 (Batch Integrate gains Rmin/Rmax exclusion +
a Detector-view preview, single-detector and Hydra. Summary:
- **Rmin/Rmax** (§7 "Integration", both tabs): exclude an inner (e.g.
  beamstop shadow) or outer radial region from integration. Rmin defaults
  to 0; Rmax defaults to "auto" (0 → the farthest detector corner from the
  beam centre, computed by the backend the same way as before) and
  auto-fills to that corner value once a calibration resolves. **Corner**/
  **Edge** buttons next to Rmax fill in the farthest-corner/farthest-edge
  radius from whichever calibration (or, in Hydra mode, whichever
  toolbar-selected panel) is active.
- **Detector view tab** (§7): a new tab next to Waterfall/Stacked profiles
  showing the current source frame with the Rmin/Rmax boundary circles
  always overlaid, plus the full (R, η) integration bin grid — concentric
  circles at each R-bin edge, spokes at each η-bin edge, thinned to ~50
  rings/~72 spokes for legibility — when the new **Show bin grid** checkbox
  is on (off by default). In Hydra mode this is one page-level tab shared
  across the 4 GE panels, refreshed for whichever panel is toolbar-selected
  (kept as a single `ImageViewer` rather than one per panel — see
  `.context/DECISIONS.md` for the pyqtgraph-widget-count teardown-crash risk
  this avoided).

Previously (2026-08-31): Analysis provenance now records the workstation it
ran on; Hydra's Overall Eta vs R Cake now correctly spans the full
azimuthal range. Summary:
- **Workstation provenance** (§17/§17): every Mask/Calibrate/Batch
  Integrate attempt's recorded `environment` snapshot gains a
  `workstation` block — hostname, OS name/release/version, CPU model,
  logical/physical core counts, and total RAM — so a project file can be
  traced back to the exact machine an analysis ran on, even years later.
  Every field is independently best-effort; a lookup failure on an
  unusual platform yields `None` for just that field, never a missing
  snapshot.
- **Hydra Overall Eta vs R Cake now actually spans -180°..180°** (§7,
  Calibrate): each GE panel's cake is now rotated by that panel's own
  `tx` (its physical installation angle about the beam axis — the same
  value the Data Viewer's windmill composite image already uses) before
  the four panels are summed. Panel calibration normally leaves `tx`
  unrefined at 0° (a full powder ring can't constrain a rotation about
  the beam axis), so every panel's cake previously landed on the same
  un-rotated η range and piled on top of each other instead of covering
  the full circle; loading each panel's real geometry (so its `tx` is the
  true ~90°-apart windmill value) now spreads them out correctly. No
  change to non-Hydra cakes or to `tx=0` panels (unchanged, still overlap
  since there's nothing to rotate by).

Previously (2026-08-29): Batch Integrate gets an opt-in **Multi-azimuth
output (cake)** checkbox — see §7 "Integration" — and Results & Export gets
an **Export for GSAS-II** feature — see §13. Summary:
- **Multi-azimuth output (cake)** (off by default, §7): `midas_integrate_v2`
  already returns a full `(η, R)` cake per frame for every kernel — Batch
  Integrate was always collapsing it to one full-circle profile before
  anything downstream saw it, even though the η bin/range fields already
  existed. Checking this keeps every azimuthal sector as a separate output
  profile instead (`profiles`/`sigmas` become `(n_frames, n_eta, n_r)`).
  Left off, nothing changes — η bin's default (5° over 360°, 72 internal
  bins) is only ever used for the existing collapse-weighting, not output
  shape, so no existing run's result size is affected.
- **Export for GSAS-II** (§13): writes one chosen Batch-Integrate attempt as
  a native MIDAS-format zarr, directly importable by GSAS-II's own MIDAS
  zarr reader, plus a provenance sidecar — see §13 for scope/limits.)

**Previously:** (2026-08-29, Mask Builder no longer has its own
Flip Y/Flip Z/Transpose **Transforms** checkboxes — see §4. Masks it produces
are now always in raw detector-space, exactly like a file/folder mask loaded
from disk; Calibrate/Batch Integrate/Refinement apply the active
calibration's `ImTransOpt` to the mask themselves at the same point they
transform (or hand off to the backend to transform) the image it pairs with.
This closes a latent coordinate-frame bug where a mask built with a flip
selected in Mask Builder could be transformed a second time downstream,
silently landing it back in raw orientation misaligned against the
transformed geometry.)

**Previously:** (2026-08-29, Open Project checkbox-tree preview: **Analysis**
now lists **Single detector**/**Hydra** headings with one **Calibrate** row
and one **Batch Integrate** row each, instead of cramming both into one grid
row per panel; **GUI Workspace** tabs are now indented under **Select all**
to show it isn't a tab itself. See §17 for full detail. Also: Project `.h5`
schema redesign — `gui_workspace` (modular, per-tab) + `analysis` (mask/
calibrate/integrate FAIR history) — and a unified Open Project dialog.
Summary:
- **Breaking schema change (clean cutover, `schema_version` 2 → 3):** the old
  single `/workspace` blob is replaced by `/gui_workspace/<tab name>/...`,
  one independently-readable/restorable group per tab; the old
  `/<panel_key>/{calib,integrate}` attempt history moves to
  `/analysis/{calibrate,integrate}/<panel_key>/...`. Project files created by
  an earlier version are **not** readable by this one — Open Project shows a
  clear warning naming the old schema version rather than silently restoring
  nothing.
- **New: Mask Builder gets FAIR provenance too** (`/analysis/mask`, global —
  not per-panel, since Mask Builder is one shared tab). A new **Log to
  Project** button (§4) records the current mask's full parameters plus its
  resulting compressed mask array as an explicit, click-when-ready action
  (unlike Calibrate/Batch Integrate there's no single "run finished" moment
  to auto-log from).
- **New unified Open Project dialog** replaces the old two-stage flow (a
  blind Yes/No "restore everything" box, then a separate calibrate/integrate
  picker): browse to a `.h5` on the left, and the moment one is clicked, the
  right pane previews everything it contains — every `gui_workspace` tab and
  every mask/calibrate/integrate attempt — as one checkbox tree, everything
  checked by default. Recent Projects entries show the same tree without the
  browser pane. Nothing is restored until you click Open on the checked
  selection.
- **File ▸ Project History…** now also lists mask attempts (a "Mask" row,
  panel column shows "—" since it isn't per-panel).)

**Previously:** (2026-08-28, Multi-panel calibration actually refines panel
shifts now, and the result persists wherever the calibration ends up — see
§5 "Export" for full detail. Summary:
- **Root-cause fix:** every pipeline behind **Multi-panel detector**
  (One-shot, First-time, Four-stage, Bayesian, Joint) was silently doing
  nothing with the panel grid beyond fixed-geometry bookkeeping — no
  per-panel δy/δz/δθ was ever actually being fit, so there was nothing to
  save regardless of how the result got exported. Each pipeline now
  registers the per-panel shift as something to refine before running, so
  Multi-panel detector runs produce real, nonzero panel corrections for
  the first time.
- **Save calibration.json** now (re)writes a companion
  `<name>_panelshifts.txt` next to wherever you actually save it, instead
  of just recording whatever path happened to be live when Fit finished —
  which, unless an Output folder was already set, was an anonymous
  temporary file with no guarantee of surviving. The saved `.json` is now
  self-contained: copy or move it together with its sidecar and the panel
  correction still resolves.
- **Save paramstest.txt** now names its sidecar `<name>_panelshifts.txt`
  too (previously a generic `panel_shifts.txt`) — saving more than one
  paramstest into the same folder no longer has them silently share (and
  overwrite) one sidecar.
- If Fit finishes with **Multi-panel detector** on and no Output folder
  set, the Log now says so explicitly (a temporary file was used and
  needs an explicit Save to keep it) instead of leaving it to be
  discovered later, once the temp file may already be gone.
- Loading a calibration file back in (**Load calibration file**, or
  anywhere else that reads a paramstest/`.json` geometry file) now also
  finds its `PanelShiftsFile`/`panel_shifts_path` sidecar when the
  recorded path is stale but the sidecar was copied/moved alongside the
  geometry file itself.
- A Project (`.h5`) calibration attempt's refined panel shifts are now
  **embedded** in the project, the same treatment a live/drawn-in-tab
  mask already got — not just a path reference to whatever (possibly
  temporary) file was live at save time. Reopening a project regenerates
  a real `_panelshifts.txt` next to the project file and points the
  restored calibration at it, so a multi-panel calibration keeps
  integrating correctly instead of silently reverting to zero panel
  shifts.)

**Previously:** (2026-08-28, Batch Integrate cosmetic + Batch-Parallel
overhaul, single-detector **and** Hydra, plus a same-day follow-up making
**Output format**, **Run mode**, and **View calibration** into on-demand
popups — see §7 "Tab 4 — Batch Integrate" for full detail. Summary:
- **Drift correction** is hidden from the GUI (single-detector only —
  Hydra never had it) — not used in production; the code/`DriftWorker`
  are untouched and it can be shown again with a one-line change.
- **Calibration values** popup (click **View calibration ▾**, next to
  Calibration source, same click-to-see-options interaction as the Data
  Viewer's λ/pixel-size labels) now shows the full geometry parameter grid
  every time — its menu is rebuilt fresh on each open, not just refreshed
  in place, so the numeric fields always render (not just the "From Tab 2 /
  From file" source note). Applies to the single-detector card and each of
  Hydra's 4 per-panel cards.
- **Output format** is a checkbox list behind a clickable **Output
  format ▾** button (its own text names the checked formats, e.g. "Output
  format: CSV, XYE ▾") instead of an always-visible list of checkboxes —
  check as many of CSV/XYE/FXYE/DAT/HDF5/2D-CSV as you want; every checked
  format is written.
- **Run mode**'s explanation of what Sequential/Batch Parallel do is now a
  hover tooltip on the "Mode:"/"Per panel:" label and the mode combo box
  itself, instead of a permanent explanatory line taking up card space.
- A green **Save** button (next to a now-red **Abort**, both narrower than
  before alongside a narrower **Start Integration**) writes out the
  lineouts already computed this run, in whichever formats are checked,
  to a folder you pick on the spot — independent of whether an Output
  folder was set before running (that still only wires up incremental
  per-frame writes during the run itself, unchanged).
- A new **Sequential** / **Batch Parallel** run-mode + worker-count
  control splits one run's frames across N concurrent workers sharing one
  detector map built once up front, auto-shrinking the worker count so
  each worker gets at least 10 frames. Hydra gets this as a *second*,
  independent level of parallelism layered under its existing per-panel
  Sequential/Parallel toggle (which is unchanged).)

**Previously:** (2026-08-28, Batch Integrate's Data field gains Browse…
parity with Calibrate/Data Viewer — **Multiple files** and **Files sharing
a name stem** are now offered there too, alongside the existing Single
file/Full folder. A filestem pick is kept as a live folder+prefix filter,
so both a one-shot run and **MONITOR** re-scan for files starting with that
prefix — a new matching file dropped in later is picked up, a non-matching
one is ignored; the info line under the field then reads e.g. `Source:
/data/scan  (filestem: scan_*)`. An arbitrary Multiple-files pick works for
a one-shot run but can't be watched by MONITOR (no folder/pattern to
re-scan) — its info line reads `Source: N file(s) — <shared folder>`. See
§7 "Data Loader panel (left)" → "Browse… options" and §1 "The Browse…
popup" for detail.)

**Previously:** (2026-08-28, Workspace and Project are merged into one
`.h5` file — see §17 "File ▸ Project (session + FAIR provenance)" for full
detail. Summary:
- `Ctrl+S`/`Ctrl+Shift+S`/`Ctrl+O` now save/save-as/open **one Project
  file**: your session (every tab's live fields) lives in its `workspace`
  slot and is overwritten each save, right alongside the append-only
  Calibrate/Batch-Integrate attempt history that already lived there.
  `File ▸ New Project…` is gone — `Save Project As…` to a new filename
  creates one in the same step.
- A Calibrate/Batch-Integrate attempt's mask/dark/bright/background inputs
  are now always **path + hash**, never embedded raw arrays — the one
  exception is a live, drawn-in-tab mask with no file of its own, which is
  still embedded since it has no path to hash. (Previously dark/bright/
  background were always embedded; they're always file-backed already, so
  nothing about reproducing a run changes.)
- A standalone Workspace JSON file from before this change can still be
  brought in via the new `File ▸ Import Legacy Workspace (.json)…`.
None of this touches an existing project's recorded attempt history —
every `attempt_NNNN` record already logged reads back exactly as before.)

**Previously:** (2026-08-27, Browse… popup polish: a Multiple-files/
Filestem pick now shows the files' shared folder — or the single file's own
path if only one matched — instead of an "N files selected" count; the
popup now opens to the `midas-gui` project root by default, with its
name column doubled in width. See "The Browse… popup" in §1 for detail.)

**Previously:** (2026-08-27, Every Data/Dark/Bright/Background field's ⋯
button — single-detector and Hydra alike — now opens a two-item menu,
**Browse…** and **Import from…**; Browse… opens a popup offering Single
file / Multiple files / Full folder / Files sharing a name stem, and Hydra
tabs gained real cross-tab Import from… for the first time.)

**Previously:** (2026-08-27, Calibrate's Multi-panel detector results —
per-panel shifts + grid geometry — now reach downstream integration
properly: see Tab 2 "Export" for detail.)

**Previously:** (2026-08-27, Error dialogs across every tab no longer
truncate the underlying error — see §14 "Common UI Conventions" for detail.
A failure now shows a one-line summary with a **Show Details…** button
revealing the complete traceback, and the same complete text is written to
that tab's log panel, not just the dialog.)

**Previously:** (2026-08-26, Project records are now more self-sufficient,
and two Hydra plots gained an Overall/summed view — see §5, §16-§17 for
full detail. Summary:
- A Project attempt's mask is embedded (compressed) only when it includes
  something hand-drawn/computed in Mask Builder — a mask assembled purely
  from files is referenced by path+hash instead, to keep project files
  smaller. Dark/bright/background are unaffected (always embedded, as
  before).
- A calibration attempt now also embeds its computed Radial Profile / Eta
  vs R Cake, so **Open Project…**'s Populate step shows them instantly with
  no recompute and no need for the original image to still be reachable
  (older attempts fall back to the previous recompute-if-image-loaded
  behavior).
- Calibrate's Hydra **Radial Profile** plot: **Overall** is now a
  green-when-active toggle button (was a checkbox) that hides GE1-4's
  curves and shows their NaN-aware summed profile, rescaling to fit;
  clicking again restores GE1-4. Its **Eta vs R Cake** plot gained its own
  GE1-4 checkboxes plus a matching **Overall** button that computes and
  shows a summed cake across all 4 panels.
- The Data Viewer's single-detector mode gained an **Eta vs R Cake** tab
  next to its Radial Profile (previously cake was only available in
  Calibrate).
- Both a saved **Workspace** and a **Project** now record the active
  beamline **Profile** (§16) alongside them, and restore it automatically
  on load/open if it still exists locally.
None of this renames or removes any existing Project `.h5` field — every
addition is a new, optional dataset/attribute, so older project files keep
reading exactly as before.
- Hydra's **Overall Eta vs R Cake** now genuinely sums all 4 GE panels onto
  a shared (η, 2θ) bin grid (NaN-aware, per-panel geometry-correct 2θ
  conversion) — since each panel physically covers only part of the full
  η range, the Overall cake ends up populated across a much wider η span
  than any single panel, with panel overlaps added together bin-for-bin.
- **Eta vs R Cake** plots' right-click-drag now zooms independently per axis
  (a purely horizontal drag zooms R only, a purely vertical drag zooms η
  only, a diagonal drag zooms both) instead of always zooming η only,
  matching the Radial Profile plot's own right-drag gesture; the mouse
  wheel is unchanged (always zooms both axes). Applies everywhere the Cake
  plot appears: single-detector and Hydra Calibrate, and the Data Viewer.)

**Previously:** (2026-08-26, File menu reworked around a clearer Project/
Workspace mental model, with a recent-files list, an unsaved-changes
indicator, autosave/crash-recovery, and a built-in provenance browser — see
§17 "File ▸ Save/Load Workspace" and §17 "File ▸ Project (FAIR
provenance)" for the full detail. Summary:
- "GUI State" is renamed **Workspace** everywhere in the UI (same
  `Ctrl+S`/`Ctrl+Shift+S`/`Ctrl+O` shortcuts, same JSON file format —
  existing saved files still load unchanged).
- File ▸ **Recent Projects** / **Recent Workspaces** submenus list the last
  10 of each, newest first.
- The window title now shows the active Workspace's name with Qt's native
  `[*]` unsaved-changes marker; closing the window with unsaved changes
  prompts to Save / Discard / Cancel instead of silently closing.
- An autosave draft is written every few minutes while the Workspace has
  unsaved changes; if MIDAS GUI is later relaunched after a crash or forced
  quit, it offers to restore that draft.
- File ▸ **New Project…** can optionally save a Workspace linked to the new
  project in the same step.
- A new read-only File ▸ **Project History…** dialog lists every recorded
  Calibrate/Batch Integrate attempt (panel, kind, timestamp, full recorded
  parameters) for the open project, so inspecting what's in it no longer
  requires an external HDF5 tool.
None of this touches the Project `.h5` file's format or its opt-in,
append-only logging behavior — see §17.)

**Previously:** (2026-08-25, Fixed a geometry/image-orientation mismatch:
Batch Integrate, Pump Probe, Calibration Refinement, and the Data Viewer /
Sim Detector radial-profile preview now correctly account for **ImTransOpt**
(the Flip Y / Flip Z / Transpose transform set on the Calibrate tab) —
previously it was silently dropped once a calibration left Tab 2, so these
consumers integrated the raw, untransformed frame against a geometry that had
actually been fit on a transformed one. The fix passes `ImTransOpt` to the
MIDAS integration backend itself (`IntegrationSpec.TransOpt`, which
`midas_integrate_v2`'s own `apply_trans_opt=True` then applies) rather than
flipping the pixel array in the GUI, so the backend is the single place the
transform is actually performed for every integration call site. The
Batch Integrate "Calibration values" card also gained a visible **ImTransOpt**
row so the active transform is no longer invisible. See §5 "Calibration
values" and §7.)

**Previously:** (2026-08-25, Calibrate's Eta vs R Cake and Batch Integrate's
Waterfall auto-level: vmin% default raised from 1 to 30, and the percentile
calculation now excludes exact-zero bins/pixels — same fix already applied to
the main image viewer — so a partially-empty cake/waterfall no longer skews
the auto-level window toward zero. See §5 and §7.)

**Previously:** (2026-08-25, Data Viewer's image toolbar gained a
**Lab-frame axes** overlay toggle — plots the APS/MIDAS lab coordinate
system (X_Lab/Y_Lab compass, beam-direction glyph, η-sweep arc) anchored at
the beam centre, for verifying orientation/ImTransOpt. See §3.)

**Previously:** (2026-08-25, Calibrate's Radial Profile/Eta vs R Cake and
Batch Integrate's Waterfall/Stacked profiles now bound pan/zoom to their own
data extent, same as the main image viewers; the Cake plot's right-click-drag
now zooms only the η (Y) axis instead of both axes; the Waterfall plot gained
a color-scale histogram sidebar like the main image viewers. See §5 and §7.)

**Previously:** (2026-08-25, Open Project's "Populate from project" now
redraws the recorded run's **results**, not just the input fields:
Calibrate's predicted-ring overlay, Radial Profile, and Eta vs R Cake
reappear immediately (rings need only the stored geometry; profile/cake also
need the run's data file to still be loadable), and Batch Integrate's
Waterfall/Stacked-profiles views are replayed from the attempt's saved
per-frame data — in Hydra mode, independently per GE panel, so switching the
GE1–GE4 toolbar shows that panel's own recorded run. A new bold, high-
contrast **Project: …** label also appears at the far right of the tab-bar
header row whenever a project is open (in addition to the existing quiet
status-bar one), so an active project is now hard to miss. See §5, §7, and
§17 "File ▸ Project (FAIR provenance)".)

**Previously:** (2026-08-25, Data Viewer and Calibrate: loading a Data file
now auto-populates **pixel size** (from the detector encoded in the
filename — `.ge1`–`.ge5` → GE, 200 µm; `.vrx` → Varex, 150 µm; `.pxrd` is
recognized as Pixirad but has no known pixel size to auto-fill) and, for an
HDF5 frame file, **wavelength** (from its recorded beam energy, `keV` →
Å). Only active for the **1-ID-E**, **20-ID-D**, and **20-ID-E** profiles —
these are beamline-specific filename/metadata conventions. Applies to both
single-detector and Hydra mode in both tabs; anything not detected is left
at its previous/default value. Batch Integrate is unaffected — it always
gets pixel size/wavelength from the calibration it's handed. See §3 and §5
"Data Loader panel".)

**Previously:** (2026-08-25, Switching profiles from the header **Profile:
[combo ▼]** now actually refreshes every dropdown/menu whose choices come
from that profile, not only tab visibility and Hydra availability: the Data
Viewer's **Live Data** PV dropdown, the Calibrate tab's **Calibrant**
dropdown (single-detector and every Hydra panel), and the pixel-size-preset /
K-edge-foil popup menus all now show the newly-active profile's values
immediately — previously these kept the *old* profile's list until the app
was restarted. Seeded numeric/path defaults (wavelength, pixel size, Lsd,
beam-centre, default files) are unchanged — they still only seed a field
once when a tab is built, so in-progress edits are never overwritten by a
profile switch. See §16 "Profiles".)

**Previously:** (2026-08-25, A prominent **Profile: [combo ▼]** dropdown now
sits at the top-left of the main window, in the same row as the tab bar —
switching profiles is instant, no more digging into Preferences. The
**Hydra** option on Data Viewer/Calibrate/Batch Integrate's mode ribbon now
only appears when the active profile is **1-ID-E** — the only beamline with
that detector — falling back to Single detector automatically if you switch
away while Hydra mode is active. See §16 "Profiles" and §3 "Mode ribbon".)

**Previously:** (2026-08-24, File ▸ **Open Project…** now offers to
*populate* the GUI from the project's own recorded attempts, not just make
it active for future logging — a **Populate from project** dialog lets you
pick, per panel (single-detector or each present Hydra GE panel), which
recorded Calibrate/Batch Integrate attempt to load. See §17 "File ▸ Project
(FAIR provenance)" → "Opening a project can populate the GUI".)

**Previously:** (2026-08-24, Tab 2 — Calibrate: Hydra mode's **Use manual
seed**/**Feed result back to seed** are now one shared choice linked across
all 4 GE panels (seed *values* stay independent per panel); both
Single-detector and Hydra Calibrate gain a new **Eta vs R Cake** tab
alongside Radial Profile — a 2-D (η, R) intensity heatmap, the standard
area-detector "cake" visualization, R on X and η on Y.)

**Previously:** (2026-08-24, File ▸ **Project** — a new, opt-in FAIR
provenance record (HDF5), separate from GUI State: while a project is open,
every completed Calibrate/Batch Integrate run (single-detector or per Hydra
panel) automatically appends a self-contained record of its exact inputs,
parameters, results, and software versions — raw multi-frame datasets are
referenced by path + checksum rather than duplicated, so this stays small
even for scans with thousands of frames. See §17 "File ▸ Project (FAIR
provenance)".)

**Previously:** (2026-08-24, Tab 4 — Batch Integrate: split into **Single
detector** / **Hydra** modes behind a leftmost mode ribbon, the same pattern
as the Calibrate/Data Viewer tabs' own splits. Hydra mode integrates each of
the 4 GE panels with its own independently fitted geometry — auto-populated
from the Calibrate tab's Hydra fits as each panel finishes, or loadable from
a file per panel — using one shared Integration/Corrections/Monitor
-normalisation/Output recipe, a **Sequential** or **Parallel** run-mode
choice, independent per-panel masks, and per-panel Waterfall/Stacked
-profile output. See §7 "Hydra mode (4-panel GE detector)".)

**Previously:** (2026-08-24, Tab 2 — Calibrate: split into **Single
detector** / **Hydra** modes behind a leftmost mode ribbon, the same pattern
as the Data Viewer tab's own split. Hydra mode fits each of the 4 GE panels'
own geometry from one calibrant dataset using a shared pipeline/refine
recipe, with independent per-panel Transforms/seed/results, a **Sequential**
or **Parallel** run-mode choice, and a **← Data Viewer** import of the
Hydra Data Viewer page's loaded panels + fitted geometry. See §5 "Hydra
mode (4-panel GE detector)".)

**Previously:** (2026-08-24, Tab 0 — Data Viewer: the **Transforms** (Flip
Y/Flip Z/Transpose[/Rotate]) checkboxes are now their own boxed
**Transforms** card, directly below the Ring simulation card, instead of an
inline "Transforms:" label + row inside it — fixes a large visual gap
between the heading and the row; this is shared code, so it applies to both
the single-detector tab's geometry card and every Hydra-tab panel card.
Hydra mode: the **Projection** card moved from the left-side loader panel to
the top of the middle panel, matching where the single-detector tab's own
Projection card sits; and the radial-integration plot's pan/zoom is now
bounded to the combined data range of the currently-visible curves (same
margin formula as the single-detector tab's radial plot), instead of being
free to scroll/zoom arbitrarily far from the data.)

**Previously:** (2026-08-24, Tab 0 — Data Viewer, Hydra mode: added a
per-panel **Projection** card (Max/Sum/Average stack reduction, mirroring
the single-detector tab) that feeds all of GE1-4 *and* the Composite view;
added a per-panel-only **Rotate** field (clockwise, degrees) next to
Flip Y/Flip Z/Transpose on GE1-4 — deliberately excluded from the Composite
view; fixed a bug where Flip Y/Flip Z/Transpose didn't actually refresh the
displayed image on GE1-4 panels; and the Material dialog's **Preset**
dropdown now also fills in the **Name** field (still editable), on both the
single-detector and Hydra tabs.)

**Previously:** (2026-08-24, Tab 0 — Data Viewer, Hydra mode: fixed a
manual (Phase 6) review's 5 bugs — λ/max 2θ/pixel size now shared across
all 5 geometry cards; dark/bright/background correction added (sibling
-aware, mirroring the single-detector tab); the Composite windmill assembly's
panel rotation stays counterclockwise (confirmed correct against real
`test_data/s1ide` data) and now also mirrors the whole canvas about its
vertical axis, which was needed to put ge2/ge4 on their correct sides; a
beam-centre edit now correctly updates the radial profile, not just the
ring overlay; and the image viewer's vmin% auto-level now excludes exact
-zero pixels app-wide, fixing a washed-out Composite view. The intensity
-range exclude mask and Top-N brightest-pixel remain single-detector-only.)

**Previously:** (2026-08-23, Tab 0 — Data Viewer, Hydra mode: the radial
plot now shows all 4 panels' profiles at once (fixed colors, independent
per-panel calibration) plus a toggleable summed **Composite** curve,
replacing the earlier single-curve placeholder. Dark/bright/mask
corrections for Hydra frames are still a planned follow-up.)

**Previously:** (2026-08-23, Tab 0 — Data Viewer: the **Hydra** mode
ribbon entry became functional — 4-panel GE detector loading, ge1-4/
composite image toolbar, independent per-panel beam-centre/ring
calibration, and a geometry-based windmill composite. Single detector mode
unchanged and still the default.)

**Previously:** (2026-08-23, Tab 0 — Data Viewer: added a leftmost **mode
ribbon** (Single detector / Hydra) — Hydra mode was a placeholder at that
point.)

**Previously:** (2026-08-23, All detector-image viewers — Data Viewer, Mask
Builder, Calibrate — now draw pixel `(0, 0)` at the **bottom-left** corner
instead of the top-left, matching MIDAS's convention that the on-screen
image match the physical world view of the detector when looking downstream
from the sample along the beam direction. This is a display-only rendering
change (`pg.ImageView`'s own default `invertY()` is overridden); it is
independent of the **Transforms** checkboxes and the `ImTransOpt` data
transform, which are unchanged. The Data Viewer ROI tool's box annotations
now label the bottom-left corner (was top-left).)

**Previously:** (2026-08-17, Tab 2 — Calibrate: the **Transforms: Flip Y /
Flip Z / Transpose** checkboxes now actually do something — toggling one
live-updates the image preview and Pick BC/Pick Ring clicks land in that same
transformed space, matching Data Viewer/Mask Builder; the calibration run
itself now applies the identical transform to the calibrant image and Dark
right before the pipeline call, fixing a real bug where several pipeline
modes (Four-stage, Bayesian, Joint-cake, panel-layout, partial
distortion-coefficient selection) silently transformed the image internally
while the seed beam centre and detector dimensions stayed untransformed;
**→ Send to Data Viewer** now carries the Transforms state too. Also: the
Mean-of-frames card's **skip** (stride) control was removed — **start**/
**end (0=all)** remain.) (2026-08-17, Data Viewer, Mask Builder, and Calibrate tabs all
gain matching **Transforms: Flip Y / Flip Z / Transpose** checkboxes applying
MIDAS's `ImTransOpt` image transform to the raw detector frame before
display/masking/calibration; the Calibrate tab's existing transform controls
now round-trip through saved/loaded `paramstest.txt` and `calibration.json`
files instead of being in-memory-only, and geometry hand-offs between the
three tabs keep the Transforms state in sync.)

**Previously:** (2026-08-16, Tab 2 — Calibrate: two fixes. (1) Picking a
strict subset of distortion coefficients (the "…" dialog) now genuinely
restricts the LM fit to just those coefficients on every pipeline, including
One-shot — which previously silently refined all 15 regardless of the
selection — and the Results tab now lists only the coefficients actually
selected for that run instead of always showing all 15 (saved
paramstest.txt/.json exports are unaffected — they still carry every slot's
real, possibly held-fixed, value). (2) Selecting/changing a Dark, Bright, or
Background field now immediately updates the image preview to the corrected
frame — it previously kept showing the raw, uncorrected image until the next
full data load).

**Previously:** (2026-08-13, Data Viewer ROI tool usability follow-ups: fixed
an OS-level minimize glitch where a popup would pop back open needing a
second click; box ROI `(x, y)`/`w = `/`h = ` annotations are now whole
pixels, not fractional; line ROIs now show their length in pixels at the
line's midpoint, and their "ROI N" label is anchored at the line's actual
start point instead of the image's top-left corner; and the box/line popup
plots plus the Intensity statistics histogram now cap zoom-out/pan at the
current data range, matching the radial-integration plot) (2026-08-13, Data Viewer ROI tool: OS-level window minimize
now tucks a popup into the ribbon (the old custom "–" button is gone); box
ROIs show live `(x, y)` / `w = ` / `h = ` annotations on the image; all
on-image ROI text gets a translucent background and a ~20% larger font;
popup stats/crop images now respect the active bad-pixel mask (excluded from
histograms/line stats, marked bright red on the crop); and the line-ROI drag
preview is a true diagonal line instead of a box) (2026-08-12, Tab 1 — Mask Builder's **Dilation (px)** control
switched from 4-connected to 8-neighbor dilation: N=1 now grows a bad pixel to
the full surrounding 3×3 block, N=2 to the full 5×5 block, etc.)
(2026-08-12, Tab 1 — Mask Builder gains a **5 · Post-processing**
card with a **Dilation (px)** control that grows bad-pixel regions from the
threshold/statistical/loaded mask by N pixels before combining with hand-drawn
shapes) (2026-08-12, Tab 1 — Mask Builder: Stack browse menu gains a
**Files (multi-select)…** entry so the temporal stack can be built from an
explicit, hand-picked set of files instead of only a whole folder/glob) (2026-08-12, Tab 6 — PDF Analysis rebuilt for the full
ROADMAP Stage 2-3 workflow: empty-cell/Paalman-Pings background subtraction,
detector-efficiency correction, absolute normalization, differentiable
multiple scattering, a fluorescence diagnostic, CIF-driven structure
refinement, and Δ-PDF significance testing, backed by a dedicated
`test_data/test_pdf/` dataset) (2026-08-10, Preferences ▸ Profile ships three bundled
beamline device presets — **20-ID-D**, **20-ID-E**, **1-ID-E** — so the Data
Viewer's Live Data PV dropdown can be switched to a beamline's detectors
without hand-editing Preferences ▸ Devices) (2026-08-09, Cross-tab data sharing: every Data Loader
panel's Data browse button, plus Mask Builder's Image/Stack browse buttons,
gain an **Import from…** menu to pull a file/folder/buffer already loaded in
another tab; **Use Buffer** gets a 💾 button to save the buffer to an HDF5
file (`buffer/data`); Data Viewer's **Project stack** button turns green while
a projection is displayed; ROI popups are now always-on-top and can be
minimized to a ribbon on the image viewer's left edge) (2026-08-09, Data Viewer: new B-PILOT auto-start bridge —
a local-socket server lets the separate B-PILOT plan-runner GUI trigger
Live Data on a scan's detector with no clicks in MIDAS GUI; see the Live
Data card section) (2026-08-03, Data Viewer/Calibrate/PDF: the clickable λ
label's popup menu gains an **Energy (keV)** entry box that converts to
wavelength on Enter, ahead of the existing K-edge foil menu; Data Viewer's
Projection card drops the **Axis** field (always 0, i.e. across frames) and
adds an **N frames** field capping how many frames after Skip frames are
included, 0 = all remaining) (2026-08-01, Data Viewer: box-ROI popup's zoomed crop image
is no longer fixed-size — it now resizes along with the popup window) (2026-08-01,
Data Viewer: line ROI is now drawn as a single
arrow shape — shaft and head recomputed together from the two endpoints on
every drag — replacing the separate arrowhead overlay item that could rotate
oddly as the endpoint moved) (2026-08-01,
Data Viewer: image toolbar gains a Box/Line
ROI tool — click-drag draws a shape, opening a small floating, freely-draggable
stats popup next to it, color/label-matched to the shape; box popups show a
linear/log intensity histogram plus a zoomed-in crop of the region, line popups
show a flippable intensity-vs-distance profile; popups stay live-linked to
their shape as it's dragged/resized but not to its screen position; multiple
ROIs supported, session-only, removable via the popup's close button, a
shape's right-click menu, or Clear ROIs (which also resets ROI numbering back
to 1); Pick BC/Pick Ring's shared Clear button now also removes the Pick BC
crosshair marker, not just Pick Ring's points) (2026-07-31,
Data Viewer: pixel-size field now accepts a
second decimal place, needed for near-field detectors' finer pixel pitches;
radial-integration plot gains a "?"
help button explaining the R-bin calculation; ring-width field widened
60->80px; Live Data card gets a "Use Buffer"
last-N-frames ring buffer — yellow while filling, green once streaming pauses,
at which point Projection and the rest of the stack analysis work on it like
a loaded HDF5/folder stack; Load-calibration card renamed/moved below Ring
simulation's Simulate button and now syncs ty/tz on load; Intensity range
card's title bar is now the on/off checkbox; Mask rows — including the
"Tab 1 mask" from the Mask Builder tab — get a checkbox to include/exclude
without deleting, and the Tab 1 mask row always stays at the top of the list;
Live Data "Use Buffer" N field capped at 100 frames to bound memory use;
Mask status line turns amber and names any source dropped for a load
failure or shape mismatch, instead of silently ignoring it; Live Data PV
dropdown gains a built-in **Sim Detector** entry — a hardware-free fake PVA
stream shaped like an Eiger2 500K with 0-60000 counts, for exercising Live
Data without a real beamline connection, see `midas_gui/sim_detector.py`;
image viewer colorbar/histogram now defaults its own zoom to the vmin%/vmax%
percentile window instead of the full data range, converts a manual
level/zoom window between Log and Linear scale on toggle, auto-resets to the
percentile defaults whenever new data is loaded — except frame-to-frame
updates within an active live stream, which keep a manual window fixed — and
always reframes its own visible window tightly around the (converted) levels
on a Log/Linear toggle, instead of carrying a manually zoomed/panned window
through the nonlinear conversion, which could leave the sliders squeezed
into a barely-visible sliver of the window; Live Data "Use Buffer" now
carries over into **Start** instead of being silently reset — if buffering
is already armed when Start is clicked, it re-arms with a fresh buffer for
the new stream rather than turning back off; Ring simulation card's λ/Lsd/pixel
size fields now accept any positive value (no more 1–5000 µm pixel-size cap
etc.) and display λ to 4 decimals, Lsd to 3, pixel size to 2 (needed for
near-field detectors' sub-µm-precision pixel pitches), BC_y/BC_z to 1, and
ty/tz to 2; "Show rings" renamed **Rings** and moved onto the same row as
**Labels** and a shrunk ring-thickness field; Exclude-out-of-range-pixels
controls moved from their own left-panel card into the radial-integration
plot's own toolbar (right end of the `X` / `Log Y` / `R bin` / `Auto` /
`Integrate` row), with the `R bin` field narrowed and the `N bins | max=...`
stats printout removed from that row to make space; Load calibration card
renamed **Load/save calibration**; Exclude-range controls further refined —
moved to sit pinned at the far right of the toolbar row, the `<`/`>` bound
fields widened (56px → 112px) and switched from float to integer spin boxes
(no decimal display), and the lower-bound comparison changed from `<=` to a
strict `<` — a pixel exactly equal to the lower bound is no longer masked)

> **Maintenance:** keep this document in sync with the code — whenever the workflow
> or a tab's controls change, update the relevant section here in the same change.

> **Tab status:** Tabs **0–4** (Data Viewer, Mask Builder, Calibrate, Calib.
> Refinement, Batch Integrate) are **verified** and ready to use. Tabs **5–8**
> (Corrections & Physics, PDF Analysis, Texture, Results & Export) are a **work in
> progress** and will be updated in the coming weeks.

---

## Table of Contents

1. [Overview and Architecture](#1-overview-and-architecture)
2. [Getting Started](#2-getting-started)
3. [Data Viewer](#3-tab-0--data-viewer)
4. [Mask Builder](#4-tab-1--mask-builder)
5. [Calibrate](#5-tab-2--calibrate)
6. [Calibration Refinement](#6-tab-3--calibration-refinement)
7. [Batch Integrate](#7-tab-4--batch-integrate)
8. [Zarr Viewer](#8-zarr-viewer)
9. [Corrections & Physics](#9-tab-5--corrections--physics)
10. [PDF Analysis](#10-tab-6--pdf-analysis)
11. [Texture / Pole Figure](#11-tab-7--texture--pole-figure)
12. [Pump Probe (time-resolved / TR-XRD)](#12-pump-probe-time-resolved--tr-xrd)
13. [Results & Export](#13-results--export)
14. [Common UI Conventions](#14-common-ui-conventions)
15. [Packaging, Deployment & Diagnostics](#15-packaging-deployment--diagnostics)
16. [Configuration & Defaults](#16-configuration--defaults)
17. [File ▸ Project (session + FAIR provenance)](#17-file--project-session--fair-provenance)

---

## 1. Overview and Architecture

The MIDAS GUI is a modular PyQt5 desktop application (up to 10 tabs) that exposes the
scientific capability of the `midas_calibrate_v2` and `midas_integrate_v2` packages
through a structured workflow. The intended order of use is:

```
Data Viewer (inspect) → Mask Builder → Calibrate → [Refine] → Batch Integrate
                                                              → Corrections preview
                                                              → PDF Analysis
                                                              → Texture / Pole Figure
                                                              → Pump Probe (TR-XRD)
                                                              → Results & Export
```

**Modular tabs.** Data Viewer, Mask Builder, Calibrate and Batch Integrate are always
shown; the remaining tabs (Calib. Refinement, Batch Queue, Zarr Viewer, Corrections,
PDF Analysis, Texture, Pump Probe, Results & Export) are optional and can be
shown/hidden from **Settings ▸ Preferences ▸ Tabs**. By default **Calib. Refinement**,
**Batch Queue**, **Zarr Viewer** and **Pump Probe** are shown; Corrections, PDF
Analysis, Texture and Results & Export ship **hidden** — turn them on when you need
them. The choice is saved per-user (`ui.visible_tabs`) and applies immediately — see
§16. Hidden tabs are only removed from the tab bar; they stay constructed, so
cross-tab wiring and state are preserved.

Every optional tab also carries an **✕** on its own label — clicking it is the same
act as unchecking that tab in Preferences ▸ Tabs, writes the same `ui.visible_tabs`
key, and sticks across restarts. The four always-on tabs have no ✕. Because closing
only removes the tab from the bar, a closed tab comes back with its state intact (a
loaded file, a running queue) when you turn it on again.

**Cross-tab shared state.** When Tab 2 (Calibrate) produces a result it is
automatically propagated to all downstream tabs. When Tab 1 (Mask Builder) computes
a mask it is sent to the consuming tabs. Tab 3 (Refinement), if applied, re-broadcasts
the refined geometry. No manual copying is needed.

**The Browse… popup.** Every **Data, Dark, Bright, Background** field's **⋯**
button — single-detector tabs (Data Viewer, Calibrate, Calib. Refinement,
Batch Integrate, Pump Probe) and their Hydra equivalents alike — opens the
same two-item menu: **Browse…** and **Import from…**. **Browse…** opens a
popup file browser with four selection modes (radio buttons at the top):
- **Single file** — one file, TIFF-family (`.tif/.tiff/.geN/.cbf/.edf`) or
  HDF5. The only mode offered for Hydra's main **Hydra data** field, whose
  frame index already comes from one anchor file's own internal frame count.
- **Multiple files** — an arbitrary multi-select. Not offered for Hydra
  fields (there's no way to derive the other 3 panels' files from an
  arbitrary pick list).
- **Full folder** — every TIFF-family file in one directory.
- **Files sharing a name stem** — every TIFF-family file in one directory
  starting with a typed prefix (click a file in the browser to prefill it).
  On **Batch Integrate's** streamed Data field specifically, a filestem pick
  is kept as a *live* filter (folder + prefix), not a frozen file list — see
  "Batch Integrate: filestem-filtered sources and MONITOR" below.

**HDF5 files are simply not shown** in Multiple-files/Full-folder/Filestem
mode — since an HDF5 file is itself a multi-frame container, only Single
file ever applies to one. For a Hydra field, Full-folder/Filestem resolve
per-panel exactly like Single-file does today: point at any one panel's
folder (or a file inside it) and the other panels' folders are found via the
same `geN` naming convention.

After a pick, the field always shows a real filesystem path — the single
file's own path (Single file mode, or a Multiple-files/Filestem pick that
resolved to exactly one file), the chosen directory (Full folder), or the
shared parent folder of every matched file (Multiple files/Filestem with
more than one match; hover the field for the full file list). The popup
opens to the `midas-gui` project's root folder by default, and its file/
folder name column is twice Qt's default width so longer names aren't
immediately truncated.

**Cross-tab data import ("Import from…").** Every Data Loader panel's **Data,
Dark, Bright, Background** fields (single-detector and Hydra alike), plus
Mask Builder's **Image** and **Stack** browse buttons, carry an **Import
from…** submenu listing whatever's currently loaded in every *other* tab of
the same field type — a file/folder path, an explicit multi-file/filestem
selection, or, if that tab's Live Data **Use Buffer** ring buffer is frozen
(green), a **Buffer (N frames)** entry. Hydra tabs are labeled distinctly
(e.g. "Data Viewer (Hydra)") so a Hydra anchor path is never confused with
its single-detector counterpart; a Hydra field's menu also silently omits
any single-detector Multiple-files selection it couldn't use. The menu is
built fresh each time it's opened, so it always reflects what's loaded right
now. Picking a path or file list just loads it like a normal Browse pick.
Picking a **buffer**:
- In a tab that keeps a live in-memory stack (Data Viewer, Calibrate,
  Refinement), the picking tab **delegates** to the source's buffer directly —
  no copy is made, so it always reflects the source buffer's current
  contents, and if the source buffer is later reset, the delegating tab clears
  itself and shows "Source buffer was reset" rather than showing stale data.
- In a tab that only ever reads from a file/HDF5 path (Batch Integrate, Pump
  Probe, and Mask Builder's Stack field), the buffer is instead **snapshotted
  once** to a temporary HDF5 file (dataset `buffer/data`) and that path is
  loaded — later changes to the source buffer are not reflected until you
  re-pick the entry.

**All heavy computation runs off the GUI thread.** Every operation that touches a
MIDAS package (calibration, integration, mask training, PDF transform, gain training,
field averaging, drift fitting) runs in a background `QThread` worker; the GUI stays
responsive and log lines stream in real time.

**Package layout.**

```
midas_gui/
├── app.py            MainWindow, dark theme, crash diagnostics, main()
├── __main__.py       enables `python -m midas_gui`
├── style.py          Dioptas-inspired QSS theme + layout helpers
├── constants.py      calibrants, colormaps, dtype sentinels, default paths
├── helpers.py        image IO, geometry parsing, spec building, no-scroll widgets
├── widgets.py        ImageViewer / PickableImageViewer / ProfileViewer / FieldSelector / …
├── workers.py        all QThread workers + the integration core
├── calib.py          calibration-pipeline dispatch
├── dialogs.py        save dialogs
└── tab_*.py          one module per tab
```

**Image orientation.** Every detector-image viewer (Data Viewer, Mask
Builder, Calibrate) draws pixel `(0, 0)` at the **bottom-left** corner,
matching MIDAS's own convention: the on-screen image then matches the
physical world view of the detector when looking downstream from the
sample along the beam direction. This is a display-only convention,
separate from the **Transforms** (Flip Y / Flip Z / Transpose) checkboxes
found in Data Viewer and Calibrate — those still flip or transpose the
underlying pixel *data* itself (persisted as MIDAS's `ImTransOpt` in saved
calibration files) to correct a detector's raw readout orientation,
independent of which corner the GUI renders as the origin. Mask Builder has
no Transforms checkboxes of its own — see §4.

**Layout pattern.** The four analysis tabs — **0 Data Viewer, 2 Calibrate, 3 Calib.
Refinement, 4 Batch Integrate** — use a **three-panel layout**:

```
[ Data Loader | Parameters | Display / results ]
```

- **Data Loader (left).** A shared `DataLoaderPanel` for selecting the five inputs —
  **Data, Dark, Bright, Background, Mask** — each as a single file, a folder, or an
  HDF5 dataset (a container dropdown appears for HDF5). Frame controls live here too
  (Tab 0 navigator, Tab 2/3 frame index, Tab 4 frame range + stride). Data/Dark/Bright/
  Background's **⋯** button opens a two-item menu — **Browse…** and **Import from…**
  (below) — see **The Browse… popup** just below for what Browse… offers.
- **Parameters (middle).** The tab's analysis controls.
- **Display / results (right).** Image viewer, plots, and/or a log panel.

The three panels are separated by **draggable splitter handles** — drag the
`Data Loader | Parameters` and `Parameters | Display` boundaries to rebalance widths
to taste. Each panel has a minimum width; drag past it and the panel shows a scrollbar
rather than clipping its controls.

The remaining tabs (1, 5–8) keep the classic two-panel layout.

**Corrections & mask (all four analysis tabs).** Dark is averaged then subtracted;
Bright is averaged then flat-field **divided** or **subtracted** (your choice);
Background is averaged then subtracted; every **checked** Mask source (files/folders plus
the auto-added "Tab 1 mask", populated from the Mask Builder tab) is **unioned** into one
composite mask that zeroes/ignores those pixels. Correction order: `(img − dark)` →
bright → `− background` → clip ≥ 0.

Each Mask row has its own **checkbox** to include/exclude it from the union without
deleting it (the **✕** button still removes the row entirely). The "Tab 1 mask" row is
always kept at the **top** of the list when other mask sources are present.

If a checked mask source fails to load, or its shape doesn't match the other sources
already unioned, it's dropped from the composite and the status line under the mask
list turns amber and names the source and the reason — it is never silently ignored.

---

## 2. Getting Started

`midas-gui` is **not on PyPI** — install it from source with conda.

```bash
git clone https://github.com/d-beniwal/MIDAS_GUI.git
cd MIDAS_GUI
conda env create -f environment.yml    # creates the 'midas-gui' env and installs the GUI
conda activate midas-gui
midas-gui                              # launch (equivalently: python -m midas_gui)
```

The GUI drives the MIDAS analysis backends (`midas-calibrate-v2`, `midas-integrate-v2`,
`midas-calibrate`, `midas-hkls`, `midas-distortion`), which are also not on PyPI —
point the `pip:` section of `environment.yml` at your MIDAS source before creating the
environment (see the comments in that file).

### Test data
Synthetic test data lives in the repo-root `test_data/` folder (git-ignored):
- `calibrant_ceria.tif` / `.h5` — CeO₂ calibrant (Eiger2 500K, 1028×512, 75 µm pixels)
- `nickel_tifs/` — 10-frame Ni scan, lattice expanding +0.1 %/frame
- `nickel_stack.h5` — the same scan as one HDF5 stack (`exchange/data`)

Default geometry: λ = 0.39 Å, Lsd = 121 mm, pixel = 75 µm, BC = (10, 10) px. Every
tab's default file paths point at this data and **auto-load on startup when present**,
so the app is usable immediately after checkout.

---

## 3. Tab 0 — Data Viewer

**Purpose:** inspect detector frames and do a quick radial integration — geometry-free
(circle binning) by default, or a full tilt/distortion-aware integration when a
calibration file is loaded.
Produces no shared state for other tabs.

### Mode ribbon (leftmost strip)
A narrow vertical strip at the very left edge of the tab switches between
**Single detector** (the view described below — unchanged) and **Hydra**
(the 1-ID-E 4-panel GE detector view). The two modes are independent:
switching does not share data or geometry between them. **The Hydra option
only appears when the active profile (top-left header dropdown — see §16
"Profiles") is 1-ID-E**, the only beamline with that detector; on any other
profile the ribbon shows Single detector only, and switching away from
1-ID-E while Hydra mode is active falls back to Single detector
automatically. Same rule applies to Calibrate's (§5) and Batch Integrate's
(§7) identical mode ribbons.

### Hydra mode (4-panel GE detector)

**Purpose:** inspect and calibrate the 1-ID-E Hydra detector — 4 separate GE
panels arranged in a "windmill" layout around a shared beam axis, each with
its own independent beam-centre/tilt calibration, composited into one
registered image for a full-coverage view.

- **Hydra data (left panel)**: point the path field at **any one** of the 4
  GE panel files (a `.geN.h5`/`.tif` or a `geN/` folder — matching this
  beamline's own naming convention) and the other 3 panels are found
  automatically. Small `ge1 ge2 ge3 ge4` status labels turn green as each
  panel is located (grayed out if not found — the view still works with as
  few as 2 panels present). A frame slider/spinbox navigates a shared frame
  index across all panels (they're synchronized frames of the same scan).
  Its **⋯** button's **Browse…** only offers Single file (see "The Browse…
  popup" in §1) — the frame index already comes from that one file's own
  internal frame count, not separate per-frame files.
- **Dark / Bright / Background (left panel, below Hydra data)**: same
  correction math as the single-detector tab (dark subtraction, bright
  flat-field divide/subtract, background subtraction). Point each field at
  **any one** panel's dark/bright/background file (or folder, or filestem —
  see "The Browse… popup" in §1) and the other panels' matching files are
  found automatically, the same way the main Hydra data path works — no need
  to pick all 4 by hand. Each field computes and applies independently per
  panel. Also gets a real **Import from…** (see "Cross-tab data import" in
  §1) — Hydra fields are labeled distinctly (e.g. "Data Viewer (Hydra)") so
  they're never confused with their single-detector counterparts.
- **Projection (top of the middle panel)**: same **Max / Sum / Average**
  stack-reduction as the single-detector tab's own Projection card (and in
  the same place — top of the middle panel, above the geometry cards), but
  computed **per panel** — clicking **Project stack** reduces every
  currently-found panel's own frame stack (honoring Skip frames/N frames and
  whatever Dark/Bright/Background correction is currently set) and shows the
  result in place of the current frame for **all of GE1-4 and the Composite
  view** (the Composite is rebuilt from the projected frames, not frame 0).
  **Back to frames** — or moving the frame slider/spinbox — returns to
  normal per-frame navigation.
- **Image toolbar**: five buttons — **GE1 / GE2 / GE3 / GE4 / Composite** —
  select what's shown in the image viewer below. GE1-4 show that panel's own
  raw frame; **Composite** shows all currently-available panels remapped
  (via each panel's own beam-centre + tilt) into one shared, registered
  canvas — a geometry-based "windmill" composite, not a raw mosaic.
- **Per-panel geometry card (middle panel, below the Projection card)**:
  switching the image-toolbar button swaps which panel's **Ring simulation +
  Transforms + Load/save calibration** cards are shown — identical in every
  way to the single-detector tab's own cards (materials list, beam-centre
  pick/ring-fit, tilt fields, calibration
  file load/save), just bound to that one panel. Loading a calibration file
  (or picking/editing a beam centre) on a GE1-4 card takes effect
  immediately: it's used for that panel's own ring overlay/radial
  integration, and — since the Composite view depends on every panel's
  geometry — the composite is automatically rebuilt the next time it's
  shown. Panels with no calibration loaded yet use a bundled default
  1-ID-E-style geometry (an example windmill layout, not your
  instrument's real calibration — load a real per-panel file to override).
  The **Composite**'s own geometry card is separate from the 4 panels': its
  beam centre is automatically seeded at the composite canvas's own centre
  the first time a given canvas size is built, so ring simulation/radial
  integration on the Composite view work immediately with no extra setup
  (though it can still be hand-edited like any other card). **λ (wavelength),
  max 2θ, and pixel size are shared across all 5 cards** — editing any one
  of them on GE1-4 or Composite applies it to the other 4 immediately (same
  X-ray beam and GE detector model, so these three are always physically
  identical); beam centre, Lsd, and tilt remain independent per panel.
  GE1-4 (not the Composite) also have a **Rotate** field on the same row as
  **Flip Y / Flip Z / Transpose** in that panel's own **Transforms** card: a
  clockwise rotation (degrees, default 0) applied only to that one panel's
  own raw display/radial-integration image — it is deliberately **not**
  applied to the Composite view, which keeps building from each panel's
  un-rotated frame and its own independent beam-centre/tilt geometry.
- **Radial integration plot (bottom right)**: shows **all 4 panels' own
  azimuthal profiles at once** — GE1-4 in fixed colors, computed
  independently from each panel's own beam centre/geometry — plus a
  toggleable **Composite** curve (white, dashed). Checkboxes above the plot
  show/hide each curve individually. The R-bin/Auto/Integrate controls are
  shared across all 5 views (one setting, not a separate control per
  panel); **Integrate** recomputes every curve immediately regardless of
  Auto. The X-axis unit selector (R/2θ/Q) applies to all curves at once.
  The **Composite** curve is the sum of the 4 panels' own profiles — each
  resampled onto a shared 2θ axis first, then added together — **not** a
  radial integration of the composited image itself (which would
  double-count any panel overlap and mix registration error into the
  profile). **Pan and zoom are bounded** to the combined X/Y extent of every
  currently-visible curve (same margin formula as the single-detector tab's
  own radial plot — see **Pan and zoom are always bounded to the current
  profile's extent** below), recomputed on every curve refresh (X-axis unit
  switch, checkbox toggle, or new data) so scrolling/zooming can't wander
  off into empty space. This plot doesn't yet have the single-detector
  plot's Auto/Manual toggle button pair.
- **Not yet supported in Hydra mode** (present in Single detector mode):
  the intensity-range exclude mask, and Top-N brightest-pixel.

### Data Loader panel (left)
Data, Dark, Bright, Background and Mask are all selected in the shared **Data Loader
panel** (see §1): each accepts a file / folder / HDF5-dataset. The **Data** card here
holds the **frame navigator** (slider, spin box, ◀/▶); **zoom/pan is preserved** as you
step through frames (the view only auto-frames on a fresh load). Dark/bright/background
and the composite mask are applied to the displayed image and to the radial integration.
**Pan and zoom are bounded to the image** (roughly half an image-width of margin on each
side) so scrolling/dragging cannot wander off into empty space or zoom out indefinitely;
the bottom-left **A** (auto-range) button re-fits the image without leaving zoom "stuck"
tracking the mouse. On the **1-ID-E** / **20-ID-D** / **20-ID-E** profiles, loading a
Data file also auto-detects **pixel size** (from a `.ge1`–`.ge5`/`.vrx`/`.pxrd` filename
tag) and, for an HDF5 file, **wavelength** (from its recorded beam energy) into the Ring
simulation card's λ/pixel fields below — whatever isn't detected is left unchanged.

### Projection card
Collapse a stack to one image: **Max** (hot-pixel hunting), **Sum** (long-exposure
equivalent), or **Average** (noise reduction), always across the stack of frames.
**Skip frames** (default 1) ignores that many leading frames before projecting (e.g. 1
drops the first frame, 4 drops the first four) — useful when the opening frames are
detector warm-up / shutter-transient exposures. **N frames** caps how many frames
(after Skip frames) are included in the projection; **0** (default) uses every
remaining frame in the stack. While a projection is being displayed, **Project
stack** stays highlighted **green** so it's obvious the image on screen is a
projection rather than a live frame; **Back to frames** (or loading new data)
reverts it to its normal styling.

### Exclude-out-of-range-pixels controls (radial-plot toolbar)
These controls used to be their own left-panel card; they now live at the **far
right** of the **radial integration plot's toolbar** (see below), past the `X`,
`Log Y`, `R bin`, `Auto`, `Integrate` controls, so there's no separate card here
anymore.

| Field | Description |
|---|---|
| Exclude range (checkbox) | When on, pixels < min or > max are drawn as a red overlay and excluded from the radial integration (removes gaps / hot / overflow). |
| < / > | Lower / upper bounds, entered as **whole-pixel-count integers** (no decimals). On load, the upper bound auto-fills to **max(99.99th percentile, 100000)**. |

### Ring simulation card
Overlays simulated Debye-Scherrer rings to check geometry. Supports **multiple
materials at once** — e.g. a sample phase overlaid on a calibrant — each with
its own lattice, visibility, and ring color.
- **Material rows**: one row per material, each with a **checkbox** (show/hide
  that material's rings on both the image and the radial-integration plot), a
  **color swatch** button (click to open a color picker; the chosen color
  drives that material's ring lines and hkl labels on both plots), and the
  **material name** as a clickable (underlined) button. A **✕** button deletes
  the row — disabled when only one material remains, since at least one row
  is always kept. **+ Add material** appends a new row (default name
  `Material N`, generic cubic lattice, next color from a 10-color palette
  cycled by row order). A single default **Ni (FCC)** row is present at
  startup, matching the pre-multi-material behavior.
- Clicking a material's name opens a **Material dialog**: an editable **Name**
  field, the **Preset** dropdown (CeO₂, LaB₆, Si, Al₂O₃, Cu, Ni, FCC-γFe,
  BCC-αFe, Au, Ag, Pt, W, Ti, or **Custom**), lattice **a, b, c** (3 decimals)
  on one row and **α, β, γ** (2 decimals) on the next, plus **SG #**, and a
  **Cubic (a=b=c, α=β=γ=90°)** checkbox that lets you enter only `a` for cubic
  crystals (b, c mirror a; angles fixed at 90°). Lattice fields are editable
  only for Custom. Picking a named preset also fills the **Name** field with
  that preset's name (still freely editable afterward — e.g. rename it
  without losing the lattice values just applied). OK applies the
  name/lattice/preset back to that material's row (renaming here updates the
  row's displayed name); Cancel discards edits.
- Geometry (λ, max 2θ, Lsd, pixel size, beam centre) is shared across all
  materials — only the lattice/space-group/color/visibility are per-material.
  Beam centre (auto = image centre, or
  manual BC_y/BC_z). The **λ** label is clickable (underlined) — click it to open a
  menu with an **Energy (keV)** entry box at the top (type a photon energy and press
  Enter, or click **↵**, to convert it to wavelength via λ = 12.398420 / E) followed
  by a menu of common K-edge foils (Pr, Sm, Yb, Lu, Hf, Ta, W, Re, Pt, Au, Pb, Bi)
  that set λ directly to that element's K absorption-edge wavelength. The same
  clickable-λ menu is on the Calibrate and PDF tabs. The **px** label is likewise clickable — a menu
  of common detectors (GE 200 µm, Varex 150 µm, Pilatus 172 µm, Eiger 75 µm) sets
  the pixel size (also on the Calibrate tab, where it sets both pxY and pxZ).
- Every field in this card steps by a fixed amount per up/down-arrow click
  (λ 0.01 Å, max 2θ 1°, Lsd 1 mm, pixel 0.1 µm, BC_y/BC_z 1 px, ty/tz 0.1°) —
  customizable from **Settings ▸ Preferences ▸ Data Viewer**.
- **ty / tz** (detector tilt about the Y/Z axes, degrees) sit right below
  BC_y/BC_z — when non-zero, the simulated rings are forward-projected through
  the tilt geometry instead of drawn as plain circles, so the overlay shows the
  same non-circular ring shape a tilted detector actually produces. Leaving
  both at 0° reproduces the previous plain-circle rendering exactly. (Rotation
  about the beam axis, `tx`, leaves a full ring's shape unchanged, so it isn't
  exposed here.)
- **Rings / Labels** toggles sit on one row together with a compact **thickness**
  spin box (0.5–10 px) that sets the line width of the simulated-ring overlay on
  the image; redraws immediately as you change it.
- **Simulate rings** is a **live toggle** (not a
  one-shot click) — while on, the button turns **green** as a visual cue, and the
  overlay + hkl table recompute automatically whenever material, lattice constants,
  space group, or geometry (λ/Lsd/px/max 2θ) change. Beam-centre and ty/tz edits
  still just reposition the existing rings (their 2θ values don't depend on BC or
  tilt). Rings are drawn out to the full **max 2θ** you set, regardless of how far
  that places them from the beam centre (rings landing off-canvas simply aren't
  visible — they aren't silently dropped at some fixed pixel-radius cutoff).
- **→ Send geometry to Calibrate** copies λ, pixel size, Lsd and beam centre into the
  Calibrate tab's detector + seed fields (the Calibrate tab has a matching
  **← Data Viewer** button that pulls the same values).

### Transforms card
A small standalone card sitting directly below the Ring simulation card,
titled **Transforms** (previously an inline "Transforms:" label + row buried
inside the Ring simulation card, with the label and row visually far apart —
now its own boxed section so the heading sits directly above the checkboxes
it labels). **Flip Y / Flip Z / Transpose** apply MIDAS's `ImTransOpt` image
transform to the raw detector frame — before display, ring overlay, and
radial integration — in that fixed order (flips, then transpose). Use this
when the detector's raw pixel orientation doesn't match the geometry model
(e.g. the beam centre would otherwise land on the wrong side of the image).
Toggling refreshes the current frame (or the active projection) immediately.
The same three checkboxes appear on the Calibrate tab and stay in sync
whenever geometry is pushed/pulled between tabs (Mask Builder has no
Transforms checkboxes of its own — see §4); saved/loaded
calibration files (`.json`/`.txt`) round-trip the codes as MIDAS's repeatable
`ImTransOpt <code>` paramstest key (1=Flip Y, 2=Flip Z, 3=Transpose). In
Hydra mode, GE1-4's Transforms cards also carry the per-panel-only **Rotate**
field (see Hydra mode below); this same Transforms card is shared code
between the single-detector and Hydra tabs.

### Load/save calibration card (optional)
Sits directly below the Transforms card.
Load geometry from a **calibration `.json`, a MIDAS `paramstest.txt`, or a pyFAI
`.poni`** (auto-detected). It fills **BC, Lsd, pixel size, wavelength, the
Ring-simulation card's ty/tz tilt fields, and the Transforms checkboxes**
(from any `ImTransOpt` lines in the file), unchecks "Beam centre = image
centre", and refreshes the overlay and radial plot.

When a calibration file carries the **full geometry (tilts + distortion)**, the radial
integration switches from simple concentric-circle binning to a **proper MIDAS-engine
integration** that maps every pixel through the calibrated tilts and distortion (the
same core as Batch Integrate) — so ring positions/intensities are geometry-correct, not
just distance-from-beam-centre. The card's status line reports which mode is active. The
binning geometry is built once and reused across frames (a fast `hard` kernel keeps the
preview responsive); without a full-geometry file, the fast circle binning is used. If no
calibration file is loaded but the Ring-simulation card's **ty/tz** tilt fields are
non-zero, the radial integration still runs the tilt-aware MIDAS engine using a geometry
built live from those fields (BC, Lsd, pixel size, wavelength) — so dialing in tilts here
without a calibration file already produces a tilt-corrected profile, not just a
tilt-shaped ring overlay.

**Save JSON / Save params (.txt) / Save PONI** (below the loader) export whatever
geometry is currently in effect — the loaded calibration file if any, otherwise the
one synthesized from the Ring-simulation widgets above — as a calibration file you can
reload here, on the Calibrate tab, or in Batch Integrate. **Save params (.txt)** writes
a standalone MIDAS `paramstest.txt`; **Save PONI** writes a pyFAI `.poni` (note: PONI's
Rot1–3 convention cannot represent MIDAS's tx/ty/tz tilts, so tilts are **not** included
in a `.poni` export — use JSON or `.txt` to keep them). Clicking any of the three before
an image is loaded warns instead of writing a file (there is no detector size to write).

### Live Data card *(experimental)*
Sits **above the Data card**, collapsed by default behind its own title-bar
checkbox — check it to reveal the live-PV controls, uncheck to hide them
again (unchecking also stops an active stream, so a hidden card can never be
left silently connected). Subscribes to a live EPICS detector-image readout
and renders each frame **inline in the Data Viewer's own image pane** — the
same view used for files/folders/HDF5 stacks. Because it feeds the normal
frame pipeline, all existing Data Viewer analysis keeps working live:
dark/bright/background/mask corrections, the intensity-range mask,
beam-centre picking, and the radial integration plot all update as new
frames arrive.
- **Two EPICS backends, picked per-device:** each entry in **Preferences ▸
  Devices** has a `backend` field — `pva` (default) subscribes to a single
  pvAccess NTNDArray PV, for beamlines whose areaDetector IOC runs an
  NDPluginPva plugin. `ca` instead reads plain EPICS Channel Access records
  off an areaDetector **NDPluginStdArrays** plugin (`ArrayData` plus its
  `ArraySize*_RBV`/`ColorMode_RBV`/`UniqueId_RBV` metadata records), for a
  beamline whose IOC has no PVA plugin (e.g. 17-BM's Varex detector, which
  ships as the bundled **17-BM** profile's `varex` device). Both backends
  share one internal contract (`midas_gui/live_sources.py`:
  `PvaLiveSource`/`CaLiveSource`) so everything else on this card — buffering,
  the frame sink, Start/Stop, the B-PILOT bridge below — behaves identically
  regardless of which one a device uses. Picking a device from the dropdown
  sets its backend automatically (shown as a small `[PVA]`/`[CA]` label next
  to the field); typing a PV by hand keeps whichever backend was last picked.
- **Dependencies:** `pvapy` (PVA) and `pyepics` (CA) are both required,
  pinned dependencies (`pyproject.toml` / `environment.yml`), installed
  automatically with the rest of the GUI's stack — no separate extra to
  install. If either is somehow missing from the active environment,
  **Start** shows an install hint instead of failing outright (only the one
  needed for the currently-selected device's backend is checked).
- **Live PV** field is an editable dropdown: pick a known device by name (the
  list comes from **Preferences ▸ Devices**, see below) to fill in its full PV
  automatically (`prefix + PVA suffix` or `prefix + CA suffix`, depending on
  that device's backend), or type any other PV by hand (placeholder shows an
  example, `20IDFF:Pva1:Image`). **Start** / **Stop** buttons and a status
  line (stopped / waiting for PV / connected / streaming with frame id /
  error). GUI updates are throttled to the tab's existing ~16 fps debounce, so
  a fast PV update rate doesn't overwhelm the interface.
- **Sim Detector** is a built-in dropdown entry (PV `midasSim:Pva1:Image`) for
  exercising Live Data with **no beamline hardware**: picking it and clicking
  **Start** lazily launches an in-process fake PVA server
  (`midas_gui/sim_detector.py`) that streams random frames shaped like a
  DECTRIS Eiger2 500K (1030×514 px) with counts in `[0, 60000]`, at 5 Hz by
  default — real `PvaLiveSource` code path, so it's indistinguishable from a
  real detector's stream. The rate, frame size and intensity range are
  constructor parameters in that file for anyone who wants a different fake
  stream. The simulator keeps running (harmless background thread) across
  repeated Start/Stop until the app closes, at which point it's stopped
  automatically.
- **Use Buffer** captures the last **N** live frames (N field next to it,
  2–100 — capped to bound memory use for large-format detectors) into
  an in-memory ring buffer, so Projection and every other stack-based analysis
  become available on live data. Click it to arm: it turns **yellow** and its
  label counts up (`Buffering… (7/20)`) while frames keep streaming in — the
  live single-frame view is unaffected during this. When no new frame arrives
  for ~2 s (streaming paused, or **Stop** clicked), it turns **green**
  (`Buffer Ready (20)`) and the buffered frames become a normal navigable
  stack: the frame slider/spin/prev-next enable, and **Project stack** runs
  max/sum/average over the buffered frames exactly as it would for a loaded
  HDF5 file or folder. If new frames resume arriving, it flips back to yellow
  and keeps rolling (oldest frame dropped once past N). Click it again to turn
  buffering off and discard the buffer. If **Use Buffer** is already armed
  (yellow or green) when **Start** is clicked, buffering carries over into the
  new stream — it re-arms with a fresh empty buffer rather than turning off,
  so you don't need to click **Use Buffer** again after Start. Loading static
  data always clears any existing buffer. A small **💾** button next to **Use
  Buffer** enables once the buffer is frozen (green) — click it to save the
  buffered frames to an HDF5 file you choose, written as a `(N,H,W)` dataset
  named `buffer/data`. The same buffer is also what **Import from…** offers to
  other tabs — see **Cross-tab data import** in §1.
- **Stopping live streaming does not auto-reload the previous file/folder/HDF5
  source** — the Data card below gets a small **⟳ Reload** button next to its
  path field for exactly this: click it after Stop to restore the static data
  that was loaded before streaming started.
- **Ring overlay is static per-frame while streaming**: rings drawn by "Simulate
  rings" (Ring simulation card) keep rendering on top of each new live frame at
  their already-computed geometry — arrival of a new frame does **not** by itself
  trigger a recompute (ring radii don't depend on frame data). With Simulate
  rings' live toggle on, changing a material/lattice/geometry field still
  recomputes and redraws immediately. **Stop** leaves the last received frame on
  screen. Closing midas-gui also stops any running stream.
- **Colormap/level changes persist across live frames**: dragging the histogram's
  level range or its own zoom (or the cmap dropdown) is remembered and reapplied
  to every new incoming live frame instead of being reset to the vmin%/vmax%
  percentile defaults. Editing vmin%/vmax%, toggling Log/Linear, or loading a
  new file/frame/dataset switches back to auto-levels — see §14 for the general
  rule shared by every image viewer in the app.
- **B-PILOT auto-start bridge**: on launch, MIDAS GUI opens a local-socket
  server (`midas_gui/bridge_server.py`) that lets **B-PILOT** (a separate
  Bluesky plan-runner GUI) trigger Live Data with no clicks here — when
  B-PILOT dispatches a scan on a known detector, it sends a `{"type":
  "live_pv", "prefix": ...}` message; MIDAS GUI resolves the prefix against
  **Preferences ▸ Devices** and, if it matches, checks/expands the Live Data
  card, fills in the PV and clicks **Start** automatically (switching PVs
  first if a different stream is already running). No-op — and silent in the
  log only — if B-PILOT never connects or the prefix isn't a known device.

### Beam-centre picking (on the image)
The image viewer has **Pick BC** (single click sets the beam centre) and **Pick Ring**
(click ≥3 points on a ring; a circle fit estimates the beam centre). Either updates
BC_y/BC_z and re-runs the overlay + radial integration. Pick Ring's picked points and
fitted circle are drawn in **blue**, distinct from the **amber** Simulate-rings overlay
so the two aren't confused when both are visible on the same image. **Clear** removes
all of it at once — Pick Ring's points/fit **and** the Pick BC crosshair marker,
regardless of which tool left it on screen.

### Pixel readout (bar under the image)
The bar below every image viewer reports the pixel under the cursor:
`x (col)`, `y (row)` and `intensity`, and — wherever a calibration is
available — the reciprocal-space position too: **2θ**, **Q**, **d** and the
azimuth **η**. It tracks incoming frames during live acquisition without the
mouse moving, and it updates when the geometry changes underneath it (editing
a beam centre or Lsd, or a fit landing) rather than waiting for the next
cursor move.

Details worth knowing:

- **2θ is tilt-aware.** It is the exact inverse of the same forward
  projection the ring overlays are drawn with, so on a tilted geometry the
  readout and the rings always agree.
- **η = 0 is straight up (+Z)** and increases towards +Y — the same
  convention the cake's η axis and the η spokes of the bin grid use.
- **d is shown as `—` at the beam centre**, where it diverges.
- **Without a wavelength** only 2θ and η appear; Q and d need λ. Without a
  distance / beam centre / pixel size the pixel cannot be placed at all and
  the bar shows just x, y and intensity, exactly as it did before.
- **Calibrate tags the value `(seed)`** while it is computed from the seed
  boxes rather than a fitted result, so the number never quietly changes
  meaning once a calibration lands. Pick BC, "Send →" from the Data Viewer
  and manual seed edits all move it immediately.
- **Mask Builder displays the raw detector image** while a calibration lives
  in the transformed frame, so the hovered pixel is mapped across before the
  geometry is applied. If the loaded image does not match the detector the
  calibration was fit on, the reciprocal-space fields are omitted rather than
  guessed.

### Top-N brightest pixels (image toolbar)
A **Top-N pixels** toggle button (with an **N** spin box) sits on the image toolbar.
When on, the **N highest-intensity pixels** of the current frame are marked with a
crosshair inside a translucent circle (cyan) centred on each pixel, and the Intensity
statistics panel switches to show the statistics of just those N pixels ("Top N
pixels"). It follows frame changes while active. Clicking the button again removes the
markers and restores the normal statistics. Useful for quickly locating saturation /
hot pixels.

An **I >** checkbox next to the N spin box enables an optional intensity floor
(a field to its right, editable once the checkbox is on): when set, pixels at
or below that value are excluded from ranking entirely — never marked, never
counted toward N — so a low-signal frame can't be forced to mark N pixels that
aren't actually meaningful. If fewer than N pixels clear the threshold, fewer
than N markers are drawn (down to none).

### Lab-frame axes overlay (image toolbar)
A **Lab-frame axes** checkbox on the image toolbar overlays the APS/MIDAS lab
coordinate system on the image, anchored at the beam centre: a red **+X_Lab
(+Y_MIDAS)** arrow, a green **+Y_Lab (+Z_MIDAS)** arrow, a blue **⊗** glyph
at BC labelled **+Z_Lab (+X_MIDAS, beam)** for the beam direction, and an
orange η-sweep arc from η=0° to η=45° with a tick at η=0°. It's a visual
sanity check for orientation/ImTransOpt — a known feature should land in the
quadrant the overlay predicts. The overlay redraws automatically whenever
the beam centre, tilt, calibration, or displayed image changes, and stays
anchored under pan/zoom since it's drawn directly on the image view.

**The convention, stated once.** η is measured **from vertical**, positive
towards screen-right:

| η | direction on screen | lab axis |
|---|---|---|
| 0° | up | +Y_Lab (+Z_MIDAS) |
| +90° | right | −Y_MIDAS (−X_Lab) |
| 180° | down | −Y_Lab |
| −90° | left | +Y_MIDAS (+X_Lab) |

with +Z_Lab (+X_MIDAS), the beam, going into the page. The displayed image
follows it: the array's **column** index runs left→right and its **row** index
runs bottom→top (row 0 is at the bottom, matching the MIDAS origin). The bin-grid
spokes, this compass, and the backend's own `eta = atan2(-Yc, Zc)` are all the
same convention, and `tests/test_lab_frame_conventions.py` asserts that they
stay that way.

What the GUI cannot check for you is whether the *file* was written in that
frame. Load a known pattern, turn the overlay on, and confirm a feature you can
identify lands where the compass says it should; if it doesn't, fix it with
ImTransOpt before calibrating or applying corrections, not after.

### Region-of-interest (ROI) tool (image toolbar)
A **ROI: Box / Line** row sits on the image toolbar, alongside a
**Clear ROIs** button. Click Box or Line to arm it, then click-drag on the
image to draw the shape (a drag shorter than a few pixels is treated as a
cancelled attempt — the mode stays armed so you can retry). Arming a shape
mode automatically disarms Pick BC/Pick Ring and vice versa, since they all
read the same click-drag. Drawing a shape opens a small **floating stats
popup** next to it and un-arms the button (one-shot, like Pick BC). While
dragging out a **line** ROI, the live preview is a true diagonal line
(previously it looked like a rectangle until the mouse was released); a
**box** ROI's drag preview is unchanged.

Each ROI gets its own color (cycled from a fixed palette) shared by the
on-image shape, its on-image label, and its popup's title/label field, so
multiple simultaneous ROIs stay easy to tell apart. The label is editable —
typing a new name in the popup updates the on-image label to match. New
ROIs are numbered "ROI 1", "ROI 2", ... in creation order; **Clear ROIs**
resets that counter, so the next ROI drawn after a clear starts back at 1.
All on-image ROI text (the "ROI N" label, and a box's corner/width/height
annotations below) is drawn with a translucent dark background so it stays
legible over bright image content, in a font ~20% larger than the app
default.

A **box** ROI additionally shows its pixel geometry directly on the image,
in the ROI's color, updating live while it's dragged or resized: the
bottom-left corner's `(x, y)` coordinate (rounded to the nearest whole pixel)
next to that corner, `w = ...` below the bottom edge, and `h = ...` to the
right of the right edge (also rounded to whole pixels — no fractional-pixel
values are shown). A **line** ROI shows its length in pixels (`NN px`,
rounded) at the line's midpoint. Every ROI's editable "ROI N" label is
anchored at the shape's actual position — for a line, that's its current
start point (the end where distance = 0 in the popup's profile, swapping
ends when **Flip direction** is used), not a fixed image corner.

- **Box** popups show a linear/log intensity histogram of the pixels inside
  the box, plus a zoomed-in crop of the boxed region rendered with the
  tab's current colormap. The crop image is not fixed-size — dragging the
  popup window's edge to resize it grows or shrinks the crop image along
  with the histogram. When a bad-pixel mask is active (composite mask file
  or the intensity-exclude controls), masked pixels are excluded from the
  histogram/stats and are marked in translucent bright red on the crop
  image, matching the main viewer's bad-pixel overlay convention.
- **Line** popups show a live **intensity-vs-distance profile** along the
  line (distance measured from one endpoint), plus N/min/max/mean of the
  sampled values. The line itself is drawn as a single arrow (shaft + head
  in one shape, recomputed from the two endpoints on every drag) pointing
  toward the end where distance = 0 → increasing runs; a **Flip direction**
  button in the popup reverses it. Masked pixels along the line are excluded
  from min/max/mean (shown as gaps in the profile); a line drawn entirely
  over masked pixels shows "(all pixels masked)" instead of the stats.

Both the box's histogram and the line's intensity-vs-distance profile cap
how far you can zoom or pan out — the same range-limiting used on the tab's
radial-integration plot — so scrolling/dragging the plot can't lose the
data in an empty view; the limit is recomputed from the current data range
each time the popup refreshes.

Dragging or resizing a shape (or right-click → drag a handle) recomputes its
popup immediately, and every popup also refreshes automatically on frame
navigation, new live-streamed frames, and dark/bright/background/mask
correction changes — the same as the rest of the tab's live readouts.

A popup's on-screen **position is independent of the shape's position**: it
opens next to its shape once, at creation time (placed on whichever monitor
the shape is on, useful for multi-detector/multi-screen setups), and after
that it's a normal free-floating window you can drag anywhere — moving it
never repositions again on its own, only its contents stay live.

Closing a popup (its window close button) removes its shape from the image;
right-clicking a shape and choosing **Remove ROI** closes its popup too.
**Clear ROIs** removes every ROI and popup at once. ROIs are session-only —
they are not saved with the rest of the tab's state.

**Always on top, minimize to ribbon.** ROI popups stay **above every other
window** (not just above the main midas-gui window) so clicking elsewhere in
the GUI, or in another application, never buries one behind something else.
Minimizing a popup with the window's own **OS-level minimize button** (the
title bar/traffic-light control) doesn't send it to the dock/taskbar as a
normal window minimize would — instead the popup hides and a small colored
square button, color-matched to its ROI, appears on a narrow **ribbon**
along the left edge of the image viewer. Clicking a ribbon button restores
that popup (shows it, raises it, gives it focus) and removes the ribbon
entry. A minimized ROI's shape stays on the image and its stats keep
updating live in the background exactly as if the popup were open —
minimizing only hides the window. Removing a minimized ROI (right-click →
**Remove ROI**, or **Clear ROIs**) also removes its ribbon entry.

### Intensity statistics (left panel, bottom)
At the bottom of the Data-Loader panel, in a **draggable pane** below the loader
cards — grab the splitter handle above the panel to make the statistics area taller or
shorter. It holds a **histogram** of the intensity distribution (full range, log-y
toggle) with a **textbox** beneath it reporting N (pixel count) and the
**p70 / p90 / p99 / p99.9 / p99.99** percentiles — each with the **number of pixels
above** that value. The histogram's lower-left corner is fixed at x = 0, y = −2 and both
axes rescale to `(0, xmax)` / `(−2, ymax)` on every refresh (frame change, scope change,
projection, new data); zooming/panning is capped at that same range (as with the ROI
popup plots and the radial-integration plot), so it can't be scrolled out until the
distribution is lost in empty space. It reflects the **corrected** image (dark / bright / background)
with masked pixels (file masks + the intensity-range mask) excluded, and updates live as
any of those change. A scope selector switches between the **current frame** (per the
slider) and **All frames** (combined over the whole stack/folder). When a **Projection**
is active the panel shows the projected image's statistics (the scope selector is
disabled); when **Top-N pixels** is active it shows those pixels' statistics.

A small **"A"/"M"** button pair sits in the histogram's bottom-left corner
(replacing pyqtgraph's native auto-range corner button), same control as on the
radial-integration plot below — see **Manual axis limits (A/M toggle)** under
that section for the full behavior (native right-click "Manual" min/max
fields, persistence across live updates, reclick-to-reset). In **Manual**,
the histogram holds those limits instead of auto-rescaling to
`(0, xmax)` / `(−2, ymax)` on every refresh (new frame, scope change,
Top-N toggle).

### Radial Profile / Eta vs R Cake plots (bottom-right)
A **Radial Profile** tab and an **Eta vs R Cake** tab sit side by side below
the image; both come from the same azimuthal-integration pass — Cake shows
the same run's full 2-D (η, R) binning as a heatmap, purely for inspection
(no separate controls of its own; it re-renders whenever a new Radial
integration runs, only when the tilt/distortion-aware MIDAS engine path is
in effect — the fast geometry-free circle-binning fallback has no cake to
show).

Below the image is a live **azimuthal mean about the beam centre** (`R bin`,
`Integrate`, and an `Auto` toggle that recomputes on frame/BC/mask change) — geometry-free
circle binning by default, or the tilt/distortion-aware MIDAS engine when a full
geometry is in effect (a loaded calibration file, or non-zero ty/tz in the
Ring-simulation card — see the Load/save calibration card above). Peak/ring markers overlaid on
the plot use the ring's true 2θ, so they line up with the profile in either mode.
**Clicking a radius on the plot draws the matching ring (magenta) on the image.** Axis
units switch between R (px) / 2θ / Q; the **X-axis lower bound defaults to 0**. A small
circular **"?"** button next to the `Radial` control opens a message box explaining how
the profile is computed (full-geometry (η, R) binning vs. the circle-binning fallback).
The **Exclude range** checkbox and its `<` / `>` integer bounds (see previous
section) sit pinned to the far right end of this same toolbar row — the `R bin`
field was narrowed and the `N bins | max=...` stats printout that used to occupy
that space was removed to fit them.

**The default view always auto-fits the current profile's X/Y extent** — it never gets
stuck showing a stale or unrelated range from an earlier profile. If you manually
zoom or pan (drag, wheel-zoom, box-zoom, or the axis context menu), that exact view is
**preserved across parameter changes** (new frame, changed BC/tilt/Lsd, etc.) instead of
snapping back to full range every time the profile updates — the curve redraws in
place under your current zoom. A manual zoom is only cleared when it would no longer
make sense: switching the X-axis unit (R/2θ/Q) drops the remembered X range, and
toggling **Log Y** drops the remembered Y range (both because the old numbers belong to
a different scale). **Pan and zoom are always bounded to the current profile's extent**
(a margin around its X/Y range) so you cannot drag or scroll off into empty space. The
splitter handle above this panel (between it and the image view) is wider than a
default Qt splitter, to make it easier to grab.

#### Manual axis limits (A/M toggle)
A small **"A"/"M"** button pair sits in the plot's bottom-left corner, in the
spot pyqtgraph's native auto-range button normally occupies (that native
button is hidden in favor of this pair). **A** (default) is the auto-fit
behavior described above. **M** switches to **Manual**, which holds exactly
the limits set via each axis's own **native right-click menu** — right-click
the plot, open **X axis** or **Y axis**, pick **Manual**, and type a min/max
(this is stock pyqtgraph, not a MIDAS-specific control). Once **M** is
active, the plot's axes show **exactly** those typed values (e.g. entering
`0` shows `0`, not a padded/rounded value) and hold them through every
live-acquisition redraw — Manual mode stops the plot from re-fitting or
re-clamping its range on each new frame, and editing the min/max fields
again takes effect immediately, live acquisition or not. Switching to **A**
does not discard the typed values — clicking **M** again restores exactly
what was last entered. **Reclicking the already-active button resets the
view to that mode's default**: **A** forces an immediate re-fit to the
current profile (discarding any manual pan/zoom drift), and **M** snaps
the view back to the held manual limits (discarding any drift from panning
around while still in Manual).

---

## 4. Tab 1 — Mask Builder

**Purpose:** identify and exclude bad pixels. The final mask is a uint8 array
(1 = bad) broadcast to Tabs 2, 3, 4, 5. Browsing an image or a mask loads it
immediately. Pointing the **Image** field at an `.h5` file reveals a **Dataset**
dropdown (auto-populated with the file's datasets and shapes, preferring a
≥2-D dataset) — the same pattern used by the Stack field below it and by the
Data Loader panels elsewhere in the app. The **Image** field's browse button
also has an **Import from…** submenu of whatever file/folder is currently
loaded in another tab (see **Cross-tab data import** in §1); the **Stack**
field's browse menu additionally lists any other tab's frozen (green) Live
Data buffer as a **Buffer (N frames)** entry — picking one snapshots that
buffer to a temporary HDF5 file and feeds it into the auto-mask stack exactly
like a browsed folder/file would. The Stack browse menu also has a
**Files (multi-select)…** entry that opens a multi-select file dialog so the
temporal stack can be built from an explicit, hand-picked set of files
(e.g. frames scattered across a directory or spread across multiple
directories) instead of only a whole folder or a single glob pattern — the
field then shows the files' shared parent folder (or the one file's own
path if only a single file was picked; hover for the full file list) and
the stride still applies to the chosen list.

**No Transforms row.** Unlike Data Viewer and Calibrate, this tab has no
Flip Y/Flip Z/Transpose checkboxes of its own — the image is always loaded
and previewed exactly as stored on disk, and every mask this tab produces
(threshold, statistical, spike, cosmic-ray, azimuthal, learnable, and
hand-drawn shapes alike) is therefore always in **raw detector-space**,
matching a file/folder mask loaded from disk. Tabs 2/3/4/5 apply the active
calibration's own `ImTransOpt` to the mask automatically, at the same point
they transform (or hand off to the backend to transform) the image it pairs
with — so a mask built here never needs to be flipped by hand to match a
flipped calibration.

### Section 1 · Threshold mask (always applied)
`pixel ≤ lower | pixel > upper`. The upper bound auto-fills from the data type on load
(e.g. 1,048,575 for uint20 Eiger, 4,294,967,295 for uint32).

### Section 2 · Statistical auto-mask
Two **independently selectable** methods — enable either or both:
- **Spatial outlier** — robust local-anomaly detection (5×5 median residual → 15×15
  local MAD → Z-score → hot/dead/saturated gates). Uses a **temporal median** over the
  stack folder when provided (cleaner reference), else the single frame. Controls:
  `K_σ` (6.0), `Hot` (1.5), `Dead` (0.5).
- **Temporal constancy** — flags frozen pixels whose frame-to-frame std is below
  `Frozen × Q75(std)` (0.05); needs a stack of ≥2 frames.

A shared **stack source + stride** feeds both (the temporal median for spatial, the
frame stack for temporal constancy). The stack can be a **folder / `*.tif` glob**, an
explicit **multi-select file list** (via the browse menu's Files (multi-select)…
entry), or a **single HDF5 file whose 3-D dataset is a time sequence of images** — for
an `.h5` a **Dataset** selector appears (auto-populated, 3-D datasets preferred). *All
σ / K_σ / n_σ fields in this tab accept any value — there is no upper limit.*

### Section 3 · Spatial spike rejection
Laplacian high-pass single-pixel-spike detector; `n_σ` threshold (5.0).

### Section 3b · Cosmic-ray rejection (temporal)
Per-pixel temporal σ-clip across a ≥3-frame stack; anomalies OR'd across frames.
`n_σ` (5.0).

### Section 4 · Calibration-based masks (need a Tab 2 result)
- **Azimuthal σ-clip** — flags (R, η) cells deviating from the azimuthal mean.
- **Learnable mask** — differentiable per-pixel mask trained against an η-uniformity
  loss (`steps`, `lr`, `sparsity`).

### Section 5 · Post-processing
**Dilation (px)** (default 0) grows every bad pixel from Sections 1-4 (and a
mask loaded via **Save / Load**) using 8-neighbor morphological dilation: at
N=1 the full 3×3 block around each bad pixel becomes bad, at N=2 the full
5×5 block, and so on (a (2N+1)×(2N+1) square centered on each pixel).
Applied once, on **Compute Mask** / mask load, before hand-drawn shapes are
combined in — hand-drawn regions are never grown by this control.

### Draw mask tools
Interactive rectangle / oval / circle / polygon / annulus / point tools, live-draggable;
**Apply shapes → mask** rasterises them (OR'd with the computed mask). **Clear shapes**
removes them.

### Save / Load
Save the combined mask as TIFF (0 = good, 1 = bad); loading a TIFF applies immediately.
A **Log to Project** button next to Save records the current mask — its full
parameters and the resulting compressed mask array — as a new FAIR-provenance
attempt in the currently-open project (see §17), enabled once a mask exists.
Unlike Calibrate/Batch Integrate there's no single "run finished" moment to
log from automatically (a mask can come from Compute, Load, hand-drawn
shapes, or any combination), so this is an explicit, click-when-ready action.

---

## 5. Tab 2 — Calibrate

**Purpose:** determine detector geometry (Lsd, BC, tilts, distortion, wavelength) from
a calibrant pattern.

### Mode ribbon (leftmost strip)
Like the Data Viewer tab, a narrow vertical strip switches between **Single
detector** (the view described below — unchanged) and **Hydra** (per-panel
calibration for the 1-ID-E 4-panel GE detector). The two modes are
independent: switching does not share data or geometry between them. As on
the Data Viewer tab, **Hydra only appears when the active profile is
1-ID-E** (see §3 "Mode ribbon", §16 "Profiles").

### Hydra mode (4-panel GE detector)

**Purpose:** fit each of the 4 GE panels' own detector geometry (BC, Lsd,
tilts, distortion) from one calibrant dataset, using one shared "recipe"
(pipeline, λ/pixel/calibrant, refine-parameter choice) applied to all 4 —
since each GE panel is a physically separate detector, its beam centre,
Lsd, and tilt are fit and shown independently.

- **Hydra data / ← Data Viewer (left panel)**: the same sibling-discovery
  data loader as the Data Viewer tab's Hydra mode (point at any one panel
  file, the other 3 are found automatically; shared frame slider; Dark /
  Bright / Background per panel). **← Data Viewer** pulls the currently
  loaded Hydra panel path *and* every present panel's fitted/seeded geometry
  (BC, Lsd, tilts, Transforms) straight from the Data Viewer tab's Hydra
  page, so you don't have to re-browse or re-pick beam centres you've
  already set up there. This mode has no stack-projection feature (like the
  single-detector Calibrate tab) — use **Mean of frames** below instead.
- **Detector & Calibrant, Threshold, Mean of frames, Refine parameters,
  Advanced (middle panel, shared across ge1–ge4)**: one copy of each,
  identical in meaning to the single-detector tab's own cards, applied to
  every panel's fit. There is no **Multi-panel detector** (tiled sub-panel
  rigid-shift) group in Hydra mode — that feature refines shifts between
  tiles *inside* one monolithic detector's readout, which doesn't apply to
  4 separate physical GE detectors.
- **Transforms / Initial seed (middle panel, switches with the active
  panel)**: independent per GE panel — physical mounting orientation
  (Flip Y/Flip Z/Transpose) and beam centre/Lsd/tilts genuinely differ
  panel to panel. **Load calibration file…** seeds one panel's BC/Lsd/tilts/
  Transforms from a file (and mirrors λ/pixel into the shared Detector card
  if the file carries them); Pick BC/Pick Ring on the image seed that
  panel's BC the same way as the single-detector tab. **Use manual seed**
  and **Feed result back to seed** are one shared choice across all 4
  panels — checking/unchecking either on any one panel's card mirrors the
  same state onto the other three — since it's a single decision about
  *how* the LM fit should start/carry forward, not a per-panel geometry
  value; the seed BC/Lsd/tilt *values* themselves remain fully independent
  per panel.
- **Run (Run mode / Run Calibration / Abort)**: **Sequential** fits one
  panel at a time (full per-line progress in the Log tab, same as the
  single-detector tab); **Parallel** starts every currently-found panel's
  fit at once — faster, but each panel's fine-grained progress prints go to
  the console rather than the Log tab (only start/finish/error lines appear
  there), since a fit's internal progress-capture can't safely be shared
  across concurrent threads.
- **Image toolbar**: **GE1 / GE2 / GE3 / GE4** (no Composite — calibration
  is inherently per-panel; see the Data Viewer tab for the windmill
  composite once all 4 are fitted) plus **Show rings**, identical in
  meaning to the single-detector tab's own predicted-ring overlay toggle,
  applied to whichever panel is active. As there, the rings are always
  drawn through the fitted tilts and distortion.
- **Radial Profile (bottom right)**: one shared multi-curve plot — GE1-4
  are ordinary checkboxes (any combination can be shown at once), same
  style/controls as the Data Viewer tab's Hydra Radial Profile plot
  (R-bin/η-bin/weighting + **Re-integrate**, X-axis unit selector). An
  **Overall** button sits alongside them instead of a fifth checkbox —
  click it and it turns green, hides GE1-4's curves, and plots their
  NaN-aware summed profile (resampled onto a shared 2θ axis first, so
  overlapping panel coverage isn't double-counted), rescaling the view to
  fit; click it again to hide the summed curve and bring GE1-4 back exactly
  as they were checked. **Ring Residuals** and **Results** switch to show
  only the currently active panel's own chart / parameter grid (Results
  has its own **→ Send to Data Viewer**, **Save .json**, and **Save
  paramstest.txt**, scoped to that one panel). Once a Hydra run finishes
  with **Overall** active and a project is open, the summed profile is also
  logged to the project as its own record (see §17).
- **Eta vs R Cake (bottom right)**: its own row of **GE1/GE2/GE3/GE4**
  checkboxes (mutually exclusive — picking one shows that panel's 2-D
  heatmap and syncs the image toolbar to match) plus an **Overall** button
  that computes and shows a single summed cake across all 4 panels
  (resampled onto a shared 2θ axis the same way the profile's Overall does;
  all 4 panels already share one η axis by construction, so only the R
  axis needs resampling). Recomputed on demand each time it's clicked, from
  whichever panels have already been integrated.
- **Log** is shared across all 4 panels, each line prefixed `[ge1]`…`[ge4]`.

### Data Loader panel (left)
The calibrant frame (Data + Frame index), Dark, Bright, Background and Mask are selected
in the shared **Data Loader panel** (see §1). Each Dark/Bright/Background field is
averaged over a chosen index range; Bright offers **Flat-field divide** or **Subtract**;
all mask sources (plus the auto-added Tab 1 mask) are unioned. Dark is passed to the
calibration pipeline; bright/background are applied to the calibrant before calibration
and post-calibration integration. The image preview updates live to show the
dark/bright/background-corrected frame as soon as a field is picked or changed (the raw
frame still feeds the actual calibration run, which applies these corrections itself).
On the **1-ID-E** / **20-ID-D** / **20-ID-E** profiles, loading the Data file also
auto-detects **pixel size** (from a `.ge1`–`.ge5`/`.vrx`/`.pxrd` filename tag) and, for
an HDF5 file, **wavelength** (from its recorded beam energy) into the Detector card
below — whatever isn't detected is left unchanged. (Hydra mode: the same detection runs
off the loaded anchor panel file and applies to the shared λ/pixel fields.)

### Detector, seed & Load calibration file
The **Detector & Calibrant** card sets λ, pixel size(s) and detector transforms
(**Flip Y / Flip Z / Transpose** — MIDAS's `ImTransOpt`, in that fixed order).
Toggling a Transforms checkbox live-updates the image preview immediately, the
same way Data Viewer does; Pick BC / Pick Ring clicks are read
straight off that (transformed) preview, so the seed beam centre always lands
in the same coordinate space the fit will actually run in. The calibration run
itself applies the identical transform (to the calibrant image and to Dark) right
before handing the array to the pipeline, so a Flip/Transpose selection is now
honoured by every pipeline mode, not just the display; the **Initial seed** card
sets the LM starting point (BC_y, BC_z, Lsd, **and tilts tx/ty/tz**). **Load calibration file…** (top of the Detector card) reads a MIDAS
**paramstest `.txt`**, a calibration **`.json`**, or a pyFAI **`.poni`** (auto-detected)
and fills λ, pixel size, the seed BC + Lsd, and the Transforms checkboxes from any
`ImTransOpt` lines — a fast way to start from a previous calibration or a known
geometry. (The synthetic test data ships
`test_data/calibration_synthetic.{json,txt,poni}` as a ready example.) **← Data Viewer**
(next to *Load calibration file…*) pulls λ, pixel size, Lsd, beam centre, the Data
Viewer's ty/tz tilt fields, and its Transforms state straight from the Data Viewer tab
into the same fields (BC and Lsd land in the seed card; ty/tz land in the seed tilt
fields). A **Feed result back to seed** checkbox (on by
default) copies the optimized BC / Lsd / tilts / distortion of each run back into the
seed fields, so a follow-up run starts from the previous solution. *Seed tilts are
honoured by the Four-stage / advanced pipelines; the One-shot / First-time paths seed
tilts only if the installed backend exposes initial-tilt options.*

### Mean of frames
For a multi-frame source (HDF5 / folder), **Combine frames into a single mean image** builds
the mean of a frame range and calibrates on that. **start** / **end (0 = all)** select
the range. The card is disabled for single-frame sources; the preview updates
live as the options change.

### Pipeline selector
One-shot (default) · First-time · Four-stage · Bayesian (Laplace σ) · Joint-cake.
*For trustworthy tilt/strain, prefer Four-stage or First-time — One-shot/Bayesian can
report a spurious self-compensated tilt on weakly-tilted data.*

### Refine flags
Which parameters vary: Lsd, BC, ty, tz, tx, Wavelength, plus a "Residual map" build
toggle. The **Distortion (n/15)** checkbox has a companion **…** button that opens a
per-coefficient dialog: the 15 distortion coefficients are grouped by η-fold (isotropic
radial + folds 1–6), and named **preset modes** (None · Isotropic only · Iso + up to
2-fold · Iso + up to 4-fold · All (15)) auto-select whole ladders. The checkbox label
shows how many coefficients are selected. Picking a strict subset genuinely restricts
which coefficients the LM fit refines — including on the One-shot pipeline, which
transparently switches to a lower-level routine for a partial selection (its own
backend otherwise only supports all-15-or-none); picking "All (15)" or leaving
Distortion unchecked runs the normal One-shot path unchanged. Advanced (E-M / LM
iters, device, output dir) and Multi-panel detector groups are collapsible.

A **Refining: … Fixed: …** line above the checkboxes states the current selection in
words, so what the fit will actually vary is readable without decoding six checkboxes.

Both halves of the Distortion row — the tick and the per-coefficient selection behind
the **…** — are saved with the project and restored on reopen. Where the GUI Workspace
and a recorded calibration attempt are both restored in one **File ▸ Open Project…**,
the workspace wins for every input field: it was saved when you pressed Ctrl+S, whereas
the attempt records what the fit used when it ran, so the workspace is the later of the
two. The attempt still supplies its fitted result, cake and panel shifts. Opening an
attempt *without* its workspace restores that run's fields, including the exact
coefficient subset it refined.

The defaults differ by calibrant kind, and the two sets are remembered separately.
A crystalline calibrant fills the detector with rings and constrains Lsd and tilt
well, so it starts at **Lsd + BC + ty + tz**. A d-spacing calibrant is fit from a
handful of hand-picked points, often on a single short arc, where floating Lsd on
top of BC is badly conditioned and tilt is not identifiable at all — so it starts at
**BC only**. Switching calibrant kind swaps the two sets; changes you make while
staying on one kind stick. See *Non-crystalline calibrants* below for why.

### Live threshold slider
Zeroes calibration-image pixels below the slider value (background suppression for
ring-finding); the preview updates instantly.

### Pick BC / Pick Ring
Click the image to seed the beam centre (single click) or fit a ring (≥3 clicks); the
Pick Ring points/fit are drawn in blue, distinct from the amber Simulate-rings overlay.

### Non-crystalline calibrants (AgBH / custom d-spacings)
Calibrants with no usable space group are fit from picked points against a list of
known d-spacings, bypassing the crystallographic backend entirely. Select **AgBH
(silver behenate)** or **Custom d-spacings…** in the Calibrant dropdown (the latter
takes a comma-separated list in Å) and the tab switches to this mode: the Distortion
row, residual-map build and multi-panel groups hide (the manual fit supports none of
them) and a **Pick d-spacing pts** tool and **Fit Geometry (manual)** button appear.

Pick points with **Ring #** set to the ring each point belongs to — ring 1 is the
innermost (largest d). Points are drawn as a **black-haloed open circle in that
ring's colour**; the halo keeps the marker readable over a bright arc (ring 1's red
would otherwise vanish into a hot-colormap ring) and the open centre leaves the
picked pixel visible. **Undo** removes the last point, **Clear** all of them, and the
status line tracks the count per ring. Picks are saved with the project, so a
reopened session can re-run the fit without re-picking.

**Fit Geometry (manual)** minimises the 2θ residual of those points over whichever
parameters the Refine flags leave free (`least_squares`, Levenberg–Marquardt when
unbounded, trust-region reflective when any limit is set).

**Why BC only by default.** At a long sample–detector distance the rings subtend a
very small 2θ, and Lsd, beam centre and tilt become nearly degenerate — several quite
different geometries fit the same picked points about equally well. The solver then
converges, reports success, and returns a value that is mostly noise. Measured on a
13.5 m SAXS geometry (λ = 0.1730 Å, 55 µm pixels, 3072×512 frame) with 8 points on the
one visible 42° arc of AgBH ring 1:

| refining | result | drawn ring error |
|---|---|---|
| Lsd + BC + ty + tz | Lsd 12886 ± 3690 mm, BC_z 157 ± 387 px, ty −2.4 ± 406° | 4.8 px |
| BC only | BC (128.40, 124.74) ± (0.37, 1.19) px | 0.95 px |

Both report `success`. Only the second is a measurement. Refining more than BC is
worth doing when the picks justify it — points on several rings, spread over a wide
arc — and the σ readout below is how you tell.

**Uncertainty and identifiability.** The fit reports a 1σ estimate per refined
parameter, from the covariance of the Jacobian at the solution. Refined rows in the
**Results** grid show `value ± σ`; held rows show `(fixed)`; a parameter resting on
one of its limits shows `(at limit)`, where σ is not meaningful. The Log lists every
refined parameter as `value ± σ` and, for any whose σ is too large to call it
measured (>1 % for Lsd/λ, >5 px for BC, >0.5° for a tilt) or that could not be
determined at all, prints:

> `WARNING: … not constrained by these picks — the value above is largely fitted
> noise. Pick points on more rings or over a wider arc, hold the parameter fixed, or
> bound it via Limits…`

This warns, never blocks — loose data may be exactly what you meant to fit.

**Limits** (the ± column in the Refine card) bounds a parameter to a window around its
seed, for holding a quantity you already know — a measured sample–detector distance —
near its true value while the fit determines the rest. Each row is `[± value] [unit]`,
with a live readout of the resulting range. The unit is either `%` of the seed or the
parameter's own absolute unit; tilts default to absolute because they seed at 0°, where
a percentage window would pin the parameter exactly (a degenerate percentage window
falls back to the row's absolute default rather than pinning it).

Both calibrant kinds are bounded, but they mean different things, so the column is
shaped for each.

*Manual (d-spacing) fit* — one row per free parameter, each with an **enable
checkbox**. All rows start off, so an untouched card leaves the fit unbounded on the
Levenberg–Marquardt path it has always used; ticking any row switches the solver to
trust-region reflective.

*Crystalline calibrants* — the MIDAS backend **always** bounds the fit
(`CalibrationParams.tolLsd` and friends become hard `(lo, hi)` constraints on the LM
solve), so there is no "off" state to offer: an untouched CeO2 fit already runs at
**±15 mm** on Lsd, **±20 px** on the beam centre, **±3°** on tilt, **±0.001 Å** on λ
and **±0.01** on the distortion coefficients. The rows are therefore always active and
prefilled with the windows actually in force, so the card shows the real constraint
rather than inviting you to add one. The backend's windows are coarser than the manual
fit's — one value covers both beam-centre coordinates, one covers both refined tilts,
and one covers all fifteen distortion slots — so those rows are merged, and `tx` has no
row at all because this backend never refines it.

Tightening a window also **shrinks that seed field's arrow step**, to 10 % of the full
range: at ±15 mm the Lsd arrows move 3 mm, at ±2 mm they move 0.4 mm. Rows with no
window in force fall back to the steps set in Preferences.

**When the fit stops on a bound**, the parameter is reporting the bound rather than a
measurement — a window that is visible but silently binding is barely better than an
invisible one. Any such parameter is marked `(at limit)` in the **Results** grid and
named in the Log, with a prompt to widen the window or check the seed. This needs the
window's centre to be known, so it is reported for a run with **Use manual seed** on;
an auto-seeded run is centred on a seed the GUI never sees, and reports nothing rather
than guessing. Parameters you held fixed are never flagged — they never moved. The Log
also records the windows in force at the start of each run, so a run's own record says
what bounded it rather than only the card, which shows whatever is set now.

One consequence worth knowing: `midas_calibrate_v2.calibrate()`, which the plain
**One-shot** pipeline calls, accepts no window arguments and hardcodes Lsd and the beam
centre as refined. So editing a limit, unchecking **Lsd** or **BC**, or refining exactly
one of **ty**/**tz** makes the GUI route One-shot through the same lower-level routine
the Four-stage / Bayesian / Joint pipelines use, which honours all of them. The Log says
when this happens and why. The trade-off is that this route skips `calibrate()`'s
STAGE-1 multi-hypothesis Lsd search and uses your seed as given, so a poor seed matters
more. **First-time** cannot take windows at all and warns if any are set.

### Predicted-ring overlay (image toolbar)
After a run, the calibrant's predicted ring positions are drawn in **lime** with a
red/yellow beam-centre marker. **Show rings** toggles the overlay.

The rings are always drawn through the *full* forward model — the fitted tilts
(tx/ty/tz) **and** the refined distortion harmonics — so what you see is where the
calibration says each ring actually lands, not a circle approximating it. On a tilted
or distorted detector the curves are visibly bent rather than round; at zero tilt and
zero distortion they reduce exactly to circles about the beam centre. There is no
longer a **Corrected** toggle: correctness was never a display preference, and having
it default to off meant the honest overlay was the one you had to go looking for.

One term is left out: the empirical `residual_corr_map` (a smooth sub-pixel ΔR(Y, Z)
absorbed after the harmonics converge, and present only if you refined **Residual
map**). When a result carries one, the status text beside the toolbar says so.

### Working directory
The **Working dir:** field (bottom of the left panel, with a **…** browse
button and **Suggest**) is the one place calibration writes on its own. Deliberate saves
are unaffected — **Save calibration.json** and **Save paramstest.txt** still ask
where to put the file — but everything generated along the way goes here.

Intermediates land in a `.midas_scratch/` subfolder inside it, one folder per
run (and, in Hydra mode, one per panel inside that): the residual-correction
map, the fit-time `<name>_panelshifts.txt`, and the backend's own scratch
output. Everything in `.midas_scratch/` is re-derivable and safe to delete
whenever you like — the GUI never deletes it for you, so a run's intermediates
are still there when you come back to them.

The field fills itself when you load data, following this order:

1. the folder the data sits in, if its name ends in `_bc`;
2. the nearest folder above it whose name ends in `_bc`;
3. mpe_wf's convention read off the path — `<outroot>/<expid>_bc`, derived from
   the standard `<outroot>/<expid>/<detector>/<froot>/<files>` layout;
4. `<data folder>/<expid>_bc`, using the Exp ID from the header;
5. otherwise **nothing** — the field stays empty and you pick a folder.

An already-`_bc` folder wins over the positional reading because data does not
always sit four levels deep: a file directly inside an `…_sep26_bc` directory
read positionally would propose a folder next to the mount root that nobody can
create. Note this is *not* the same default as Batch Integrate, which appends
`/<froot>/<detector>/` to the same `_bc` root — calibration writes one run's
scratch, not a tree of per-detector outputs.

Autofill never overwrites a path you typed yourself, and never fills in a
folder it can't write to — if the derived default isn't writable it is left out
and the Log says why. **Suggest** re-derives it on demand, so a default you
cleared or overwrote is always recoverable; unlike autofill it fills the field
even when the folder isn't writable, and warns, so you can see what it picked.

Leaving the field empty is allowed: intermediates then go to a temporary folder
that is deleted when the GUI exits, and the Log says so. You lose the residual
map and the fit-time panel shifts on exit unless you Save.

If the folder can't be written to, **Run Calibration** stops before starting
rather than discovering it minutes later with a finished fit that has quietly
failed to record its residual map. Reopening a project whose stored working
directory no longer resolves (a different machine, a mount that isn't there)
logs a warning at open time — the stored path is left alone for you to correct.

### Run / Abort
**Run Calibration** launches the worker; **Abort** terminates it and frees the slot so
you can immediately start a new run (the calibration is one uninterruptible library
call, so abort hard-terminates the worker thread rather than waiting).

### Results (right panel, bottom tabs)
Radial Profile (with ring markers and an **Azim. mean** selector — see Tab 4), **Eta
vs R Cake**, Ring Residuals bar chart, **Results**, and Log. Integration runs
automatically after calibration. The **Eta vs R Cake** tab shows the same
integration as a 2-D heatmap instead of collapsed to a 1-D curve — R (px) on the
X-axis, η (°) on the Y-axis, intensity as color — the routine "cake" visualization
for area-detector diffraction data; it has its own Log/colormap/vmin%/vmax%
controls (vmin% defaults to **30**) plus a color-scale histogram sidebar (same
conventions as the main image viewer) and updates on every **Re-integrate**. The
vmin%/vmax% auto-level calculation excludes exact-zero bins (unfilled η/R bins),
same as the main image viewer, so a partially-empty cake doesn't skew the level
window toward zero. Pan/zoom is bounded to the cake's
own (R, η) extent, like the main image viewers, so you can't scroll/zoom out into
empty space; the mouse wheel zooms **both axes together**, while a right-click-drag
zooms **only the axis you actually dragged in** — a purely horizontal drag zooms R,
a purely vertical drag zooms η, a diagonal drag zooms both — matching the
**Radial Profile** plot's own right-drag behavior. The **Radial Profile** plot is bounded the same
way, to its own data range. The **Results** tab shows the parameter set exactly as it is written to
`paramstest.txt` (Lsd, BC, tx/ty/tz, the distortion coefficients, Parallax, Wavelength,
px, NrPixelsY/Z, RhoD, SpaceGroup, LatticeConstant, and any `ImTransOpt` codes from the
Transforms checkboxes) as **plain text laid out in multiple
columns** so the wide-but-short panel stays readable — no table widget. The distortion
`p0–p14` slots are labelled with their coefficient names (iso_R2, a1, phi1, …), and only
the coefficients actually selected for that run are listed (not all 15), so the display
matches what you asked to refine; the saved `paramstest.txt`/`.json` exports still carry
every slot's real value, including any held-fixed value carried over from a prior
calibration. A strain/timing line sits below. **→ Send to Data Viewer** pushes the *full*
calibrated geometry (λ, pixel size, Lsd, beam centre, **tilts, distortion, and the
Transforms state**) into the Data Viewer tab — where it drives the tilt/distortion-aware
radial integration, not just circle binning, and sets Data Viewer's own Flip/Transpose
checkboxes to match — the reverse of the Data Viewer's "→ Send geometry to Calibrate". The
bottom tab area is fully resizable (drag the horizontal splitter) and the Log fills its
tab.

### Export
`→ Send to Data Viewer`, `Save .json` and `Save paramstest.txt` are all gated on a
completed fit. To use a geometry you have dialled in by hand rather than fitted, build
it in the **Data Viewer** tab — its Ring simulation card exists for exactly that, and
`Geometry: [← Get]` there pulls this tab's calibrated values as a starting point.

**Save calibration.json** and **Save paramstest.txt** (standalone or from a template).
Both carry the Transforms checkboxes' `ImTransOpt` codes (one `ImTransOpt <code>` line
per checked transform in the `.txt`; an `im_trans` list in the `.json`), so reloading
either file elsewhere restores the same detector orientation.

**Multi-panel detector** registers each panel's rigid shift (δy, δz, δθ) as something
every pipeline (One-shot, First-time, Four-stage, Bayesian, Joint) actually refines,
not just a fixed grid the forward geometry accounts for — so a Fit with it checked
produces a real, nonzero per-panel correction. Both exports also carry the refined
per-panel geometry: `NPanelsY`/`NPanelsZ`/`PanelSizeY`/`PanelSizeZ`/`PanelGapsY`/`PanelGapsZ`
(the panel grid) plus `PanelShiftsFile`/`panel_shifts_path` pointing at a companion
`<name>_panelshifts.txt` (one line per panel — id, δy, δz, δθ, δLsd, δp₂) — the exact
fields `midas_integrate_v2`'s detector mapper needs to apply the panel corrections, so
a saved paramstest.txt or calibration.json is immediately usable standalone. This
sidecar is (re)written **at Save time**, next to wherever the `.json`/`.txt` actually
lands — so it always travels with the file you saved, rather than pointing back at the
possibly-temporary file the live Fit run first wrote it to (Fit itself writes into
`.midas_scratch/` inside the **Working dir** — see above; with no working directory
set, panel shifts land in a temporary folder that is deleted when the GUI exits and
the Log says so explicitly, as a reminder that Save is needed to make them
permanent). Each saved file gets its own uniquely-named sidecar (derived
from that file's own name), so saving several calibrations into the same folder never
has them overwrite each other's panel data. If a saved calibration/paramstest and its
sidecar are later copied or moved together to somewhere the originally-recorded path
no longer resolves, loading the file back in still finds the sidecar as long as it's
sitting right beside it.

The same panel geometry is also attached to the in-memory calibration result the
moment a Multi-panel run finishes (not only on explicit Save), so **Results tab
preview** (Radial Profile / Eta vs R Cake) and **Batch Integrate**'s "Use Tab 2
calibration" run both apply the panel corrections automatically; loading a saved
calibration file back in (Batch Integrate's "Load calibration file", or anywhere else
that reads a paramstest/json geometry file) round-trips the same panel fields. A
calibration attempt logged to a **Project** (`.h5`, see §17) also embeds the refined
panel-shift values directly — reopening the project and populating Tab 2 from a
recorded multi-panel attempt regenerates a real `_panelshifts.txt` next to the project
file and points the restored result at it, so the panel correction survives even if
the original run's file is long gone or the project was moved to another machine.

---

## 6. Tab 3 — Calibration Refinement

**Purpose:** optional post-calibration geometry refinement against an **η-uniformity**
criterion (rings should be azimuthally uniform). Uses a **derivative-free Nelder-Mead**
optimiser (the differentiable integrator returns NaN geometry gradients at the
beam-centre singularity on this build).

The sample frame and Dark/Bright/Background/Mask are selected in the shared **Data Loader
panel** (see §1) — the refinement runs on the corrected image. Middle-panel controls:
calibration source (Tab 2 result, or start-from/continue radios), parameters to refine
(BC_y, BC_z, Lsd, ty, tz), optimiser + iterations. **Run Refinement** shows a live loss
curve; **Apply** broadcasts the refined geometry downstream. If Apply is never clicked,
the Tab 2 result flows through unchanged.

---

## 7. Tab 4 — Batch Integrate

**Purpose:** integrate a stack of frames into 1-D profiles using the calibrated
geometry and optional corrections.

### Mode ribbon (leftmost strip)
Like the Data Viewer and Calibrate tabs, a narrow vertical strip switches
between **Single detector** (the view described below — unchanged) and
**Hydra** (per-panel integration for the 1-ID-E 4-panel GE detector). The
two modes are independent: switching does not share data or geometry
between them. As on the other two tabs, **Hydra only appears when the
active profile is 1-ID-E** (see §3 "Mode ribbon", §16 "Profiles").

### Hydra mode (4-panel GE detector)

**Purpose:** integrate each of the 4 GE panels' own frame stack with its
own independently fitted geometry, using one shared integration "recipe"
(kernel, bins, corrections, monitor normalisation, output format) applied
to all 4 — since each GE panel is a physically separate detector with its
own geometry, but the same beam and the same choice of what to compute.

- **Hydra data (left panel)**: the same sibling-discovery data loader as
  the other Hydra pages (point at any one panel file, the other 3 are found
  automatically), but in **streaming** form for batch runs — a shared
  frame **range + stride** (frames are synchronized across panels, so one
  range applies to all) instead of a frame navigator, Dark/Bright/
  Background per panel, and an independent **Mask** per panel (its own
  file/folder sources, unioned — no cross-panel auto-discovery, since mask
  files are physically panel-specific and may not follow the ge{n} naming
  convention data files do).
- **Integration, Physics corrections, Monitor normalisation, Output
  (middle panel, shared across ge1–ge4)**: one copy of each, identical in
  meaning to the single-detector tab's own cards, applied to every panel's
  run. Each panel writes its output into its own `ge{n}/` subfolder under
  the shared output directory, so the 4 panels' files never collide. There
  is no Drift correction or live **MONITOR** folder-watch in Hydra mode
  (both single-detector-only for now).
- **ge1 – ge4 toggle + Calibration source / values / progress (middle
  panel, switches with the active panel)**: each panel picks **From
  Calibrate tab** (auto-populated the moment that panel's fit finishes on
  the Calibrate tab's own Hydra page — no action needed here) or **From
  file** (a calibration `.json`, MIDAS `paramstest.txt`, or pyFAI `.poni`,
  same auto-detection as the single-detector tab); a per-panel **View
  calibration ▾** popup (see "Calibration values" below) and a compact
  progress bar are per panel.
- **Run — two independent levels of parallelism**: a **Panels:**
  Sequential/Parallel choice (unchanged) controls how many of ge1–ge4
  integrate concurrently; a separate **Per panel:** Sequential/Batch
  Parallel + worker-count control (new, shared across panels) decides
  whether *each* running panel's own frames are further split across
  chunk workers — see "Run / Abort / Save" below for how Batch Parallel
  works. **Start Integration** / **Abort** (red) / **Save** (green, saves
  every panel's already-computed lineouts, each into its own `ge{n}/`
  subfolder) sit in one row. Abort stops every running panel after its
  current frame, keeping frames already written.
- **Waterfall / Stacked profiles (right panel, switches with the active
  panel)**: each panel has its own pair, same controls as the
  single-detector tab's own viewers. A shared **Log** below is prefixed
  `[ge{n}]` per line so all 4 panels' activity can be read from one place.

### Data Loader panel (left)
The streaming **Data** source (folder/glob or HDF5 dataset) with **frame range + stride**,
plus Dark/Bright/Background and Mask, are selected in the shared **Data Loader panel**
(see §1). Dark/bright/background are applied per frame; all mask sources are unioned.

**Browse… options (Multiple files / Filestem), and how MONITOR sees them.**
The Data field's **⋯ → Browse…** popup (see §1 "The Browse… popup") offers
all four selection modes here too:
- **Full folder** — unchanged; MONITOR watches it for new TIFF frames.
- **Files sharing a name stem** — kept as a *live* filter (the chosen folder
  + typed prefix), not a frozen file list: both a one-shot *Start
  Integration* run and MONITOR re-scan the folder for files whose name
  starts with that prefix, so a new matching file dropped in later is
  picked up by MONITOR and a non-matching file (e.g. from an unrelated scan
  sharing the same folder) is ignored. Once loaded, the small info line
  under the field reads e.g. `Source: /data/scan  (filestem: scan_*)`.
- **Multiple files** — an arbitrary multi-select (may not share a prefix).
  This runs through *Start Integration* like any other source, but **cannot
  be watched by MONITOR** (there's no folder/pattern to re-scan) — the
  info line reads `Source: N file(s) — <shared folder>`, and turning
  MONITOR on with this kind of pick selected shows a "Folder needed"
  warning instead of starting.
- **Single file** — unchanged.

### Calibration source (middle)
- **From Tab 2** — the calibration (or refined) result.
- **From file** — a **calibration `.json`, MIDAS `paramstest.txt`, or pyFAI `.poni`**
  (auto-detected; both GUI and MIDAS-pipeline json key styles supported). Entering a
  path auto-selects this option.

### Calibration values (click "View calibration ▾" to see them)
Next to Calibration source, a **View calibration ▾** link — the same
click-to-see-options interaction as the Data Viewer's clickable λ/pixel-size
labels — pops up the geometry actually in use, instead of an always-visible
grid taking up space in the middle panel: λ, Lsd (mm), BC_y/BC_z (px), tilts
tx/ty/tz (°), pixel sizes, detector size, a distortion summary (number of
non-zero coefficients), and an **ImTransOpt** row (e.g. "Flip Y, Flip Z" or
"None") naming the image transform the active geometry was calibrated in — the
same transform Batch Integrate applies internally (via the MIDAS integration
backend, not a GUI-side pixel flip) to every streamed frame so it matches. The
popup's entire content (not just the field values) is rebuilt fresh every
time it opens, so it always reflects whichever calibration source (Tab 2
result, or a parsed file) is currently active and always shows the full
parameter grid alongside a note line reporting that source (or any
file-read error) — never just the note on its own. Hydra mode has the same
popup per panel, next to each `ge{n}` card's calibration-source radios.

### Cake parameters… (middle, under "View calibration")

A **Cake parameters…** button opens a small editor holding all nine columns
of an `mpe_wf_saxs_waxs`-style `cake_parameters` CSV in one place, with
**Load CSV… / Save CSV… / Apply / Close**. Seven of them are fields you
already have elsewhere on this tab, gathered together:

| CSV column | Where it otherwise lives |
|---|---|
| `R_MIN` / `R_MAX` / `R_STEP` | **R bins…** (Integration → Bin type) |
| `ETA_MIN` / `ETA_MAX` / `ETA_STEP` | **Azimuthal bins…** |
| `OME_SUM` | the data loader's **Combine sub-frames** |
| `OME_START` / `OME_STEP` | nowhere else — they drive the per-frame ω, see below |

Editing a value here or in the corresponding popup is the same setting either
way; **Apply** is what pushes the editor's numbers back onto the tab, so you
can open it, try values, and close without having changed the next run. The
`R bins…` / `Azimuthal bins…` popups stay for adjusting one axis mid-run;
this is the view of the whole file at once.

#### `OME_START` / `OME_STEP` — the rotation angle on every frame

These two describe the rotation the frames were collected over: `OME_START` is
the angle of raw sub-frame 0, `OME_STEP` the increment per raw sub-frame. (At
the beamline mpe_wf reads them straight off `20idaSoft:userTran9.H` and `.I`.)
`midas_integrate_v2` itself knows nothing about rotation, but the zarr writer
takes a per-frame omega, so Batch Integrate turns the two into angles:

> **ω(frame) = `OME_START` + mean(the raw sub-frame indices that frame was
> built from) × `OME_STEP`**

The raw indices are counted **from the start of the rotation the frame came
from**, and one HDF5 sub-frame stack is one rotation — so on a multi-file pick
each file restarts at `OME_START` rather than continuing the previous file's
ramp. One-frame-per-file data (TIFF, `.ge*`) is the other way round: a single
such file is not a rotation, the series is, so there the whole selection is
counted through. A raw-frame filter *shifts* the angles rather than rebasing
them in both cases: skipping the first ten sub-frames does not move where the
rotation began, so the survivors keep the angles they physically had.

Counting per file is also what lets the computed ramp and a **measured**
omega channel (below) mean the same thing. A measured channel is a 1-D dataset
stored inside each file and indexed from 0 in each, so it has no choice but to
be file-local; the ramp is aligned to it rather than the other way round, and
both now read the same window for a given frame.

That one expression covers every case: with no combining (`OME_SUM` = 1) it is
`OME_START + k·OME_STEP`; with `OME_SUM` = *n* it reproduces mpe_wf's own
`ome_start + (idx·ome_sum + (ome_sum−1)/2)·ome_step` exactly; with `OME_SUM` =
0 ("combine everything selected into one frame") it is the mean angle the
collapsed exposure actually covered.

#### Frame numbers and angles are the same axis

`start` / `end` / **Combine sub-frames** in the data loader and
`OME_START` / `OME_STEP` / `OME_SUM` in the cake parameters describe *one*
thing — the raw sub-frames of the acquisition. One set counts them, the other
puts degrees on them, and `OME_SUM` **is** the **Combine sub-frames** spin box
(the editor edits that widget, not a copy of it, so the two cannot disagree).

Two readouts now state the correspondence for whatever is loaded, so the
conversion never has to be done in your head:

* The **cake summary line** under *Cake parameters…* gains a second line:

  ```
  R 0–auto px  ΔR 1 px   η -180…180°  Δη 5°   sum 25 sub-frames (OME_SUM)   ω 0°  Δω 0.25°/sub-frame = 6.25°/frame
     sub-frames 0…1441 → ω 0°…360.25°   58 frame(s)
  ```

  The `Δω …/sub-frame = …/frame` pair is the easily-missed multiplication:
  combining 25 sub-frames leaves consecutive *output* frames 25 × `OME_STEP`
  apart. On a multi-file pick the line says *of each file* and *in every
  file*, matching the per-file rule above. If a rotation is loaded and no
  angles are configured, it says so outright —
  `ω 0° on all 58 frames (OME_START/OME_STEP not set)` — which is the state
  that otherwise produces a perfectly valid-looking run of all-zero `/Omegas`.
  With a measured channel picked, it names the channel instead of inventing a
  range from the unused ramp.

* The **start/end hint** in the data loader card gains the short form of the
  same thing — `… → ω 0°…360.25°` — right under the boxes where the
  sub-frame range is set.

Both are recomputed from the source's own frame windows, so they cannot report
a range the run will not produce; editing `OME_START` or `OME_STEP` re-renders
them without re-reading any file.

Two more controls sit under the nine columns in the same editor. They are
**not** CSV columns — they are saved and restored with the GUI state, not
written into the file:

| Control | What it does |
|---|---|
| **Omega channel** | An editable drop-down listing the 1-D datasets in the loaded HDF5, with omega-looking names (`/omegas`, `samry`, `omega`) sorted to the top. Pick one to use the *measured* angle instead of the computed ramp, reduced over each output frame's window the same way the pixels are. Blank — the default — falls back to `OME_START`/`OME_STEP`; nothing is ever auto-selected, the name hint only orders the list. A channel that turns out to be missing or unreadable logs one line and falls back to the ramp rather than failing the run. |
| **Averaged/summed — one ω for all frames** | The override for data that was averaged or summed *outside* this loader (pre-averaged TIFFs, or frames that are already sums while `OME_SUM` still reads 1): every output frame gets the single run-wide mean angle instead of its own position in a ramp the pixels no longer have. When the combining happened here (`OME_SUM` = 0) the one window already spans the run, so the override changes nothing. |

The resulting angles go into the zarr's `/Omegas` (which the backend labels
`Units: Degrees`), into an `omegas` dataset alongside the profiles in the
combined HDF5 (both layouts — the plain 1-D one and the multi-azimuth cake
file, see the Output formats section below), and into the logged attempt — so a GSAS-II export of that
attempt re-exports with the same angles the run used, including a measured
channel the export path can no longer reach. Each run logs one
`[batch] omega: …` line naming the source it actually used.

Both run paths carry these settings: **Start Integration** passes them
in-process, and **Run as background job** serialises them onto the
`batch_cli` command line as `--ome-start` / `--ome-step` / `--ome-channel` /
`--ome-collapse`. Start and step are always on that command line, even at the
0/0 default, so the launched command echoed into the **Logs** tab always
states the angles the detached job will record.

**Leaving both at 0 now means a genuine ω = 0 on every frame**, not "unset" —
a stationary sample really is at zero. This is a visible change for anyone who
never opens this editor: `/Omegas` used to be filled with the frame index
(`0, 1, 2, …`) labelled as degrees, and is now all zeros until you set the two
keys. Nothing downstream read it yet, which is why the old behaviour went
unnoticed, but peak fits and pole figures take ω as the independent variable.

**Save** always goes through a Save-As dialog, pre-filled with
`<expid>_bc/cake_parameters.<beamline>.<detector>.csv` — mpe_wf's own filename
convention, with the beamline token taken from the active profile (`20-ID-E` →
`20ide`) and the detector from the loaded source path. mpe_wf writes that path
silently because it runs as `S20IDUSER`; this GUI runs as you, so it shows the
path before writing into a shared beamline directory. The suggestion only ever
names a directory that already exists, falling back through the Output
directory field and the source folder to your home directory — nothing is
created until you save. The file itself is the same nine-column, header-plus-
one-row layout mpe_wf's tools read, so it is interchangeable with one written
there.

Immediately below the button, a muted one-line summary shows the cake
parameters currently in force (R range/bin, η range/bin, Q output range when
Q-uniform bins are on, the sub-frame sum when it's greater than 1, and the
omega source when it isn't the default zero),
wherever they came from — CSV, typed by hand, restored with a project, or
auto-filled.

### Integration
| Field | Description |
|---|---|
| Kernel | Hard (fastest) · Subpixel K=2 · Subpixel K=4 · Polygon (exact). |
| R bin (px) / η bin (°) | Radial and azimuthal bin sizes. |
| **Rmin / Rmax (px)** | Exclude an inner (e.g. beamstop shadow) or outer radial region from integration. Rmin defaults to 0. Rmax defaults to 0, meaning **auto** — left at 0 it's passed through unset so the backend picks the farthest-detector-corner radius itself (same value the **Corner** button below fills in); once a calibration is resolved, Rmax auto-fills to that corner value the first time (a manual edit or preset click after that always wins). **Corner** sets Rmax to the farthest detector corner from the beam centre; **Edge** sets it to the farthest straight detector edge (smaller than Corner — excludes the corner regions beyond it). |
| **Azim. mean** | How the (η, R) cake becomes a 1-D profile: **Pixel-weighted** (default) `Σ(mean·count)/Σ(count)` — independent of η-bin size and robust to partial azimuthal coverage / **off-detector beam centres**; or **η-bin mean (legacy)** — the unweighted mean of per-η-bin means, which can distort with a coarse η bin when the beam centre is off the detector. |
| Per-bin variance (σ) | Error model poisson / azimuthal / hybrid (ignored when corrections are on → σ = √I). |
| Q-uniform bins | Integrate in R then rebin onto a uniform-Q grid (Qmin, Qmax, ΔQ). |
| **Multi-azimuth output (cake)** | Off by default. Keeps every azimuthal (η) sector from the η bin/range above as a **separate** output profile per frame (`profiles`/`sigmas` become `(n_frames, n_eta, n_r)`) instead of collapsing to one full-circle mean profile — needed for per-azimuth GSAS-II/texture work. Off, η bin still exists (default 5° over the full 360°, i.e. 72 internal bins) but is used only to control the collapse's weighting resolution, so turning this on repurposes that same field rather than changing any existing run's output. Text-format Save/live writes become one file per `(frame, η bin)`, named `<id>_etaNNN.<fmt>`; HDF5 output switches to the cake layout below (`midas_gui/cake_hdf5.py`) rather than being skipped, since `midas_integrate_v2.write_h5` only accepts a 1-D profile per frame. Not yet combinable with Q-uniform bins. |
| **Show bin grid** | Off by default. Overlays the full (R, η) integration bin grid — concentric circles at each R-bin edge, spokes at each η-bin edge — on the **Detector view** tab, thinned to at most ~50 rings / ~72 spokes so a fine bin size stays legible. |

A new **Detector view** tab (alongside Waterfall/Stacked profiles — one page-level
tab in Hydra mode, shared across the 4 GE panels and refreshed for whichever
panel is toolbar-selected) shows the current source frame with the Rmin/Rmax
boundary circles always overlaid (once a calibration resolves), plus the bin
grid when **Show bin grid** is checked — lets you confirm the excluded
region/binning geometry visually before running a batch.

**On a large sub-frame stack the preview now appears in seconds, not minutes.**
Showing one frame used to decode the whole file it lives in: on a 1442-sub-frame
VAREX file (2880x2880 uint16, 23.9 GB) that was about **230 s and ~1.9 GB of
memory to draw a single 33 MB frame**, and the same cost was paid again by each
parallel worker at the start of a batch run. Only the sub-frames behind the
requested frame are read now — the same file previews in about **4 s / 415 MB**,
and the frame count still comes from the HDF5 header without reading any pixels
at all. Nothing about the output changes: frames, their ids and the resulting
filenames are identical to before. This is the same defect behind a run that
looked frozen before it logged anything; that half was fixed earlier, for
counting only.

### Physics corrections
Polarization and solid-angle (pixel-domain, via `integrate_with_corrections`).

**Polarization plane — MIDAS η is measured from VERTICAL.** The **Plane** spin
box is the azimuth of the polarization plane in MIDAS η, and it defaults to
**90°**, which is *horizontal*: the storage ring's X_Lab–Z_Lab plane, where the
beam is polarized. This is almost always the right value at an APS beamline.

The trap is that 0° looks like the horizontal answer and is not. η = 0 is
straight **up** (+Z_MIDAS / +Y_Lab); η = ±90 is horizontal (∓Y_MIDAS / ±X_Lab).
Because the correction goes as `cos(2(η − plane))`, setting the plane to 0
doesn't merely fail to flatten a ring's azimuthal intensity variation — it
roughly doubles it. (±90 are equivalent; the factor has period 180° in the
plane.) The GUI shipped a 0.0 default before 2026-09-24, so **a project saved
before then restores its own stored 0.0** rather than being silently rewritten;
the batch log prints the plane and fraction at every run start and flags an
off-plane value, so check that line if you are re-running old work.

**Fraction** is the polarized fraction: 0 = unpolarized, 1 = fully polarized in
the ring plane. 0.99 is the usual synchrotron value and is the default.

It is up to you to load images consistent with the lab frame — see
[Lab-frame axes overlay](#lab-frame-axes-overlay-image-toolbar) for how to check
that with your own data before trusting a correction.

### Monitor normalisation
Divide each processed frame's profile/σ by a per-frame scalar from a text file (one
value per *processed* frame).

### Drift correction (long scans) — hidden from the GUI
Not shown in the GUI (not used in production) — per-frame geometry
interpolated from an anchor JSON (`{frame_idx: {Lsd, BC_y, BC_z}}`) with
spline/linear/constant parametrization, each frame then integrated with its
own geometry, is still fully implemented underneath (`DriftWorker` and all
its state save/restore keys), so it can be shown again by removing one
`setVisible(False)` line if it's ever needed.

### Run mode: Sequential / Batch Parallel
A **Run mode** card above Start Integration chooses how this run's frames
are processed. What each mode does is a **hover tooltip** on the "Mode:"
label and the mode combo box itself (not a permanent explanatory line —
keeps the card compact):
- **Sequential** *(default)* — one `BatchWorker`, unchanged from before.
- **Batch Parallel** — a **Workers** spin box (1..CPU count) splits the
  run's frames into that many contiguous, near-equal chunks, each handled
  by its own concurrent worker sharing **one** detector map built once up
  front (the "detector mapping happens on one process first" step) rather
  than rebuilt per worker. The worker count **auto-shrinks** so every
  worker gets at least **10 frames** — e.g. asking for 8 workers on a
  15-frame run silently drops to 1 worker (logged in the tab's Log panel).
  Waterfall/Stacked-profiles and the progress bar behave identically to
  Sequential; frames may just arrive slightly out of order as chunks
  finish at different times. Workers are threads within the app, not
  separate OS processes — the same approach Hydra's own per-panel
  Sequential/Parallel toggle already uses for its concurrency.

Hydra's **Per panel:** Sequential/Batch Parallel combo (its own frame-level
parallelism, independent of the **Panels:** Sequential/Parallel toggle)
carries the same tooltip-instead-of-permanent-note treatment.

### Run / Abort / Save / Clear
**Start Integration**, a red **Abort**, and a green **Save** sit in one row
(Start Integration is narrower than before to make room). Abort first asks
the worker(s) to stop cleanly between frames (keeping frames already
written); if a worker does not stop promptly it is detached to finish on
its own. **Save** writes the lineouts already computed this run to a folder
you pick on the spot, in whichever output format(s) are checked. Save is
only enabled for a run that had **no Output folder** set — if an Output
folder was set, the run already wrote everything there as it went (into
its per-format subfolders — see "Output folder" below), so Save is left
**disabled** afterward to avoid dumping a second, flat, unsubfoldered copy
on top of what's already on disk; the completion message notes this
explicitly ("Save is disabled — results are already saved above..."). 2D
CSV (cake) is the one format Save can't produce — per-frame cake arrays
aren't kept in memory after a run to avoid bloating RAM for large batches;
re-run with an Output folder set and 2D CSV checked to get that format.
**Clear results** removes the profiles/plots computed **this session** for
the current data (waterfall, stacked profiles, and the integrated-frame
tracking) so a fresh integration can start — it stops any active monitor
but does **not** delete raw data or any files already written to disk.

### Live folder monitoring (MONITOR)
The **MONITOR** button at the bottom of the left Data Loader panel (folder/glob sources
only — including a filestem-filtered folder, see "Browse… options" above; not an
explicit Multiple-files pick) starts a live watch: while active it turns **green** and
the GUI polls the data folder for **new** TIFF frames — matching the chosen filestem
prefix when one is set, otherwise every TIFF frame in the folder — integrating each one
**as it appears** and adding it to the display. It does **not** re-run the whole batch — it
reuses the already-built
**detector map** (the binning geometry / pixel-count cakes; reused from a prior *Start
Integration* run when the calibration, kernel, bins, mask and folder are unchanged, or
built once on first use) and integrates **only the new files** (tracked by frame id, so
frames already shown are skipped). New frames honour the current kernel, corrections,
Dark/Bright/Background, mask and Q-uniform settings, and are saved to the output folder
when a 1-D format is selected. Click MONITOR again to stop; starting a fresh *Start
Integration* also stops it. (Distinct from **Monitor normalisation** above, which divides
profiles by a scalar file.)

### Output folder
The **Folder:** field (Output card) is a plain text field + **Browse…**,
plus a **Suggest** button that fills it in automatically from whatever
data source is loaded, mirroring `mpe_wf_saxs_waxs`'s own
`outroot/<expid>_bc/<froot>/<detector>/` output-folder convention:
- **Suggest** reads `<outroot>/<expid>/<detector>/<froot>/<files>`
  positionally off the loaded source's own path (four directories deep,
  counting the files' own containing folder) and fills in
  `<outroot>/<expid>_bc/<froot>/<detector>/` — no need to type the header
  Exp ID field first. When the source doesn't sit that deep (e.g. files in
  a flat folder), it falls back to `<source folder>/<expid>_bc/<froot>/`
  using the header **Exp ID** field instead (see below), dropping the
  `_bc` segment entirely if that field is empty.
- The folder is also **auto-filled** the moment a data source loads,
  using the same logic as clicking Suggest — but only when the field is
  still empty; typing or Browse-picking a folder yourself always wins and
  is never overwritten.
- Once an Output folder is set (by any of the above, or a run started
  with one), output is written into **per-format subfolders** underneath
  it — `csv/`, `xye/`, `fxye/`, `dat/`, `2d_csv/`, `h5/`, `zarr/` — rather
  than all formats side by side in one folder.
- The header-level **Exp ID** field (top of the window, next to the
  Profile selector) is a free-text label — e.g. `park_may26` — used as the
  fallback expid for Suggest above and saved with the Project (§17) and
  across restarts (last-used value, independent of Profile).
- The Calibrate tab's **Working dir** (§5) is derived from the same `_bc`
  convention but is deliberately *not* the same folder: it stops at the bare
  `<expid>_bc` root, without the `/<froot>/<detector>/` tail, and it prefers
  an existing `_bc` folder in the path over the positional reading. Batch
  keeps the positional reading, which is correct for the layout Batch is
  pointed at. See §5's "Working directory" for why they differ.

### Output formats — checkbox list behind a popup button (multi-select)
Click the **Output format ▾** button to reveal a checkbox per format —
CSV (R,I,σ) · XYE (2θ) · FXYE (centideg) · DAT (Q) · HDF5 (full stack) ·
2D-CSV (η×R cake) · Zarr (cake, REtaMap) — check as many as you want and
every checked one is written for every frame (HDF5 as one combined
full-stack file, Zarr as described under **Zarr grouping** below, the rest
one file per frame). One format (CSV) is checked by default. The button's own
text names whichever formats are currently checked (e.g. "Output format:
CSV, XYE ▾") so the selection is visible without opening the menu — the
checkboxes themselves no longer take up permanent space in the Output
card.

#### Zarr grouping — how many frames share one archive

A **Zarr grouping** drop-down sits under the format button and applies to the
`zarr` format alone (it greys out, with a tooltip saying so, while Zarr is
unchecked):

| Setting | One `.zarr.zip` per | Named |
|---|---|---|
| **One zarr per output frame** (default) | combined output frame | `<frame-id>.ave.zarr.zip` |
| **One zarr per source file** | rotation | `<source-stem>.ave.zarr.zip` |
| **One zarr for the whole run** | run | `<source-stem>.<start>_<end>.ave.zarr.zip` |

Per-frame is the original behaviour, matching mpe_wf's one-zarr-per-scan-point
convention, and stays the default so every existing project keeps writing what
it always wrote. It does not scale: three 1442-sub-frame VAREX files at
`OME_SUM 10` produce **435 single-cake archives**.

**A group is one rotation — the same unit ω is measured from** (§7's omega
rule). One HDF5 sub-frame stack is one rotation, so "per source file" gives one
archive per file; one-frame-per-file data (TIFF/`.ge*`) only becomes a rotation
as a series, so there the whole selection is one group and "per source file"
produces a single archive. Grouping and ω read the same answer from the same
code (`zarr_group_key` is defined through `omega_channel_window`), so an
archive can never disagree with the angles inside it.

Three consequences worth knowing:

- **`/Omegas` is per frame regardless.** Grouping changes packaging, not
  physics — a per-file archive carries its rotation's three (or 145) angles,
  each file still restarting at `OME_START`.
- **`/SumFrames` becomes the sum over the group.** The backend writes it over
  every frame in an archive; per-frame that is just the frame, per-file it is
  the summed rotation — which is what the field is for.
- **Batch Parallel cannot split a group**, since two workers would open the
  same path. Chunk boundaries fall on group boundaries instead, so the worker
  count is capped at the number of groups: a 3-file folder uses 3 workers
  however many were requested, and the run log says so. "One zarr for the
  whole run" is a single group, so it runs sequentially.

A run-spanning archive is built as `…ave.zarr.zip.part` and renamed once the
processed frame range is known, so a run that dies mid-write leaves an
obviously incomplete file rather than a plausible-looking one.

Both run paths carry the setting: **Start Integration** passes it in-process,
**Run as background job** serialises it as `--zarr-grouping frame|file|run`,
always present on the command line echoed into the **Logs** tab.

**2D-CSV (η×R cake) — one file per frame, in either mode.** Each frame's
whole cake is written to `2d_csv/<frame>_cake.csv`: a header row of R values
and one row per η bin, prefixed by that bin's centre angle. It is a picture of
the cake rather than a lineout, so **Multi-azimuth output** does not change
it — that checkbox controls whether the *other* formats fan out into one file
per η bin (`<frame>_etaNNN.<fmt>`), and 2D-CSV is written the same way with it
on or off. (Until 2026-09-29 it was written only with multi-azimuth on: with
the box off the run produced an empty `2d_csv/` folder while still reporting a
file — under the wrong name at that. If you have a run whose `2d_csv/` folder
is empty, that is the bug, and re-running is all that is needed.) The one case
where it genuinely cannot be produced is **Save** after a 1-D run, which is
explained above.

**HDF5 in multi-azimuth mode — the cake layout.** With **Multi-azimuth
output (cake)** on, the combined HDF5 is written by `midas_gui/cake_hdf5.py`
instead of `midas_integrate_v2.write_h5`, which only accepts a 1-D profile per
frame. It is a flat file — every dataset at a root-level path, no NeXus
nesting — holding `cake`/`cake_sigma` `(N, n_eta, n_r)`, the real
engine-collapsed `profiles`/`sigmas`, the `r_px`/`two_theta_deg`/`d_angstrom`/
`q_invA` radial axes, `eta_deg`, `frame_ids`, `omegas` (degrees, one per frame
— see §7's OME_START/OME_STEP), and the `bin_area` pixel-count weight. The
stored cake is reweighted by each bin's share of that area, so
`cake.sum(axis=eta)` reproduces `profiles` exactly. This is a GUI-native
archive, **not** a GSAS-II input: GSAS-II's importer only ever opens
`.zarr.zip`, so nothing here mirrors the zarr `REtaMap`/`OmegaSumFrame`
convention. Provenance is stamped afterwards, in the root attrs *and*
mirrored into a root-level `provenance_history` dataset so it shows up in a
plain `h5ls`/tree view.

**Zarr's environmental metadata (stopgap).** The `zarr` output format's
per-frame `OmegaSumFrame` attrs, and its `provenance_history` entry, also try
to carry the real beam-monitor ion chamber (`I0`, and `I` when the setup has
one) and sample-stage motor positions — not just Temperature/Pressure. Since
the file's own `active_instrument` field is currently always empty at this
beamline (a known, open DAQ-side gap — not something this GUI can read
around), **which hutch a source belongs to is inferred from its path**
(`varexE`/`varexD`, case-insensitive) rather than read from the file. An
unrecognized path attempts none of this — same graceful "not available for
this source" behavior Temperature/Pressure already have. Full per-station
mapping, and why each simplification was chosen, is in
`.context/DECISIONS.md`.

**The source HDF5's instrument metadata comes along.** A VAREX/areaDetector
file arrives with a few hundred EPICS PVs snapshotted under `instrument/` at
acquisition time — motor positions, slit gaps, monochromator angles,
insertion-device gap, every scaler channel. MIDAS's own pipeline carries them
into its output (`integrator.py:_enrich_zarr_with_metadata`); MIDAS_GUI reads
frames straight into memory and so used to drop all of them, leaving the
`.zarr.zip` with only what the backend writer has named slots for. It now
reopens the source file after the write and copies `instrument/` and
`active_instrument/` in wholesale — nothing curated, so a PV the DAQ adds
tomorrow comes along without a code change. A real tree is about 24 KiB,
negligible against a 300 MB source.

Three things to know about it:

- **HDF5 sources only.** A TIFF stack has no such tree; nothing is copied and
  nothing is missing.
- **Per-frame arrays are averaged, not copied raw.** A 1-D array with one
  entry per acquisition is reduced to one value per output frame, over exactly
  the raw sub-frames that frame combines. It is aligned to the *light* block,
  found via the timestamp gap — the DAQ records lights and darks in one flat
  array, so a 10-frame scan carries length-20 metadata and a naive
  length-match would blend the dark tail into every average.
- **GSAS-II never reads any of it.** Its importer looks at
  `InstrumentParameters`/`REtaMap`/`OmegaSumFrame` and ignores every other
  group, so this is pure provenance: it cannot perturb an import, and it is
  not a substitute for one (see §13).
- **It is not free, and the cost scales with frame count.** Measured on a
  300-PV tree: about **+0.15 s and +130 KiB per output frame**. The bytes are
  mostly zarr's own per-array bookkeeping, not the ~24 KiB of readings, and
  the time is the archive repack — a zip can't be edited in place, so the
  several hundred extra members have to be written out with it. On a
  10-frame scan you won't notice; on a 3600-frame one that is roughly nine
  extra minutes and half a gigabyte across the run.

The GSAS-II export (§13) copies the same tree from the same file, reopening it
via the source path recorded in the attempt's `inputs.src_cfg`, so which
writer produced a given archive still isn't visible in its layout.

**The calibration that produced a zarr is recorded in it.** A `.zarr.zip`
always contained the geometry, but only *applied* — baked into `REtaMap`'s
per-bin Radius/2θ/Eta/Q columns, recoverable only by inverting the map. The
one readable trace was `InstrumentParameters/`, and that holds just
`Distance` (Lsd) and `Lam`; its other entries (`U`/`V`/`W`, `Polariz`,
`SH_L`, `X`/`Y`/`Z`) are **GSAS-II peak-profile defaults written by the
backend, not anything MIDAS refined** — easy to mistake for fit output. The
`provenance_history` entry now carries an `instrument_params` block with the
geometry actually used: `Lsd`, `BC_y`/`BC_z`, `tx`/`ty`/`tz`, `pxY`/`pxZ`,
`NrPixelsY`/`NrPixelsZ`, `Wavelength`, `RhoD`, the `TransOpt` flips, all
fifteen distortion harmonics (recorded even when zero — "no distortion" is a
statement worth being able to read back), and, when in use, the panel
layout and the residual-correction map path.

**Both zarr paths write the same layout.** Batch Integrate's `zarr` format
and the GSAS-II export (§13) call the same backend writer, so their arrays
and groups always matched; their provenance did not — the export path used
to stamp nothing inside the zip, only a sidecar. Both now write an identical
`provenance_history` entry shape, `instrument_params` included, differing
only in the `tool` field that names the writer.
`tests/test_zarr_layout_parity.py` writes a file by each path and diffs
them, so the two can't drift apart again unnoticed.

Right panel: live **Waterfall** and **Stacked profiles** — both have an **x**
selector to show the axis in **R (px) / 2θ (°) / Q (Å⁻¹)** (converted from the run's
calibration). Both plots are bounded to their own data extent (like the main image
viewers) so you can't scroll/zoom out into empty space. **Waterfall** now has a
color-scale histogram sidebar on the right, same as the main image viewers, driven
by its **cmap** selector; its vmin%/vmax% auto-level defaults to the **30th/99th**
percentile and, like the main image viewer, excludes exact-zero pixels from the
percentile calculation so zero-padded/unfilled rows don't skew the level window
toward zero.

The **Stacked profiles** view is a publication-quality plot: each curve tags its
**source file name inline, just below the curve at its left edge** (toggle with the
*Labels* checkbox; an optional corner *Legend* is also available). A **theme** selector
switches between two saved presets:
- **White (publication)** *(default)* — white background, **point + line** markers, a
  colour-blind-friendly categorical palette (matplotlib *tab10*), a boxed frame and a
  light grid — ready to drop into a paper.
- **Dark** — the classic on-screen look (dark background, line-only, vivid
  golden-angle colours).

An **x** selector plots the axis in **R (px)**, **2θ (°)** or **Q (Å⁻¹)** (converted
from the run's calibration). The **Grid** checkbox (off by default) toggles the
horizontal + vertical grid. The *spacing* box shifts successive curves vertically
(0 = overlay). Top-right controls adjust the **line width** (`− line +`), **symbol
size** (`− sym +`, for the point+line markers — also turns markers on if a line-only
theme is active) and **label font size** (`− font +`).

---

## 8. Zarr Viewer

Browse and plot any MIDAS `.zarr.zip` — the same file format Batch Integrate's
own "zarr" output writes (§7) and `midas_gui/gsas_export.py`'s GSAS-II export
produces, via the shared `midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip`
backend, so a file this tab opens needs no format-detection step. Ported from
`mpe_wf_saxs_waxs/gui_view_zarr.py`.

**Layout.** A tree of the file's groups/arrays on the left; a matplotlib plot
canvas, display controls, and a metadata/attributes inspector on the right.

**Tree.** Click a group to see its attributes; click an array to plot it.
Groups are shown bold; arrays show their shape and dtype inline.

**Plotting, by array rank:**
- **0-D** — the scalar value is appended to the metadata pane as text.
- **1-D** — a line plot. The Y-axis label is drawn from the array's own
  `Header`/`Units` attributes when present (e.g. `Omegas` → "Degrees").
- **2-D** — either a "2-D map" (image/pcolormesh) or, for an integration-shaped
  array (`(nR, nEta)` matching the file's own `REtaMap`), "1-D lines" — one
  line per azimuth bin, chosen via the **Azimuth bins** field (`0,5,10` or
  `all`).
- **3-D** — a **slice selector** (axis-0 index) picks one 2-D slice, then
  plotted the same way as the 2-D case above. Selecting the file's own
  `REtaMap` array this way steps through its five channels (Radius, 2θ, Eta,
  BinArea, Q).

**Display controls:**
- **Colormap** / **Scale** (Linear/Log/Sqrt) / **Clim lo/hi** (blank = auto
  1st/99th percentile) — **Lock** preserves the current Clim across array/slice
  changes so successive images share a color scale.
- **Xlim lo/hi** — manual x-axis range; **Lock zoom** preserves the full
  current view (x *and* y) across array/slice/option changes, so panning or
  zooming with the matplotlib toolbar carries over to the next plot.
- **X axis** — R bin / 2θ (deg) / Q (Å⁻¹) / d (Å), computed by averaging the
  matching channel of the file's own `REtaMap` array (or `REtaMap_corrected`,
  preferred automatically when present) across the other axis. Falls back to
  plain R-bin indices when no REtaMap is found or its shape doesn't match the
  selected array.
- **REtaMap** selector — choose `REtaMap` vs `REtaMap_corrected` when a file
  has both (a distortion-corrected file typically does).

**Metadata & attributes.** Selecting the root, a group, or an array shows a
header summary (shape/dtype/chunks for an array; child count for a group), an
expandable attribute tree, and a JSON-pretty-printed raw value pane — click any
attribute (including nested list/dict entries) to see its full value.
`provenance_history` — the append-only list every MIDAS_GUI/mpe_wf_saxs_waxs
writer appends to (§17) — shows up here at the root.

**File-format assumptions**, inherited from the source tool and shared with
mpe_wf_saxs_waxs's own zarr toolchain: a root `REtaMap` array, shape
`(5, nR, nEta)`, channel order Radius/2θ/Eta/BinArea/Q; an optional
`REtaMap_corrected` of the same shape; wavelength from
`InstrumentParameters/Lam`. Any file `write_gsas_zarr_zip` produces —
including everything Batch Integrate's "zarr" output format and the GSAS-II
export write — already matches this.

---

## 9. Tab 5 — Corrections & Physics  *(work in progress)*

Preview physics corrections on a single frame (auto-loads on browse). Pixel-domain:
polarization, solid angle, empty subtraction. Profile-domain: cylindrical absorption
(μR), Compton subtraction (composition:fraction). **Compute corrected profile** overlays
corrected vs uncorrected and shows the correction factor vs 2θ.

Also hosts **per-pixel gain training (LearnableGain)**: from a clean reference frame and
a drifted frame, learn a spatial gain map `g_i = 1 + scale·r_i` by minimising
`MSE(profile) + unity·Σ(g−1)² + smooth·TV(g)`; save as NPZ and apply with
`corrected = raw / gain_map`.

An **Output:** field (text + **…** browse button, in the gain card) sets where the save dialog
opens, so a trained gain map doesn't default to whatever directory the app was
launched from. It's a starting directory only — **Save gain map** still asks,
and you can put the file anywhere. The path is saved with the tab's state, and
is flagged in place if it stops resolving.

---

## 10. Tab 6 — PDF Analysis

Polyatomic **total-scattering** workflow, powered by the `midas_pdf` backend:
I(Q) → Faber-Ziman structure function S(Q) → pair-distribution G(r) (Stage 1:
real composition-weighted normalization, Compton subtraction, end-to-end σ
propagation, optional differentiable scale/background refinement) **plus** a
full Stage 2-3 corrections/analysis surface — empty-cell/Paalman-Pings
absorption subtraction, detector-efficiency correction, absolute (electron-unit)
normalization, differentiable multiple-scattering correction, a fluorescence
diagnostic, CIF-driven small-box structure refinement (PDFfit-style,
error-aware), and Δ-PDF significance testing between two saved reductions. The
tab is organized as a **4-tab left panel** (Data & Reduction / Corrections /
Structure Fit / Δ-PDF) driving a **4-tab right panel** (Reduction / Structure
Fit / Δ-PDF / Log).

Ships with a dedicated, ready-to-run dataset at `test_data/test_pdf/` (real
beamline frames + pre-integrated I(Q) for Ni/CeO₂/IPA/Kapton/air-scatter, a
rasterized beamstop mask, a MIDAS calibration file, and an authored `Ni.cif`)
— kept out of git via `.gitignore` (~320 MB raw frames) but present on this
machine, so the tab opens ready to run against real data with no setup.

### Background — what I(Q), S(Q) and G(r) mean

PDF analysis is three transforms that move from *what the detector measured* to
*where the atoms are*. They are exactly the three stacked plots (top → bottom).

- **I(Q) — measured scattered intensity.** Intensity vs. momentum transfer
  `Q = 4π·sin(θ)/λ` (Å⁻¹), a wavelength-independent rescaling of the angle 2θ;
  high Q = wide angle = fine spatial detail. Obtained by azimuthally integrating
  the detector rings to a 1-D curve. I(Q) is **not** pure structure — it mixes the
  interference (coherent) signal with big smooth backgrounds: per-atom
  self-scattering (form factors `⟨f²⟩`) and inelastic **Compton** scattering.

- **S(Q) — structure function.** I(Q) with the self-scattering and Compton removed
  and normalized away (the **Faber-Ziman** step), leaving only interference:
  `S(Q) = [I_coh − (⟨f²⟩ − ⟨f⟩²)] / ⟨f⟩²`, where `⟨f⟩, ⟨f²⟩` are
  composition-weighted atomic form factors (hence the **Composition** input). S(Q)
  oscillates about **1** and → 1 at high Q where correlations wash out (the dashed
  guide line). `F(Q) = Q·[S(Q)−1]` is the same thing re-weighted by Q to emphasize
  the structurally rich high-Q oscillations.

- **G(r) — reduced pair distribution function.** The real-space answer: a
  histogram of interatomic distances, via a sine Fourier transform
  `G(r) = (2/π)·∫ Q·[S(Q)−1]·sin(Qr) dQ`. A **peak at r** means many atom pairs are
  separated by that distance; the **first peak is the nearest-neighbour bond**
  (Ni ≈ 2.49 Å, CeO₂ Ce–O ≈ 2.34 Å), later peaks are further coordination shells.
  Below the first bond `G(r) = −4πρ₀r` (nothing can be closer) — a constraint that
  needs the number density **ρ₀**. Because it uses *total* scattering (Bragg **and**
  diffuse), the PDF captures local structure even in disordered / nanoscale /
  amorphous materials, unlike a Bragg-only Rietveld fit.

```
 detector image / I(Q) file
        │  azimuthal integration
        ▼
     I(Q)   raw intensity = interference + form factors + Compton      (top plot)
        │  subtract Compton, subtract Laue (⟨f²⟩−⟨f⟩²), divide by ⟨f⟩²  (needs composition)
        ▼
     S(Q)   pure interference, oscillates about 1                      (middle plot)
        │  Fourier sine transform over [Qmin, Qmax] with a window (Lorch)
        ▼
     G(r)   interatomic distances; peaks = coordination shells         (bottom plot)
```

The **±1σ band** on G(r) is the counting uncertainty σ propagated from I(Q) through
every step, so real peaks can be told from noise. The **G/g/T/R** family (Output
selector) is the same information re-weighted: **G** oscillates about 0 (refinement
standard); **g** → 1 at large r; **T** is the total correlation; **R** is the radial
distribution whose *peak area = coordination number*. g/T/R all need ρ₀.

### Left tab 1 — Data & Reduction

**I(Q) source** (choose one):
- **Integrate detector frame** — a calibrated frame is integrated (hard/polygon
  binning, Poisson σ) and mapped to Q. Uses the Tab 2 calibration (λ, geometry)
  and an optional local **Mask** (`.tif`, nonzero = masked — same convention as
  Tab 1, loaded independently so this tab needs no cross-tab wiring to run).
- **Load I(Q) file** — a pre-integrated 2- or 3-column `Q, I, σ` text/CSV
  (comment/header lines tolerated). The tab **opens on this source by default**,
  pointed at `test_data/test_pdf/iq/04_iq_Nickel.csv` — just press
  **Compute G(r)** for the Ni PDF (first shell ≈ 2.5 Å). The same folder also
  has CeO₂, IPA, Kapton (container), and air-scatter I(Q) at λ 0.1839 Å for use
  as backgrounds or empty-cell references (see Corrections, below).

**Calibration.** The geometry/wavelength comes from Tab 2 automatically, or use
**Load calibration file…** to read a MIDAS `paramstest.txt`, a `.json`, or a pyFAI
`.poni` (auto-detected). This is required for *Integrate detector frame* and also
sets λ used by *Load I(Q) file* mode.

**Sample.** Composition (e.g. `Ni` or `C:3,H:8,O:1`; ions like `Ni2+` allowed),
number density ρ₀ (atoms/Å³; needed for refinement and for g/T/R), wavelength λ
(auto-filled from calibration), and a **Compton subtraction** toggle.

**Background subtraction (empty-cell).** Optional. Loads a second I(Q) file
(e.g. the Kapton or air-scatter references) and subtracts it from the sample:
**Manual scale** (a fixed transmission factor `s`, `I_corr = I − s·I_empty`) or
**Fit high-Q** (least-squares `s` over a chosen high-Q window). Enabling
**Paalman-Pings cylinder-in-cylinder correction** replaces the flat scale with
the Q-dependent self-/container-absorption ratio, using sample/container linear
attenuation μ (1/µm) and cylinder radii; **Estimate μ from composition** fills μ
from the sample/container composition, λ, and container density automatically.

**Normalization.** Optional **Refine scale + background** (L-BFGS) — reveals a
background polynomial degree (0–3), a low-r cutoff `r_min` (Å), and an iteration
count. Refinement requires ρ₀ > 0. It fits scale and a smooth b(Q) against two
model-free constraints: high-Q ⟨S⟩→1 and G(r)=−4πρ₀r below `r_min`.

**Q range / r range + FT / Output.** Unchanged from Stage 1: Qmin/Qmax trim,
rmin/rmax/Δr + window (Lorch/none) + binning (hard/polygon), and the bottom-plot
convention family — **G(r)** (reduced PDF), **g(r)** (pair distribution),
**T(r)** (total correlation), or **R(r)** (radial distribution, peak integral =
coordination number); g/T/R need ρ₀. Middle plot toggles between **S(Q)**
(with the S=1 guide) and **F(Q)=Q(S−1)**.

**Save G(r)** writes a three-column `r, G(r), σ` file (diffpy-CMI / PDFgui /
RMCProfile compatible); **Save S(Q)** writes `Q, S(Q)`.

### Left tab 2 — Corrections

All four stages are opt-in checkboxes; leaving all off reproduces Stage-1
behavior exactly.

- **Detector efficiency** — divides I(Q) by the sensor's quantum efficiency
  `η(Q)` for a given material/thickness (e.g. 500 µm Si), computed from the
  photoelectric absorption at each Q's implied path length. At hard X-ray
  energies a thin high-Z-poor sensor can be only a few percent efficient, so
  this legitimately amplifies I(Q) severalfold — that is physically expected,
  not a bug.
- **Multiple scattering** — differentiable cylinder-transport correction
  (`slab_transport_ms`/`ms_background_on_grid`): given an effective optical
  depth τ (from μ × R, μ either auto-estimated from composition/density or set
  manually), a single-scattering albedo, and a Q grid, computes and subtracts
  an MS background before normalization. The Log reports the median MS
  fraction β.
- **Absolute (electron-unit) normalization** — anchors the mean I(Q) over a
  high-Q window to the composition's `⟨f²⟩+⟨S_inc⟩` baseline, putting I(Q) on
  an absolute per-electron scale (needed before meaningfully comparing
  intensities across samples/geometries).
- **S(Q) tail-flatten** — a **display-only** PDFgetX3-style iterative
  MAD-clipped polynomial baseline flatten over a high-Q window; toggled on the
  Reduction plot via **Show tail-flattened S(Q)**. It never feeds back into
  G(r) — the reduction always uses the un-flattened S(Q).
- **Fluorescence diagnostic** — **Check fluorescence** reports which
  sample/container element K/L emission lines fall near the incident energy
  (a source of spurious background near an absorption edge), via
  `expected_fluorescence` / `fluorescence_report_sample_and_container`.

### Left tab 3 — Structure Fit

CIF-driven (or manually-specified lattice + atom table) small-box structure
refinement against the observed G(r), in the style of PDFfit/PDFgui:

- **Crystal** — load a `.cif` (e.g. `test_data/test_pdf/structures/Ni.cif`,
  authored FCC Ni, space group Fm-3m #225, a=3.524 Å) or build one manually
  (lattice a/b/c/α/β/γ, space-group number, an atom table with add/remove rows).
- **Fit range + parameters** — pair-list cutoff (`build_pair_list` r_max), the
  `[fit rmin, fit rmax]` window actually fit, a **σ inflate** factor (real
  beamline G(r) is often dominated by shape-mismatch systematics rather than
  counting noise, so σ can be scaled up before fitting — exposed as a tunable,
  not hardcoded), initial guesses for lattice constant / isotropic ADP (u_iso)
  / scale, an optional background polynomial order, and optimizer steps/lr/
  posterior-sample count.
- **Run structure fit** calls `refine_structure` (gradient-based, PyTorch) and
  reports fitted values ± uncertainty, χ²/ndof, and observed/model/residual
  curves on the right panel's **Structure Fit** tab.

### Left tab 4 — Δ-PDF

Significance testing between two saved G(r) snapshots (e.g. before/after a
correction, or two temperatures): **Save current result as State A/B** each
capture `(r, G, σ)` from the last Compute; **Compute ΔG(r) = B − A** calls
`delta_pdf`/`significant_mask` and plots ΔG(r) with an n·σ band, marking points
that exceed the chosen threshold as a red scatter overlay — a fast way to see
*where* two reductions genuinely differ vs. where the difference is just noise.

### Right panel

- **Reduction** — I(Q) (+ background overlay), S(Q)/F(Q) (with the
  tail-flatten toggle), and the selected G/g/T/R with a shaded **±1σ** band —
  same three-stacked-plot layout as Stage 1.
- **Structure Fit** — observed G(r) (±1σ band) with the fitted model overlaid,
  a `(obs−calc)/σ` residual sub-plot, and a fitted-parameter table with χ²/ndof.
- **Δ-PDF** — ΔG(r) with its uncertainty band and significant-point scatter,
  plus an "N/Ntotal points > nσ" summary label.
- **Log** — shared run log for reduction, corrections, structure fit, and
  Δ-PDF, in the same `LogPanel` used elsewhere.

### Functions behind the tab

The tab is a thin UI over two background workers and the `midas_pdf` backend.

**UI — `PDFTab` (`midas_gui/tab_pdf.py`)**

| Method | Role |
|--------|------|
| `_build_ui` | Builds the 4-tab left panel and 4-tab right panel described above. |
| `set_calibration(result, source)` | Receives a Tab-2 result (or a loaded geometry) → sets λ and enables Compute. |
| `_load_calib_file` | Loads a `paramstest`/`.json`/`.poni` → builds a calibration result → `set_calibration`. |
| `_load_img` / `_load_mask` | Loads a detector frame / local `.tif` mask for *Integrate detector frame* mode. |
| `_estimate_mu` | Fills sample/container μ (1/µm) from composition + λ (+ container density) for the Paalman-Pings correction. |
| `_check_fluorescence` | Calls `fluorescence_report_sample_and_container` and reports the result in the Log + a message box. |
| `_run` | Validates inputs, assembles the full Stage-1 + Stage-2/3 `cfg` dict, and starts a `PDFWorker`. |
| `_on_done` / `_redraw_mid` / `_redraw_bottom` | Draw I(Q)+bg, the S(Q)↔F(Q) toggle (incl. tail-flattened), and the G/g/T/R curve with its ±1σ band. |
| `_collect_fit_cfg` / `_run_structure_fit` / `_on_fit_done` / `_redraw_fit` | Build the CIF/manual crystal + fit-parameter `cfg`, run `PDFStructureFitWorker`, and draw the observed/model/residual plots + parameter table. |
| `_save_state` / `_run_delta_pdf` / `_redraw_delta_pdf` | Snapshot `(r, G, σ)` into State A/B and compute/plot `delta_pdf`/`significant_mask`. |
| `_save_gr_file` / `_save_sq_file` | Export `r,G,σ` and `Q,S`. |

**Workers — `midas_gui/workers.py`** run off the GUI thread:

| Worker / function | Role |
|----------|------|
| `PDFWorker._acquire_iq` | Produces `(q, I, σ)` — either by integrating the frame (image mode) or by reading a file (`_load_iq_file`, tolerant of comma/space and 2- or 3-columns). |
| `PDFWorker.run` | Trims to `[Qmin,Qmax]`, builds `Composition`, then runs the opt-in Stage 2-3 stages in order — background subtraction (manual or fit-scale, optional Paalman-Pings), detector efficiency, absolute normalization, multiple scattering — before the existing refine/non-refine normalization, then an optional display-only S(Q) tail-flatten; emits `q,Iq,background,S,S_flat,Fq,r,Gr,sigma_Gr,Gr_family,scale,bg_coef,refine_loss,bg_scale_used,ms_beta_median`. |
| `_fit_subtraction_scale` / `_absolute_normalize` / `_flatten_sq_tail` | Hand-written recipe helpers (numpy-only, not part of `midas_pdf` itself) for the fit-scale empty-cell mode, absolute normalization, and the tail-flatten display transform. |
| `PDFStructureFitWorker.run` | Builds a `Crystal` (from CIF via `read_cif_to_crystal`, or manually from `Lattice`/`SpaceGroup.from_number`/`Atom`), builds the pair list (`build_pair_list`), masks `(r,G,σ)` to the fit window (with `sigma_inflate`), calls `refine_structure`, and emits `fitted,uncertainty,chi2_reduced,history,r_fit,G_obs,G_calc,sigma_fit,posterior,cov`. |

Image-mode integration reuses the shared core `_build_spec`, `build_geom`,
`integrate_frame`, `axis_conversions` (same path as the other tabs); geometry-file
parsing uses `geometry_fields_from_file` / `result_ns_from_geometry_file`
(`midas_gui/helpers.py`).

**Reduction — `midas_pdf` via `midas_gui/pdf_backend.py`** (re-exported symbols):

| Function | Does | Maps to plot |
|----------|------|--------------|
| `Composition(fractions, number_density)` | Composition layer: `⟨f⟩, ⟨f²⟩` form-factor averages, Laue term `⟨f²⟩−⟨f⟩²`, and per-atom Compton. | — |
| `faber_ziman_S(I, q, comp, …)` | I(Q) → S(Q) with σ (subtract Compton/Laue, divide by `⟨f⟩²`). | S(Q) |
| `structure_function_F(q, S)` | `F(Q) = Q·(S−1)`. | F(Q) |
| `fourier_sine_transform(q, S, r, window)` | S(Q) → G(r) with σ (Lorch / none window). | G(r) |
| `i_of_q_to_Gr(q, I, comp, r, …)` | End-to-end I(Q) → G(r) (composes the two above); the default Compute path. | S(Q) + G(r) |
| `refine_normalization(q, I, comp, r, …)` | L-BFGS fit of scale + background polynomial against high-Q ⟨S⟩→1 and low-r `G=−4πρ₀r`. | refined S(Q)/G(r) + bg |
| `pair_distribution_g` / `total_correlation_T` / `radial_distribution_R` | Convention family from G(r) + ρ₀. | g/T/R |
| `paalman_pings_cylinder_in_cylinder` / `paalman_pings_cell_only` | Q-dependent self-/container-absorption correction for empty-cell subtraction. | background |
| `apply_detector_efficiency` / `detector_efficiency` / `linear_attenuation_um` | Detector-efficiency correction + the μ used by it and by Paalman-Pings/MS. | I(Q) |
| `cylinder_effective_tau` / `slab_transport_ms` / `ms_background_on_grid` | Differentiable cylinder multiple-scattering transport → MS background. | I(Q) |
| `expected_fluorescence` / `fluorescence_report_sample_and_container` | Predicted fluorescence emission lines near the incident energy. | diagnostic only |
| `read_cif_to_crystal` / `build_pair_list` / `refine_structure` | CIF loading, pair-list construction, and the gradient-based small-box structure fit. | Structure Fit tab |
| `delta_pdf` / `significant_mask` | Difference + σ-significance test between two G(r) snapshots — **require `torch.Tensor` inputs**, not numpy arrays. | Δ-PDF tab |

> **Packaging note.** `midas_pdf` is the public PyPI package (pinned in
> `environment.yml` / `pyproject.toml`), imported directly through
> `midas_gui/pdf_backend.py`. Earlier versions of the GUI carried a vendored
> copy under `midas_gui/_vendor/` plus a `midas_hkls.absorption` compatibility
> shim, needed before `midas-pdf` was published and while `midas-hkls` was
> pinned below the release that added `absorption`; both were retired once
> `midas-hkls>=0.5.0` and `midas-pdf` were available. Deliberately **not**
> re-exported (out of scope, see `.context/ROADMAP.md`): the Bayesian SVI/NUTS
> posterior path, RMC big-box refinement, and the non-differentiable
> Monte-Carlo multiple-scattering variants.

---

## 11. Tab 7 — Texture / Pole Figure  *(work in progress)*

Per-ring azimuthal analysis for preferred orientation. Controls: calibration source,
sample frame, R/η bins, ring index, χ (tilt) / φ (rotation). **Compute Pole Figure**
integrates to a cake and extracts I(η) at the selected ring; the right panel shows the
stereographic pole figure and the raw I(η). **Save pole figure (.pol)** exports POPLA
format.

---

## 12. Pump Probe (time-resolved / TR-XRD)

Analyses time-resolved (pump-probe) diffraction the way the TRR group does: a folder of
raw detector frames is pooled by a filename prefix, the pump-probe **delay** is parsed
from each name, every frame is integrated to I(q) with the **MIDAS engine** (the same
core as Batch Integrate, driven by a calibration), repeats at each delay are averaged,
and a reference (mean of the pre-time-zero / negative delays) is subtracted to give
**ΔI(q, delay)**.

**File-naming / pooling.** Frames follow `PREFIX-<fshw>fshw<delay>delay<id>.tif`, where
`PREFIX` (e.g. `Ex01_Sa01_Sc17` = Experiment / Sample / Scan) is one opaque grouping key.
**Scan folder** globs `PREFIX*.tif` in the loaded folder, parses `fshw` and `delay`
(seconds) from each name, and reports how many frames and unique delays were found. The
delay sign is flipped on load so positive delay = after the pump.

The tab uses the same three-panel layout as Batch Integrate: **data loader** (left),
**settings** (middle), **plots** (right).

**Left — data loader** (the shared loader panel, as on Calibrate / Batch):
- **Data** — the folder of raw frames. Defaults to the bundled TRR test set in
  `test_data/trr_s7id/pump_probe_BTO/detimages/` (125 Pilatus2M frames; a large,
  git-ignored local asset), so the tab is populated on open. An index range / stride
  can subset the pooled frames.
- **Dark / Bright / Background** — per-frame field corrections (compute each field, then
  it is applied to every frame before integration).
- **Mask** — a mask file and/or the mask from the Mask Builder tab. For the TRR data the
  tab pre-loads `invert_mask.tif` (0 = valid pixel, 1 = bad pixel / module gap), which
  matches MIDAS's convention that non-zero pixels are masked out.

**Middle — settings.**
- **Calibration source** — *From Tab 2* (the live calibration) or *From file*
  (`calibration.json` / `paramstest.txt` / `.poni`). Same convention as Batch Integrate.
  For the shipped TRR test data the field defaults to a ready MIDAS `paramstest`
  (`Ex01_Sa01_Sc17_midas.txt`, converted from the dataset's pyFAI/Fit2D geometry), so
  the tab integrates out of the box.
- **Data pooling (TRR)** — the pooling **prefix** + **Scan folder** (the folder itself
  comes from the loader on the left).
- **Integration** — kernel, R/η bins, the plot axis (Q / 2θ / R), and optional
  Q-uniform binning. Identical engine and options to Batch Integrate.
- **Physics corrections** — polarization / solid-angle.
- **Pump-probe options** — the reference-delay set (defaults to all negative delays;
  multi-select to override), optional per-pattern normalization over a q-window, the ΔI
  colour range (auto or fixed ±) and the diverging colormap.

**Views (right panel).** Every axis shows **real physical values**, not indices — the
radial axis is in Q (Å⁻¹), 2θ (°) or R (px) per the plot-axis selector, and delays are in
seconds.
1. **ΔI heatmap** — ΔI over the radial axis (a true, continuous Q/2θ/R axis) versus
   delay, diverging colour centred at zero, with a labelled ΔI colour-bar and the
   reference I(q) lineout beside it (shared radial axis). Delays span several decades and
   are irregular, so they are laid out as columns labelled with their real delay values.
2. **ΔI vs q** — one curve per delay, coloured along a rainbow by delay. With ≤ 8 delays
   each curve is named in the legend; with more, a continuous **delay colour-bar**
   (min → max, in seconds) replaces the legend.
3. **Kinetics** — ΔI versus delay for user-defined q-bands (**Add band** over a q range);
   x-axis as real delay (**linear**), **log** (post-t₀ delays, natural for the decade-wide
   delay range), or **rank**.
4. **Mean patterns** — the averaged I(q) at each delay plus the dashed reference, with an
   optional **±1σ band** (spread of I(q) across delays) and a **Log Y** toggle to reveal
   weak features across the full dynamic range; a signal-level / stability check.

All views use a publication-style white theme with large, clearly-labelled axes, readable
legends and a subtle grid. A toolbar above the plots sets the **draw** mode (lines /
lines+points / points) and the **line**, **point (sym)** and **font** sizes (the −/+
groups), as on the Batch Integrate tab. Pan and zoom are bounded to the data so you cannot
lose the plot area.

Integration runs off the GUI thread (`PumpProbeWorker`) with a progress bar and
**Abort**. There is no peak fitting — peak position / width / area are read from the
plots, matching the reference workflow.

---

## 13. Results & Export  *(work in progress)*

Session summary + one-click export. Checkboxes select which products (calibration.json,
paramstest.txt, mask.tif, integrated profiles, G(r), pole figures, session log) to copy
to an output directory. A provenance block (package versions, geometry hash, mask
fraction, correction flags) can be copied to the clipboard for a Methods section.

### Export for GSAS-II

Writes ONE chosen Batch-Integrate attempt from the open project — picked from
a dropdown (newest first, via the project's attempt history; nothing is
silently assumed) — as a native MIDAS-format GSAS-II zarr (`<name>.zarr.zip`),
directly importable via GSAS-II's own **Import → Powder Data → from MIDAS
zarr file**. Uses `midas_integrate_v2.io.zarr_gsas.write_gsas_zarr_zip`
directly, so the layout is bit-for-bit what MIDAS's own C integrator
produces — not a GUI-specific approximation. Works whether the attempt used
plain (single full-circle profile) or Multi-azimuth (cake) Batch Integrate
output (§7); degenerates to one azimuth in the plain case. The zip's root
attrs carry a `provenance_history` entry in exactly the shape Batch
Integrate writes (§7), `instrument_params` geometry snapshot included, so a
file reads the same way whichever path produced it. In addition, a
`<name>.zarr.zip.provenance.json` sidecar carries the attempt's full
provenance (params, hashed input paths, environment snapshot, calibration
snapshot) verbatim — attempt-level history the Batch Integrate path has no
equivalent for, kept alongside rather than inside so the zip's structure
stays exactly what GSAS-II expects.

**What GSAS-II actually reads** (from its own `G2pwd_MIDAS.py`, checked
against the current upstream source): the three groups its validator
requires — `InstrumentParameters`, `REtaMap`, `OmegaSumFrame` — and within
them only `REtaMap` rows 1/2/3 (2θ, η, bin area, where area == 0 is the
mask), each `OmegaSumFrame/<k>` array plus its `Number Of Frames Summed` /
`FirstOme` / `LastOme` / `Temperature` / `Pressure` attrs, and
`InstrumentParameters/<key>[0]` (with `Polariz`→`Polariz.`, `SH_L`→`SH/L`,
and `Distance` read as Gonio. radius, µm→mm). We write all of it.

Three things follow that are easy to get wrong:

- **Our `.provenance.json` is invisible to GSAS-II, by design.** GSAS-II does
  read sidecars, but plain-text ones at `<name>.zarr.samprm` /
  `<name>.zarr.instprm`, and whatever is in them *overrides* the zip. We
  write neither, so nothing we put beside a file can perturb an import.
- **Temperature/Pressure don't currently survive the import**, through no
  fault of the file: `readMidas` reads them off the attrs and then assigns
  into a list of tuples, a `TypeError` caught by a bare `except`. `.samprm`
  is today the only route by which sample metadata reaches a GSAS-II
  histogram.
- **GSAS-II defaults `InstrName` to `'APS 1-ID'`**, so an un-annotated 20-ID
  histogram silently claims the wrong beamline until you set it.

A zarr written elsewhere in the APS toolchain may carry a much larger
metadata tree (`instrument/GSAS2_PVS/*`, `misc/*`, `Detector/*`,
`StorageRing/*`). Despite the name, GSAS-II reads none of it — that tree is
an EPICS PV snapshot the areaDetector plugin writes into the *source HDF5*,
copied forward by MIDAS's `integrator.py`. See `.context/DECISIONS.md`
(2026-09-28) for the full comparison.

Single-detector only for v1 (Hydra composite is a possible fast-follow). Not
yet supported: an attempt run with Q-uniform bins (its stored radial axis is
Q-rebinned, not a plain function of the calibration geometry) or a
file-backed (not embedded) mask — both raise a clear error naming the
attempt and what to change, rather than exporting something silently wrong.

---

## 14. Common UI Conventions

- **Browse = load.** Selecting a file/folder (or pressing Enter in a path field) loads
  it immediately; there are no separate "Load" buttons. HDF5 dataset / frame-index
  changes reload automatically.
- **No accidental scroll changes.** Spin boxes and drop-downs ignore the mouse wheel —
  values change only by clicking/typing; the wheel scrolls the panel instead.
- **Readable right-click menus.** pyqtgraph plot context menus use the dark theme.
- **Error dialogs never truncate the underlying error.** A worker/background-task
  failure ("Calibration failed", "Integration failed", "PDF failed", etc.) shows a
  one-line summary with a **Show Details…** button revealing the complete, scrollable
  error text (traceback included) — and the same complete text is written to that
  tab's log panel, not just the dialog. This applies across every tab; earlier
  versions truncated both to a fixed character count, which could cut an error off
  mid-word and hide the actual exception (see `documentation/development_history.md`).
- **Dark theme, orange accent** (Dioptas-inspired); off-white text, light input fields.
- **Sample-to-detector distance is entered/shown in mm** (Data Viewer, Calibrate seed,
  Preferences ▸ Geometry). Internally — all calculations — and in written calibration
  files (`paramstest.txt`, `calibration.json`) it is always in **microns**; the mm↔µm
  conversion happens only at the display boundary. The config key remains `lsd_um` (µm).
- **Image viewer color scale (Log/cmap/vmin%/vmax%)**, shared by the Data Viewer,
  Calibrate, and Mask Builder image viewers (including Hydra mode): the colorbar's own
  zoom defaults to the vmin%/vmax% percentile window rather than the full data range,
  which a single bad pixel can otherwise stretch into an unreadable sliver. The
  percentile calculation excludes exact-zero pixels, so a mostly-empty frame (e.g. the
  Hydra Composite view's unfilled canvas background) doesn't skew the window toward
  looking washed out. Dragging the LUT region or
  zooming/panning the histogram's own axis is remembered and reapplied on redraw
  instead of resetting to the percentile defaults, and toggling Log/Linear converts a
  manually-set window into the other scale so it keeps pointing at the same data
  (rather than the same raw numbers) — and always reframes the histogram's own
  visible window tightly around the converted levels rather than carrying a
  manually zoomed/panned window through the same nonlinear conversion, which
  would otherwise leave the sliders technically in range but squeezed into a
  barely-visible sliver of the window. Loading new data — a new file, a different
  frame/dataset, a projection, or a correction/mask change — resets to the percentile
  defaults; only frame-to-frame updates within an active live stream (Data Viewer's
  Live Data card) keep a manual window fixed, matching the live-streaming behavior
  described in §3.

---

## 15. Packaging, Deployment & Diagnostics

**Packaging.** `midas-gui` is a MIDAS-style package: `pyproject.toml` (BSD-3-Clause),
a `tests/` smoke suite, and `release.sh` for cutting versioned releases (see
`RELEASING.md`). Once published it can join the `midas-suite` meta-package as an
optional `gui` extra (`pip install midas-suite[gui]`).

**Deployment on another system.** Clone the repo, create the conda environment
(`environment.yml`), and run `midas-gui`. The MIDAS analysis backends are installed via
the `pip:` section of `environment.yml` (editable from a local MIDAS checkout, from a
git subdirectory, or from a private index). `midas_gui/_paths.py` is an inert stub in an
installed package (no `sys.path` manipulation).

**Crash diagnostics.** On startup the app installs a logging exception hook and
`faulthandler`; any uncaught Python exception or native fault is written to
`~/midas_gui_error.log` and shown in a dialog. Installing the hook also prevents PyQt5
from hard-aborting on an exception inside a slot. Each tab is built in isolation — a tab
that fails becomes an error placeholder instead of taking the window down. If the app
ever "pops up and dies" (typically on Windows), send that log file.

**Freeze diagnostics (`kill -USR1`).** A crash leaves a traceback; a *freeze*
leaves nothing, and the moment the window stops repainting is exactly the
moment you can no longer ask the app anything. `launch.py` therefore arms a
`SIGUSR1` handler at startup (Linux/macOS; Windows has no `SIGUSR1`, so the
guard skips it). While the GUI is unresponsive, from another terminal:

```
pgrep -u $USER -f launch.py          # or -f midas-gui
kill -USR1 <pid>
tail -60 ~/midas_gui_hang.log
```

Every thread's Python stack is appended to `~/midas_gui_hang.log`, naming the
exact call that is blocking the event loop. The process is **not** killed or
interrupted — unlike `faulthandler`'s fatal-error handlers, this one only
prints, so you can signal a stuck GUI repeatedly and watch whether the stack
moves (slow but progressing) or stays put (genuinely wedged). It is inert
until signalled and costs nothing at runtime.

Worth knowing what is *not* a freeze: Batch Integrate's Detector-view preview
is fetched by a background `StreamPreviewWorker` and never blocks the event
loop, and the ω mapping readout reads HDF5 headers only (no pixels),
coalesced through a 150 ms timer and skipped entirely when the source has not
changed. A window that stops repainting during either is a bug worth a stack
dump, not expected behaviour.

---

## 16. Configuration & Defaults

Every default in the GUI — detector geometry, data/output paths, ring-simulation
**materials**, **calibrants**, the **pixel-preset** and **K-edge** menus, the
calibration / integration **algorithms**, and **which tabs are visible** — can be set
**without editing code**, via a single per-user JSON config that overrides the shipped
built-in defaults.

### Where it lives
One per-user file, auto-located per OS:
`~/.config/midas_gui/config.json` (Linux),
`~/Library/Application Support/midas_gui/config.json` (macOS),
`%APPDATA%\midas_gui\config.json` (Windows). If it's absent, the shipped built-in
defaults are used. Sharing between machines/users is done by exporting/importing a
JSON file (below) — copy it wherever you like.

### Profiles — header dropdown + Settings ▸ Preferences… ▸ Profile row
A large **Profile: [combo ▼]** control sits at the top-left of the main window,
in the same row as the tab bar (a vertical rule separates it from the tabs) —
the fastest way to switch profiles day-to-day. The mirror-image top-right
corner of that same row shows the active **Project** name (bold, high-
contrast green) whenever File ▸ Project has one open — see §17 "File ▸
Project (FAIR provenance)". Picking a different profile there
switches instantly (no confirmation prompt): live `DEFAULT_*` globals reload,
tab visibility re-applies, Hydra mode's availability updates, and every
option-list dropdown/menu backed by that profile (Data Viewer's Live PV
device list, Calibrate's Calibrant dropdown, and the pixel-size-preset /
K-edge-foil popup menus) repopulates from the newly-active profile
immediately (see §3 "Mode
ribbon"). Profile *management* — creating, duplicating, renaming, deleting —
still lives in the Preferences dialog's own Profile row —
`Profile: [combo ▼]  [New…] [Duplicate…] [Rename…] [Delete]` — which stays in
sync with the header dropdown either way you switch:
- Each profile is its own config file under `<config dir>/profiles/<name>.json`; the
  active one is remembered in `<config dir>/profile_meta.json`. Existing single-config
  installs are migrated transparently into a profile named **Default** the first time
  this runs — no data is lost.
- Four beamline device presets ship bundled and appear in the combo alongside
  **Default**: **20-ID-D**, **20-ID-E**, **1-ID-E**, **17-BM** — each differs only
  in its **Devices** list (below), so picking one just swaps the Live Data PV
  dropdown's detectors for that beamline's. They're seeded once, the first time
  the app runs on a machine; deleting one doesn't bring it back. Fresh installs
  still start on **Default** (same detectors as 20-ID-D) so existing setups are
  unaffected.
- **New…** seeds a blank profile from the shipped built-in defaults.
- **Duplicate…** seeds a new profile from whatever is currently shown in the dialog
  (including unsaved edits), then switches to it.
- **Rename… / Delete** — Delete refuses to remove the last remaining profile; deleting
  the active profile switches to another one first.
- Switching profiles (via the combo) prompts to confirm if the dialog has unsaved
  edits, then reloads the dialog's fields and the live `DEFAULT_*` globals from the
  new profile. Option lists — the Live Data device dropdown, the Calibrant
  dropdown, and the pixel-size-preset / K-edge-foil menus — refresh immediately
  everywhere they're shown. Seeded default *values* (wavelength, pixel size,
  Lsd, beam-centre, viewer step sizes, default file paths) only take effect on
  fields built after the switch — an already-open tab keeps whatever it was
  showing, since a profile switch should never silently overwrite a value
  you're in the middle of editing; a restart applies them everywhere.
- The "Local config: …" label under the row becomes **"Profile '\<name\>': \<path\>"**
  so you always know which file you're editing.

### Editing — Settings ▸ Preferences…
The **Settings** menu opens **Preferences…**, a dialog whose tabs are **pre-filled
with the full shipped defaults** so you edit from a complete starting point:
- **Geometry** — λ, pixel size, Lsd, beam centre.
- **Data Viewer** — the up/down-arrow step size for each field in the Data Viewer's
  Ring simulation card (λ, max 2θ, Lsd, pixel size, BC_y/BC_z, ty/tz). Shipped
  defaults: 0.01 Å, 1°, 1 mm, 0.1 µm, 1 px, 0.1° respectively.
- **Paths** — default data / calibration / output files & folders.
- **Materials** / **Calibrants** — add / remove / modify (name + lattice + SG).
- **Devices** — the detector devices offered in the Data Viewer's **Live Data**
  PV dropdown (name, prefix, PVA suffix, backend, CA suffix). `backend` picks
  which suffix column builds the live PV: `pva` (default, or blank) uses
  `prefix + PVA suffix`; `ca` uses `prefix + CA suffix` instead — read over
  plain EPICS Channel Access via an areaDetector **NDPluginStdArrays** plugin,
  for a beamline whose IOC has no PVA plugin (see the Live Data card section
  above). All PVA suffixes are `Pva1:Image`; the CA suffix convention is
  `image1:`. A built-in **Sim Detector** entry (`midasSim:` prefix, PVA) is
  always included for hardware-free testing; add / remove / edit rows for your
  own beamline's devices, or switch to one of the bundled beamline **Profiles**
  above instead of hand-editing:
  - **20-ID-D** — `20iddNF` (`20idOR1:`), `s20idPil` (`20idPil:`), `pg4`
    (`1idPG4:`), `20iddTomo` (`20idGH1s:`), `20iddFF` (`20IDFF:`).
  - **20-ID-E** — `pimega` (`PITEC:D:RAD1_5Mh:`), `spl1` (`20idsp1:`),
    `s20varex2` (`20idVarex2:`), `pg6` (`20idPG6s:`), `gh2` (`20idGH2S:`).
  - **1-ID-E** — `ge1`–`ge5` (`GE1:`–`GE5:`), `pixirad` (`s1_pixirad2:`),
    `gh1` (`1idGH1:`), `pg1` (`1idPG1:`), `pg5` (`1idSP5:`), `s1varex1`
    (`1idVarex1:`). Names match each detector's variable name in B-PILOT
    (`mpe_bluesky/instrument/devices/`), the beamline's Bluesky/ophyd device
    definitions, so entries are traceable back to source.
  - **17-BM** — `varex` (`17bmVarex:`, `backend: ca`, CA suffix `image1:`) —
    **placeholder prefix/plugin name**, pending confirmation with 17-BM staff
    that the IOC actually runs an NDPluginStdArrays instance at that name.
- **Menus** — the pixel-size presets and K-edge foils.
- **Algorithms** — default calibration pipeline, integration kernel, output format,
  error model, colormap/theme.
- **Tabs** — which tabs are visible. Data Viewer, Mask Builder, Calibrate and Batch
  Integrate are always shown (their boxes are locked on); tick/untick the rest. **Tab
  visibility applies immediately** on save (no restart), unlike the other settings.
- **Display** — the **interface scale**, a whole-application zoom (layout *and* fonts)
  for HiDPI / 4K monitors where the default looks too small. Pick a value or a preset
  (100 % ≈ 1080p, 150 % ≈ 1440p, 200 % ≈ 4K); it applies after a restart (you're
  offered one on save). Also reachable from **Settings ▸ Interface scaling…**. It is
  implemented with Qt's `QT_SCALE_FACTOR`, so the whole layout scales uniformly.

Buttons: **Save as my defaults** (write your config), **Save current GUI state**
(capture the Data Viewer's live λ/pixel/Lsd/BC into the fields), **Save config to
JSON…** / **Load config (JSON)…** (export/import a config to share), and **Reset to
shipped defaults** (delete your config). Also on the menu: **Open config folder** and
**Reload config**. Saving writes your per-user file; **changes take effect on the
next launch**.

### File format
```json
{
  "geometry": { "wavelength_A": 0.39, "pixel_um": 75.0, "lsd_um": 121000.0,
                "bc_y": 10.0, "bc_z": 10.0,
                "pixel_presets": [["Eiger", 75.0], ["Pilatus", 172.0]],
                "k_edge_foils": [["Au", 80.725], ["Pb", 88.005]] },
  "viewer_steps": { "wavelength": 0.01, "two_theta": 1.0, "lsd_mm": 1.0,
                    "pixel": 0.1, "bc": 1.0, "tilt": 0.1 },
  "materials":  { "Ni (FCC)": {"a":3.5238,"b":3.5238,"c":3.5238,"alpha":90,"beta":90,"gamma":90,"sg":225} },
  "calibrants": { "CeO2": {"a":5.4116,"b":5.4116,"c":5.4116,"alpha":90,"beta":90,"gamma":90,"sg":225} },
  "devices": [
    {"name": "s20varex1", "prefix": "20IDFF:", "pva_suffix": "Pva1:Image"},
    {"name": "varex", "prefix": "17bmVarex:", "backend": "ca", "ca_suffix": "image1:"}
  ],
  "paths": { "nickel_h5": "/data/mygroup/sample.h5", "calib_file": "/data/mygroup/calibration.json" },
  "ui": { "calibration_pipeline": "one_shot", "integration_kernel": "subpixel2",
          "output_format": "csv", "azimuthal_method": "poisson", "plot_theme": "hot",
          "visible_tabs": ["Calib. Refinement", "Pump Probe"],
          "ui_scale": 1.0 }
}
```
- `ui.visible_tabs` lists the **optional** tabs to show (the four always-on tabs are
  implicit). The shipped default is `["Calib. Refinement", "Pump Probe"]` — Corrections,
  PDF Analysis, Texture and Results & Export are hidden until you add them. Edit it from
  **Preferences ▸ Tabs**.
- `ui.ui_scale` is the whole-interface zoom (0.5–4.0) applied at startup via
  `QT_SCALE_FACTOR`; ~1.5 for 1440p, ~2.0 for 4K. Edit it from **Preferences ▸
  Display** or **Settings ▸ Interface scaling…** (takes effect on restart).
- When a **list section is present it fully replaces** that built-in list — so
  `materials`, `calibrants`, `devices`, `pixel_presets` and `k_edge_foils` in the file
  are your complete lists. (The Preferences dialog pre-fills them with the shipped
  entries, so you always start from the full set; use **Reset to shipped defaults** to
  get them back.) `sg` is the space-group number.
- Geometry scalars, `paths` and `ui` are simple overrides; any section or key may be
  omitted. A malformed config is ignored (built-ins are used) rather than blocking
  startup.
- A minimal template lives at `documentation/config.example.json`; the step-by-step
  guide is `documentation/config_gui.md`.

---

## 17. File ▸ Project (session + FAIR provenance)

A **Project** is a single, long-lived `.h5` file with two top-level headers
— beyond a saved **profile** of defaults (§16):

- **`gui_workspace`** — the live, in-progress state of every tab, stored
  **modularly, one group per tab** (Data Viewer, Mask Builder, Calibrate,
  …), so a tab's saved snapshot can be inspected or restored independently
  of every other tab's. This is simply **overwritten** each time you save —
  a freely-editable draft, not a permanent record.
- **`analysis`** — three sub-headers (**mask**, **calibrate**, **integrate**),
  each an append-only, self-contained FAIR-provenance history: every time
  Mask Builder logs a mask, or a Calibrate/Batch Integrate run finishes, a
  full record of what actually *happened* (inputs, parameters, results,
  software versions) is added. This never gets overwritten — it accumulates
  across many separate launches of the GUI over the course of an experiment.

> **Breaking change:** this two-header layout (`schema_version 3`) replaced
> the older single-`workspace`-slot layout in a clean cutover — **project
> files created by an earlier version of MIDAS GUI are not readable by this
> one.** Opening one shows a clear warning (rather than a silently empty
> project) naming its old schema version; there is no automatic migration.

- **File ▸ Save Project** (`Ctrl+S`) — if a project is already open,
  silently overwrites its `gui_workspace` header with the current session
  (no dialog), one group per tab; the `analysis` history elsewhere in the
  file is untouched. If no project is open yet, falls through to Save
  Project As…. A completion dialog lists which tabs (if any) failed to save.
  Every such overwrite is crash-safe: the new content is built alongside
  the old (not deleted-then-rebuilt in place) and swapped in only once
  it's complete, and a rolling backup copy (`<name>.h5.bak`) of the file is
  made first, so a crash mid-save can't take the whole project — including
  its `analysis` history — down with it.
- **File ▸ Save Project As…** (`Ctrl+Shift+S`) — always prompts for a
  destination `.h5`, and always creates a **fresh** project there: an
  existing file at that path is completely overwritten (after an explicit
  confirmation, and a `.bak` backup of it), never merged into. If the
  currently open project has any recorded `analysis` history, a second
  prompt lets you choose how much of it to carry into the new file:
  **include the full history** (default), **only the most recent attempt
  of each kind**, or **none** (workspace only). Either way, the new file
  becomes the target for future plain `Ctrl+S`.
- **File ▸ Open Last Project** — a one-click shortcut for the top entry of
  Recent Projects below: opens the most recently opened/saved project
  through the same confirmation flow, without going through the submenu.
- **File ▸ Open Project…** (`Ctrl+O`) — opens a single combined dialog: a
  file-tree browser on the left to navigate to a `.h5`, and the moment one
  is clicked, the right pane previews exactly what that project contains —
  which `gui_workspace` tabs have a saved snapshot, and which mask/
  calibrate/integrate attempts are recorded — as a checkbox tree,
  **everything checked by default**. Uncheck anything you don't want
  restored, then **Open**; nothing is restored until you do. See "Opening a
  project can populate the GUI" below for exactly what each checked item
  restores. If the current session has unsaved changes, this (and picking
  a Recent Projects entry) prompts **Save / Discard / Cancel** first, the
  same as closing the window does — opening a project can no longer
  silently discard in-progress edits.
- **File ▸ Recent Projects** — the last 10 opened/saved projects, newest
  first; picking one shows the same checkbox-tree preview (without the
  file-browser pane, since the path is already known).
- **File ▸ Project History…** — a read-only browser listing every attempt
  recorded in the active project, including mask attempts; see below.
  Disabled when no project is open.
- **File ▸ Close Project** — detaches from the file (stops future
  Calibrate/Batch-Integrate/Mask-Builder logging and the `Ctrl+S` target)
  without touching any tab's current values. If the session has unsaved
  changes, prompts **Save / Discard / Cancel** first, the same as closing
  the window does.
- **File ▸ Import Legacy Workspace (.json)…** — reads a standalone
  Workspace JSON file saved before Workspace and Project were merged into
  one `.h5` (same validation and confirmation as the old Load Workspace…),
  and applies it into whichever project is — or isn't — currently open.
- **File ▸ Quit** (`Ctrl+Q`) — closes the main window (same as the window's
  own close button), so quitting no longer requires the OS window-manager
  shortcut.
- The active project's filename is always shown in the status bar
  (bottom-right, "Project: …" / "Project: none") **and**, in a bold,
  high-contrast green, at the far right of the tab-bar header row (empty
  when no project is open) — the header indicator is the one meant to catch
  your eye; the status-bar one is the quiet, always-present record.

### Unsaved-changes indicator and autosave
The window title shows the active project's name (or "Untitled") and grows a `*`
the moment any tab's value differs from what was last saved/loaded — a plain,
Qt-native marker, not tied to any one field, so it stays accurate across every tab.
Closing the window while it's showing prompts **Save / Discard / Cancel** instead of
silently discarding your changes.

While the `*` is showing, MIDAS GUI periodically (every few minutes) writes an
internal autosave draft JSON in the background — not into the open project's `.h5`.
If the application is later relaunched after a crash or a forced quit and a draft is
found, it offers **"Restore unsaved session?"**; accepting restores it into every tab
and detaches from whatever project was open (so the next Save prompts you for a real
destination, rather than silently overwriting an unrelated project), and declining
discards the draft.

### What is restored automatically
Every field (text boxes, spin boxes, checkboxes, combo/dropdown selections) in every
tab is restored as typed. In addition, **path-backed data is reloaded from disk**, the
same way it loads when you type/browse to a path by hand — images, masks-by-path,
dark/bright/background frames, and HDF5 datasets all re-read their file automatically
after a state load (guarded so a moved/deleted file is skipped quietly rather than
popping a warning). The beamline **Profile** (§16) active at save time is recorded
alongside the tab fields and restored the same way a manual profile switch would be
(header combo synced, tab visibility and calibrant/device dropdowns refreshed) — if it
still exists locally and differs from the one currently active; otherwise this is
silently skipped.

### What is *not* re-run
Restoring a session **does not re-run any long-running pipeline** — Calibrate's Fit,
Batch Integrate, the PDF transform, and Calibration Refinement all keep their inputs
restored but require **one manual click of the tab's own Run/Fit button** to reproduce
their result. This keeps a load fast and avoids silently kicking off a multi-minute
job in the background.

### Sidecars for in-progress derived data
A tab's `gui_workspace` snapshot can hold computed data that hasn't been exported to a
file of its own yet — a drawn/computed **mask** (Mask Builder) and a just-fit
**calibration result** (Calibrate). Saving a session embeds small sidecars into that
tab's own group (never as loose files on disk) so this in-progress work isn't silently
lost:
- the current in-memory **mask**, if any; reloaded automatically the next time this
  session is restored.
- the last fit's **calibration result**, flattened, kept for the record and to reseed
  the manual/seed geometry fields. This does **not** reconstruct the in-memory fitted
  result object (it isn't a plain, re-loadable structure) — re-run **Fit** after
  restoring to reproduce it.

### What gets logged, and when
While a project is open, **every completed Calibrate run** (single-detector
Tab 2, or each of the 4 panels in Hydra mode) and **every completed Batch
Integrate run** (single-detector Tab 4, or each Hydra panel) appends one
record automatically, with no extra action needed. **Mask Builder** is the
one exception — since there's no single "run finished" moment to log from
(a mask can come from Compute, Load, hand-drawn shapes, or any combination),
clicking its **Log to Project** button (§4) explicitly records the current
mask. Nothing is logged when no project is open (the default), so all of
this is entirely opt-in.

Each record is self-contained and includes: the exact parameters used
(pipeline/refine choices for a calibration, kernel/binning/corrections for
an integration, every threshold/method toggle for a mask); the fitted
calibration result **and its computed radial profile / Eta-vs-R cake** (for
a calibration, so re-opening the project never needs to recompute or even
re-locate the original image — see "Opening a project can populate the GUI"
below); the mask/dark/bright/background **files actually used, referenced
by path + hash** — never duplicated into the record, the same as the raw
calibrant image / raw scan data (not duplicated — a scan can be many
thousands of frames), since they're always backed by a real file on disk.
The one exception is a mask that includes anything hand-drawn/computed
directly in Mask Builder with no file of its own to point back to — that
one *is* embedded (compressed) in the calibrate/integrate record that used
it, since there's no path to hash — a **mask attempt** itself (`analysis/
mask`) always embeds its resulting mask array, compressed, since recording
that array *is* the point of logging one. Each record also carries the
midas-gui / MIDAS package versions active at the time, **plus the beamline
Profile active for that run** (see §16), **plus a workstation snapshot** —
hostname, OS name/release/version, CPU model, logical/physical core
counts, and total RAM — recorded automatically, so a run can always be
traced back to exactly which machine it was analyzed on, even long
afterward. A Batch Integrate record
additionally embeds the exact calibration values it used and links back to
the specific Calibrate run that produced them, when that run was itself
logged to the same project.

Records are never overwritten — recalibrating, re-integrating, or logging
another mask adds a new, separately numbered record rather than replacing
the previous one, so the full history of what was tried stays in the file.
A failed provenance write (e.g. disk full) is logged to the tab's own Log
panel (a message box, for Mask Builder) and never blocks or alters the
run's on-screen result.

The file is plain HDF5 (`h5py`/`h5dump`/HDFView can browse it directly) —
no separate tool is required to inspect what a project contains.

### Opening a project can populate the GUI

**File ▸ Open Project…**'s checkbox-tree preview (see above) is where you
choose what to restore — nothing happens until you click **Open** on the
checked selection. The tree has two parts, matching the file's two headers:

- **GUI Workspace** — a **Select all** convenience toggle above a slightly
  indented list of one checkbox per tab that has a saved snapshot in this
  project, the indent making clear "Select all" is the toggle, not a tab of
  its own. Checking a tab restores its fields exactly as **File ▸ Save
  Project** left them, including its sidecars (see above) — unchecking a
  tab leaves it exactly as it currently is.
- **Analysis** — a **Mask** row (checkbox + a picker over every recorded
  mask attempt, defaulting to the latest), then one heading per detector
  group actually present in the file — **Single detector** and/or **Hydra**
  — each followed by its own indented **Calibrate** and **Batch Integrate**
  rows (Hydra additionally labels each row with its panel, **ge1**–**ge4**,
  since a Hydra project can have more than one), every row its own
  checkbox + attempt picker (older attempts stay selectable if the project
  has more than one). Only panel/kind combinations with a recorded attempt
  appear at all. (The Hydra **Overall** composite attempt has no
  corresponding tab to restore into, so it never appears here — it's still
  fully visible, read-only, via **File ▸ Project History…**.)

Checking a **Calibrate**/**Batch Integrate** row switches that tab's mode
ribbon (Single-detector vs Hydra) to match, and restores:

- **Calibrate:** the data path/dataset (and, single-detector only, the frame
  index), wavelength, pixel size, calibrant, refine-parameter checkboxes,
  iteration counts, device, and the fitted BC/Lsd/tilts as a manual seed
  (**Use manual seed** is turned on) — a single click of **Fit** reproduces
  the recorded result. The predicted-ring overlay reappears immediately
  (pure geometry — no image needed), and the **Radial Profile**/**Eta vs R
  Cake** tabs are populated **directly from the record itself** — no
  recompute, and the original image doesn't even need to still be
  reachable. In Hydra mode this happens per panel.
- **Batch Integrate:** the data path/dataset and frame range, kernel, output
  format, and monitor file — plus, unlike a plain GUI-state load, a live,
  ready-to-use **calibration** built from the attempt's own recorded
  geometry, so **Run** works immediately without first re-running Tab 2.
  The **Waterfall** and **Stacked profiles** views are replayed from the
  attempt's own saved per-frame profiles — no re-run needed. In Hydra mode
  each GE panel keeps its own independent pair of views.

Checking the **Mask** row restores Mask Builder's threshold/statistical/
spike/cosmic/azimuthal/learnable field settings and its data-source fields
(image path, stack source), reloading the image if its path still resolves
— and sets the tab's mask directly from the recorded attempt's embedded
array (already fully processed — dilation etc. already applied when it was
first computed), broadcasting it to every other tab exactly as a fresh
**Compute Mask** would.

A **Multi-panel detector** calibration attempt's refined panel shifts are
re-created automatically on restore: populating Tab 2 from such an attempt
writes a fresh `_panelshifts.txt` next to the project file (in a `<project
name>_panel_shifts/` folder) and points the restored calibration at it, so
the panel correction is usable immediately, the same as any other geometry
field. See §5 "Export" for how panel shifts are saved/embedded in the first
place.

### File ▸ Project History…

A read-only browser over everything the active project has recorded, so
checking what's in it doesn't require an external HDF5 tool (`h5dump`,
HDFView). The table lists one row per attempt — panel (or "—" for a mask
attempt, which isn't per-panel), kind (**Mask** / Calibrate / Batch
Integrate), attempt name, and its UTC timestamp — across every panel the
project contains; selecting a row shows that attempt's full recorded
parameters (the same JSON the Open Project checkbox-tree reads from) in a
detail pane below. Purely a viewer: nothing here can modify the project file.

---

*For bugs or questions, see the MIDAS GUI repository.*

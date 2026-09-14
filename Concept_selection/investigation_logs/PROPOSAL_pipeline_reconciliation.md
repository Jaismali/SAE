# Proposal: Reconciling Concept_selection/ and concept_selection_v2/

Status: Ordering (Section 1), manifest.py reconciliation (table),
threshold (2a), and naming (Section 5) are RESOLVED/LOCKED, with real
evidence behind each. The unified-pipeline BUILD ITSELF has not
started -- no files have been changed toward that merge yet. Written
for researcher review and sign-off on the remaining execution steps
(Section 4), given the earlier session incident where a consolidation
move was made without first checking for live dependents.

## 1. Structural-filter-first ordering -- CONFIRMED EMPIRICALLY, not just reasoned about

`concept_filter_pipeline.py` carries this tag: "ORDERING-DEPENDENT-ON:
heuristic auto-interp placeholder in real_backend.py... If the
heuristic is replaced, this ordering needs to be explicitly
re-examined, not silently carried forward." That replacement has now
happened (Qwen-based auto-interp, built and pilot-tested this
session). Per explicit instruction, the reasoning below was NOT
accepted on its plausibility alone -- it was checked against real
behavior first (`check_structural_outlier_auto_interp.py`), because a
plausible-sounding mechanism proposed to replace another plausible-
sounding mechanism, without verification, was exactly the failure
pattern already caught twice elsewhere this session.

**RESULT (confirmed, not hypothesized):** `feat_4` and `feat_11` --
real, confirmed `<bos>`-only structural artifacts from the earlier
diagnostic run -- were run through the actual Qwen explainer+simulator
chain. Qwen CORRECTLY identified both as structural artifacts ("fires
on the beginning-of-sequence token... a start-of-sentence marker," NOT
a semantic theme) -- and BOTH still scored a PERFECT 1.000 correlation.

**This settles the question more conclusively than the original
reasoning did.** The mechanism is not "an honest judge might rate this
coherent-but-trivial" (a hedge). It is: **correlation-based
monosemanticity scoring is mathematically blind to the concept-vs-
artifact distinction, regardless of explanation quality or judge
correctness.** A perfectly-predictable `<bos>`-firing pattern IS,
arithmetically, a perfectly-scoring one -- correctly naming the
artifact does not and cannot change that, because the score never
looks at the explanation text, only at simulated-vs-true numeric
agreement.

**LOCKED (no longer just proposed): structural-filter-first, for this
confirmed reason.** The efficiency argument (avoid spending auto-interp
compute on likely-artifact candidates) remains true and additional,
but is now secondary to the primary, evidence-based reason above. The
ORDERING-DEPENDENT-ON tag in `concept_filter_pipeline.py` should be
updated to reference this confirmed result, not the superseded
heuristic-blind-spot reasoning or the once-hedged replacement reasoning.

## 2. Where stratification fits

Month 1's `selection_pipeline.py` runs stratification LAST (after
monosemanticity + dedup), with explicit reasoning: tier composition
should reflect the final, cleaned candidate pool, since that's what
tier-balanced sampling actually draws from. This reasoning holds
regardless of which filters run before it or in what order -- LOCKED:
stratification LAST in the unified pipeline: structural ->
monosemanticity (real Qwen-scored) -> dedup -> stratification.

## 2a. Monosemanticity threshold -- LOCKED, real value now exists

Per the real n=50 pilot batch and its own decision record
(`DECISION_monosemanticity_threshold.md`): **0.3585**, the median of
n=36 real Qwen auto-interp scores, explicitly a median-split default
(not a distributional-gap threshold -- checked directly, no real gap
exists in the observed distribution). This supersedes Month 1's
synthetic-data-tuned 0.5 for the unified pipeline. Per that decision
record's own re-validation trigger, this MUST be re-derived once the
full 40-50 concept target pool is scored -- not assumed to hold at
that scale just because it held at n=36.

## 2b. NEW open item, surfaced by finishing this document: auto-interp scoring is not yet wired into any pipeline

Neither `concept_filter_pipeline.py` (v2) nor a hypothetical unified
pipeline currently calls the real Qwen auto-interp module
(`llm_auto_interp_local_client.py` + `held_out_context_sampling.py`)
as part of candidate scoring -- `apply_monosemanticity_filter()`
currently reads whatever `candidate.auto_interp_score` already
contains, which for v2's `real_backend.py` output is still the OLD
HEURISTIC score, not a real Qwen score. The real scoring pipeline only
exists today as the standalone `run_pilot_auto_interp_batch.py`
script. **Wiring real auto-interp scoring into the actual candidate-
production path (so `fetch_candidates()` or a post-processing step
populates real scores before `apply_monosemanticity_filter()` ever
runs) is a required, not-yet-done step before the unified pipeline can
run end-to-end on real data.** Flagged here rather than assumed
already covered by the pilot script's existence.

## 3. File-by-file reconciliation table

| Concern | Concept_selection/ (Month 1) | concept_selection_v2/ (this session) | Proposed resolution |
|---|---|---|---|
| `contract.py` | Original | Copied verbatim (needs byte-diff check, not assumed identical) | Single canonical copy, location TBD by researcher preference |
| `monosemanticity_filter.py` | Original, tested (part of Month 1's 60/60) | Copied verbatim | Keep logic identical either way; single canonical copy |
| `deduplication.py` | Original, tested | Copied verbatim | Same as above |
| `stratification.py` | Exists, tested | Does NOT exist in v2 | Must be added to the unified pipeline -- currently missing from v2 entirely |
| `structural_token_filter.py` | Does not exist | Exists, evidence-locked (5x/50%), tested | New filter, must be added to Month 1's side of the merge |
| `real_backend.py` | Still the ORIGINAL UNIMPLEMENTED STUB (confirmed by direct inspection this session -- Month 1 never built this) | Real, GPU-verified SAELens implementation, layer 13 | v2's version is the only working one -- becomes canonical |
| `manifest.py` (Month 1) vs `manifest_freeze.py` (v2) | **NOW INSPECTED.** Auto-incrementing version numbers per directory (`concept_manifest_v1.json`, `v2.json`...); requires `deviation_reason` embedded IN each version's own metadata for version > 1; atomic write via temp-file + rename (protects against interruption mid-write); `load_manifest()` has NO tamper/integrity check. | Single fixed path per call (no auto-versioning); SHA-256 hash-based tamper detection on load (Month 1's version has no equivalent); append-only sidecar `.deviations.log` for full override history; no atomic write. | **Neither strictly dominates -- genuinely complementary, not a pick-one situation.** RECOMMENDATION (needs sign-off, not decided here): merge = Month 1's auto-versioning + atomic write, PLUS this session's hash-based tamper detection added to the load function. Deviation reasoning: keep it colocated in each version's own metadata (Month 1's style, git-tracked, immutable per-version) rather than also maintaining a separate sidecar log -- a versioned JSON per freeze already is a complete audit trail; a second logging mechanism would be redundant rather than additive. |
| Layer target | Unknown whether Month 1's pipeline references layer 14 or 13 anywhere | Layer 13, corrected, documented | Must check whether Month 1's code has any hardcoded layer references needing the same correction |
| `synthetic_backend.py` | Exists, used for Month 1's pipeline testing | Does not exist in v2 | Likely still useful for testing the unified pipeline without GPU access -- probably keep, not delete |

## 4. Safety process for actually executing this (learned from the recent incident)

1. ~~Read `manifest.py`'s actual contents~~ -- DONE, see table above.
2. **Before deleting or moving any file:** run
   `grep -rn "import <module>"` across BOTH folders to find every
   importer, not just the ones already known about. The recent
   incident happened specifically because this check was skipped.
3. **Before merging:** run both folders' full test suites and record
   the exact passing counts as a before-state to compare against.
   Current known counts: Month 1's `Concept_selection/`: 60 passed
   (confirmed after the deletion-incident revert). `concept_selection_v2/`:
   84 passed (as of the held-out-sampling-bug-fix commit).
4. **Wire real Qwen auto-interp scoring into candidate production**
   (Section 2b) -- a genuinely new prerequisite, not part of the
   original plan, surfaced only while finishing this document.
5. **Build the unified pipeline as NEW code** (e.g. a new
   `unified_selection_pipeline.py`) rather than editing either
   existing pipeline file in place, so both old pipelines keep working
   untouched until the new one is confirmed correct.
6. **Test the unified pipeline independently**, including a case that
   exercises all four stages (structural -> monosemanticity -> dedup ->
   stratification) against real-shaped data.
7. **Re-derive the monosemanticity threshold** at whatever scale the
   unified pipeline is first run at for real (per Section 2a's
   re-validation trigger) -- do not carry 0.3585 forward unchecked.
8. **Only after the unified pipeline is confirmed working:** rename
   `concept_selection_v2/` to `concept_selection/` (per Section 5's
   locked naming decision) and retire Month 1's original folder,
   following the same dependent-check discipline as step 2 -- not
   before, and not the reverse order (renaming while both folders
   still coexist would recreate the exact confusion this whole
   reconciliation exists to resolve).

## 5. Decisions locked (were open questions, now resolved)

1. **Structural-filter ordering:** confirmed empirically, locked --
   see Section 1.
2. **`manifest.py`:** read and reconciled -- see table above.
3. **Canonical naming:** rename to `concept_selection/` ONLY once
   Month 1's original folder is fully retired and archived -- never
   keep `_v2` as a permanent name (a migration-artifact name would
   permanently signal "patched-up version of something else" to any
   future reader, including a paper reviewer inspecting a code
   release), and never rename prematurely while both folders still
   coexist (would recreate the exact two-live-folders confusion this
   reconciliation is meant to close out). Do the rename as the FINAL
   step (safety process step 8), not before.

## 6. Still open, surfaced by finishing this document

1. Wiring real Qwen auto-interp scores into actual candidate
   production (Section 2b) -- not started.
2. The `contract.py`/`monosemanticity_filter.py`/`deduplication.py`
   "copied verbatim" status in the table above still has NOT had an
   actual byte-diff check run to confirm the v2 copies haven't
   drifted from Month 1's originals since they were copied -- flagged
   as an assumption in the table, not yet verified.

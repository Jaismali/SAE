# Audit Trail Addendum: Auto-Interp Investigation & Filter Lock-In

(Append to GEOMETRIC_DORMANCY_MASTER_ARCHIVE.md. DECISION-IDs below are
placeholders -- assign real sequential numbers when merging, per the
archive's existing numbering.)

---

## DECISION-XXX: Structural-token/magnitude-outlier filter locked

**Status:** Locked, evidence-based, pending full-scale re-validation.

**Criterion:** Exclude a candidate SAE feature if BOTH hold: (a)
`mean_activation_magnitude` > 5x the batch's median nonzero magnitude,
AND (b) >50% of its top-10 activating tokens are structural/special
tokens (same "dominated" definition used elsewhere, not a separate
bespoke fraction).

**Evidence:** n=20 pilot sample (40 documents, layer 13). 5/20
candidates were structural-token-dominated. Magnitude ratios to batch
median: 0.44x, 0.70x, 1.58x -- [real gap, confirmed via full ratio
listing, not just the >10x outliers] -- 15.85x, 19.02x. Structural
fractions: 90%, 60%, 60%, 100%, 100%. feat_3 (90% structural fraction,
0.44x median magnitude) demonstrated that high structural-token
fraction does NOT imply magnitude-outlier status -- justifying the
conjunction over a fraction-only rule. Both actual magnitude outliers
were at 100% structural fraction, supporting reuse of the existing
50%-dominance definition rather than a second bespoke threshold. 5x
(not 8x) chosen because it sits at the edge of the observed gap
(1.58x-15.85x), the more literal reading of where the gap sits, rather
than padding further for conservatism the evidence didn't call for.

**Limitation:** n=5 structural-dominated candidates out of a 20-feature
pilot. Real, observed gap, not noise -- but small sample. MUST be
re-validated at the full 40-50 concept scale, same as monosemanticity
(0.5) and dedup (0.95) thresholds.

**Implementation:** `structural_token_filter.py`, 12/12 tests passing
(real-data reconstruction + boundary cases).

---

## DECISION-XXX: Filter pipeline ordering locked

**Status:** Locked, CONTINGENT on current auto-interp heuristic.

**ORDERING-DEPENDENT-ON:** heuristic auto-interp placeholder in
`real_backend.py` (`_heuristic_auto_interp`). Search this exact tag
when the heuristic is replaced (e.g. by real Neuronpedia data) -- the
ordering reasoning below may not survive that change and must be
re-examined, not silently carried forward.

**Order:** structural-token filter -> monosemanticity filter -> dedup.

**Mechanism (not a general principle):** the current heuristic scores
`1 - unique_token_fraction`, so LOW token diversity produces a HIGH
monosemanticity score. An attention-sink feature (fires almost
entirely on one structural token) has very low diversity and would be
falsely rated highly monosemantic if monosemanticity ran first,
letting it reach dedup and potentially become a duplicate cluster's
kept representative. Running structural-token filtering first closes
this specific blind spot.

**Verified via causal test** (`test_concept_filter_pipeline.py`): a
synthetic attention-sink candidate (`['<bos>']*10`) scores 1.0 under
the current heuristic (comfortably above the 0.5 threshold, confirming
the blind spot is real) but is excluded by the structural filter
before ever reaching monosemanticity, in the actual pipeline order.
3/3 tests passing.

**Implementation:** `concept_filter_pipeline.py`.

---

## FINDING: Auto-interp heuristic has a structural, two-directional flaw

**Not a calibration problem -- a design problem.** Confirmed via a
real n=20 pipeline run on GPU-derived candidates (not synthetic):
monosemanticity filter rejected 16/18 structural-survivors (89%).
Score distribution was collapsed toward zero (11/18 scored exactly
0.000).

Manual spot-check of rejected candidates found BOTH predicted failure
directions present in the same run:
- feat_5: genuinely coherent theme (`controversy, Debate, Fault,
  Failure, pain, error, weaknesses, fault, painful, compare`) --
  fully lexically diverse, scored 0.000. Confirms the heuristic
  penalizes real, coherent concepts purely for not repeating tokens.
- feat_0/1/6/7: shared a near-identical token fragment traced to a
  SINGLE document (`doc[0]` of 40) containing "It is done, and
  submitted. You can play 'Survival of the Tastiest'...". Confirmed via
  distinctive-substring search (1/40 documents matched "Tastiest" /
  "Survival of the"), ruling out cross-document duplication (0 exact,
  0 near-duplicates found across all 40 docs) as the cause.

## FINDING: Reference corpus is under-sampled, and does not converge with more of the same

Diagnostic 8 (`diagnostic_corpus_convergence_check.py`) tested
whether growing the corpus resolves the sparse-sampling artifact
found above. Fixed 10 features tracked across three NESTED corpus
sizes (40 -> 150 -> 400 documents, same fixed stream order, confirmed
by inspection of `real_backend.py`'s unshuffled `enumerate()`-based
streaming -- no code change was needed to support this comparison).

**Result: turnover did NOT decrease.** 60.2% average top-10-token
turnover between 40->150 docs; 61.4% between 150->400 docs (flat,
slightly higher, not declining, across a 10x corpus increase).
Excluding the one degenerate always-<bos> feature (trivially 0%
turnover), the remaining 9 features averaged ~67-68% turnover at both
steps. This is a genuine negative result against the 1/sqrt(N) null
hypothesis for sampling convergence, not an inconclusive one.

**Decision: do NOT pursue further local corpus scaling.** Reasoning
(researcher's explicit call): even if a much larger local corpus (e.g.
1500-2000 docs) eventually showed declining turnover, this would be
reconstructing -- on a single laptop GPU with a generic corpus --
something the original Gemma Scope 2 authors computed with dedicated
infrastructure at a scale plausibly 2-3 orders of magnitude beyond
what's locally tractable, with no local ground truth to confirm
convergence was actually reached. Bad risk/cost tradeoff before even
counting GPU-hours across 40-50 concepts.

## DECISION-XXX: Pursue real Neuronpedia auto-interp access request

**Status:** In progress. Request submitted (see email draft, session
log) requesting access to gemma-3-1b-it Gemma Scope 2 sources
(layer 13 primary) and asking explicitly for expected turnaround time.

**Known schedule risk:** Neuronpedia's own Support & Contact page
lists channels (GitHub Issues, Slack, email) but publishes NO stated
SLA for access-request turnaround, unlike AIKosh's explicit "3 working
days" (Month 1 precedent). This is an open-ended timeline risk,
flagged explicitly rather than assumed to resolve quickly.

**Explicitly NOT done while pending, and why:** no further local
corpus-scaling experiments (see finding above -- the risk/cost
analysis holds regardless of wait time), and no parallel "better
heuristic" build-out (would risk solving a problem that may not exist
in its current form once real data or Neuronpedia's own response is in
hand; if the request is rejected, that is the correct moment to design
a replacement heuristic with full context, not before).

**Parallel work performed while request is pending** (all genuinely
non-blocked by the auto-interp question): manifest-freezing/
timestamping module built and tested against placeholder data
(`manifest_freeze.py`, 8/8 tests) -- this is real forward progress on
Part A's actual deliverable, since the freeze logic is agnostic to
whether the auto-interp scores it freezes are placeholder or real.

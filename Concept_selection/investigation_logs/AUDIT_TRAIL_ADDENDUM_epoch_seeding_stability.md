# Audit Trail Addendum: Epoch Schedule, Reproducibility, and Induction Stability

(Append to GEOMETRIC_DORMANCY_MASTER_ARCHIVE.md. DECISION-IDs below are
placeholders -- assign real sequential numbers when merging.)

---

## DECISION-XXX: Asymmetric induction/correction epoch schedule locked

**Status:** Locked, explicitly scoped -- read the scope limits below
before treating this as broadly validated.

**Locked values:** induction=4 epochs, correction=1 epoch. Supersedes
Month 1's untested default of a shared 3 epochs for both stages.

**Literature anchor:** Qi et al. (2023), "Fine-tuning Aligned Language
Models Compromises Safety" (Llama-2-7B-Chat). Their behavior-instilling
attacks typically use ~5 epochs; their correction-analog (standard
benign Alpaca/Dolly fine-tuning) uses 1 epoch by official default, with
their own ablation showing no benefit beyond 1. This indicated the
right SHAPE (asymmetric, induction-high/correction-low), not
transferable numbers -- confirmed by two explicitly stated translation
gaps: (1) model scale (Llama-2-7B vs. our Gemma 3 1B, ~7x difference,
direction of effect on required epochs not obvious a priori), and (2)
task-analogy is directional only (their induction is an adversarial
attack fighting safety-training resistance; ours instills a neutral
theme with no competing resistance; their correction-analog doesn't
actively suppress anything, ours does).

**Empirical procedure:** tested epochs 1, 3 (Month 1 baseline), 4, and
5 on `dog_theme` and `ocean_theme`, with induction and correction
epoch counts varied independently. 1 epoch: gate failed (induction
delta 0.0074, ~6x weaker than 3-epoch baseline). 5 epochs: strong
deltas (0.187 dog_theme, 0.421 ocean_theme) but visible fluency
degradation in generated text, including a genuine content-collapse
case (ocean_theme, one held-out prompt produced zero theme tokens) --
this was identified as a real confound: TokenFrequencyMeasure can be
distorted in EITHER direction by degenerate generation (word-salad
repetition inflates the score, content collapse deflates it), so the
5-epoch delta increases could not be trusted as purely "cleaner,
stronger induction." 4 epochs: deltas interpolated smoothly and
monotonically between 3- and 5-epoch results for both concepts
(dog_theme: 0.046 -> 0.144 -> 0.187; ocean_theme: 0.046 -> 0.137 ->
0.421), with dog_theme's syntactic breakdown mostly resolved.
Correction=1 confirmed clean via manual generation inspection at every
epoch setting tested -- no verbatim overfitting to the tiny correction
set, fluent and varied text across all held-out prompts.

**Confidence is NOT symmetric between the two numbers, stated
explicitly per instruction:** correction=1 has two independent lines
of support (Qi et al.'s literature default AND this project's own
diagnostic finding that 3-epoch correction saturates by epoch 1,
observed in the original Diagnostic 6 loss curve). induction=4 has
only the epoch-interpolation evidence above -- a real but narrower
basis.

**SCOPE LIMIT, explicit:** this is an n=2-concepts decision
(`dog_theme`, `ocean_theme`), appropriately scoped given induction is
purely instrumental (no RQ reads off induction-stage text quality
directly) and correction's stronger evidentiary basis above. NOT yet
validated across a broader concept sample. Must be re-examined once
the real 40-50 concept manifest exists, same as every other
synthetic/pilot-tuned threshold in this project.

---

## FINDING: No training run before this session was reproducible

**Confirmed by inspection, not assumed:** grep of `pilot_model_io.py`
found zero calls to `torch.manual_seed` or any other seeding mechanism
anywhere in the training loop, before this session. Every run in the
epoch/hyperparameter investigation above -- including the evidence
underlying the locked epoch schedule -- was conducted on UNSEEDED
runs. This is a real, standalone gap against the project's
reproducibility standard, independent of the epoch decision itself.

**Fix implemented:** `set_deterministic_seed(seed)` added to
`pilot_model_io.py`, called from `load_base_model(seed=...)` before
LoRA adapter initialization (the specific point unseeded randomness
entered). Also sets cuDNN to deterministic mode (disables benchmark-
mode kernel selection), since `torch.manual_seed` alone does not
guarantee GPU reproducibility. Seed values are logged per-run via
`log_seed_used()` to a `seeds_used.csv` alongside loss logs, kept
external to `pilot_runner.py`'s tested `PilotConceptResult` (see
convention below).

**Verified, not just built:** `dog_theme` and `ocean_theme` each run
twice at seed=42 via `single_concept_pilot_check.py` produced BIT-FOR-
BIT identical loss trajectories and scores across both runs of each
concept.

**A related bug found and fixed during this verification:**
`inspect_generations.py` generated "baseline" text without ever
calling `model.eval()` -- `get_peft_model()` leaves a fresh model in
train() mode by default, so baseline generation happened with dropout
ACTIVE. This was conceptually wrong (a baseline read should reflect
clean pretrained behavior) and was confirmed to be the actual cause of
an apparent same-seed non-reproducibility (two seed=7 runs on
`ocean_theme` initially produced different post-induction text).
Fixed by calling `model.eval()` immediately after `load_base_model()`
before baseline generation. Re-verified: two post-fix seed=7 runs on
`ocean_theme` produced word-for-word identical output across all three
stages (baseline, post-induction, post-correction) and identical
stability-gate verdicts.

**IMPLICATION FOR THE EPOCH-SCHEDULE DECISION ABOVE, stated explicitly
per instruction:** the induction=4/correction=1 decision was made
using UNSEEDED runs (seeding didn't exist yet during that
investigation). It should be treated as PROVISIONAL pending a seeded
re-confirmation, not silently upgraded to "controlled" after the fact
just because seeding now exists for later work.

---

## DECISION-XXX: Induction-stage stability gate -- exclude, not retry

**Status:** Locked and built (`induction_stability_gate.py`).

**Motivating finding:** `ocean_theme` at induction=4/correction=1
showed degenerate verbatim n-gram repetition (e.g. "I noticed coral."
repeated 5-6x verbatim) on a MAJORITY of post-induction held-out
prompts, reproducibly, across two DIFFERENT explicit seeds (42: 4/5
degenerate; 7: 4/5 degenerate, confirmed identical across two
reproducible re-runs post-eval-mode-fix). `dog_theme` showed zero such
degeneracy across multiple seeds at the same settings. This is a
controlled replication across genuinely different random
initializations, not a single unlucky draw -- ruling out "retry with a
different seed" as the fix, since two different seeds both failed.

**Why the existing sanity gate does not catch this:** `TokenFrequencyMeasure`
averages across 5 held-out prompts into one aggregate score. A
collapsed/degenerate prompt can be diluted into a still-plausible-
looking aggregate pass rather than surfaced -- this is suspected to be
a PRE-EXISTING gap (not proven to be new at any specific epoch count),
since `ocean_theme`'s original Month 1 3-epoch checkpoint no longer
exists on disk and could not be retroactively inspected. This missing-
checkpoint gap is itself worth noting: checkpoints (or at minimum
generation samples) should be retained longer specifically to permit
this kind of retrospective check in the future.

**Mechanism built:** `repetition_detector.py` -- flags a generated
continuation as degenerate if any 2-4 word n-gram repeats 3+ times.
Validated against real session text (both genuine failures and Month
1's accepted "repetitive-but-coherent-across-different-sentences"
pattern, which must NOT be flagged). `induction_stability_gate.py`
composes this over a concept's post-induction generations, applying a
majority threshold (>=3 of 5 prompts degenerate -- both real failures
observed, 4/5 and 4/5, clear this comfortably) and returns a full
per-prompt record (not just a boolean), since the raw pattern (which
prompts, how many, which failure mode) was what made distinguishing
"concept problem" from "seed problem" possible in the first place.

**Policy: auto-exclude, NOT auto-retry.** Two independent, different,
explicit seeds both failed for `ocean_theme` -- this is a concept-level
property of the induction stage at these settings, not a bad-seed
problem a retry would likely fix. Post-correction was clean in both
`ocean_theme` failures observed -- the instability is specific to the
induction stage, not a general "this concept is broken" verdict.

**Integration status:** wired into `inspect_generations.py` (prints
the gate verdict after post-induction generation). NOT wired into the
automated `run_pilot.py` pipeline -- that pipeline never generates raw
text at all (only aggregate scores via `TokenFrequencyMeasure`), so
automatic exclusion during a real run would require either exposing
generation from within `pilot_runner.py`'s flow or restructuring
`TokenFrequencyMeasure` to produce generations as a byproduct. Left as
an open design question for when the real 40-50 concept run is
unblocked, not solved here.

---

## PROJECT CONVENTION: External composition over pilot_runner.py by default

Written as a standing rule after this question recurred twice in one
session (the fine-tune-call stage-labeling closure, and this stability
gate):

> New sanity/diagnostic checks are built as external, composable
> functions over pipeline outputs BY DEFAULT. Modifying
> `pilot_runner.py` directly requires an explicit reason why external
> composition doesn't work, stated and signed off -- not just
> convenience.

Rationale: `pilot_runner.py` is validated Month 1 orchestration code.
Every change considered against it in this session was additive risk
for no demonstrated need, since nothing built so far required running
mid-training or accessing anything `run_pilot_for_concept()` has that
a function operating on its output doesn't already get.

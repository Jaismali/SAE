# Scoping: LLM-Based Auto-Interpretation Module

Status: SCOPING IN PROGRESS, started in parallel with the Neuronpedia
follow-up window. Not yet built.

## Framing (stated explicitly, not left implicit)

**This is the PERMANENT primary auto-interpretation methodology for
this project, not a contingency fallback.** Neuronpedia access was
always a shortcut for obtaining a legitimate auto-interp signal "for
free" rather than generating one directly -- Month 1's `real_backend.py`
stub was always intended to eventually wire in a real auto-interp
method, and Neuronpedia was simply the path that required the least
new work if it panned out. It didn't pan out on a usable timeline.

The method scoped here (Bills et al. 2023, "Language models can
explain neurons in language models") is the actual published standard
for this task, independently citable, and arguably MORE defensible in
the eventual paper than depending on a third party's pre-computed,
unvalidated explanations would have been -- this project generates and
validates its own auto-interp signal end-to-end. If Neuronpedia
replies before this is built, that's a nice-to-have cross-check or
time-saver against an independently-built method -- not a methodology
the paper should be described as depending on either way.

## Method (per Bills et al. 2023, confirmed via literature search, not assumed)

Two-stage design:

1. **Explainer step:** an LLM is shown a feature's top-activating
   examples (tokens/context windows) and asked to describe, in natural
   language, what pattern the feature responds to.
2. **Simulator step:** a (possibly separate) LLM is given the
   explanation and a set of HELD-OUT text contexts (not used to
   generate the explanation), and asked to predict the feature's
   activation strength at each token, typically on a discretized scale
   (e.g. 0-10).
3. **Scoring:** Pearson correlation between the simulated and the true
   (real SAE-measured) activation values across the held-out contexts.
   This correlation IS the auto-interp score, replacing both the
   current heuristic's score and Neuronpedia's stored score.

Both explainer and simulator roles can be filled by Claude via the
Anthropic API, which this project already has access to (per the
Anthropic API section of the project's tooling).

## Design decision 1: self-consistency scoring, locked BEFORE building

Per explicit instruction, deciding this now rather than letting it be
implicitly decided by whatever's easiest to code:

- **Held-out context sampling:** "top-and-random" mix, matching the
  literature standard (not a from-scratch invention) -- half the
  held-out contexts drawn from the feature's own top-activating
  examples (excluding whichever ones were shown to the explainer), half
  sampled uniformly at random from the reference corpus. PROPOSED: 10
  held-out contexts total (5 top-activating-but-unseen, 5 random),
  scaled down from typical published setups (which often use more) to
  fit this project's compute/API budget -- flagged as a deliberate,
  logged scope reduction, not an oversight.
- **Activation discretization:** 0-10 integer scale for the simulator's
  predictions, matching Bills et al.'s original design, so the
  correlation calculation is well-defined and comparable to published
  practice.
- **Threshold, CALIBRATION RISK FLAGGED UP FRONT:** Bills et al.'s own
  paper found simulator correlation scores are GENERALLY LOW across
  the board (a small fraction of neurons scored highly at all). This
  means adopting a naive fixed threshold (e.g. "correlation > 0.5 =
  monosemantic") risks discarding most real, legitimate features. DO
  NOT pick an absolute cutoff before seeing this project's own actual
  score distribution. Instead: compute scores for an initial pilot
  batch of candidates first (mirroring how the 0.5/0.95/5x thresholds
  were all locked against real observed distributions elsewhere in
  this project, not picked in advance), then set the monosemanticity
  threshold from that real distribution -- e.g. a percentile-based cut
  within this project's own candidate pool, not an absolute number
  copied from a different paper's differently-scaled study.

## Design decision 2: cost / rate-limit estimate, done BEFORE building

**Known unknown, flagged rather than guessed past:** the exact number
of RAW candidate features that will be scored before filtering down to
the final 40-50 concept manifest is not yet fixed -- this depends on
Month 1's stratified frequency-quartile sampling design, which
determines how many candidates get pulled per tier before the
monosemanticity/dedup/structural filters reduce that pool. This number
needs to come from the actual Concept_selection/ pipeline configuration,
not be assumed here.

**Rough estimate, pending that real number:**
- Each candidate feature needs: 1 explainer call + 1 simulator call
  (assuming a single simulator call can score all ~10 held-out contexts
  at once, which the literature's prompting approach supports) = 2 API
  calls per candidate.
- Per-call size: explainer prompt includes top-activating examples
  (~10-20 short snippets) -- roughly 500-1500 input tokens, a few
  hundred output tokens for the explanation. Simulator prompt includes
  the explanation plus 10 held-out contexts -- roughly 800-2000 input
  tokens, up to a few hundred output tokens for the per-token
  predictions.
- **Illustrative scenario (numbers ARE placeholders pending the real
  candidate count):** if ~300-500 raw candidates need scoring before
  filtering to 40-50 final concepts, that's ~600-1000 API calls total,
  roughly 1-2M total tokens combined input+output. This is a small
  fraction of what a single real GPU pilot run already costs in time,
  and should be inexpensive in absolute API cost terms -- but the
  exact dollar figure depends on current Anthropic API pricing, which
  should be checked at build time (pricing can change) rather than
  quoted from memory here.
- **ACTION ITEM before building:** get the real candidate-pool count
  from the actual Concept_selection/ pipeline config, and re-run this
  estimate with real numbers before starting the API-calling loop, not
  just an illustrative scenario.

## Neuronpedia deadline: 2 weeks, reasoning stated explicitly

Matching the standard already set for every other threshold this
session (5x, 50%, induction=4 -- always with stated reasoning, never a
bare round number): 2 weeks is long enough to be a fair wait for what
Neuronpedia's own site describes as a project with no revenue model
run by a small team/volunteer effort, and short enough not to block
Part A indefinitely while the actual permanent methodology (above) is
buildable in parallel regardless of their answer.

**Interim step before the deadline:** follow up via Neuronpedia's
Slack (#neuronpedia) or GitHub Issues -- a different channel than the
original email, in case that's simply sitting unseen.

## Immediate next steps

1. Follow up with Neuronpedia via Slack/GitHub this week (independent
   of the rest of this plan).
2. Get the real raw-candidate-pool count from Concept_selection/'s
   pipeline design, to replace the illustrative cost estimate above
   with real numbers.
3. Build the explainer + simulator API-calling module (not started).
4. Run it against an initial pilot batch of real candidates.
5. Set the monosemanticity threshold from that real score distribution
   -- not before.
6. Only then, re-run the full structural -> monosemanticity -> dedup
   pipeline with the new, real auto-interp scores in place of the
   heuristic.

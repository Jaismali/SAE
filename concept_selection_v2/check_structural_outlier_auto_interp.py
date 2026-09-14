"""
Empirical check for the pipeline-reconciliation proposal's Section 1
(structural-filter-ordering re-justification).

NOT a unit test -- a manual sanity script, run once. Tests whether
PROPOSAL_pipeline_reconciliation.md's Section 1 reasoning ("a correct
LLM judge would legitimately score a <bos>-dominated feature as
coherent-but-trivial, so structural-filter-first is still justified,
just for a different reason") actually holds, rather than accepting
it as a plausible-sounding hypothesis. Per explicit instruction: this
is exactly the "propose a mechanism instead of checking it" pattern
already flagged twice this session (the epoch-count fork, the
retry-vs-exclude fork) -- checking it against real behavior instead of
trusting the reasoning alone.

Uses REAL data, not invented examples: feat_4 and feat_11 from the
n=20 real SAELens backend diagnostic run earlier this session, both
confirmed structural-token-dominated magnitude outliers
(top_activating_tokens = ['<bos>']*10, magnitude ratios 19.02x and
15.85x respectively -- the exact candidates that originally motivated
building the structural-token filter).

Three possible outcomes, all informative:
  1. Qwen correctly identifies these as BOS/structural artifacts AND
     scores them low -- the ordering question needs a THIRD pass, since
     real auto-interp might already catch this without a separate
     structural filter running first.
  2. Qwen scores them as coherent/monosemantic (mistakenly or not) --
     confirms Section 1's proposed reasoning: structural-filter-first
     remains necessary even with a correct-in-principle judge.
  3. Qwen's explanation and/or score is unstable/nonsensical for this
     degenerate input -- a different, separate finding about how the
     auto-interp module handles degenerate candidates, worth its own
     follow-up regardless of the ordering question.

Usage:
    python check_structural_outlier_auto_interp.py
"""

from llm_auto_interp_local_client import load_auto_interp_model, run_llm_auto_interp_local

# Real data from the n=20 real backend diagnostic run (earlier this session).
# Both are the confirmed structural-token-dominated magnitude outliers that
# motivated the structural_token_filter.py build.
STRUCTURAL_OUTLIER_CANDIDATES = {
    "feat_4": {
        "top_activating_tokens": ["<bos>"] * 10,
        "magnitude_ratio": 19.02,
    },
    "feat_11": {
        "top_activating_tokens": ["<bos>"] * 10,
        "magnitude_ratio": 15.85,
    },
}

# Held-out contexts + hand-assigned true activations: since these are
# real <bos>-only features, the "true" behavior is that they fire near
# the START of any sequence (where <bos> sits) and NOT on ordinary
# mid-sequence content -- reflected here as high activation on a
# stand-in for "sequence start" and low on ordinary unrelated content.
# This is a simplification (we don't have per-token real activation
# traces for arbitrary new text for these specific features), flagged
# as a real limitation of this check, not hidden.
HELD_OUT_CONTEXTS = [
    "<bos> This is the start of a new document.",
    "The quarterly earnings report showed strong growth.",
    "<bos> Beginning of a fresh conversation here.",
    "She walked to the store to buy some milk.",
]
TRUE_ACTIVATIONS = [8.0, 0.0, 8.0, 0.0]  # high at <bos>-containing starts, low otherwise


def main():
    print("=== Empirical check: does real Qwen auto-interp change the ordering reasoning? ===")
    print("Loading local auto-interp model...")
    model, tokenizer = load_auto_interp_model()
    print("  Loaded.")
    print()

    for feature_id, data in STRUCTURAL_OUTLIER_CANDIDATES.items():
        print(f"--- {feature_id} (magnitude_ratio={data['magnitude_ratio']}x, "
              f"top_activating_tokens all '<bos>') ---")
        try:
            result = run_llm_auto_interp_local(
                model=model,
                tokenizer=tokenizer,
                top_activating_examples=data["top_activating_tokens"],
                held_out_contexts=HELD_OUT_CONTEXTS,
                true_activations=TRUE_ACTIVATIONS,
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print("  (This itself is informative -- see outcome 3 in the module docstring.)")
            print()
            continue

        print(f"  Explanation: {result.explanation!r}")
        print(f"  Score (Pearson correlation): {result.score:.3f}")
        print()

    print("=== Read this yourself against the three possible outcomes in the module docstring ===")
    print("Does the explanation correctly identify this as a BOS/structural artifact")
    print("(e.g. mentions 'beginning of sequence', 'start token', 'structural', etc.)")
    print("rather than describing a semantic theme? Does the SCORE end up low despite")
    print("a correct-sounding explanation, or does correctly identifying the pattern")
    print("still produce a high 'monosemantic' score? Both are possible and both are")
    print("informative for which version of Section 1's reasoning is actually true.")


if __name__ == "__main__":
    main()

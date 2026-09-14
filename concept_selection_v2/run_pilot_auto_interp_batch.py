"""
Part A pilot batch: real candidates -> structural filter -> REAL Qwen
auto-interp scoring -> observed score distribution.

NOT a unit test -- a manual GPU run, executed once by hand. This is
the actual pilot batch called for in SCOPING_llm_auto_interp_module.md:
does NOT assume a raw candidate count or a monosemanticity threshold
in advance -- both get set from what this run actually observes.

Pipeline order (per PROPOSAL_pipeline_reconciliation.md Section 1,
now evidence-backed rather than reasoned-about only -- see
check_structural_outlier_auto_interp.py's result: Qwen scored two
confirmed <bos>-only structural artifacts at a PERFECT 1.000
correlation despite correctly naming them as artifacts, proving
monosemanticity-as-correlation cannot by itself distinguish real
concepts from trivial structural regularities):

    1. Fetch N raw candidates (real SAELens backend, layer 13)
    2. Structural-token filter FIRST (mandatory given the above finding
       -- correlation-based scoring alone will NOT catch these)
    3. Real Qwen auto-interp scoring for structural survivors
    4. Report the observed score distribution (no threshold applied
       yet -- that gets set from this distribution, not assumed)

Dedup and stratification are NOT run in this pilot batch -- this run's
purpose is scoring-pipeline validation and threshold-setting evidence,
not producing a final manifest.

Usage:
    python run_pilot_auto_interp_batch.py [num_candidates] [num_documents]
    Example:
    python run_pilot_auto_interp_batch.py 50 40
"""

import sys

from real_backend import RealConceptBackend
from structural_token_filter import apply_structural_token_filter
from held_out_context_sampling import sample_held_out_contexts, has_sufficient_variance
from llm_auto_interp_local_client import load_auto_interp_model, run_llm_auto_interp_local

LAYER = 13
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
MODEL_ID = "google/gemma-3-1b-it"

NUM_HELD_OUT_CONTEXTS = 10
HELD_OUT_WINDOW_SIZE = 5
HELD_OUT_SAMPLING_SEED = 42  # explicit, logged, reproducible sampling


def main():
    num_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    num_documents = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    print(f"=== Pilot auto-interp batch: {num_candidates} candidates, "
          f"{num_documents} reference documents ===")
    print()

    backend = RealConceptBackend(
        model_id=MODEL_ID,
        sae_id=SAE_ID,
        sae_release=SAE_RELEASE,
        num_documents=num_documents,
    )

    print("Loading Gemma 3 1B-IT + SAE...")
    gemma_model, sae = backend._load_model_and_sae()
    print("  Loaded.")

    print(f"Fetching {num_candidates} raw candidates and running activation extraction...")
    candidates = backend.fetch_candidates(layer=LAYER, max_candidates=num_candidates)
    print(f"  Got {len(candidates)} candidates.")
    print()

    print("Re-collecting the full per-token activation matrix for held-out sampling "
          "(reuses the same extraction method, run once more since fetch_candidates() "
          "doesn't expose the raw matrix externally -- a known, flagged inefficiency, "
          "acceptable for a pilot-scale run)...")
    activations, tokens = backend._collect_activations_for_all_features(gemma_model, sae, LAYER)
    print(f"  Activations shape: {activations.shape}")
    print()

    print("Freeing Gemma 3 1B + SAE from GPU memory before loading the judge model -- "
          "the 8GB card cannot hold both simultaneously (confirmed by a real failure "
          "on the first attempt at this pilot run, not assumed in advance).")
    print("NOTE: backend._load_model_and_sae() caches the model/SAE on the backend")
    print("object itself (backend._model/backend._sae) -- deleting only the local")
    print("gemma_model/sae variables would NOT free GPU memory, since the backend")
    print("would still hold live references. Clearing the backend's own attributes too.")
    import gc
    import torch
    backend._model = None
    backend._sae = None
    del gemma_model
    del sae
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed.")
    print()

    print("=== Stage 1: structural-token filter ===")
    structural_result = apply_structural_token_filter(candidates)
    print(f"  {len(structural_result.accepted)}/{len(candidates)} candidates passed "
          f"structural filtering.")
    print(f"  Rejected: {list(structural_result.rejection_reasons.keys())}")
    print()

    print("Loading local auto-interp model (Qwen2.5-7B-Instruct)...")
    judge_model, judge_tokenizer = load_auto_interp_model()
    print("  Loaded.")
    print()

    print("=== Stage 2: real Qwen auto-interp scoring for structural survivors ===")
    print("Three-way categorization, per real evidence from an earlier pilot run:")
    print("  SCORED: got a real, meaningful correlation score")
    print("  INSUFFICIENT_DATA: corpus lacks enough activating examples for this")
    print("    feature to produce a meaningful score -- NOT a low-monosemanticity")
    print("    judgment, just missing data. Checked BEFORE calling the LLM, saving")
    print("    compute on candidates that cannot be scored regardless of judge quality.")
    print("  PARSE_FAILED: simulator's response could not be parsed even after retries")
    print("    -- real evidence of judge-model limitation, not silently retried past")
    print("    indefinitely or hidden inside a misleading 0.0 score.")
    print()

    scored = []
    insufficient_data = []
    parse_failed = []

    for i, candidate in enumerate(structural_result.accepted):
        feature_idx = int(candidate.feature_id.split("_")[-1])
        feature_activations = activations[:, feature_idx]

        top_indices = sorted(
            range(len(tokens)), key=lambda idx: feature_activations[idx], reverse=True
        )[:10]

        held_out = sample_held_out_contexts(
            feature_activations,
            tokens,
            excluded_indices=top_indices,
            num_samples=NUM_HELD_OUT_CONTEXTS,
            window_size=HELD_OUT_WINDOW_SIZE,
            seed=HELD_OUT_SAMPLING_SEED,
        )
        if not held_out:
            print(f"  [{i+1}/{len(structural_result.accepted)}] {candidate.feature_id}: "
                  f"INSUFFICIENT_DATA (no held-out contexts available at all)")
            insufficient_data.append(candidate.feature_id)
            continue

        held_out_contexts = [context for context, _ in held_out]
        true_activations = [activation for _, activation in held_out]

        # Check variance BEFORE calling the LLM -- saves compute on
        # candidates that cannot produce a meaningful score regardless
        # of judge quality, per the real finding that 10/48 candidates
        # in an earlier run had all-zero true activations even after
        # the top-and-random sampling fix (genuine corpus exhaustion).
        if not has_sufficient_variance(true_activations):
            print(f"  [{i+1}/{len(structural_result.accepted)}] {candidate.feature_id}: "
                  f"INSUFFICIENT_DATA (true activations have ~zero variance: "
                  f"{[round(v, 2) for v in true_activations]}) -- skipping LLM call")
            insufficient_data.append(candidate.feature_id)
            continue

        try:
            result = run_llm_auto_interp_local(
                model=judge_model,
                tokenizer=judge_tokenizer,
                top_activating_examples=candidate.top_activating_tokens,
                held_out_contexts=held_out_contexts,
                true_activations=true_activations,
            )
        except ValueError as e:
            print(f"  [{i+1}/{len(structural_result.accepted)}] {candidate.feature_id}: "
                  f"PARSE_FAILED (even after retries) -- {e}")
            parse_failed.append(candidate.feature_id)
            continue
        except Exception as e:
            print(f"  [{i+1}/{len(structural_result.accepted)}] {candidate.feature_id}: "
                  f"UNEXPECTED FAILURE -- {type(e).__name__}: {e}")
            parse_failed.append(candidate.feature_id)
            continue

        scored.append((candidate.feature_id, result.score, result.explanation))
        print(f"  [{i+1}/{len(structural_result.accepted)}] {candidate.feature_id}: "
              f"SCORED score={result.score:.3f} -- {result.explanation[:80]!r}")

    print()
    print("=== Summary ===")
    total = len(structural_result.accepted)
    print(f"  Total structural survivors: {total}")
    print(f"  SCORED: {len(scored)}")
    print(f"  INSUFFICIENT_DATA (excluded from distribution -- not a real judgment): {len(insufficient_data)}")
    print(f"  PARSE_FAILED (excluded from distribution -- real judge-model limitation): {len(parse_failed)}")
    print()

    if not scored:
        print("  No usable scores obtained -- investigate the failures above.")
        return

    print("=== Observed score distribution, SCORED candidates ONLY "
          "(no threshold applied -- read this yourself) ===")
    sorted_scores = sorted(scored, key=lambda x: x[1])
    for feature_id, score, explanation in sorted_scores:
        print(f"  {feature_id}: {score:.3f}")

    values = [s[1] for s in scored]
    print()
    print(f"  n={len(values)} (usable, out of {total} structural survivors), "
          f"min={min(values):.3f}, max={max(values):.3f}, "
          f"median={sorted(values)[len(values)//2]:.3f}")
    print()
    print("Use this REAL, PROPERLY-CATEGORIZED distribution to set the monosemanticity")
    print("threshold -- per the scoping plan, do not assume a threshold in advance.")
    print("The INSUFFICIENT_DATA and PARSE_FAILED counts are themselves real findings:")
    print("report them honestly rather than folding them into the scored distribution")
    print("or silently discarding them from the record.")


if __name__ == "__main__":
    main()

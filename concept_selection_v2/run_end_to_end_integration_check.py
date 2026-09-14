"""
End-to-end integration check: real candidates -> real Qwen rescoring ->
full filter pipeline (structural -> monosemanticity -> dedup).

NOT a unit test -- a manual GPU run, executed once by hand. This is
the first time ALL of this session's real-data pieces run together in
one chain: RealConceptBackend (SAELens, layer 13) ->
rescore_with_llm_auto_interp (Qwen2.5-7B-Instruct) ->
concept_filter_pipeline (structural/monosemanticity/dedup, using the
locked 0.3585 threshold instead of the old synthetic-tuned 0.5).

Confirms the chain works BEFORE building unified_selection_pipeline.py
on top of it -- per the same evidence-first pattern used throughout
this session (verify before building further).

Usage:
    python run_end_to_end_integration_check.py [num_candidates] [num_documents]
    Example:
    python run_end_to_end_integration_check.py 50 40
"""

import sys

from real_backend import RealConceptBackend
from rescore_with_llm_auto_interp import rescore_candidates_with_llm_auto_interp
from llm_auto_interp_local_client import load_auto_interp_model
from monosemanticity_filter import MONOSEMANTICITY_SCORE_THRESHOLD

LAYER = 13
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
MODEL_ID = "google/gemma-3-1b-it"

# The locked, evidence-based threshold from DECISION_monosemanticity_threshold.md --
# NOT the old 0.5 synthetic-data default still hardcoded as
# monosemanticity_filter.py's own MONOSEMANTICITY_SCORE_THRESHOLD constant.
LOCKED_MONOSEMANTICITY_THRESHOLD = 0.3585


def main():
    num_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    num_documents = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    print(f"=== End-to-end integration check: {num_candidates} candidates, "
          f"{num_documents} documents ===")
    print(f"NOTE: monosemanticity_filter.py's own default threshold is still "
          f"{MONOSEMANTICITY_SCORE_THRESHOLD} (Month 1's synthetic-data value) -- "
          f"this script explicitly overrides it to the locked real value "
          f"{LOCKED_MONOSEMANTICITY_THRESHOLD}. The module's own default has NOT "
          f"been changed, flagged as a follow-up item, not silently fixed here.")
    print()

    backend = RealConceptBackend(
        model_id=MODEL_ID, sae_id=SAE_ID, sae_release=SAE_RELEASE, num_documents=num_documents,
    )

    print("Step 1: Loading Gemma 3 1B-IT + SAE, fetching raw candidates...")
    gemma_model, sae = backend._load_model_and_sae()
    candidates = backend.fetch_candidates(layer=LAYER, max_candidates=num_candidates)
    print(f"  Got {len(candidates)} raw candidates.")

    print("Step 2: Collecting full activation matrix for real rescoring...")
    activations, tokens = backend._collect_activations_for_all_features(gemma_model, sae, LAYER)
    print(f"  Activations shape: {activations.shape}")
    print()

    print("Step 3: Freeing Gemma 3 1B + SAE before loading the judge model "
          "(8GB card cannot hold both -- confirmed necessary earlier this session)...")
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

    print("Step 4: Loading Qwen2.5-7B-Instruct judge model...")
    judge_model, judge_tokenizer = load_auto_interp_model()
    print("  Loaded.")
    print()

    print("Step 5: Rescoring ALL raw candidates with real Qwen auto-interp "
          "(replacing heuristic scores)...")
    rescore_result = rescore_candidates_with_llm_auto_interp(
        candidates, activations, tokens, judge_model, judge_tokenizer, seed=42,
    )
    print(f"  {rescore_result.summary()}")
    print()

    if not rescore_result.scored:
        print("No SCORED candidates survived rescoring -- cannot proceed to the filter "
              "pipeline. Investigate the INSUFFICIENT_DATA/PARSE_FAILED counts above.")
        return

    print("Step 6: Running the full filter pipeline (structural -> monosemanticity "
          f"[threshold={LOCKED_MONOSEMANTICITY_THRESHOLD}] -> dedup) on the "
          "REAL-SCORED candidates...")

    # concept_filter_pipeline.run_concept_filter_pipeline() calls
    # apply_monosemanticity_filter() with ITS OWN default threshold
    # internally -- it does not currently accept an override parameter.
    # Rather than silently accept the wrong (0.5) threshold, or modify
    # concept_filter_pipeline.py blindly, calling the stages directly
    # here with the correct threshold, and flagging that
    # concept_filter_pipeline.py needs a threshold-override parameter
    # added as a real, separate follow-up.
    from structural_token_filter import apply_structural_token_filter
    from monosemanticity_filter import apply_monosemanticity_filter
    from deduplication import deduplicate_by_decoder_vector

    structural_result = apply_structural_token_filter(rescore_result.scored)
    print(f"  Structural filter: {len(structural_result.accepted)}/{len(rescore_result.scored)} passed.")

    monosemanticity_result = apply_monosemanticity_filter(
        structural_result.accepted, threshold=LOCKED_MONOSEMANTICITY_THRESHOLD,
    )
    print(f"  Monosemanticity filter (threshold={LOCKED_MONOSEMANTICITY_THRESHOLD}): "
          f"{len(monosemanticity_result.accepted)}/{len(structural_result.accepted)} passed.")

    dedup_result = deduplicate_by_decoder_vector(monosemanticity_result.accepted)
    print(f"  Dedup: {len(dedup_result.kept)}/{len(monosemanticity_result.accepted)} kept "
          f"({len(dedup_result.discarded)} discarded as duplicates).")
    print()

    print("=== Final result ===")
    print(f"  Raw candidates fetched: {len(candidates)}")
    print(f"  Rescored (SCORED / INSUFFICIENT_DATA / PARSE_FAILED): {rescore_result.summary()}")
    print(f"  After structural filter: {len(structural_result.accepted)}")
    print(f"  After monosemanticity filter: {len(monosemanticity_result.accepted)}")
    print(f"  FINAL (after dedup): {len(dedup_result.kept)}")
    print()
    print("Final surviving candidates:")
    for c in dedup_result.kept:
        print(f"  {c.feature_id}: score={c.auto_interp_score:.3f} -- {c.auto_interp_text[:80]!r}")
    print()
    print("Read this yourself: does the end-to-end pipeline behave sensibly? Does the")
    print("final surviving set look like real, distinct concepts, not duplicates or")
    print("artifacts? This confirms (or doesn't) that the full chain is ready to be")
    print("formalized into unified_selection_pipeline.py.")


if __name__ == "__main__":
    main()

"""
Confirmation run: calls the NEW, FORMALIZED unified_selection_pipeline.py
against the same n=50 batch already run via the ad-hoc
run_end_to_end_integration_check.py, to prove the formalization
produces the SAME result (18 final survivors, same feature IDs) --
not just "runs without crashing."

Usage:
    python confirm_unified_pipeline_matches_integration_check.py [num_candidates] [num_documents]
    Example:
    python confirm_unified_pipeline_matches_integration_check.py 50 40
"""

import sys

from real_backend import RealConceptBackend
from llm_auto_interp_local_client import load_auto_interp_model
from unified_selection_pipeline import run_unified_selection_pipeline

LAYER = 13
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
MODEL_ID = "google/gemma-3-1b-it"
LOCKED_MONOSEMANTICITY_THRESHOLD = 0.3585

# The 18 feature IDs the ad-hoc script found as final survivors, for
# direct comparison -- this is the actual expected result being checked
# against, not just "some result."
EXPECTED_FINAL_FEATURE_IDS = {
    "gemma-scope-2-1b-it-res/layer13/feat_3", "gemma-scope-2-1b-it-res/layer13/feat_5",
    "gemma-scope-2-1b-it-res/layer13/feat_6", "gemma-scope-2-1b-it-res/layer13/feat_9",
    "gemma-scope-2-1b-it-res/layer13/feat_14", "gemma-scope-2-1b-it-res/layer13/feat_15",
    "gemma-scope-2-1b-it-res/layer13/feat_16", "gemma-scope-2-1b-it-res/layer13/feat_25",
    "gemma-scope-2-1b-it-res/layer13/feat_26", "gemma-scope-2-1b-it-res/layer13/feat_27",
    "gemma-scope-2-1b-it-res/layer13/feat_34", "gemma-scope-2-1b-it-res/layer13/feat_36",
    "gemma-scope-2-1b-it-res/layer13/feat_37", "gemma-scope-2-1b-it-res/layer13/feat_38",
    "gemma-scope-2-1b-it-res/layer13/feat_39", "gemma-scope-2-1b-it-res/layer13/feat_41",
    "gemma-scope-2-1b-it-res/layer13/feat_47", "gemma-scope-2-1b-it-res/layer13/feat_49",
}


def main():
    num_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    num_documents = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    print(f"=== Confirming unified_selection_pipeline.py matches the ad-hoc "
          f"integration check ({num_candidates} candidates, {num_documents} docs) ===")
    print("NOTE: Qwen's generation is unseeded internally (only the held-out")
    print("SAMPLING is seeded), so exact scores may differ slightly run-to-run --")
    print("checking whether the FINAL SURVIVING FEATURE SET matches, not")
    print("bit-for-bit identical scores.")
    print()

    backend = RealConceptBackend(
        model_id=MODEL_ID, sae_id=SAE_ID, sae_release=SAE_RELEASE, num_documents=num_documents,
    )
    print("Loading Gemma 3 1B-IT + SAE, fetching candidates...")
    gemma_model, sae = backend._load_model_and_sae()
    candidates = backend.fetch_candidates(layer=LAYER, max_candidates=num_candidates)
    print(f"  Got {len(candidates)} candidates.")

    activations, tokens = backend._collect_activations_for_all_features(gemma_model, sae, LAYER)
    print(f"  Activations shape: {activations.shape}")

    import gc
    import torch
    backend._model = None
    backend._sae = None
    del gemma_model
    del sae
    gc.collect()
    torch.cuda.empty_cache()
    print("  Freed Gemma + SAE from GPU.")
    print()

    print("Loading Qwen judge model...")
    judge_model, judge_tokenizer = load_auto_interp_model()
    print("  Loaded.")
    print()

    print("Running the FORMALIZED unified pipeline (single function call)...")
    result = run_unified_selection_pipeline(
        candidates, activations, tokens, judge_model, judge_tokenizer,
        monosemanticity_threshold=LOCKED_MONOSEMANTICITY_THRESHOLD, seed=42,
    )
    print(f"  {result.summary()}")
    print()

    actual_final_ids = {c.feature_id for c in result.filter_result.final_kept}

    print("=== Comparison against the ad-hoc script's earlier result ===")
    print(f"  Ad-hoc result: {len(EXPECTED_FINAL_FEATURE_IDS)} final survivors")
    print(f"  Unified pipeline result: {len(actual_final_ids)} final survivors")

    only_in_expected = EXPECTED_FINAL_FEATURE_IDS - actual_final_ids
    only_in_actual = actual_final_ids - EXPECTED_FINAL_FEATURE_IDS

    if not only_in_expected and not only_in_actual:
        print("  MATCH: identical final feature sets. Formalization confirmed correct.")
    else:
        print("  DIFFERENCE FOUND -- read this carefully, don't assume it's a bug:")
        if only_in_expected:
            print(f"    In ad-hoc result but NOT in unified pipeline result: {only_in_expected}")
        if only_in_actual:
            print(f"    In unified pipeline result but NOT in ad-hoc result: {only_in_actual}")
        print("  Possible causes: Qwen's unseeded generation genuinely producing a")
        print("  different score for a borderline candidate near the 0.3585 threshold")
        print("  (expected, minor variance), OR a real bug in the formalization")
        print("  (unexpected, needs investigation). Check individual scores for any")
        print("  differing feature IDs before concluding either way.")


if __name__ == "__main__":
    main()

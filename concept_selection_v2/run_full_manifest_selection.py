"""
Part A real deliverable: the actual concept manifest selection run.

Fetches raw candidates at target scale, rescores with real Qwen
auto-interp, runs the full filter pipeline (structural ->
monosemanticity [locked threshold 0.3585] -> dedup), stratifies the
survivors, and FREEZES the result as a real, timestamped manifest file
via manifest_freeze.py -- the actual Part A deliverable, not just an
in-memory result.

Target size reasoning: the n=50 pilot batch yielded 18/50 final
survivors (~36% pass rate). To reach the 40-50 concept target,
starting from ~125 raw candidates is a reasonable first attempt
(36% x 125 =~ 45) -- but this ratio is based on ONE 40-document corpus
sample and may not hold exactly at a larger scale. This run's own
result should be read as evidence for or against that estimate, not
assumed correct in advance.

Usage:
    python run_full_manifest_selection.py [num_candidates] [num_documents]
    Example:
    python run_full_manifest_selection.py 125 40
"""

import sys

from real_backend import RealConceptBackend
from llm_auto_interp_local_client import load_auto_interp_model
from unified_selection_pipeline import run_unified_selection_pipeline
from manifest_freeze import freeze_manifest_to_file

LAYER = 13
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
MODEL_ID = "google/gemma-3-1b-it"
LOCKED_MONOSEMANTICITY_THRESHOLD = 0.3585
MANIFEST_OUTPUT_PATH = "manifests/concept_manifest_v1.json"


def main():
    num_candidates = int(sys.argv[1]) if len(sys.argv) > 1 else 125
    num_documents = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    print(f"=== Part A real manifest selection run: {num_candidates} candidates, "
          f"{num_documents} documents ===")
    print(f"Target: 40-50 final concepts. Estimate based on n=50 pilot's 36% pass rate "
          f"(not guaranteed to hold at this scale -- this run tests that estimate).")
    print()

    backend = RealConceptBackend(
        model_id=MODEL_ID, sae_id=SAE_ID, sae_release=SAE_RELEASE, num_documents=num_documents,
    )

    print("Step 1: Loading Gemma 3 1B-IT + SAE, fetching raw candidates...")
    gemma_model, sae = backend._load_model_and_sae()
    candidates = backend.fetch_candidates(layer=LAYER, max_candidates=num_candidates)
    print(f"  Got {len(candidates)} raw candidates.")

    print("Step 2: Collecting full activation matrix...")
    activations, tokens = backend._collect_activations_for_all_features(gemma_model, sae, LAYER)
    print(f"  Activations shape: {activations.shape}")
    print()

    print("Step 3: Freeing Gemma 3 1B + SAE before loading the judge model...")
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

    print(f"Step 5: Running the unified pipeline (rescore -> filter "
          f"[threshold={LOCKED_MONOSEMANTICITY_THRESHOLD}] -> stratify) on all "
          f"{len(candidates)} candidates. This will take a while -- each candidate "
          f"needs 1-2 real Qwen generation calls...")
    result = run_unified_selection_pipeline(
        candidates, activations, tokens, judge_model, judge_tokenizer,
        monosemanticity_threshold=LOCKED_MONOSEMANTICITY_THRESHOLD, seed=42,
    )
    print(f"  {result.summary()}")
    print()

    final_candidates = result.filter_result.final_kept
    final_count = len(final_candidates)

    print("=== Result vs. target ===")
    print(f"  Final candidate count: {final_count}")
    if 40 <= final_count <= 50:
        print(f"  WITHIN TARGET RANGE (40-50). Proceeding to freeze the manifest.")
    elif final_count < 40:
        print(f"  BELOW TARGET (need 40-50, got {final_count}). The pass-rate estimate "
              f"was optimistic for this run, or corpus/candidate composition differs. "
              f"Consider re-running with more raw candidates before freezing, or "
              f"freeze this as a smaller v1 manifest and document the shortfall "
              f"explicitly -- your call, not decided here.")
    else:
        print(f"  ABOVE TARGET (need 40-50, got {final_count}). Consider whether to "
              f"freeze all {final_count} or trim to the target range -- your call, "
              f"not decided here. Freezing as-is below; trim manually if desired "
              f"before the NEXT freeze (this one is immutable once written).")
    print()

    print(f"Step 6: Freezing manifest to {MANIFEST_OUTPUT_PATH}...")
    notes = (
        f"Real Part A manifest, layer {LAYER}, {SAE_RELEASE}. "
        f"Raw candidates: {len(candidates)} (from {num_documents}-doc corpus). "
        f"Monosemanticity threshold: {LOCKED_MONOSEMANTICITY_THRESHOLD} "
        f"(median of n=36 pilot scores, see DECISION_monosemanticity_threshold.md -- "
        f"PENDING RE-VALIDATION at this scale, per that decision's own trigger). "
        f"Rescore summary: {result.rescore_result.summary()}. "
        f"Tier counts: {result.summary()['final_tier_counts']}."
    )
    try:
        manifest = freeze_manifest_to_file(final_candidates, MANIFEST_OUTPUT_PATH, notes=notes)
        print(f"  FROZEN. Manifest hash: {manifest.metadata.manifest_hash}")
        print(f"  Frozen at: {manifest.metadata.frozen_at_utc}")
        print(f"  Concept count: {manifest.metadata.concept_count}")
    except Exception as e:
        print(f"  FREEZE FAILED: {type(e).__name__}: {e}")
        print("  (If this is ManifestAlreadyFrozenError, a manifest already exists at "
              "this path -- pass allow_overwrite_with_reason explicitly if this is a "
              "deliberate new version, don't just retry blindly.)")
        return

    print()
    print("=== Final concepts ===")
    for c in final_candidates:
        print(f"  {c.feature_id}: score={c.auto_interp_score:.3f} -- {c.auto_interp_text[:80]!r}")


if __name__ == "__main__":
    main()

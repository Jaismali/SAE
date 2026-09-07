"""
Diagnostic 8: reference corpus size convergence check.

NOT a unit test, a manual sanity script -- run once, by hand. Follows
directly from Diagnostic 7, which confirmed the 89% monosemanticity
rejection rate was partly explained by corpus under-sampling: a single
document (doc[0], out of only 40) dominated multiple unrelated
features' top-10 token lists, purely from having too few total tokens
(1,280) in the sample.

This script answers the follow-up question EMPIRICALLY rather than by
guessing a bigger number: for a FIXED set of features, how much does
their top-10 activating-token list change as the corpus grows through
several nested sizes? Turnover between consecutive sizes is the
convergence signal -- when it drops to a low, stable level rather than
continuing to shift substantially, that's evidence of having crossed
out of the sparse-sampling-artifact regime.

Nesting requirement: real_backend.py's _stream_reference_texts() uses
datasets.load_dataset(..., streaming=True) with NO shuffle, just plain
enumerate() taking the first N documents in a fixed stream order. This
means a larger num_documents run already contains the smaller run's
documents as an exact prefix -- confirmed by inspection, not assumed.
No changes to real_backend.py were needed to support this comparison.

Cost note: this runs the model/SAE forward pass at THREE corpus sizes
(model and SAE are loaded only ONCE and reused across sizes to avoid
redundant load time, but the forward-pass cost still scales with
corpus size). The largest size (400 docs) is meaningfully more compute
than the smoke tests run so far -- if this is slow or hits memory
issues, that itself is useful information for deciding the real
corpus size, not just an inconvenience.

Usage:
    python diagnostic_corpus_convergence_check.py
"""

from real_backend import RealConceptBackend
from activation_statistics import get_top_activating_tokens

MODEL_ID = "google/gemma-3-1b-it"
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
LAYER = 13
MAX_TOKENS_PER_DOC = 32  # held fixed across all sizes, per the researcher's instruction

# Nested, increasing corpus sizes. Each larger size's documents are a
# strict superset (same fixed stream order, no shuffling) of the
# smaller size's documents -- required for a clean convergence
# comparison, confirmed true by inspection above.
CORPUS_SIZES_TO_TEST = [40, 150, 400]

# A small, fixed set of feature indices to track across all three
# sizes. Using the same 10 feature indices Diagnostic 5/6 already
# looked at, so this connects directly to the earlier finding rather
# than introducing a fresh, uncompared set of features.
FIXED_FEATURE_INDICES = list(range(10))

TOP_K = 10


def _compute_top_tokens_at_size(backend: RealConceptBackend, model, sae, num_documents: int):
    """Runs activation extraction at a specific corpus size and returns
    {feature_index: top_10_tokens} for the fixed feature set.
    """
    backend.num_documents = num_documents
    activations, tokens = backend._collect_activations_for_all_features(model, sae, LAYER)

    result = {}
    for feature_idx in FIXED_FEATURE_INDICES:
        feature_acts = activations[:, feature_idx]
        result[feature_idx] = get_top_activating_tokens(feature_acts, tokens, k=TOP_K)
    return result


def _turnover(list_a, list_b) -> float:
    """Fraction of the top-10 list that CHANGED between two runs.
    0.0 = identical lists, 1.0 = completely different lists.
    """
    set_a, set_b = set(list_a), set(list_b)
    overlap = len(set_a & set_b)
    return 1.0 - (overlap / max(len(set_a), 1))


def main():
    print("=== Diagnostic 8: reference corpus size convergence check ===")
    print(f"Corpus sizes tested (nested, same fixed stream order): {CORPUS_SIZES_TO_TEST}")
    print(f"Fixed feature indices tracked: {FIXED_FEATURE_INDICES}")
    print(f"max_tokens_per_doc held fixed at: {MAX_TOKENS_PER_DOC}")
    print()

    backend = RealConceptBackend(
        model_id=MODEL_ID,
        sae_id=SAE_ID,
        sae_release=SAE_RELEASE,
        num_documents=CORPUS_SIZES_TO_TEST[0],
        max_tokens_per_doc=MAX_TOKENS_PER_DOC,
        device="cuda",
    )

    print("Loading model and SAE once (reused across all corpus sizes)...")
    model, sae = backend._load_model_and_sae()
    print("  Loaded.")
    print()

    results_by_size = {}
    for size in CORPUS_SIZES_TO_TEST:
        print(f"Running extraction at corpus size {size} documents...")
        results_by_size[size] = _compute_top_tokens_at_size(backend, model, sae, size)
        print(f"  Done. Sample (feature 0): {results_by_size[size][0]}")
    print()

    print("=== Turnover between consecutive corpus sizes ===")
    print("(0% = top-10 list identical to the smaller size; 100% = completely different)")
    print()
    all_turnovers_by_step = []
    for i in range(1, len(CORPUS_SIZES_TO_TEST)):
        smaller_size = CORPUS_SIZES_TO_TEST[i - 1]
        larger_size = CORPUS_SIZES_TO_TEST[i]
        print(f"--- {smaller_size} -> {larger_size} documents ---")
        turnovers = []
        for feature_idx in FIXED_FEATURE_INDICES:
            list_smaller = results_by_size[smaller_size][feature_idx]
            list_larger = results_by_size[larger_size][feature_idx]
            turnover = _turnover(list_smaller, list_larger)
            turnovers.append(turnover)
            print(f"  feature {feature_idx}: turnover={turnover:.0%}  "
                  f"({smaller_size}-doc list: {list_smaller[:5]}... vs "
                  f"{larger_size}-doc list: {list_larger[:5]}...)")
        avg_turnover = sum(turnovers) / len(turnovers)
        all_turnovers_by_step.append((smaller_size, larger_size, avg_turnover))
        print(f"  AVERAGE turnover across {len(FIXED_FEATURE_INDICES)} tracked features: {avg_turnover:.1%}")
        print()

    print("=== Summary ===")
    for smaller_size, larger_size, avg_turnover in all_turnovers_by_step:
        flag = "STABILIZING" if avg_turnover < 0.20 else "STILL SHIFTING"
        print(f"  {smaller_size} -> {larger_size} docs: {avg_turnover:.1%} average turnover [{flag}]")
    print()
    print("Read this yourself: if turnover keeps dropping toward <20% as size increases,")
    print("that's empirical evidence of approaching a stable regime. If turnover stays")
    print("high even at 400 documents, that means MORE than 400 is likely needed --")
    print("report the actual numbers back before any corpus size gets decided, since")
    print("this needs explicit cost/benefit sign-off (400+ documents x 40-50 concepts")
    print("is a real, sizeable compute cost) same as every other threshold this session.")


if __name__ == "__main__":
    main()

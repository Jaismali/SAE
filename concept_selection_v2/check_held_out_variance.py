"""
Cheap diagnostic for the pilot batch's suspicious 45/48 exactly-0.000
scores: checks whether the TRUE activation values in the held-out
sample have near-zero variance (which would trigger
pearson_correlation()'s zero-variance safeguard regardless of how good
the simulator's guess is) -- WITHOUT needing to re-run Qwen, since this
only needs the already-fast Gemma+SAE activation extraction.

Usage:
    python check_held_out_variance.py
"""

from real_backend import RealConceptBackend
from held_out_context_sampling import sample_held_out_contexts

LAYER = 13
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
MODEL_ID = "google/gemma-3-1b-it"

# A mix of exactly-0.000-scoring candidates and the two high-scoring
# ones, for direct comparison.
FEATURE_INDICES_TO_CHECK = [0, 1, 2, 25, 45]  # feat_0, feat_1, feat_2 (0.000), feat_25, feat_45 (high)


def main():
    backend = RealConceptBackend(
        model_id=MODEL_ID, sae_id=SAE_ID, sae_release=SAE_RELEASE, num_documents=40
    )
    print("Loading model + SAE, collecting activations (same 40-doc corpus as the pilot run)...")
    model, sae = backend._load_model_and_sae()
    activations, tokens = backend._collect_activations_for_all_features(model, sae, LAYER)
    print(f"  Activations shape: {activations.shape}")
    print()

    for feature_idx in FEATURE_INDICES_TO_CHECK:
        feature_activations = activations[:, feature_idx]
        top_indices = sorted(
            range(len(tokens)), key=lambda idx: feature_activations[idx], reverse=True
        )[:10]

        held_out = sample_held_out_contexts(
            feature_activations, tokens, excluded_indices=top_indices,
            num_samples=10, window_size=5, seed=42,
        )
        true_activations = [v for _, v in held_out]
        variance = (
            sum((v - sum(true_activations) / len(true_activations)) ** 2 for v in true_activations)
            / len(true_activations)
            if true_activations else 0.0
        )

        print(f"feat_{feature_idx}: held-out true_activations = "
              f"{[round(v, 3) for v in true_activations]}")
        print(f"  variance: {variance:.6f} {'<<< ZERO/NEAR-ZERO VARIANCE' if variance < 0.01 else ''}")
        print(f"  For reference, top-10 activation values (excluded from held-out): "
              f"{[round(float(feature_activations[i]), 2) for i in top_indices]}")
        print()

    print("=== Interpretation ===")
    print("If the flagged (near-zero-variance) features are the ones that scored exactly")
    print("0.000 in the pilot run, and the top-10 values show these ARE sparse features")
    print("(most held-out positions genuinely non-activating), that confirms the")
    print("hypothesis: pure random held-out sampling produces degenerate all-near-zero")
    print("true activations for sparse features, regardless of simulator quality --")
    print("the sampling STRATEGY needs fixing (top-and-random, not pure random), not")
    print("the model or the scoring code.")


if __name__ == "__main__":
    main()

"""
Smoke test for RealConceptBackend -- NOT a unit test, a manual sanity
script. Run this once, by hand, before attempting a full fetch_candidates()
call across all features. Loads the real model and SAE, runs a modest
number of documents through, and prints shapes, sample values, and two
diagnostic checks so any API mismatch or pipeline-level artifact
surfaces cheaply, before committing to the full 2,000-document run.

This does not assert anything automatically -- read the printed output
yourself and sanity-check it before proceeding. Report back what it
actually printed, not just "it ran."

Iteration 2 (30-50 doc checkpoint): bumped from the initial 2-document/
5-candidate smoke test after that run raised two open questions:
  1. Feature 4 showed an anomalously large magnitude (~25-30x the
     others), dominated by <bos> tokens -- plausible attention-sink
     effect, not yet confirmed at scale.
  2. The same handful of tokens ('\\r', '>', 'name', '1', '<bos>')
     appeared across nearly all 5 features' top-token lists -- plausible
     sample-size artifact (only 64 tokens total), not yet confirmed.
This version adds explicit diagnostics for both, rather than relying on
eyeballing the raw candidate dump.

Also bumped MAX_CANDIDATES from 5 to 20 (not requested, but reasoned):
with only 5 features we can't distinguish "a few outlier features" from
"a widespread pattern" for the structural-token-dominance question --
flagging this change explicitly rather than making it silently.

Usage:
    python smoke_test_real_backend.py
"""

from collections import Counter

from real_backend import RealConceptBackend, REFERENCE_CORPUS_DATASET
from concept_filter_pipeline import run_concept_filter_pipeline

MODEL_ID = "google/gemma-3-1b-it"
SAE_ID = "layer_13_width_16k_l0_medium"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
LAYER = 13
SMOKE_TEST_NUM_DOCUMENTS = 40  # bumped from 2, per the 30-50 doc checkpoint
SMOKE_TEST_MAX_TOKENS_PER_DOC = 32
SMOKE_TEST_MAX_CANDIDATES = 20  # bumped from 5, to distinguish "few" vs "widespread"

# A token counts as "structural" if it's a special/control token or is
# made up entirely of punctuation/whitespace once stripped. This is a
# deliberately narrow, mechanical definition for this diagnostic only --
# it is NOT the pre-filter rule itself. Per the researcher's explicit
# instruction, no filtering rule is being built yet; this just measures
# how often the pattern shows up, so a real filter can be designed later
# with actual evidence behind it.
SPECIAL_TOKENS = {"<bos>", "<eos>", "<pad>", "<unk>"}


def _is_structural_token(token: str) -> bool:
    stripped = token.strip()
    if stripped in SPECIAL_TOKENS:
        return True
    if stripped == "":
        return True  # whitespace-only, e.g. '\r', '\n'
    return all(not ch.isalnum() for ch in stripped)  # pure punctuation


def _get_structural_dominated_candidates(candidates) -> list:
    """Returns the actual list of candidates whose top-activating-token
    lists are majority (>50%) structural/special tokens -- not just a
    count. Needed so the magnitude-correlation check (added after the
    n=20 checkpoint) can inspect all of them, not only the ones that
    happened to also be >10x-magnitude outliers.
    """
    dominated = []
    for c in candidates:
        if not c.top_activating_tokens:
            continue
        structural_count = sum(
            1 for t in c.top_activating_tokens if _is_structural_token(t)
        )
        fraction = structural_count / len(c.top_activating_tokens)
        if fraction > 0.5:
            dominated.append(c)
    return dominated


def _report_structural_dominance(candidates) -> None:
    """What fraction of candidates have top-activating-token lists
    dominated (majority, i.e. >50%) by structural/special tokens."""
    dominated = _get_structural_dominated_candidates(candidates)
    overall_fraction = len(dominated) / len(candidates) if candidates else 0.0
    print(
        f"  Structural-token-dominated candidates: {len(dominated)}/{len(candidates)} "
        f"({overall_fraction:.1%})"
    )
    print(
        "  (This is a diagnostic count only -- no filtering rule has been "
        "built or applied based on this number.)"
    )


def _report_top_token_overlap(candidates) -> None:
    """Distinguishes 'sample-size artifact' from 'structural extraction
    issue' using two angles:
      (a) exact-set overlap: how many candidates share an IDENTICAL
          ordered top-5 list (a strong signal if high).
      (b) vocabulary overlap: across all candidates' top-5 slots
          combined, how concentrated is the token vocabulary -- this
          catches the case the researcher actually raised (the same
          handful of tokens reappearing across DIFFERENT candidates'
          lists, even when no two lists are perfectly identical).
    """
    top5_lists = [c.top_activating_tokens[:5] for c in candidates if c.top_activating_tokens]

    # (a) exact-set overlap
    top5_sets = [tuple(lst) for lst in top5_lists]
    distinct_sets = Counter(top5_sets)
    most_common_set, most_common_count = distinct_sets.most_common(1)[0] if distinct_sets else (None, 0)
    print(f"  Distinct EXACT top-5 sets across {len(top5_sets)} candidates: {len(distinct_sets)}")
    if most_common_count > 1:
        print(
            f"  Most repeated exact top-5 set ({most_common_count} candidates share it): "
            f"{most_common_set}"
        )

    # (b) vocabulary overlap -- flattened token pool across all slots
    all_tokens_flat = [tok for lst in top5_lists for tok in lst]
    total_slots = len(all_tokens_flat)
    token_counts = Counter(all_tokens_flat)
    unique_token_count = len(token_counts)
    print(
        f"  Vocabulary check: {unique_token_count} unique tokens fill "
        f"{total_slots} total top-5 slots across {len(top5_lists)} candidates."
    )
    if token_counts:
        top_shared = token_counts.most_common(5)
        print(f"  Most frequently repeated tokens across DIFFERENT candidates' lists: {top_shared}")
        vocab_concentration = unique_token_count / total_slots if total_slots else 1.0
        if vocab_concentration < 0.4:
            print(
                f"  WARNING: only {vocab_concentration:.1%} of top-5 slots are filled by "
                "distinct tokens -- a small shared vocabulary is dominating most "
                "candidates' top-activating lists. At 30-50 documents this is no longer "
                "well-explained by sample size alone; worth re-checking the extraction "
                "code's indexing (e.g. whether tokens are being correctly re-aligned per "
                "document) before scaling to the full run."
            )
        else:
            print(
                f"  Vocabulary concentration ({vocab_concentration:.1%} unique) looks "
                "reasonable -- consistent with the sample-size-artifact explanation."
            )


def main():
    print("=== RealConceptBackend smoke test (iteration 2: 30-50 doc checkpoint) ===")
    print(f"Model: {MODEL_ID}")
    print(f"SAE release: {SAE_RELEASE}, sae_id: {SAE_ID}, layer: {LAYER}")
    print(f"Reference corpus: {REFERENCE_CORPUS_DATASET}")
    print(f"Documents: {SMOKE_TEST_NUM_DOCUMENTS}, "
          f"max tokens/doc: {SMOKE_TEST_MAX_TOKENS_PER_DOC}, "
          f"candidates: {SMOKE_TEST_MAX_CANDIDATES}")
    print()

    backend = RealConceptBackend(
        model_id=MODEL_ID,
        sae_id=SAE_ID,
        sae_release=SAE_RELEASE,
        num_documents=SMOKE_TEST_NUM_DOCUMENTS,
        max_tokens_per_doc=SMOKE_TEST_MAX_TOKENS_PER_DOC,
        device="cuda",
    )

    print("Loading model and SAE (this is the first real checkpoint -- "
          "if the sae_id or release string is wrong, this is where it fails)...")
    model, sae = backend._load_model_and_sae()
    print(f"  Model loaded. d_model (from SAE decoder shape): {sae.W_dec.shape}")
    print(f"  SAE decoder matrix shape (num_features, d_model): {sae.W_dec.shape}")
    print()

    print("Running activation extraction over "
          f"{SMOKE_TEST_NUM_DOCUMENTS} documents...")
    activations, tokens = backend._collect_activations_for_all_features(
        model, sae, LAYER
    )
    print(f"  Activations shape (num_tokens_total, num_features): {activations.shape}")
    print(f"  Total tokens collected: {len(tokens)}")
    print(f"  First 10 tokens: {tokens[:10]}")
    print()

    print(f"Fetching {SMOKE_TEST_MAX_CANDIDATES} candidates from full pipeline...")
    candidates = backend.fetch_candidates(layer=LAYER, max_candidates=SMOKE_TEST_MAX_CANDIDATES)
    print(f"  Got {len(candidates)} candidates.")
    for c in candidates:
        print(f"  --- {c.feature_id} ---")
        print(f"      activation_frequency: {c.activation_frequency:.4f}")
        print(f"      mean_activation_magnitude: {c.mean_activation_magnitude:.4f}")
        print(f"      top_activating_tokens: {c.top_activating_tokens[:5]}")
        print(f"      auto_interp_score: {c.auto_interp_score:.4f}")
        print(f"      auto_interp_text: {c.auto_interp_text}")
        print(f"      decoder_vector length: {len(c.decoder_vector)}")
        print(f"      contract violations: {c.validate()}")
    print()

    print("=== Diagnostic 1: structural/special-token dominance ===")
    _report_structural_dominance(candidates)
    print()

    print("=== Diagnostic 2: top-5 token overlap across features ===")
    _report_top_token_overlap(candidates)
    print()

    magnitudes = [c.mean_activation_magnitude for c in candidates if c.mean_activation_magnitude > 0]
    if magnitudes:
        max_mag = max(magnitudes)
        median_mag = sorted(magnitudes)[len(magnitudes) // 2]
        print("=== Diagnostic 3: magnitude outlier check (feature-4-style pattern) ===")
        print(f"  Median nonzero mean_activation_magnitude: {median_mag:.2f}")
        print(f"  Max nonzero mean_activation_magnitude: {max_mag:.2f}")
        if median_mag > 0 and max_mag / median_mag > 10:
            outliers = [c for c in candidates if c.mean_activation_magnitude > 10 * median_mag]
            print(
                f"  {len(outliers)} candidate(s) have magnitude >10x the median -- "
                "checking whether they're <bos>/structural-token-dominated:"
            )
            for c in outliers:
                structural = sum(1 for t in c.top_activating_tokens if _is_structural_token(t))
                print(
                    f"    {c.feature_id}: magnitude={c.mean_activation_magnitude:.2f}, "
                    f"structural tokens in top-5: {structural}/{len(c.top_activating_tokens)}"
                )
        else:
            print("  No magnitude outliers >10x the median at this sample size.")
        print()

        print("=== Diagnostic 4: magnitude ratio for ALL structural-dominated candidates ===")
        print("  (Not just the >10x outliers -- this is the evidence needed before")
        print("  locking any specific threshold, per researcher's explicit request.)")
        structural_dominated = _get_structural_dominated_candidates(candidates)
        if not structural_dominated:
            print("  No structural-dominated candidates in this sample.")
        else:
            ratios = []
            for c in structural_dominated:
                ratio = c.mean_activation_magnitude / median_mag if median_mag > 0 else float("nan")
                structural_count = sum(1 for t in c.top_activating_tokens if _is_structural_token(t))
                structural_fraction = (
                    structural_count / len(c.top_activating_tokens)
                    if c.top_activating_tokens
                    else float("nan")
                )
                ratios.append((c.feature_id, c.mean_activation_magnitude, ratio, structural_fraction))
            ratios.sort(key=lambda x: x[2])
            print(f"  {len(ratios)} structural-dominated candidates, magnitude ratio to median, ascending:")
            for feature_id, magnitude, ratio, structural_fraction in ratios:
                print(
                    f"    {feature_id}: magnitude={magnitude:.2f}, ratio={ratio:.2f}x median, "
                    f"structural_fraction={structural_fraction:.1%}"
                )
            print()
            print("  Read this list yourself: do the ratios cluster into two separated")
            print("  groups (e.g. several near ~1x, a couple far above 10x, a real gap")
            print("  between them), or are they scattered continuously across the range?")
            print("  A real gap supports the conjunction rule as designed. A continuum")
            print("  means the magnitude cutoff is an arbitrary cut through one")
            print("  phenomenon, not evidence of two distinct populations.")
            print("  Also check structural_fraction the same way: does it separate the")
            print("  same way magnitude does, or does it sit >50% for all five without a")
            print("  clear break -- which would mean the fraction threshold isn't doing")
            print("  independent work and the conjunction may reduce to magnitude alone.")
        print()

    print("=== Diagnostic 5: real candidates through the wired filter pipeline ===")
    print("  (Previously only tested against synthetic data -- this is the first")
    print("  time the ACTUAL locked pipeline order runs on real GPU-derived candidates.)")
    pipeline_result = run_concept_filter_pipeline(candidates)
    print(f"  Input: {len(candidates)} candidates")
    print(
        f"  After structural-token filter: {len(pipeline_result.structural_filter_result.accepted)} "
        f"kept, {len(pipeline_result.structural_filter_result.rejected)} rejected"
    )
    if pipeline_result.structural_filter_result.rejected:
        print("    Rejected by structural filter:")
        for feature_id, reason in pipeline_result.structural_filter_result.rejection_reasons.items():
            print(f"      {feature_id}: {reason}")
    print(
        f"  After monosemanticity filter: {len(pipeline_result.monosemanticity_result.accepted)} "
        f"kept, {len(pipeline_result.monosemanticity_result.rejected)} rejected"
    )
    if pipeline_result.monosemanticity_result.rejected:
        rejected_ids = list(pipeline_result.monosemanticity_result.rejection_reasons.keys())
        print(f"    Rejected by monosemanticity filter: {rejected_ids}")
        print(
            "    NOTE: at this sample size, check whether monosemanticity rejected "
            "candidates for a substantive reason or because the current heuristic "
            "auto-interp score doesn't separate real signal well yet -- this "
            "threshold (0.5) is itself still pending re-validation against real "
            "score distributions, per Part A's own requirement."
        )
    print(
        f"  After dedup: {len(pipeline_result.dedup_result.kept)} kept, "
        f"{len(pipeline_result.dedup_result.discarded)} discarded"
    )
    if pipeline_result.dedup_result.discarded:
        print("    Discarded by dedup:")
        for feature_id, reason in pipeline_result.dedup_result.discard_reasons.items():
            print(f"      {feature_id}: {reason}")
    print(f"  FINAL: {len(pipeline_result.final_kept)}/{len(candidates)} candidates survived all three filters.")
    print()

    print("=== Diagnostic 6: auto_interp_score distribution + manual spot-check ===")
    print("  (Added after the 89% monosemanticity rejection rate raised a real question:")
    print("  is the 0.5 threshold miscalibrated, or is the heuristic itself structurally")
    print("  flawed? This diagnostic gathers evidence for that question -- it does not")
    print("  answer it by itself.)")
    structural_survivors = pipeline_result.structural_filter_result.accepted
    scored = sorted(
        [(c.feature_id, c.auto_interp_score) for c in structural_survivors],
        key=lambda x: x[1],
    )
    print(f"  auto_interp_score across {len(scored)} structural-survivors, ascending:")
    for feature_id, score in scored:
        print(f"    {feature_id}: {score:.3f}")
    print()

    rejected_candidates = pipeline_result.monosemanticity_result.rejected
    spot_check_n = min(6, len(rejected_candidates))
    print(f"  Manual spot-check: top-10 tokens for {spot_check_n} of the "
          f"{len(rejected_candidates)} monosemanticity-rejected candidates")
    print("  (Read these yourself: do they show a diverse-but-coherent theme")
    print("  -- e.g. country names, motion verbs -- or genuinely incoherent,")
    print("  unrelated words? This is the direct test of the hypothesis, not")
    print("  just an inference from the score shape above.)")
    for c in rejected_candidates[:spot_check_n]:
        print(f"    {c.feature_id} (score={c.auto_interp_score:.3f}): {c.top_activating_tokens}")
    print()

    print("=== Smoke test complete. Read the output above before trusting it. ===")
    print("No filtering rules have been built or applied based on any of the "
          "above diagnostics -- they are for your review only.")


if __name__ == "__main__":
    main()


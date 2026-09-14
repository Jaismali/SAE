"""
Resolves both open questions from check_bos_cluster_magnitude.py at
once: pulls the REAL structural_fraction (via the actual tested
compute_structural_fraction() function, not reimplemented) for all 45
manifest candidates, sorted -- checking (1) whether a genuine gap
exists between the 9 <bos>-cluster candidates (100% fraction) and the
36 "legitimate" ones, or whether it's a smooth continuum (same
gap-vs-gradient distinction already applied to the monosemanticity
threshold), and (2) for any legitimate candidate sitting at high
fraction, what token actually dominates it -- since a rule based on
"high concentration on ANY token" is different from a rule based on
"high concentration specifically on <bos>/structural tokens."

Pure JSON + existing tested code -- no GPU needed.

Usage:
    python check_structural_fraction_distribution.py
"""

import json
import sys

from structural_token_filter import compute_structural_fraction

MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else "manifests/concept_manifest_v1.json"
HIGH_FRACTION_INSPECTION_THRESHOLD = 0.80


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    all_concepts = manifest["concepts"]

    scored = []
    for c in all_concepts:
        fraction = compute_structural_fraction(c["top_activating_tokens"])
        scored.append((c["feature_id"], fraction, c["top_activating_tokens"]))

    scored.sort(key=lambda x: x[1])

    print(f"=== structural_fraction for all {len(scored)} manifest candidates, sorted ===")
    for feature_id, fraction, tokens in scored:
        print(f"  {feature_id}: {fraction:.1%}")
    print()

    print("=== Consecutive gaps in the sorted fraction list ===")
    gaps = []
    for i in range(1, len(scored)):
        gap = scored[i][1] - scored[i - 1][1]
        gaps.append((gap, scored[i - 1][0], scored[i][0], scored[i - 1][1], scored[i][1]))
    gaps.sort(reverse=True)
    print("  Largest gaps (top 5):")
    for gap, id_a, id_b, frac_a, frac_b in gaps[:5]:
        print(f"    {gap:.1%} gap: {frac_a:.1%} ({id_a}) -> {frac_b:.1%} ({id_b})")
    print()

    print(f"=== High-fraction (>{HIGH_FRACTION_INSPECTION_THRESHOLD:.0%}) candidates: "
          f"what token dominates each? ===")
    high_fraction = [s for s in scored if s[1] > HIGH_FRACTION_INSPECTION_THRESHOLD]
    print(f"  {len(high_fraction)} candidates exceed {HIGH_FRACTION_INSPECTION_THRESHOLD:.0%} "
          f"structural fraction:")
    for feature_id, fraction, tokens in high_fraction:
        # Report the actual token(s) present, not just the fraction --
        # this is the check for "dominated by <bos> specifically" vs
        # "dominated by some OTHER structural token (punctuation, etc)."
        unique_tokens = sorted(set(tokens), key=tokens.count, reverse=True)
        print(f"    {feature_id}: fraction={fraction:.1%}, tokens={tokens}, "
              f"unique (most common first)={unique_tokens}")
    print()

    print("Read this yourself against BOTH questions:")
    print("  1. Is there a real gap near the top of the distribution (between the")
    print("     highest legitimate-looking fraction and the <bos> cluster's 100%),")
    print("     or is it a smooth continuum? Check the 'largest gaps' list above --")
    print("     if the biggest gap near the top is small, that's a continuum, not a")
    print("     real boundary, same distinction already applied to the")
    print("     monosemanticity threshold.")
    print("  2. For any non-<bos> candidate at high fraction, is it dominated by a")
    print("     DIFFERENT structural token (e.g. punctuation), or does <bos> uniquely")
    print("     characterize the problem cluster? This determines whether the fix")
    print("     should be a general fraction-based rule or a <bos>-specific one.")


if __name__ == "__main__":
    main()

"""
Follow-up check: confirms whether the 9 pure-<bos> candidates found in
check_sentence_start_cluster_diversity.py have NORMAL (not outlier)
mean_activation_magnitude -- which would explain exactly why the
structural filter's conjunction rule (>5x magnitude AND >50%
structural fraction) did not exclude them, since it's designed to
require BOTH conditions (per feat_3's original evidence that
high-structural-fraction-but-normal-magnitude candidates should NOT be
excluded).

Pure JSON analysis of the already-frozen manifest -- no GPU needed.

Usage:
    python check_bos_cluster_magnitude.py
"""

import json
import statistics

MANIFEST_PATH = "manifests/concept_manifest_v1.json"


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    all_concepts = manifest["concepts"]

    bos_cluster = [
        c for c in all_concepts
        if c["top_activating_tokens"] == ["<bos>"] * 10
    ]
    non_bos = [c for c in all_concepts if c not in bos_cluster]

    print(f"Pure <bos> cluster size: {len(bos_cluster)}")
    print(f"Rest of manifest: {len(non_bos)}")
    print()

    bos_magnitudes = [c["mean_activation_magnitude"] for c in bos_cluster]
    non_bos_magnitudes = [c["mean_activation_magnitude"] for c in non_bos]

    print("=== Magnitude comparison (using the FINAL manifest's own 45 candidates ===")
    print("=== as the reference population -- NOT the original 125-candidate batch ===")
    print("=== median, which isn't stored anywhere post-hoc; flagged as an ===")
    print("=== approximation, not the literal value the structural filter used) ===")
    print()
    print(f"  <bos> cluster magnitudes: {[round(m, 2) for m in bos_magnitudes]}")
    print(f"  <bos> cluster median: {statistics.median(bos_magnitudes):.2f}")
    print()
    print(f"  Rest-of-manifest median: {statistics.median(non_bos_magnitudes):.2f}")
    print(f"  Rest-of-manifest range: {min(non_bos_magnitudes):.2f} - {max(non_bos_magnitudes):.2f}")
    print()

    combined_median = statistics.median([c["mean_activation_magnitude"] for c in all_concepts])
    print(f"  Whole-manifest (all 45) median: {combined_median:.2f}")
    for c in bos_cluster:
        ratio = c["mean_activation_magnitude"] / combined_median
        flag = "OUTLIER (>5x)" if ratio > 5 else "normal range"
        print(f"    {c['feature_id']}: magnitude={c['mean_activation_magnitude']:.2f}, "
              f"ratio-to-whole-manifest-median={ratio:.2f}x [{flag}]")

    print()
    print("Read this yourself: if the <bos> cluster's magnitudes sit within or close")
    print("to the rest of the manifest's normal range (not >5x), that CONFIRMS these")
    print("are exactly the 'feat_3 case' -- high structural fraction, normal magnitude,")
    print("correctly NOT excluded by the current conjunction rule, but still receiving")
    print("inflated correlation scores from Qwen for the reason already proven in")
    print("check_structural_outlier_auto_interp.py. This means the REAL fix is not a")
    print("corpus fix and not a semantic-diversity filter -- it's recognizing that the")
    print("structural filter's conjunction rule has a genuine coverage gap: it was")
    print("designed to catch MAGNITUDE outliers specifically, but never claimed to")
    print("catch ALL structural artifacts regardless of magnitude. A pure <bos>")
    print("feature at normal magnitude was never something the 5x/50% rule was")
    print("evidenced to handle -- this is new evidence, not a broken promise.")


if __name__ == "__main__":
    main()

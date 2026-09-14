"""
Diagnostic: does the "beginning of sentences" cluster (10 of 45
concepts in the real v1 manifest) show the SAME single-document-
domination pattern already diagnosed in the corpus-convergence
investigation (Diagnostic 7/8 -- doc[0]'s distinctive vocabulary
bleeding into multiple unrelated features' top-activating-token lists
purely from corpus sparsity, not genuine semantic overlap)?

This determines which fix is the REAL one, not a bandage on the wrong
problem:
  - If the same corpus-domination signature shows up here -> the fix
    is corpus scale/diversity (though Diagnostic 8 already found more
    local documents doesn't converge -- this would mean escalating
    beyond what's locally tractable, same conclusion as before).
  - If NOT the same signature (i.e. these 10 features have genuinely
    DIFFERENT top tokens that just all happen to describe the same
    general syntactic category) -> the fix is a semantic-diversity
    filter on top of dedup, since the underlying signal is real and
    distinct, just redundant at the category level.

Pure JSON analysis of the already-frozen manifest -- no GPU needed.

Usage:
    python check_sentence_start_cluster_diversity.py
"""

import json
from collections import Counter

MANIFEST_PATH = "manifests/concept_manifest_v1.json"
CLUSTER_MARKER = "beginning of sentences"


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    all_concepts = manifest["concepts"]
    print(f"Total concepts in manifest: {len(all_concepts)}")

    cluster = [c for c in all_concepts if CLUSTER_MARKER in c["auto_interp_text"]]
    print(f"'{CLUSTER_MARKER}' cluster size: {len(cluster)}")
    print()

    print("=== Per-candidate top_activating_tokens ===")
    for c in cluster:
        print(f"  {c['feature_id']}: {c['top_activating_tokens']}")
    print()

    # --- Check 1: exact-list overlap (are they literally the same tokens?) ---
    print("=== Check 1: exact top_activating_tokens list overlap ===")
    token_lists = [tuple(c["top_activating_tokens"]) for c in cluster]
    distinct_lists = Counter(token_lists)
    print(f"  Distinct EXACT token lists among {len(cluster)} candidates: {len(distinct_lists)}")
    if len(distinct_lists) < len(cluster):
        for token_list, count in distinct_lists.items():
            if count > 1:
                print(f"    IDENTICAL list shared by {count} candidates: {list(token_list)}")
    print()

    # --- Check 2: vocabulary concentration (same style as the original
    # corpus-convergence diagnostic in smoke_test_real_backend.py) ---
    print("=== Check 2: vocabulary concentration across the cluster ===")
    all_tokens_flat = [tok for c in cluster for tok in c["top_activating_tokens"]]
    total_slots = len(all_tokens_flat)
    token_counts = Counter(all_tokens_flat)
    unique_token_count = len(token_counts)
    vocab_concentration = unique_token_count / total_slots if total_slots else 1.0

    print(f"  {unique_token_count} unique tokens fill {total_slots} total slots "
          f"across {len(cluster)} candidates.")
    print(f"  Vocabulary concentration: {vocab_concentration:.1%}")
    print(f"  Most frequently repeated tokens across DIFFERENT candidates: "
          f"{token_counts.most_common(10)}")
    print()

    if vocab_concentration < 0.4:
        print("  SIGNATURE MATCHES the earlier corpus-domination finding: a small")
        print("  shared vocabulary dominates most candidates' lists -- consistent with")
        print("  a small number of documents' distinctive tokens bleeding across many")
        print("  features, not genuinely independent signal. This points to a CORPUS")
        print("  problem (same conclusion as Diagnostic 8: local scaling didn't help")
        print("  before, escalating further has the same bad cost/benefit tradeoff).")
    else:
        print("  SIGNATURE DOES NOT MATCH the corpus-domination pattern: vocabulary")
        print("  is reasonably diverse across the cluster (each candidate's top tokens")
        print("  are mostly distinct from the others'). This suggests these ARE")
        print("  genuinely different features that each independently detect some")
        print("  flavor of 'sentence start' -- real, distinct signal, just redundant")
        print("  at the CATEGORY level. This points to needing a semantic-diversity")
        print("  filter on top of dedup, NOT a corpus fix.")
    print()

    print("Read the per-candidate token lists above yourself too -- the aggregate")
    print("stats can miss a pattern that's obvious by eye (e.g. all sharing one")
    print("distinctive rare token even if overall vocabulary looks diverse).")


if __name__ == "__main__":
    main()

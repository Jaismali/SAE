"""
Diagnostic 7: reference corpus document-level duplication check.

NOT a unit test, a manual sanity script -- run once, by hand. Follows
from Diagnostic 6's spot-check, which found a suspicious pattern: four
different SAE features (feat_0, feat_1, feat_6, feat_7) shared a nearly
identical fragment of top-activating tokens (' "', ' play', ' can',
' You', '.', ' submitted', 'Survival') -- looking like shared boilerplate
/template text bleeding into multiple UNRELATED features' top-token
lists, rather than each feature having its own coherent semantic theme.

This script checks the actual reference corpus sample itself (same
dataset, same document count as the smoke test) for:
  1. Exact duplicate documents.
  2. Near-duplicate documents (high character-level overlap).
  3. Specifically, which document(s) contain the suspicious fragment
     found in Diagnostic 6, to confirm or rule out that a small number
     of template/boilerplate documents are disproportionately
     contaminating the token-frequency signal across many features.

If a small number of documents dominate, this is a CORPUS problem
(sample too small / not deduplicated), not an auto-interp-method
problem -- it would need fixing regardless of which path (better
heuristic vs. real Neuronpedia access) gets chosen for the
auto-interp question.

Usage:
    python diagnostic_corpus_document_check.py
"""

from collections import Counter
from difflib import SequenceMatcher

from real_backend import REFERENCE_CORPUS_DATASET, DEFAULT_NUM_DOCUMENTS

# Match the actual smoke test's document count, so this checks the
# EXACT corpus slice that produced Diagnostic 6's suspicious pattern --
# not some other arbitrary sample size.
NUM_DOCUMENTS_TO_CHECK = 40

SUSPICIOUS_FRAGMENT_TOKENS = [" play", " can", " You", " submitted", "Survival"]

# CORRECTION (added after first run): the tokens above were a mistake --
# they're common English words that will match most documents in any
# sample, making the "N/40 documents match" count nearly meaningless.
# The first run confirmed 30/40 documents matched, which is not
# informative on its own. What actually matters is whether a RARE,
# distinctive substring is concentrated in one document rather than
# scattered everywhere. Checking that specifically now.
DISTINCTIVE_SUBSTRINGS = ["Tastiest", "Survival of the"]


def _stream_documents(dataset_name: str, num_documents: int):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    documents = []
    for i, example in enumerate(dataset):
        if i >= num_documents:
            break
        documents.append(example["text"])
    return documents


def _similarity(a: str, b: str) -> float:
    # Truncate for speed -- we only need to detect near-duplication,
    # not do a precise diff on potentially long documents.
    return SequenceMatcher(None, a[:2000], b[:2000]).ratio()


def main():
    print("=== Diagnostic 7: reference corpus document-level duplication check ===")
    print(f"Dataset: {REFERENCE_CORPUS_DATASET}")
    print(f"Documents checked: {NUM_DOCUMENTS_TO_CHECK}")
    print()

    documents = _stream_documents(REFERENCE_CORPUS_DATASET, NUM_DOCUMENTS_TO_CHECK)
    print(f"Fetched {len(documents)} documents.")
    print()

    # --- Check 1: exact duplicates ---
    print("--- Check 1: exact duplicate documents ---")
    doc_counts = Counter(documents)
    exact_dupes = {doc: count for doc, count in doc_counts.items() if count > 1}
    if exact_dupes:
        print(f"  WARNING: {len(exact_dupes)} document(s) appear more than once verbatim.")
        for doc, count in exact_dupes.items():
            print(f"    Appears {count}x, first 100 chars: {doc[:100]!r}")
    else:
        print("  No exact duplicate documents found.")
    print()

    # --- Check 2: near-duplicates (pairwise similarity, expensive but n=40 is small) ---
    print("--- Check 2: near-duplicate documents (character-level similarity > 0.7) ---")
    near_dupe_pairs = []
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = _similarity(documents[i], documents[j])
            if sim > 0.7:
                near_dupe_pairs.append((i, j, sim))
    if near_dupe_pairs:
        print(f"  WARNING: {len(near_dupe_pairs)} near-duplicate pair(s) found:")
        for i, j, sim in near_dupe_pairs:
            print(f"    doc[{i}] <-> doc[{j}]: similarity={sim:.3f}")
            print(f"      doc[{i}] first 100 chars: {documents[i][:100]!r}")
            print(f"      doc[{j}] first 100 chars: {documents[j][:100]!r}")
    else:
        print("  No near-duplicate document pairs found (threshold 0.7).")
    print()

    # --- Check 3: which documents contain the suspicious fragment from Diagnostic 6 ---
    print("--- Check 3: documents containing the Diagnostic 6 suspicious fragment ---")
    print(f"  Searching for any of: {SUSPICIOUS_FRAGMENT_TOKENS}")
    matching_doc_indices = []
    for i, doc in enumerate(documents):
        if any(token.strip() in doc for token in SUSPICIOUS_FRAGMENT_TOKENS):
            matching_doc_indices.append(i)
    print(f"  {len(matching_doc_indices)}/{len(documents)} documents contain at least one "
          f"of these fragment tokens.")
    if matching_doc_indices:
        print("  Matching document indices and first 150 chars each:")
        for i in matching_doc_indices:
            print(f"    doc[{i}]: {documents[i][:150]!r}")
    print()

    # --- Check 4 (corrected): distinctive substrings, not generic words ---
    print("--- Check 4: RARE/distinctive substring concentration (corrected search) ---")
    print(f"  Searching for: {DISTINCTIVE_SUBSTRINGS}")
    print("  (Check 3 above used overly generic words like ' can'/' You' -- that was a")
    print("  design mistake, not a real finding; ~75% of any English text sample will")
    print("  contain those words regardless of any real contamination. This check uses")
    print("  the actual distinctive phrase instead, to test the real hypothesis.)")
    distinctive_matches = []
    for i, doc in enumerate(documents):
        if any(sub in doc for sub in DISTINCTIVE_SUBSTRINGS):
            distinctive_matches.append(i)
    print(f"  {len(distinctive_matches)}/{len(documents)} documents contain a distinctive substring.")
    for i in distinctive_matches:
        print(f"    doc[{i}]: {documents[i][:150]!r}")
    if len(distinctive_matches) <= 2:
        print()
        print("  This is the informative result: if only 1-2 documents contain the")
        print("  distinctive phrase (not 30/40 like the generic-word search), that")
        print("  confirms the real mechanism is NOT cross-document duplication, but a")
        print("  single document's vocabulary appearing in multiple UNRELATED features'")
        print("  top-10 lists -- because the total token pool (num_documents x")
        print("  max_tokens_per_doc) is too small for 'top activating tokens' to be a")
        print("  reliable per-feature signal yet, independent of which auto-interp")
        print("  method (heuristic or real Neuronpedia) is eventually used.")
    print()

    print("=== Diagnostic 7 complete. ===")
    print("Read the output above yourself: if a small number of documents (or one")
    print("template repeated with minor variations) account for most of the matches")
    print("in Check 3, or if Check 1/2 found real duplication, that confirms a CORPUS")
    print("problem at this sample size -- worth fixing (larger sample, explicit")
    print("dedup) before deciding between a better auto-interp heuristic or")
    print("escalating real Neuronpedia access, since neither fixes contaminated")
    print("input data.")


if __name__ == "__main__":
    main()

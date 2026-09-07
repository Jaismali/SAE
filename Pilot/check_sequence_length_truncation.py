"""
Diagnostic: sequence-length truncation check.

NOT a unit test -- a manual sanity script, run once. Checks a concrete,
previously-unaddressed fact: do any of the actual induction/correction
training texts, once tokenized, exceed max_sequence_length (64)? If so,
_training_step()'s tokenizer call (truncation=True, max_length=64)
silently cuts them off -- training on malformed, mid-thought-truncated
inputs without anyone knowing. This is a binary fact to check, not a
judgment call: this script reports the actual max token length found
per concept per stage and flags explicitly whether it exceeds 64.

Uses the REAL tokenizer (google/gemma-3-1b-it) via load_base_model's
same AutoTokenizer call, not an approximation -- word count or
character count would not accurately reflect token count for this
model's vocabulary.

Usage:
    python check_sequence_length_truncation.py
"""

from transformers import AutoTokenizer

from pilot_model_io import MODEL_ID, DEFAULT_MAX_SEQUENCE_LENGTH
from pilot_concepts import get_pilot_concepts
from training_data import build_induction_dataset, build_correction_dataset


def check_texts(tokenizer, label: str, texts: list) -> None:
    print(f"\n--- {label} ---")
    max_length_found = 0
    max_length_text = ""
    any_exceeds = False

    for text in texts:
        token_ids = tokenizer(text, truncation=False)["input_ids"]
        length = len(token_ids)
        if length > max_length_found:
            max_length_found = length
            max_length_text = text
        if length > DEFAULT_MAX_SEQUENCE_LENGTH:
            any_exceeds = True
            print(f"  EXCEEDS {DEFAULT_MAX_SEQUENCE_LENGTH}: {length} tokens -- {text!r}")

    print(f"  Max token length found: {max_length_found} "
          f"(limit: {DEFAULT_MAX_SEQUENCE_LENGTH})")
    print(f"  Longest text: {max_length_text!r}")
    if any_exceeds:
        print(f"  *** TRUNCATION IS HAPPENING for {label} -- at least one text "
              f"exceeds {DEFAULT_MAX_SEQUENCE_LENGTH} tokens and is being cut off. ***")
    else:
        margin = DEFAULT_MAX_SEQUENCE_LENGTH - max_length_found
        print(f"  No truncation -- {margin} token margin below the {DEFAULT_MAX_SEQUENCE_LENGTH} limit.")


def main():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"max_sequence_length in use: {DEFAULT_MAX_SEQUENCE_LENGTH}")

    concepts = get_pilot_concepts()
    overall_max = 0
    overall_any_exceeds = False

    for concept in concepts:
        print(f"\n=== Concept: {concept.name} ===")
        induction_texts = build_induction_dataset(concept)
        correction_texts = build_correction_dataset(concept)

        check_texts(tokenizer, f"{concept.name} induction", induction_texts)
        check_texts(tokenizer, f"{concept.name} correction", correction_texts)

    print("\n=== Summary ===")
    print("Read the per-concept output above for the actual max lengths found.")
    print("If ANY '*** TRUNCATION IS HAPPENING ***' line appeared, that's a real,")
    print("concrete gap requiring a fix (raise max_sequence_length or shorten")
    print("the templates) -- not a judgment call, just a fact now known.")


if __name__ == "__main__":
    main()

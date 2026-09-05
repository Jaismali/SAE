"""
Diagnostic: print the actual generated continuations at each pilot stage
(base / post-induction / post-correction) for one concept, so results
can be sanity-checked by eye, not just trusted from the aggregate score.

Motivation: the first full pilot run showed baseline and post-correction
scores of EXACTLY 0.0 for every concept. That could mean genuine, clean
suppression -- or it could mean the model overfit to the tiny correction
dataset and started producing near-verbatim, generic text regardless of
prompt. Only reading the actual text can distinguish these.

Usage:
    python inspect_generations.py <checkpoint_base_dir> <concept_name>
    Example:
    python inspect_generations.py D:\\...\\checkpoints\\qlora dog_theme
"""

import sys

from pilot_concepts import get_pilot_concepts
from pilot_model_io import load_base_model, fine_tune_on_texts
from training_data import build_induction_dataset, build_correction_dataset


def generate_for_prompts(model, tokenizer, prompts, max_new_tokens=30):
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        continuation = full_text[len(prompt):]
        results.append((prompt, continuation))
    return results


def print_generations(stage_label, generations):
    print(f"\n--- {stage_label} ---")
    for prompt, continuation in generations:
        print(f"  Prompt: {prompt!r}")
        print(f"  Continuation: {continuation!r}\n")


def inspect_one_concept(checkpoint_base_dir: str, concept_name: str) -> None:
    concept = next(c for c in get_pilot_concepts() if c.name == concept_name)
    print(f"Inspecting concept: {concept.name}")
    print(f"Theme tokens: {concept.theme_tokens}")

    print("\nLoading fresh base model...")
    model, tokenizer = load_base_model()

    base_generations = generate_for_prompts(model, tokenizer, concept.held_out_prompts)
    print_generations("BASELINE (before any fine-tuning)", base_generations)

    print("Running induction fine-tuning...")
    induction_texts = build_induction_dataset(concept)
    model = fine_tune_on_texts(model, tokenizer, induction_texts)

    induced_generations = generate_for_prompts(model, tokenizer, concept.held_out_prompts)
    print_generations("POST-INDUCTION", induced_generations)

    print("Running correction fine-tuning...")
    correction_texts = build_correction_dataset(concept)
    model = fine_tune_on_texts(model, tokenizer, correction_texts)

    corrected_generations = generate_for_prompts(model, tokenizer, concept.held_out_prompts)
    print_generations("POST-CORRECTION", corrected_generations)

    print("\nCorrection training texts used (for reference, to check for verbatim overfitting):")
    for text in correction_texts[:5]:
        print(f"  - {text!r}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_generations.py <checkpoint_base_dir> <concept_name>")
        print("Available concept names: dog_theme, ocean_theme, music_theme, "
              "accounting_theme, volcano_theme")
        sys.exit(1)

    inspect_one_concept(checkpoint_base_dir=sys.argv[1], concept_name=sys.argv[2])

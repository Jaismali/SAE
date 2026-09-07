"""
Diagnostic: print the actual generated continuations at each pilot stage
(base / post-induction / post-correction) for one concept, so results
can be sanity-checked by eye, not just trusted from the aggregate score.
Motivation: the first full pilot run showed baseline and post-correction
scores of EXACTLY 0.0 for every concept. That could mean genuine, clean
suppression -- or it could mean the model overfit to the tiny correction
dataset and started producing near-verbatim, generic text regardless of
prompt. Only reading the actual text can distinguish these.

PRE-EXISTING ISSUE, FLAGGED (found while extending this script, not
introduced by this change): the checkpoint_base_dir parameter below
was accepted but NEVER USED anywhere in inspect_one_concept()'s body --
this script has always retrained from scratch rather than loading a
saved checkpoint from that directory, despite what its usage/docstring
implies. This is a real, pre-existing gap in code from before this
session, not something to silently paper over. It's left unfixed here
(loading a saved PEFT adapter would require exposing raw base-model
loading separately from load_base_model()'s bundled adapter-attachment
step -- a bigger change, flagged rather than made blind) -- callers
should be aware this always retrains fresh, not inspects a specific
saved checkpoint's exact weights.

Added induction_num_epochs / correction_num_epochs (optional, default
None = fine_tune_on_texts's own default for both) to support inspecting
the SAME asymmetric epoch configuration used in a comparison run (e.g.
induction=5, correction=1) -- since this always retrains anyway, this
lets that retrain match the configuration actually being evaluated.

Added seed (optional, default None = UNSEEDED, matching every run in
this investigation before seeding was added). Pass an explicit seed to
make this specific inspection run reproducible and auditable.

Usage:
    python inspect_generations.py <checkpoint_base_dir> <concept_name> [induction_num_epochs] [correction_num_epochs] [seed]
    Example (default epochs, matches original behavior, unseeded):
    python inspect_generations.py D:\\...\\checkpoints\\qlora dog_theme
    Example (asymmetric 5/1 configuration, unseeded):
    python inspect_generations.py D:\\...\\checkpoints\\qlora_5_1_test dog_theme 5 1
    Example (asymmetric 4/1 configuration, EXPLICIT SEED for reproducibility check):
    python inspect_generations.py D:\\...\\checkpoints\\qlora_4_1_ocean_seeded ocean_theme 4 1 42
"""
import sys
from pilot_concepts import get_pilot_concepts
from pilot_model_io import load_base_model, fine_tune_on_texts
from training_data import build_induction_dataset, build_correction_dataset
from repetition_detector import detect_ngram_repetition
from induction_stability_gate import evaluate_induction_stability
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
        print(f"  Continuation: {continuation!r}")
        repetition = detect_ngram_repetition(continuation)
        if repetition.is_degenerate:
            print(f"  [REPETITION FLAG] {repetition.ngram_size}-gram "
                  f"{repetition.worst_ngram!r} repeated {repetition.repeat_count}x -- "
                  "visibility only, this does not change the aggregate score")
        print()
def inspect_one_concept(
    checkpoint_base_dir: str,
    concept_name: str,
    induction_num_epochs: int = None,
    correction_num_epochs: int = None,
    seed: int = None,
) -> None:
    concept = next(c for c in get_pilot_concepts() if c.name == concept_name)
    print(f"Inspecting concept: {concept.name}")
    print(f"Theme tokens: {concept.theme_tokens}")
    if induction_num_epochs is not None or correction_num_epochs is not None:
        print(f"Epoch override: induction={induction_num_epochs}, correction={correction_num_epochs}")
    if seed is not None:
        print(f"SEED: {seed} (explicit, reproducible)")
    else:
        print("SEED: none given -- this run is UNSEEDED and cannot be reproduced or audited later")
    print("NOTE: this always retrains from scratch (see module docstring) -- it does")
    print(f"NOT load the saved checkpoint at {checkpoint_base_dir!r}.")
    print("\nLoading fresh base model...")
    model, tokenizer = load_base_model(seed=seed)
    # BUG FOUND AND FIXED: model was never put in eval() mode before
    # generating baseline text, meaning dropout was ACTIVE during the
    # "baseline" read (get_peft_model leaves a fresh model in train()
    # mode by default). This is conceptually wrong (a baseline should
    # reflect clean pretrained behavior, not dropout-noised output) and
    # is a plausible source of the seed=7 run-to-run discrepancy
    # observed: dropout-active generation is exactly the step most
    # exposed to residual GPU-kernel nondeterminism (see
    # set_deterministic_seed's honest caveat), and any resulting RNG-
    # state shift there could cascade into different dropout masks
    # during the SUBSEQUENT training, changing the final trained
    # weights despite an identical top-level seed. fine_tune_on_texts
    # already correctly calls model.eval() at its end, so post-induction
    # and post-correction generation were NOT affected by this -- only
    # the baseline read, and only as concerns discrepancy IN BASELINE
    # text itself plus this downstream cascading risk.
    model.eval()
    base_generations = generate_for_prompts(model, tokenizer, concept.held_out_prompts)
    print_generations("BASELINE (before any fine-tuning)", base_generations)
    print("Running induction fine-tuning...")
    induction_texts = build_induction_dataset(concept)
    induction_kwargs = {}
    if induction_num_epochs is not None:
        induction_kwargs["num_epochs"] = induction_num_epochs
    model = fine_tune_on_texts(model, tokenizer, induction_texts, **induction_kwargs)
    induced_generations = generate_for_prompts(model, tokenizer, concept.held_out_prompts)
    print_generations("POST-INDUCTION", induced_generations)

    stability_result = evaluate_induction_stability(concept.name, induced_generations)
    print(f"=== Induction stability gate: {stability_result.degenerate_count}/"
          f"{stability_result.total_prompts} degenerate (threshold: "
          f"{stability_result.exclusion_threshold}) ===")
    print(f"  Verdict: {'EXCLUDE' if stability_result.is_excluded else 'PASS'}")
    print(f"  {stability_result.reason}")
    print()

    print("Running correction fine-tuning...")
    correction_texts = build_correction_dataset(concept)
    correction_kwargs = {}
    if correction_num_epochs is not None:
        correction_kwargs["num_epochs"] = correction_num_epochs
    model = fine_tune_on_texts(model, tokenizer, correction_texts, **correction_kwargs)
    corrected_generations = generate_for_prompts(model, tokenizer, concept.held_out_prompts)
    print_generations("POST-CORRECTION", corrected_generations)
    print("\nCorrection training texts used (for reference, to check for verbatim overfitting):")
    for text in correction_texts[:5]:
        print(f"  - {text!r}")
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_generations.py <checkpoint_base_dir> <concept_name> "
              "[induction_num_epochs] [correction_num_epochs] [seed]")
        print("Available concept names: dog_theme, ocean_theme, music_theme, "
              "accounting_theme, volcano_theme")
        sys.exit(1)
    induction_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
    correction_override = int(sys.argv[4]) if len(sys.argv) > 4 else None
    seed_override = int(sys.argv[5]) if len(sys.argv) > 5 else None
    inspect_one_concept(
        checkpoint_base_dir=sys.argv[1],
        concept_name=sys.argv[2],
        induction_num_epochs=induction_override,
        correction_num_epochs=correction_override,
        seed=seed_override,
    )


"""
Part C Pilot - Step 6: Full pilot run script.

Wires the real model I/O (pilot_model_io.py) into the tested
orchestration logic (pilot_runner.py) and runs the complete induction ->
correction pilot across all 5 pilot concepts on this machine.

This is the actual Part C deliverable: confirms whether the QLoRA
induction/correction training loop works correctly end-to-end, per the
Month 1 brief's stated purpose for this pilot.

METRIC NOTE (see concept_strength_measure.py): scores reported here are
BEHAVIORAL/TOKEN-BASED proxies, NOT SAE-based dormancy measurements.
This is stated explicitly in every result printed and logged.

Usage:
    python run_pilot.py <base_checkpoint_dir>
    Example:
    python run_pilot.py D:\\Devlopment\\Projects\\geometric-dormancy\\checkpoints\\qlora
"""

import json
import sys
from datetime import datetime, timezone

from concept_strength_measure import TokenFrequencyMeasure, METRIC_TYPE_LABEL
from pilot_concepts import get_pilot_concepts
from pilot_model_io import load_base_model, fine_tune_on_texts, save_lora_checkpoint
from pilot_runner import run_pilot_for_all_concepts, PilotConceptResult


def _make_measure_for_concept(concept) -> TokenFrequencyMeasure:
    return TokenFrequencyMeasure(theme_tokens=concept.theme_tokens)


def _make_save_checkpoint_fn(base_checkpoint_dir: str):
    def save_checkpoint_fn(model, tag: str) -> str:
        return save_lora_checkpoint(model, tag, base_checkpoint_dir)
    return save_checkpoint_fn


def _make_logging_fine_tune_fn(
    concept_name: str,
    loss_log_dir: str,
    num_epochs: int = None,
    induction_num_epochs: int = None,
    correction_num_epochs: int = None,
):
    """Wraps fine_tune_on_texts() so each call's loss is logged to a
    separate CSV, one per (concept, stage).

    Epoch overrides, in order of precedence:
      - induction_num_epochs / correction_num_epochs: independent
        per-stage overrides. Added to test an ASYMMETRIC epoch
        hypothesis grounded in Qi et al. (2023) ("Fine-tuning Aligned
        Language Models Compromises Safety"): their attacks that
        INSTILL a behavior typically use ~5 epochs, while their
        correction-analog (standard benign fine-tuning) uses 1 epoch
        by official default. Our induction/correction stages are
        structurally similar in role but NOT the same mechanism as
        Qi et al.'s -- two translation gaps, stated explicitly so they
        don't get overclaimed later:
          (1) Model scale: Qi et al. used Llama-2-7B-Chat; we use
              Gemma 3 1B. A ~7x parameter difference could plausibly
              change how many epochs it takes to instill/saturate a
              behavior in either direction (smaller models sometimes
              overfit a narrow pattern FASTER, not slower, having less
              capacity to distribute the update).
          (2) Task analogy is directional, not exact: Qi et al.'s
              induction is an ADVERSARIAL attack fighting against
              safety-training resistance; ours instills a neutral,
              arbitrary theme with no competing resistance. Their
              correction-analog (Alpaca/Dolly fine-tuning) isn't
              actively correcting anything, just not reinforcing the
              attack; ours actively tries to suppress something just
              induced. Structurally similar roles, not the same
              mechanism.
        Epoch counts here are INFORMED BY, not directly transferred
        from, Qi et al.'s findings -- the literature indicates the
        right SHAPE (asymmetric, induction-high/correction-low), and
        our own pilot data confirms or adjusts the actual numbers.
      - num_epochs: single shared override for BOTH stages (original
        behavior, e.g. the 1-epoch symmetric comparison run).
      - If none of the above given, fine_tune_on_texts's own default
        (3 epochs) is used for both stages, unchanged.

    IMPLICIT COUPLING, FLAGGED EXPLICITLY: pilot_runner.py's
    run_pilot_for_concept() calls the injected fine_tune_fn exactly
    twice per concept, ALWAYS in the order (induction, correction) --
    per its own docstring's documented sequence. This closure counts
    calls to infer which stage each one is, rather than pilot_runner.py
    passing an explicit stage argument.

    This was a deliberate choice (Option A, lower risk) over changing
    pilot_runner.py's tested interface to pass an explicit stage
    argument (Option B), which would have required modifying
    test_pilot_runner.py without having reviewed it first. If
    pilot_runner.py's call order or call count per concept EVER changes,
    this labeling will silently become wrong rather than erroring --
    revisit this closure if that orchestration logic changes.

    Added specifically to make Part B's pilot-hyperparameter review
    possible: Month 1's pilot run captured NO loss data at all
    (confirmed by inspection of the training loop, not assumed).
    """
    import os

    call_count = {"n": 0}
    stage_names = ["induction", "correction"]
    stage_epoch_overrides = {
        "induction": induction_num_epochs if induction_num_epochs is not None else num_epochs,
        "correction": correction_num_epochs if correction_num_epochs is not None else num_epochs,
    }

    def logging_fine_tune_fn(model, tokenizer, texts):
        stage_index = call_count["n"]
        if stage_index >= len(stage_names):
            raise RuntimeError(
                f"fine_tune_fn called {stage_index + 1} times for concept "
                f"'{concept_name}', but only {len(stage_names)} calls "
                f"(induction, correction) were expected. pilot_runner.py's "
                "call pattern may have changed -- this logging wrapper's "
                "assumption is now stale and must be fixed before trusting "
                "any loss log produced by it."
            )
        stage_name = stage_names[stage_index]
        call_count["n"] += 1

        log_path = os.path.join(loss_log_dir, f"{concept_name}_{stage_name}_loss.csv")
        kwargs = {"loss_log_path": log_path}
        stage_epochs = stage_epoch_overrides[stage_name]
        if stage_epochs is not None:
            kwargs["num_epochs"] = stage_epochs
        return fine_tune_on_texts(model, tokenizer, texts, **kwargs)

    return logging_fine_tune_fn


def run_full_pilot(base_checkpoint_dir: str, loss_log_dir: str = None) -> list:
    """Runs the pilot across all 5 concepts, one at a time. Each concept
    gets a freshly-loaded base model (so concepts don't contaminate each
    other's induction/correction results), and its own
    TokenFrequencyMeasure scoped to its own theme tokens.

    loss_log_dir (optional): if provided, per-concept per-stage loss CSVs
    are written here via _make_logging_fine_tune_fn(). Default None
    preserves the exact previous behavior (no logging) for any other
    caller of this function.
    """
    concepts = get_pilot_concepts()
    save_checkpoint_fn = _make_save_checkpoint_fn(base_checkpoint_dir)

    results = []
    for concept in concepts:
        print(f"\n{'=' * 60}")
        print(f"Running pilot for concept: {concept.name}")
        print(f"{'=' * 60}")

        measure = _make_measure_for_concept(concept)

        if loss_log_dir is not None:
            fine_tune_fn = _make_logging_fine_tune_fn(concept.name, loss_log_dir)
        else:
            fine_tune_fn = fine_tune_on_texts

        concept_results = run_pilot_for_all_concepts(
            concepts=[concept],
            measure=measure,
            load_base_model_fn=load_base_model,
            fine_tune_fn=fine_tune_fn,
            save_checkpoint_fn=save_checkpoint_fn,
        )
        results.extend(concept_results)
        _print_concept_result(concept_results[0])

    return results


def _print_concept_result(result: PilotConceptResult) -> None:
    gate = result.gate_result
    print(f"\nConcept: {result.concept_name}")
    print(f"Metric type: {result.metric_type_label}")
    print(f"  Baseline score:        {gate.baseline_score:.4f}")
    print(f"  Post-induction score:  {gate.post_induction_score:.4f}  "
          f"(delta: {gate.induction_delta:+.4f})")
    print(f"  Post-correction score: {gate.post_correction_score:.4f}  "
          f"(delta: {gate.correction_delta:+.4f})")
    print(f"  Gate outcome: {gate.outcome.value}")
    print(f"  Reason: {gate.reason}")
    print(f"  Checkpoints: {result.checkpoint_paths}")


def _result_to_dict(result: PilotConceptResult) -> dict:
    gate = result.gate_result
    return {
        "concept_name": result.concept_name,
        "metric_type_label": result.metric_type_label,
        "baseline_score": gate.baseline_score,
        "post_induction_score": gate.post_induction_score,
        "post_correction_score": gate.post_correction_score,
        "induction_delta": gate.induction_delta,
        "correction_delta": gate.correction_delta,
        "gate_outcome": gate.outcome.value,
        "gate_reason": gate.reason,
        "checkpoint_paths": result.checkpoint_paths,
    }


def write_results_log(results: list, results_dir: str) -> str:
    import os
    os.makedirs(results_dir, exist_ok=True)

    log_document = {
        "pilot_run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metric_type_label": METRIC_TYPE_LABEL,
        "scope_note": (
            "This is a Part C plumbing-test pilot confirming the QLoRA induction/"
            "correction training loop works correctly. Scores are a behavioral/"
            "token-frequency proxy, NOT an SAE-based dormancy measurement. "
            "See concept_strength_measure.py for the interface a future SAE-based "
            "measure should implement."
        ),
        "concept_results": [_result_to_dict(r) for r in results],
        "summary": {
            "total_concepts": len(results),
            "passed": sum(1 for r in results if r.gate_result.passed),
            "induction_failed": sum(
                1 for r in results if r.gate_result.outcome.value == "induction_failed"
            ),
            "correction_failed": sum(
                1 for r in results if r.gate_result.outcome.value == "correction_failed"
            ),
            "both_failed": sum(
                1 for r in results if r.gate_result.outcome.value == "both_failed"
            ),
        },
    }

    log_path = os.path.join(results_dir, "part_c_pilot_results.json")
    with open(log_path, "w") as f:
        json.dump(log_document, f, indent=2)

    return log_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pilot.py <base_checkpoint_dir> [results_dir] [loss_log_dir]")
        print(r"Example: python run_pilot.py D:\Devlopment\Projects\geometric-dormancy\checkpoints\qlora")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "../results"
    # loss_log_dir defaults to a fresh directory under results_dir if not given,
    # so re-running the pilot after this change ALWAYS produces loss data unless
    # explicitly disabled below -- this is the whole point of this addition.
    loss_log_dir = sys.argv[3] if len(sys.argv) > 3 else None
    if loss_log_dir is None:
        import os
        loss_log_dir = os.path.join(results_dir, "loss_logs")

    print("Starting Part C pilot run across all pilot concepts.")
    print(f"METRIC TYPE: {METRIC_TYPE_LABEL}")
    print(f"Loss logs will be written to: {loss_log_dir}\n")

    all_results = run_full_pilot(checkpoint_dir, loss_log_dir=loss_log_dir)

    log_path = write_results_log(all_results, results_dir)

    print(f"\n{'=' * 60}")
    print("PILOT RUN COMPLETE")
    print(f"{'=' * 60}")
    passed = sum(1 for r in all_results if r.gate_result.passed)
    print(f"{passed}/{len(all_results)} concepts passed the sanity gate.")
    print(f"Full results log written to: {log_path}")
    print(f"Per-concept, per-stage loss logs written to: {loss_log_dir}")

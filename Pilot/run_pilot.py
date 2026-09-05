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


def run_full_pilot(base_checkpoint_dir: str) -> list:
    """Runs the pilot across all 5 concepts, one at a time. Each concept
    gets a freshly-loaded base model (so concepts don't contaminate each
    other's induction/correction results), and its own
    TokenFrequencyMeasure scoped to its own theme tokens."""
    concepts = get_pilot_concepts()
    save_checkpoint_fn = _make_save_checkpoint_fn(base_checkpoint_dir)

    results = []
    for concept in concepts:
        print(f"\n{'=' * 60}")
        print(f"Running pilot for concept: {concept.name}")
        print(f"{'=' * 60}")

        measure = _make_measure_for_concept(concept)

        concept_results = run_pilot_for_all_concepts(
            concepts=[concept],
            measure=measure,
            load_base_model_fn=load_base_model,
            fine_tune_fn=fine_tune_on_texts,
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
        print("Usage: python run_pilot.py <base_checkpoint_dir>")
        print(r"Example: python run_pilot.py D:\Devlopment\Projects\geometric-dormancy\checkpoints\qlora")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "../results"

    print("Starting Part C pilot run across all pilot concepts.")
    print(f"METRIC TYPE: {METRIC_TYPE_LABEL}\n")

    all_results = run_full_pilot(checkpoint_dir)

    log_path = write_results_log(all_results, results_dir)

    print(f"\n{'=' * 60}")
    print("PILOT RUN COMPLETE")
    print(f"{'=' * 60}")
    passed = sum(1 for r in all_results if r.gate_result.passed)
    print(f"{passed}/{len(all_results)} concepts passed the sanity gate.")
    print(f"Full results log written to: {log_path}")

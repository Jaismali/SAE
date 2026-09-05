"""
Part C Pilot - Step 4b: Pilot run orchestration.

Ties together training_data, training_config, concept_strength_measure,
and sanity_gate into the actual per-concept pilot sequence:

    load base model (fresh, 4-bit + LoRA)
      -> measure baseline concept strength
      -> induction fine-tune
      -> measure post-induction concept strength
      -> correction fine-tune
      -> measure post-correction concept strength
      -> evaluate sanity gate
      -> save checkpoints (base / post-induction / post-correction)

Model loading, tokenization, and the actual train() calls are injected
as callables (not hardcoded imports at the top of this module), so the
ORCHESTRATION logic -- the order of operations, what gets measured when,
how results are assembled -- can be fully tested with fake/mocked
components, independent of whether a real GPU is available. The real
model-loading and training functions live in pilot_model_io.py and are
wired in only when actually running on the target machine.
"""

from dataclasses import dataclass
from typing import Callable, List

from concept_strength_measure import ConceptStrengthMeasure, METRIC_TYPE_LABEL
from pilot_concepts import PilotConcept
from sanity_gate import evaluate_sanity_gate, SanityGateResult
from training_data import build_induction_dataset, build_correction_dataset


@dataclass
class PilotConceptResult:
    concept_name: str
    metric_type_label: str  # always carried alongside the scores -- see concept_strength_measure.py
    gate_result: SanityGateResult
    checkpoint_paths: dict  # {"base": ..., "post_induction": ..., "post_correction": ...}


def run_pilot_for_concept(
    concept: PilotConcept,
    measure: ConceptStrengthMeasure,
    load_base_model_fn: Callable[[], tuple],
    fine_tune_fn: Callable[[object, object, List[str]], object],
    save_checkpoint_fn: Callable[[object, str], str],
) -> PilotConceptResult:
    """Runs the full induction -> correction pilot sequence for one
    concept. All model/training operations are injected as callables so
    this function's ORDER OF OPERATIONS can be tested without a real
    model. `load_base_model_fn` returns (model, tokenizer).
    `fine_tune_fn(model, tokenizer, texts)` returns the fine-tuned model.
    `save_checkpoint_fn(model, tag)` returns the saved checkpoint's path.
    """
    model, tokenizer = load_base_model_fn()
    checkpoint_paths = {"base": save_checkpoint_fn(model, f"{concept.name}_base")}

    baseline_score = measure.measure(model, tokenizer, concept.held_out_prompts)

    induction_texts = build_induction_dataset(concept)
    model = fine_tune_fn(model, tokenizer, induction_texts)
    checkpoint_paths["post_induction"] = save_checkpoint_fn(model, f"{concept.name}_post_induction")
    post_induction_score = measure.measure(model, tokenizer, concept.held_out_prompts)

    correction_texts = build_correction_dataset(concept)
    model = fine_tune_fn(model, tokenizer, correction_texts)
    checkpoint_paths["post_correction"] = save_checkpoint_fn(model, f"{concept.name}_post_correction")
    post_correction_score = measure.measure(model, tokenizer, concept.held_out_prompts)

    gate_result = evaluate_sanity_gate(
        baseline_score=baseline_score,
        post_induction_score=post_induction_score,
        post_correction_score=post_correction_score,
    )

    return PilotConceptResult(
        concept_name=concept.name,
        metric_type_label=measure.metric_type_label,
        gate_result=gate_result,
        checkpoint_paths=checkpoint_paths,
    )


def run_pilot_for_all_concepts(
    concepts: List[PilotConcept],
    measure: ConceptStrengthMeasure,
    load_base_model_fn: Callable[[], tuple],
    fine_tune_fn: Callable[[object, object, List[str]], object],
    save_checkpoint_fn: Callable[[object, str], str],
) -> List[PilotConceptResult]:
    """Runs run_pilot_for_concept() across every concept, continuing even
    if one concept's gate fails -- per the standards note, a failed or
    ambiguous result must be reported honestly, not treated as a reason
    to abort the whole pilot."""
    return [
        run_pilot_for_concept(
            concept, measure, load_base_model_fn, fine_tune_fn, save_checkpoint_fn
        )
        for concept in concepts
    ]

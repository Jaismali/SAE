"""
Test Step 4b in isolation: does run_pilot_for_concept() execute the
correct sequence of operations (load -> baseline -> induce -> measure ->
correct -> measure -> gate -> checkpoint), using fake injected
model/training/checkpoint functions -- no real GPU or model needed?
"""

from concept_strength_measure import ConceptStrengthMeasure
from pilot_concepts import PilotConcept
from pilot_runner import run_pilot_for_concept, run_pilot_for_all_concepts
from sanity_gate import GateOutcome


class ScriptedMeasure(ConceptStrengthMeasure):
    """Fake measure returning a pre-scripted sequence of scores, one per
    call, so tests control exactly what baseline/post-induction/
    post-correction scores the gate sees."""

    def __init__(self, scripted_scores):
        self._scores = list(scripted_scores)
        self._call_count = 0

    def measure(self, model, tokenizer, prompts):
        score = self._scores[self._call_count]
        self._call_count += 1
        return score

    @property
    def metric_type_label(self) -> str:
        return "scripted_test_measure"


def _dummy_concept() -> PilotConcept:
    return PilotConcept(
        name="test_concept",
        theme_tokens=["alpha", "beta", "gamma"],
        held_out_prompts=["prompt one", "prompt two"],
    )


def _fake_pipeline(scripted_scores):
    """Builds a full set of fake injected functions plus a call-order
    tracker, so tests can assert both the sequence and the final result."""
    call_order = []

    def fake_load_base_model():
        call_order.append("load_base_model")
        return ("fake_model", "fake_tokenizer")

    def fake_fine_tune(model, tokenizer, texts):
        call_order.append(f"fine_tune({len(texts)}_texts)")
        return model  # pretend fine-tuning returns an "updated" model

    saved_checkpoints = []

    def fake_save_checkpoint(model, tag):
        call_order.append(f"save_checkpoint({tag})")
        path = f"/fake/checkpoints/{tag}.pt"
        saved_checkpoints.append(path)
        return path

    measure = ScriptedMeasure(scripted_scores)
    return measure, fake_load_base_model, fake_fine_tune, fake_save_checkpoint, call_order, saved_checkpoints


def test_operations_execute_in_the_correct_order():
    measure, load_fn, tune_fn, save_fn, call_order, _ = _fake_pipeline(
        scripted_scores=[0.02, 0.30, 0.05]  # baseline, post-induction, post-correction
    )

    run_pilot_for_concept(_dummy_concept(), measure, load_fn, tune_fn, save_fn)

    expected_order = [
        "load_base_model",
        "save_checkpoint(test_concept_base)",
        "fine_tune(9_texts)",        # induction: 3 theme tokens * 3 sentences each (default)
        "save_checkpoint(test_concept_post_induction)",
        "fine_tune(15_texts)",       # correction: default target_size=15
        "save_checkpoint(test_concept_post_correction)",
    ]
    assert call_order == expected_order, f"Expected {expected_order}, got {call_order}"
    print("PASS: pilot executes load -> checkpoint -> induce -> checkpoint -> "
          "correct -> checkpoint in the exact correct order")


def test_result_carries_the_metric_type_label():
    measure, load_fn, tune_fn, save_fn, _, _ = _fake_pipeline(
        scripted_scores=[0.02, 0.30, 0.05]
    )
    result = run_pilot_for_concept(_dummy_concept(), measure, load_fn, tune_fn, save_fn)

    assert result.metric_type_label == "scripted_test_measure", (
        "Result must carry the measure's metric_type_label, so it's never mistaken "
        "for an SAE-based result later"
    )
    print(f"PASS: result carries the metric_type_label ({result.metric_type_label!r}), "
          f"preventing later misattribution")


def test_result_includes_all_three_checkpoint_paths():
    measure, load_fn, tune_fn, save_fn, _, saved_checkpoints = _fake_pipeline(
        scripted_scores=[0.02, 0.30, 0.05]
    )
    result = run_pilot_for_concept(_dummy_concept(), measure, load_fn, tune_fn, save_fn)

    assert set(result.checkpoint_paths.keys()) == {"base", "post_induction", "post_correction"}
    assert len(saved_checkpoints) == 3
    print("PASS: result includes all three required checkpoint stages (base, post_induction, post_correction)")


def test_clear_success_case_produces_passed_gate():
    measure, load_fn, tune_fn, save_fn, _, _ = _fake_pipeline(
        scripted_scores=[0.02, 0.30, 0.05]  # strong induction, strong correction
    )
    result = run_pilot_for_concept(_dummy_concept(), measure, load_fn, tune_fn, save_fn)

    assert result.gate_result.outcome == GateOutcome.PASSED
    print("PASS: a concept with strong induction and strong correction produces a PASSED gate")


def test_failed_induction_case_produces_induction_failed_gate():
    measure, load_fn, tune_fn, save_fn, _, _ = _fake_pipeline(
        scripted_scores=[0.10, 0.105, 0.08]  # induction barely moved
    )
    result = run_pilot_for_concept(_dummy_concept(), measure, load_fn, tune_fn, save_fn)

    assert result.gate_result.outcome == GateOutcome.INDUCTION_FAILED
    print("PASS: a concept where induction barely moved the score produces an INDUCTION_FAILED gate, "
          "not a silently-passed result")


def test_pilot_continues_across_concepts_even_if_one_fails():
    # Two concepts: first fails induction, second succeeds fully. Both
    # should still be run and reported -- a failure must not abort the batch.
    concepts = [
        PilotConcept(name="failing_concept", theme_tokens=["x"], held_out_prompts=["p"]),
        PilotConcept(name="succeeding_concept", theme_tokens=["y"], held_out_prompts=["p"]),
    ]

    call_log = []

    class SequencedMeasure(ConceptStrengthMeasure):
        def __init__(self):
            # First concept: failing scores. Second concept: succeeding scores.
            self._sequences = [
                iter([0.10, 0.105, 0.08]),  # failing_concept
                iter([0.02, 0.30, 0.05]),   # succeeding_concept
            ]
            self._concept_index = -1
            self._current_iter = None

        def measure(self, model, tokenizer, prompts):
            return next(self._current_iter)

        @property
        def metric_type_label(self):
            return "sequenced_test_measure"

        def advance_concept(self):
            self._concept_index += 1
            self._current_iter = self._sequences[self._concept_index]

    measure = SequencedMeasure()

    def load_fn():
        measure.advance_concept()
        call_log.append("load")
        return ("model", "tokenizer")

    def tune_fn(model, tokenizer, texts):
        return model

    def save_fn(model, tag):
        return f"/fake/{tag}.pt"

    results = run_pilot_for_all_concepts(concepts, measure, load_fn, tune_fn, save_fn)

    assert len(results) == 2, "Both concepts must be run and reported, even though the first fails"
    assert results[0].gate_result.outcome == GateOutcome.INDUCTION_FAILED
    assert results[1].gate_result.outcome == GateOutcome.PASSED
    print("PASS: a failing concept does not abort the batch -- both concepts run and "
          "are reported with their true, distinct outcomes")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        raise SystemExit(1)

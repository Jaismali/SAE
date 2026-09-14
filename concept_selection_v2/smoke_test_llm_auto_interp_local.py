"""
Smoke test for llm_auto_interp_local_client.py -- NOT a unit test, a
manual sanity script. Run once, by hand, before running the real
auto-interp module against any real SAE candidate data.

Uses the SAME deliberately unambiguous example as
smoke_test_llm_auto_interp.py (the API version), so results are
directly comparable if the API version is ever run too -- a feature
that clearly fires on ocean-related words, so a human can immediately
judge whether the explainer and simulator are behaving sensibly.

HONEST EXPECTATION, stated before running rather than after: per
llm_auto_interp_local_client.py's own documented caveat, a 7B local
model is likely to be noisier and weaker at this task than a frontier
API model would be. Do not be surprised by a lower score or a rougher
explanation than you might expect -- that is exactly the kind of
real evidence this smoke test exists to surface, not a sign something
is necessarily broken.

Requires: transformers, torch, bitsandbytes, peft (already installed,
same stack as pilot_model_io.py). No API key, no license acceptance --
Qwen2.5-7B-Instruct is ungated.

Usage:
    python smoke_test_llm_auto_interp_local.py
"""

from llm_auto_interp_local_client import (
    LOCAL_MODEL_ID,
    load_auto_interp_model,
    run_llm_auto_interp_local,
)

# Identical to smoke_test_llm_auto_interp.py's example, for direct comparability.
TOP_ACTIVATING_EXAMPLES = [
    "The waves crashed against the rocky shore.",
    "A pod of dolphins swam near the boat.",
    "The tide was coming in fast.",
    "Coral reefs are home to thousands of species.",
    "The ocean current pulled the swimmer further out.",
]

HELD_OUT_CONTEXTS = [
    "The whale breached near the fishing boat.",       # ocean-related -> high
    "The stock market dropped sharply today.",          # unrelated -> low
    "Seashells were scattered along the beach.",         # ocean-related -> high
    "The committee voted on the new budget proposal.",   # unrelated -> low
    "A storm was forming over the open sea.",            # ocean-related -> high
]
TRUE_ACTIVATIONS = [8.0, 0.0, 7.0, 0.0, 8.0]


def main():
    print(f"=== LLM auto-interp smoke test (LOCAL model: {LOCAL_MODEL_ID}) ===")
    print("Loading model (first run will download it -- may take a while)...")

    try:
        model, tokenizer = load_auto_interp_model()
    except Exception as e:
        print(f"FAILED to load model: {type(e).__name__}: {e}")
        return
    print("  Loaded.")
    print()

    print(f"Top-activating examples shown to explainer: {len(TOP_ACTIVATING_EXAMPLES)}")
    print(f"Held-out contexts shown to simulator: {len(HELD_OUT_CONTEXTS)}")
    print()

    try:
        result = run_llm_auto_interp_local(
            model=model,
            tokenizer=tokenizer,
            top_activating_examples=TOP_ACTIVATING_EXAMPLES,
            held_out_contexts=HELD_OUT_CONTEXTS,
            true_activations=TRUE_ACTIVATIONS,
        )
    except Exception as e:
        import traceback
        print(f"FAILED: {type(e).__name__}: {e}")
        print()
        print("=== Full traceback (needed to diagnose this properly) ===")
        traceback.print_exc()
        print()
        print("If this is a parsing error (wrong count of numbers in the simulator's")
        print("response), that is real, informative evidence -- not a bug to silently")
        print("retry past. It means this prompt, written with a frontier model in mind,")
        print("may need tightening for a smaller local model's weaker instruction-")
        print("following. Report the raw response shown in the error message.")
        return

    print(f"Explanation generated: {result.explanation!r}")
    print()
    print("Read this yourself: does the explanation correctly identify an")
    print("ocean/sea-related pattern? If it says something unrelated or vague,")
    print("that's a real problem with the explainer step, not a minor issue.")
    print()

    for i, context in enumerate(HELD_OUT_CONTEXTS):
        print(f"  Context: {context!r}")
        print(f"    True activation (hand-assigned):  {TRUE_ACTIVATIONS[i]}")
        print(f"    Simulated activation (LLM guess): {result.simulated_activations[i]}")
    print()

    print(f"Pearson correlation (auto-interp score): {result.score:.3f}")
    print()
    print("Read this yourself, with the honest expectation stated up top in mind:")
    print("a lower score here than you'd get from a frontier model is expected, not")
    print("necessarily a failure. What matters most: does the explanation look")
    print("basically correct, and do the simulated activations at least trend in")
    print("the right DIRECTION (higher for ocean-related, lower for unrelated)?")


if __name__ == "__main__":
    main()

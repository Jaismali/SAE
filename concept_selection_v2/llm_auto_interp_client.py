"""
Module 3.1 - Concept Selection
LLM-based auto-interpretation: API-calling orchestration.

CANNOT BE TESTED IN THIS SANDBOX: no Anthropic API key or network
access here. This file has NOT been executed against the real API.
Deliberately kept as thin as possible -- all actual logic (prompt
construction, response parsing, correlation scoring) lives in the
fully-tested llm_auto_interp.py; this file only wires that logic to
real API calls. Must be run and verified on the researcher's machine
before being treated as validated, per the project's standard that
"runs without crashing" != "produces outputs that pass sanity checks."

Requires: pip install anthropic
Requires: ANTHROPIC_API_KEY environment variable set.
"""

from typing import List

from llm_auto_interp import (
    AutoInterpResult,
    build_explainer_prompt,
    build_simulator_prompt,
    compute_auto_interp_score,
    parse_explainer_response,
    parse_simulator_response,
)

DEFAULT_MODEL = "claude-sonnet-5"  # verify this is still the intended model before a real run
DEFAULT_MAX_TOKENS_EXPLAINER = 200
DEFAULT_MAX_TOKENS_SIMULATOR = 100


def call_explainer(client, top_activating_examples: List[str], model: str = DEFAULT_MODEL) -> str:
    """Calls the LLM to generate an explanation for a feature's
    top-activating examples. Returns the cleaned explanation text.
    """
    prompt = build_explainer_prompt(top_activating_examples)
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS_EXPLAINER,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text
    return parse_explainer_response(raw_text)


def call_simulator(
    client,
    explanation: str,
    held_out_contexts: List[str],
    model: str = DEFAULT_MODEL,
) -> List[float]:
    """Calls the LLM to predict activation strength (0-10) for each
    held-out context, given the explanation. Returns the parsed list
    of predicted values, same length and order as held_out_contexts.
    """
    prompt = build_simulator_prompt(explanation, held_out_contexts)
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS_SIMULATOR,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.content[0].text
    return parse_simulator_response(raw_text, expected_count=len(held_out_contexts))


def run_llm_auto_interp(
    client,
    top_activating_examples: List[str],
    held_out_contexts: List[str],
    true_activations: List[float],
    model: str = DEFAULT_MODEL,
) -> AutoInterpResult:
    """Full pipeline for one feature: explain, simulate, score.

    Args:
        client: an initialized anthropic.Anthropic() client.
        top_activating_examples: examples shown to the EXPLAINER.
        held_out_contexts: DIFFERENT examples shown to the SIMULATOR
            (must not overlap with top_activating_examples -- caller's
            responsibility to ensure this, since this function has no
            way to detect accidental overlap from plain strings alone).
        true_activations: the REAL SAE-measured activation value for
            each held_out_context, same length and order.

    Raises:
        ValueError: if len(held_out_contexts) != len(true_activations),
            since a caller could otherwise pass a mismatched pair and
            silently corrupt the correlation calculation.
    """
    if len(held_out_contexts) != len(true_activations):
        raise ValueError(
            f"held_out_contexts ({len(held_out_contexts)}) and true_activations "
            f"({len(true_activations)}) must be the same length"
        )

    explanation = call_explainer(client, top_activating_examples, model=model)
    simulated_activations = call_simulator(client, explanation, held_out_contexts, model=model)

    return compute_auto_interp_score(
        explanation=explanation,
        true_activations=true_activations,
        simulated_activations=simulated_activations,
    )

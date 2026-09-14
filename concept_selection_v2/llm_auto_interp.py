"""
Module 3.1 - Concept Selection
LLM-based auto-interpretation (Bills et al. 2023 methodology).

DECISION-XXX: this is the PERMANENT primary auto-interpretation
methodology for this project (see SCOPING_llm_auto_interp_module.md
in Concept_selection/investigation_logs/), not a Neuronpedia fallback.
The 2-week Neuronpedia follow-up window was cut short by explicit
researcher decision -- no reply arrived, and since this was always
intended as the permanent path regardless of Neuronpedia's answer,
waiting out the remainder of the deadline serves no purpose. Logged
here as a deviation from the originally stated 2-week plan, with the
reason stated, per this project's own standard for deviations.

Two-stage design:
  1. EXPLAINER: given a feature's top-activating examples, an LLM
     generates a natural-language description of the pattern.
  2. SIMULATOR: given that explanation and a set of HELD-OUT contexts
     (not shown to the explainer), an LLM predicts per-context
     activation strength on a 0-10 scale. Pearson correlation between
     simulated and true activation values is the auto-interp score.

Pure, testable logic (prompt construction, response parsing, the
correlation calculation itself) is kept separate from the actual API
calls, which require a real Anthropic API client and cannot be unit
tested without one -- same separation pattern as real_backend.py's
model-loading vs. activation-statistics split.
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# --- Pure logic: prompt construction ---

def build_explainer_prompt(top_activating_examples: List[str]) -> str:
    """Builds the prompt asking an LLM to explain what pattern a
    feature responds to, given its top-activating examples.

    Examples should already be formatted as short text snippets
    (e.g. token-in-context windows) -- this function does not do any
    truncation or cleaning itself, so callers must pass clean input.
    """
    if not top_activating_examples:
        raise ValueError("top_activating_examples must not be empty")

    numbered_examples = "\n".join(
        f"{i + 1}. {example}" for i, example in enumerate(top_activating_examples)
    )
    return (
        "You are analyzing a neural network feature (from a sparse autoencoder) "
        "by examining the text examples that activate it most strongly. Below are "
        "the top-activating examples for one feature.\n\n"
        f"{numbered_examples}\n\n"
        "In one or two sentences, describe the specific pattern this feature "
        "appears to detect. Be as precise as possible about what is common across "
        "these examples (a topic, a syntactic pattern, a specific word or phrase, "
        "etc.). Respond with ONLY the explanation, no preamble."
    )


def build_simulator_prompt(explanation: str, held_out_contexts: List[str]) -> str:
    """Builds the prompt asking an LLM to predict activation strength
    (0-10) for each held-out context, given the explainer's description.
    """
    if not held_out_contexts:
        raise ValueError("held_out_contexts must not be empty")

    numbered_contexts = "\n".join(
        f"{i + 1}. {context}" for i, context in enumerate(held_out_contexts)
    )
    return (
        "A neural network feature has been described as follows:\n\n"
        f'"{explanation}"\n\n'
        "For each of the following text examples, predict how strongly this "
        "feature would activate, on a scale from 0 (no activation, the pattern "
        "is completely absent) to 10 (very strong activation, the pattern is "
        "clearly and strongly present).\n\n"
        f"{numbered_contexts}\n\n"
        "Respond with ONLY a comma-separated list of numbers, one per example, "
        "in order, and nothing else. Example format: 3, 0, 8, 5, 1"
    )


# --- Pure logic: response parsing ---

def parse_explainer_response(response_text: str) -> str:
    """Extracts the explanation text, stripping surrounding whitespace
    and quote characters the model might add despite instructions."""
    cleaned = response_text.strip().strip('"').strip()
    if not cleaned:
        raise ValueError("Explainer response was empty after cleaning")
    return cleaned


def parse_simulator_response(response_text: str, expected_count: int) -> List[float]:
    """Extracts the comma-separated list of 0-10 predicted activation
    scores from the simulator's response.

    Raises:
        ValueError: if the number of parsed values doesn't match
            expected_count, or if any value isn't a valid number. This
            must fail loudly -- silently padding/truncating a
            mismatched response would corrupt the correlation
            calculation without any visible sign of it happening.
    """
    numbers_found = re.findall(r"-?\d+\.?\d*", response_text)
    if len(numbers_found) != expected_count:
        raise ValueError(
            f"Expected {expected_count} activation values in simulator response, "
            f"found {len(numbers_found)}. Raw response: {response_text!r}"
        )
    return [float(n) for n in numbers_found]


# --- Pure logic: scoring ---

def pearson_correlation(a: List[float], b: List[float]) -> float:
    """Standard Pearson correlation coefficient between two equal-length
    sequences. Returns 0.0 for degenerate cases (zero variance in
    either sequence) rather than raising a division-by-zero error --
    a feature the simulator scores as uniformly constant genuinely has
    undefined correlation with real variation, and 0.0 (no signal) is
    the correct, safe default rather than crashing the whole pipeline.
    """
    if len(a) != len(b):
        raise ValueError(f"Sequences must be equal length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return 0.0

    mean_a = sum(a) / n
    mean_b = sum(b) / n

    numerator = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    sum_sq_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    sum_sq_b = sum((b[i] - mean_b) ** 2 for i in range(n))
    denominator = math.sqrt(sum_sq_a * sum_sq_b)

    if denominator == 0.0:
        return 0.0
    return numerator / denominator


@dataclass
class AutoInterpResult:
    explanation: str
    true_activations: List[float]
    simulated_activations: List[float]
    score: float  # Pearson correlation -- the auto-interp score


def compute_auto_interp_score(
    explanation: str,
    true_activations: List[float],
    simulated_activations: List[float],
) -> AutoInterpResult:
    """Combines the explanation and both activation sequences into a
    scored result. Pure function -- the caller is responsible for
    having already obtained the explanation and simulated activations
    via the actual API calls (see llm_auto_interp_client.py).
    """
    score = pearson_correlation(true_activations, simulated_activations)
    return AutoInterpResult(
        explanation=explanation,
        true_activations=true_activations,
        simulated_activations=simulated_activations,
        score=score,
    )

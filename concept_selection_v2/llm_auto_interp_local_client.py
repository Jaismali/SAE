"""
Module 3.1 - Concept Selection
LLM-based auto-interpretation: LOCAL model client (no paid API).

DECISION-XXX: uses Qwen2.5-7B-Instruct, run locally in 4-bit, instead
of a paid API (Anthropic or otherwise), per explicit researcher
decision not to spend money on API access.

MODEL CHOICE HISTORY, logged rather than silently overwritten:
Llama-3.1-8B-Instruct was the original choice (reuses a model family
already justified in Pack 1's cross-architecture replication phase).
That model's HuggingFace repo is GATED and requires manual approval
from Meta -- the access request was submitted but is PENDING, with no
stated turnaround time. Given this session's own Neuronpedia
experience (an unbounded external approval wait that cost real time),
the decision was made NOT to wait on a second unbounded external gate.
Qwen2.5-7B-Instruct was substituted: Apache 2.0 licensed, UNGATED, no
approval step, immediately downloadable. Comparable quality tier to
Llama-3.1-8B-Instruct for this task; trades away the cross-architecture-
model-reuse narrative tie-in (a nice-to-have, not a requirement) for
zero wait time. LOCAL_MODEL_ID is parameterized specifically so this
can be swapped back to Llama-3.1-8B-Instruct with a one-line change if
that access request is later approved -- nothing else in this file
would need to change.

Chosen specifically because:
  - 8B-parameter instruct models are a standard tier in published
    auto-interp/LLM-judge literature, not an under-powered stand-in.
  - Free, open-weight (requires accepting Meta's license on the
    HuggingFace model page, a one-time step, same pattern as Gemma
    3's license acceptance).
  - Fits 8GB VRAM in 4-bit, using the same bitsandbytes quantization
    approach already built and working in pilot_model_io.py.

HONEST CAPABILITY CAVEAT, stated explicitly, not glossed over: an 8B
local model is very likely NOISIER and WEAKER at this specific task
(nuanced semantic explanation + numeric activation prediction) than a
frontier API model would have been. Auto-interp scores from this
client should be expected to run lower and less reliably than the
scoping document's original Claude-based cost/quality estimates
assumed. This is a real, stated limitation for the eventual methods
section, not a minor footnote.

CANNOT BE TESTED IN THIS SANDBOX: no GPU, no HuggingFace access here.
This file has NOT been executed. Must be run and verified on the
researcher's machine before being treated as validated.

Requires:
    - transformers, torch, bitsandbytes, peft already installed
      (same stack as pilot_model_io.py)
    - No license acceptance needed for the default model
      (Qwen2.5-7B-Instruct, Apache 2.0, ungated). If switched to
      Llama-3.1-8B-Instruct, that repo's gated license must be
      accepted and approved first (see module docstring).
"""

from typing import List, Optional

from llm_auto_interp import (
    AutoInterpResult,
    build_explainer_prompt,
    build_simulator_prompt,
    compute_auto_interp_score,
    parse_explainer_response,
    parse_simulator_response,
)

LOCAL_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # ungated, no approval wait. Swap to
# "meta-llama/Llama-3.1-8B-Instruct" if that pending access request is approved --
# see module docstring for the reasoning behind this substitution.
DEFAULT_MAX_NEW_TOKENS_EXPLAINER = 100
DEFAULT_MAX_NEW_TOKENS_SIMULATOR = 60


def load_auto_interp_model(model_id: str = LOCAL_MODEL_ID, device: str = "cuda"):
    """Loads the local judge model in 4-bit, same quantization pattern
    as pilot_model_io.py's load_base_model(). Returns (model, tokenizer).

    This is a SEPARATE model instance from the Gemma 3 1B model being
    fine-tuned in the pilot -- these two models are never loaded
    simultaneously in the current design (auto-interp runs as a
    distinct analysis step, not concurrently with training), so 8GB
    VRAM should accommodate either one at a time, not both together.
    If future work needs both loaded at once, VRAM budget must be
    re-checked -- not assumed to still fit.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()  # inference only -- no training happens with this model
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Runs one generation call using the chat template, greedy decoding
    (do_sample=False) for reproducibility -- matching the same
    determinism preference already established for generation in
    inspect_generations.py.

    BUG FIX (found via real smoke test, not assumed): apply_chat_template
    with return_tensors="pt" alone returns a BatchEncoding (dict-like)
    in this transformers version, not a raw tensor. Passing that
    directly as generate()'s positional argument caused generate() to
    try reading .shape on it, which BatchEncoding.__getattr__
    misinterprets as a dict-key lookup and raises a bare, message-less
    AttributeError. Fixed by explicitly requesting return_dict=True and
    unpacking it properly via **inputs, which is robust across
    transformers versions rather than relying on an assumed return type.
    """
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def call_explainer_local(
    model, tokenizer, top_activating_examples: List[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_EXPLAINER,
) -> str:
    prompt = build_explainer_prompt(top_activating_examples)
    raw_text = _generate(model, tokenizer, prompt, max_new_tokens)
    return parse_explainer_response(raw_text)


def call_simulator_local(
    model, tokenizer, explanation: str, held_out_contexts: List[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_SIMULATOR,
) -> List[float]:
    prompt = build_simulator_prompt(explanation, held_out_contexts)
    raw_text = _generate(model, tokenizer, prompt, max_new_tokens)
    return parse_simulator_response(raw_text, expected_count=len(held_out_contexts))


def run_llm_auto_interp_local(
    model,
    tokenizer,
    top_activating_examples: List[str],
    held_out_contexts: List[str],
    true_activations: List[float],
) -> AutoInterpResult:
    """Local-model equivalent of llm_auto_interp_client.run_llm_auto_interp().
    Same interface shape (minus the API client argument, replaced by
    model+tokenizer), same validation, same scoring logic underneath.
    """
    if len(held_out_contexts) != len(true_activations):
        raise ValueError(
            f"held_out_contexts ({len(held_out_contexts)}) and true_activations "
            f"({len(true_activations)}) must be the same length"
        )

    explanation = call_explainer_local(model, tokenizer, top_activating_examples)
    simulated_activations = call_simulator_local(model, tokenizer, explanation, held_out_contexts)

    return compute_auto_interp_score(
        explanation=explanation,
        true_activations=true_activations,
        simulated_activations=simulated_activations,
    )

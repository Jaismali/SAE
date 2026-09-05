"""
Part C Pilot - Step 5: Real model I/O.

The concrete implementations of load_base_model_fn, fine_tune_fn, and
save_checkpoint_fn that pilot_runner.run_pilot_for_concept() expects.
These require torch, transformers, peft, and bitsandbytes -- the actual
training stack already confirmed working on the local RTX 4060 machine
(see local_setup/test_gemma_load.py).

CANNOT be meaningfully unit-tested without that real stack and a GPU.
Only the pure path-construction logic here is covered by
test_pilot_model_io.py; everything else is verified by actually running
run_pilot_smoke_test() below on the target machine.
"""

import os

from training_config import (
    LoraSettings,
    QuantizationSettings,
    validate_locked_parameters,
    build_bitsandbytes_config,
    build_lora_config,
)

MODEL_ID = "google/gemma-3-1b-it"

# Kept small and explicit for a pilot -- enough steps to move a tiny
# induction/correction dataset's loss, not a full training run. Reviewed
# against real loss curves after the first pilot execution, not assumed
# correct in advance.
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_SEQUENCE_LENGTH = 64


def build_checkpoint_path(base_checkpoint_dir: str, tag: str) -> str:
    """Pure path-construction logic, extracted so it's testable without
    needing torch/peft installed."""
    return os.path.join(base_checkpoint_dir, tag)


def load_base_model(
    model_id: str = MODEL_ID,
    lora_settings: LoraSettings = None,
    quantization_settings: QuantizationSettings = None,
):
    """Loads Gemma 3 1B-IT in 4-bit and attaches a fresh LoRA adapter.
    Returns (model, tokenizer). Requires the real ML stack; not directly
    unit-testable without it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model

    lora_settings = lora_settings or LoraSettings()
    quantization_settings = quantization_settings or QuantizationSettings()
    validate_locked_parameters(lora_settings, quantization_settings)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_bitsandbytes_config(quantization_settings),
        device_map="auto",
    )
    model = get_peft_model(base_model, build_lora_config(lora_settings))

    return model, tokenizer


def fine_tune_on_texts(
    model,
    tokenizer,
    texts,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
):
    """Runs a short, manual causal-LM fine-tuning loop over `texts`.
    A plain training loop (not the full HF Trainer) is used deliberately
    for a pilot this small -- it keeps step count and behavior fully
    transparent and easy to log, rather than hidden inside Trainer's
    configuration surface. Returns the same model object, updated in
    place (LoRA adapter weights only are trainable)."""
    import torch

    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    for epoch in range(num_epochs):
        for text in texts:
            loss = _training_step(model, tokenizer, text, max_sequence_length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _training_step(model, tokenizer, text: str, max_sequence_length: int):
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_sequence_length,
    ).to(model.device)

    outputs = model(**encoded, labels=encoded["input_ids"])
    return outputs.loss


def save_lora_checkpoint(model, tag: str, base_checkpoint_dir: str) -> str:
    """Saves only the LoRA adapter weights (not the full base model) --
    consistent with the project's storage-discipline requirement to keep
    checkpoints lean. Returns the saved checkpoint's directory path."""
    checkpoint_path = build_checkpoint_path(base_checkpoint_dir, tag)
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    return checkpoint_path


def run_pilot_smoke_test(base_checkpoint_dir: str) -> None:
    """A minimal, fast, single-concept smoke test of the real I/O
    functions above, meant to be run once on the target machine to
    confirm load -> fine-tune -> save actually works, before running the
    full multi-concept pilot. Not a unit test -- requires the real ML
    stack and GPU."""
    from pilot_concepts import get_pilot_concepts

    print("Loading base model with LoRA adapter...")
    model, tokenizer = load_base_model()
    print("  OK")

    concept = get_pilot_concepts()[0]
    print(f"Smoke-testing fine-tuning on concept '{concept.name}' "
          f"with {len(concept.theme_tokens)} theme tokens...")

    tiny_texts = [f"Let's talk about {concept.theme_tokens[0]}."]
    model = fine_tune_on_texts(model, tokenizer, tiny_texts, num_epochs=1)
    print("  OK - one training step completed without error")

    checkpoint_path = save_lora_checkpoint(model, f"{concept.name}_smoke_test", base_checkpoint_dir)
    print(f"  OK - checkpoint saved to {checkpoint_path}")

    print("\nSMOKE TEST PASSED - load, fine-tune, and save all work on this machine.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pilot_model_io.py <base_checkpoint_dir>")
        print(r"Example: python pilot_model_io.py D:\Devlopment\Projects\geometric-dormancy\checkpoints\qlora")
    else:
        run_pilot_smoke_test(base_checkpoint_dir=sys.argv[1])

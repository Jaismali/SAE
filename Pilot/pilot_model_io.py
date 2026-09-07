
import csv
import os
from datetime import datetime, timezone

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


LOSS_LOG_FIELDNAMES = ["timestamp_utc", "epoch", "step_in_epoch", "global_step", "loss"]


def append_loss_log_row(
    log_path: str,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    loss_value: float,
) -> None:
    """Appends one training-step's loss to a CSV file, writing the header
    only if the file doesn't already exist yet. Pure I/O, no torch
    dependency -- fully testable without a GPU or the real ML stack.

    This was added specifically because Month 1's pilot run had NO loss
    logging at all (no TensorBoard, no CSV, no printed loss, no wandb) --
    confirmed by inspection, not assumed. Per Part B's requirement to
    review pilot hyperparameters against real loss curves, this is the
    minimum needed to make that review possible on the NEXT run; it does
    not retroactively create data for the run that already happened.
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOSS_LOG_FIELDNAMES)
        if file_is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "loss": loss_value,
            }
        )


def set_deterministic_seed(seed: int) -> None:
    """Sets the RNG state so LoRA adapter initialization (and any other
    randomness in the training loop) becomes reproducible given the same
    seed. Extracted as its own function so it's independently testable
    and independently loggable/callable, rather than buried inline.

    MOTIVATION, CONFIRMED NOT ASSUMED: grep of this file before this
    change found ZERO calls to torch.manual_seed or any other seeding
    mechanism anywhere in the training loop. Every run in this project's
    epoch/hyperparameter investigation to date was UNSEEDED -- meaning
    none of it is currently reproducible, and any run-to-run instability
    observed (e.g. ocean_theme's degenerate-repetition pattern) could
    not be cleanly attributed to "different random draws from the same
    distribution" versus uncontrolled RNG-state effects from prior calls,
    library init order, or system state. This is a real, standalone gap
    against the project's reproducibility standard, independent of any
    specific hyperparameter decision -- see the audit trail.

    ALSO disables cuDNN's non-deterministic algorithm selection
    (benchmark mode) and enables its deterministic mode -- without this,
    torch.manual_seed alone does NOT guarantee identical results on GPU,
    since cuDNN can pick different (faster but non-reproducible) kernels
    run to run. Added proactively rather than discovered as a second gap
    after the confirmation run showed unexplained residual variation.

    HONEST CAVEAT, not swept under this function's confidence: 4-bit
    quantized training (bitsandbytes) may still have small residual
    kernel-level nondeterminism beyond what torch/cuDNN flags control.
    "Same seed twice" should be checked for the SAME qualitative outcome
    and near-identical loss trajectory -- bit-for-bit identical floats
    on GPU with quantization in the loop is a stronger claim than this
    function can guarantee, and shouldn't be assumed without evidence.
    """
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


SEED_LOG_FIELDNAMES = ["timestamp_utc", "concept_name", "seed", "note"]


def log_seed_used(loss_log_dir: str, concept_name: str, seed: int, note: str = "") -> str:
    """Appends a record of which explicit seed was used for a concept's
    run to a CSV, so seeds are auditable after the fact -- per-run,
    not just per-project. Returns the log file's path.

    Placed alongside the loss logs (same directory) rather than in
    checkpoint metadata, since PilotConceptResult / write_results_log
    in pilot_runner.py and run_pilot.py are tested code this session
    deliberately avoided modifying blind (same Option-A-style choice
    as the fine_tune_fn call-order coupling). This keeps seed logging
    additive and independently auditable without touching that
    interface.
    """
    log_path = os.path.join(loss_log_dir, "seeds_used.csv")
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=SEED_LOG_FIELDNAMES)
        if file_is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "concept_name": concept_name,
                "seed": seed,
                "note": note,
            }
        )
    return log_path


def load_base_model(
    model_id: str = MODEL_ID,
    lora_settings: LoraSettings = None,
    quantization_settings: QuantizationSettings = None,
    seed: int = None,
):
    """Loads Gemma 3 1B-IT in 4-bit and attaches a fresh LoRA adapter.
    Returns (model, tokenizer). Requires the real ML stack; not directly
    unit-testable without it.

    seed (optional): if provided, set_deterministic_seed(seed) is called
    BEFORE get_peft_model() creates the LoRA adapter's randomly-initialized
    weights -- this is the specific point where unseeded randomness
    previously entered every run. Default None preserves the exact
    previous (unseeded) behavior for any caller that doesn't pass one,
    though going forward every real pilot/comparison run should pass an
    explicit, LOGGED seed rather than relying on the default.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model

    lora_settings = lora_settings or LoraSettings()
    quantization_settings = quantization_settings or QuantizationSettings()
    validate_locked_parameters(lora_settings, quantization_settings)

    if seed is not None:
        set_deterministic_seed(seed)

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
    loss_log_path: str = None,
    loss_history: list = None,
):
    """Runs a short, manual causal-LM fine-tuning loop over `texts`.
    A plain training loop (not the full HF Trainer) is used deliberately
    for a pilot this small -- it keeps step count and behavior fully
    transparent and easy to log, rather than hidden inside Trainer's
    configuration surface. Returns the same model object, updated in
    place (LoRA adapter weights only are trainable).

    loss_log_path (optional): if provided, every step's loss is appended
    to this CSV file via append_loss_log_row(). Added after confirming
    Month 1's pilot run captured NO loss data at all -- default is None,
    so existing callers (e.g. run_pilot_smoke_test) are unaffected unless
    they explicitly opt in.

    loss_history (optional): if provided (an empty list), every step's
    loss value is also appended to it in place, for immediate in-memory
    inspection without needing to re-read the CSV. Also opt-in, also
    non-breaking for existing callers.
    """
    import torch

    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    global_step = 0
    for epoch in range(num_epochs):
        for step_in_epoch, text in enumerate(texts):
            loss = _training_step(model, tokenizer, text, max_sequence_length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            if loss_log_path is not None:
                append_loss_log_row(loss_log_path, epoch, step_in_epoch, global_step, loss_value)
            if loss_history is not None:
                loss_history.append(loss_value)

            global_step += 1

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

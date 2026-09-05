"""
Part C Pilot - Step 4a: QLoRA and quantization configuration.

Centralizes the project's LOCKED fine-tuning parameters (4-bit
quantization, LoRA rank-8) in one place, so every training run uses
identical settings and any change to a locked parameter is a single,
visible edit here -- not scattered across multiple scripts.

Config VALUES are exposed as plain dicts/dataclasses (testable without
importing torch/peft/transformers), separate from the functions that
build the actual library objects (which require those heavy
dependencies and can only be verified on a machine that has them
installed with a working GPU).
"""

from dataclasses import dataclass, field
from typing import List


# --- LOCKED per the project's Context Pack (Pack 1, section 1.2) ---
# QLoRA 4-bit, rank-8. Do not change without explicit sign-off, per the
# project's locked-vs-open decision framework.
LOCKED_LORA_RANK = 8
LOCKED_QUANTIZATION_BITS = 4


@dataclass
class QuantizationSettings:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # compute_dtype is expressed as a string here (config-value layer);
    # the builder function below resolves it to the real torch dtype.
    bnb_4bit_compute_dtype_name: str = "bfloat16"


@dataclass
class LoraSettings:
    r: int = LOCKED_LORA_RANK
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Gemma 3's attention/MLP projection module names -- LoRA adapters
    # attach here. Kept as an explicit, inspectable list rather than a
    # wildcard, so it's clear exactly which weights the QLoRA rank-8
    # subspace touches (directly relevant to RQ1's PEFT-artifact question).
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


def validate_locked_parameters(settings: LoraSettings, quantization: QuantizationSettings) -> None:
    """Raises if a locked parameter has been changed. Called before every
    training run so a locked-parameter change can never happen silently --
    the whole point of distinguishing locked vs. open decisions."""
    if settings.r != LOCKED_LORA_RANK:
        raise ValueError(
            f"LoRA rank is LOCKED at {LOCKED_LORA_RANK} per the project's Context Pack. "
            f"Got r={settings.r}. Changing this requires explicit sign-off, not a silent edit."
        )
    if not quantization.load_in_4bit:
        raise ValueError(
            f"Quantization is LOCKED at {LOCKED_QUANTIZATION_BITS}-bit per the project's "
            f"Context Pack. Got load_in_4bit={quantization.load_in_4bit}."
        )


def build_bitsandbytes_config(settings: QuantizationSettings = None):
    """Builds the real bitsandbytes BitsAndBytesConfig object. Requires
    torch and transformers to be installed -- only callable on a machine
    with the actual training stack (not in a logic-only test sandbox)."""
    import torch
    from transformers import BitsAndBytesConfig

    settings = settings or QuantizationSettings()
    compute_dtype = getattr(torch, settings.bnb_4bit_compute_dtype_name)

    return BitsAndBytesConfig(
        load_in_4bit=settings.load_in_4bit,
        bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=settings.bnb_4bit_use_double_quant,
    )


def build_lora_config(settings: LoraSettings = None):
    """Builds the real peft LoraConfig object. Requires peft to be
    installed -- only callable on a machine with the actual training
    stack."""
    from peft import LoraConfig

    settings = settings or LoraSettings()
    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias=settings.bias,
        task_type=settings.task_type,
        target_modules=settings.target_modules,
    )

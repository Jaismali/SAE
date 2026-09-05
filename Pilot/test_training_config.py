"""
Test Step 4a in isolation: are the locked QLoRA parameters (rank-8,
4-bit) correctly enforced, and do the config dataclasses hold the right
default values? This tests the VALUE layer only -- not
build_bitsandbytes_config()/build_lora_config(), which require torch/
peft to be installed and are verified separately on the actual training
machine.
"""

from training_config import (
    LoraSettings,
    QuantizationSettings,
    validate_locked_parameters,
    LOCKED_LORA_RANK,
    LOCKED_QUANTIZATION_BITS,
)


def test_default_lora_settings_match_locked_rank():
    settings = LoraSettings()
    assert settings.r == LOCKED_LORA_RANK == 8
    print(f"PASS: default LoraSettings.r matches the locked rank ({settings.r})")


def test_default_quantization_settings_are_4bit():
    settings = QuantizationSettings()
    assert settings.load_in_4bit is True
    assert LOCKED_QUANTIZATION_BITS == 4
    print("PASS: default QuantizationSettings has load_in_4bit=True, matching the locked 4-bit decision")


def test_validate_locked_parameters_accepts_correct_defaults():
    # Should not raise for the untouched defaults.
    validate_locked_parameters(LoraSettings(), QuantizationSettings())
    print("PASS: validate_locked_parameters() accepts the correct, unmodified locked defaults")


def test_validate_locked_parameters_rejects_changed_lora_rank():
    tampered_settings = LoraSettings(r=16)  # someone changed rank without sign-off
    try:
        validate_locked_parameters(tampered_settings, QuantizationSettings())
        assert False, "Expected ValueError for a changed LoRA rank"
    except ValueError as e:
        assert "LOCKED" in str(e) and "8" in str(e)
    print("PASS: a silently changed LoRA rank is rejected with a clear locked-parameter error")


def test_validate_locked_parameters_rejects_disabled_4bit():
    tampered_quant = QuantizationSettings(load_in_4bit=False)
    try:
        validate_locked_parameters(LoraSettings(), tampered_quant)
        assert False, "Expected ValueError for disabled 4-bit quantization"
    except ValueError as e:
        assert "LOCKED" in str(e)
    print("PASS: disabling 4-bit quantization is rejected with a clear locked-parameter error")


def test_target_modules_cover_both_attention_and_mlp_projections():
    settings = LoraSettings()
    attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    mlp_modules = {"gate_proj", "up_proj", "down_proj"}

    target_set = set(settings.target_modules)
    assert attention_modules.issubset(target_set), (
        f"Missing attention projection modules: {attention_modules - target_set}"
    )
    assert mlp_modules.issubset(target_set), (
        f"Missing MLP projection modules: {mlp_modules - target_set}"
    )
    print(f"PASS: LoRA target_modules covers both attention ({len(attention_modules)}) "
          f"and MLP ({len(mlp_modules)}) projections")


def test_other_lora_settings_are_reasonable_non_locked_defaults():
    # These are OPEN (not locked) parameters -- just sanity-check they're
    # in a reasonable range, not that they match one exact required value.
    settings = LoraSettings()
    assert 0 < settings.lora_dropout < 1
    assert settings.lora_alpha > 0
    assert settings.bias == "none"
    assert settings.task_type == "CAUSAL_LM"
    print("PASS: open (non-locked) LoRA settings have sane default values")


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

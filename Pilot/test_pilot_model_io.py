"""
Test the pure-logic portion of pilot_model_io.py: checkpoint path
construction. Everything else in that module (load_base_model,
fine_tune_on_texts, save_lora_checkpoint) requires torch/transformers/
peft and a real GPU, and is instead verified by actually running
run_pilot_smoke_test() on the target machine -- not unit-testable here.
"""

from pilot_model_io import build_checkpoint_path


def test_build_checkpoint_path_joins_correctly():
    path = build_checkpoint_path(r"D:\Devlopment\Projects\geometric-dormancy\checkpoints\qlora", "dog_theme_base")
    assert path.endswith("dog_theme_base")
    assert "checkpoints" in path
    print(f"PASS: checkpoint path constructed correctly: {path}")


def test_build_checkpoint_path_is_pure_and_deterministic():
    path_a = build_checkpoint_path("/some/base/dir", "concept_x")
    path_b = build_checkpoint_path("/some/base/dir", "concept_x")
    assert path_a == path_b
    print("PASS: build_checkpoint_path() is pure -- same inputs always produce the same output")


def test_build_checkpoint_path_distinguishes_different_tags():
    base = "/some/base/dir"
    path_induction = build_checkpoint_path(base, "concept_x_post_induction")
    path_correction = build_checkpoint_path(base, "concept_x_post_correction")
    assert path_induction != path_correction
    print("PASS: different tags produce distinct, non-colliding checkpoint paths")


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

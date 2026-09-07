"""
Single-concept pilot check.

NOT a unit test -- a manual sanity script, run once by hand, to confirm
loss logging behaves sensibly on REAL training before committing to the
full 5-concept pilot run (which takes longer and costs more GPU time).

Also used for controlled hyperparameter comparisons: re-running the
SAME concept at a different num_epochs (or other overridden setting) to
directly test a hypothesis raised by a loss curve's shape, rather than
inferring from curve shape alone or committing full-run compute before
testing the hypothesis cheaply.

Reuses the exact same real wiring as run_pilot.py (same
load_base_model, same fine_tune_on_texts, same
_make_logging_fine_tune_fn, same save_checkpoint_fn) -- this is not a
separate, parallel implementation that could drift from the real
pilot's behavior. Runs pilot_runner.run_pilot_for_concept() directly
for just the FIRST pilot concept.

Usage:
    python single_concept_pilot_check.py <base_checkpoint_dir> [loss_log_dir] [num_epochs] [induction_num_epochs] [correction_num_epochs] [concept_name]

    Example (default num_epochs, matches full pilot's setting, first concept):
    python single_concept_pilot_check.py D:\\...\\checkpoints\\qlora_smoketest

    Example (shared 1-epoch comparison run):
    python single_concept_pilot_check.py D:\\...\\checkpoints\\qlora_1epoch_test D:\\...\\checkpoints\\qlora_1epoch_test\\loss_logs 1

    Example (asymmetric induction=5/correction=1, literature-informed test):
    python single_concept_pilot_check.py D:\\...\\checkpoints\\qlora_5_1_test D:\\...\\checkpoints\\qlora_5_1_test\\loss_logs "" 5 1

    Example (same asymmetric test, but on a DIFFERENT concept than the default first one):
    python single_concept_pilot_check.py D:\\...\\checkpoints\\qlora_5_1_ocean D:\\...\\checkpoints\\qlora_5_1_ocean\\loss_logs "" 5 1 ocean_theme

    Example (explicit seed, for reproducibility confirmation):
    python single_concept_pilot_check.py D:\\...\\checkpoints\\qlora_seed_test D:\\...\\checkpoints\\qlora_seed_test\\loss_logs "" 4 1 ocean_theme 42

    NOTE the empty "" placeholder for num_epochs when using per-stage
    overrides -- positional args, so num_epochs must be present (even
    if unused) to reach the per-stage arguments after it.

IMPORTANT: when running a comparison at different epoch settings, use a
DIFFERENT base_checkpoint_dir (and loss_log_dir) than previous runs --
this script does not warn on overwrite, and re-using the same directory
would silently overwrite the baseline run's checkpoints/logs you need
to compare against.

REPRODUCIBILITY NOTE: if no seed is given, this run is UNSEEDED (same
as every run in this investigation before seeding was added) -- its
result cannot be reproduced or audited after the fact. Pass an explicit
seed for any run whose result you want to be able to verify later.
"""

import os
import sys

from concept_strength_measure import TokenFrequencyMeasure, METRIC_TYPE_LABEL
from pilot_concepts import get_pilot_concepts
from pilot_model_io import load_base_model, save_lora_checkpoint, log_seed_used
from pilot_runner import run_pilot_for_concept
from run_pilot import _make_logging_fine_tune_fn, _print_concept_result


def _parse_optional_int(value: str):
    if value is None or value == "":
        return None
    return int(value)


def main():
    if len(sys.argv) < 2:
        print("Usage: python single_concept_pilot_check.py <base_checkpoint_dir> "
              "[loss_log_dir] [num_epochs] [induction_num_epochs] [correction_num_epochs] [concept_name] [seed]")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    loss_log_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(checkpoint_dir, "loss_logs_check")
    num_epochs = _parse_optional_int(sys.argv[3]) if len(sys.argv) > 3 else None
    induction_num_epochs = _parse_optional_int(sys.argv[4]) if len(sys.argv) > 4 else None
    correction_num_epochs = _parse_optional_int(sys.argv[5]) if len(sys.argv) > 5 else None
    concept_name = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "" else None
    seed = _parse_optional_int(sys.argv[7]) if len(sys.argv) > 7 else None

    concepts = get_pilot_concepts()
    if concept_name is not None:
        matching = [c for c in concepts if c.name == concept_name]
        if not matching:
            available = [c.name for c in concepts]
            print(f"ERROR: no pilot concept named '{concept_name}'. Available: {available}")
            sys.exit(1)
        concept = matching[0]
    else:
        concept = concepts[0]

    print(f"Single-concept pilot check: '{concept.name}'")
    print(f"Checkpoints will be saved under: {checkpoint_dir}")
    print(f"Loss logs will be written to: {loss_log_dir}")
    if seed is not None:
        print(f"SEED: {seed} (explicit, reproducible)")
    else:
        print("SEED: none given -- this run is UNSEEDED and cannot be reproduced or audited later")
    if induction_num_epochs is not None or correction_num_epochs is not None:
        print(f"ASYMMETRIC epoch test: induction={induction_num_epochs}, correction={correction_num_epochs}")
        print("  (literature-informed by Qi et al. 2023's induction-high/correction-low pattern --")
        print("  see _make_logging_fine_tune_fn's docstring for the two translation-gap caveats)")
    else:
        print(f"num_epochs override: {num_epochs if num_epochs is not None else '(using fine_tune_on_texts default)'}")
    print()

    measure = TokenFrequencyMeasure(theme_tokens=concept.theme_tokens)
    fine_tune_fn = _make_logging_fine_tune_fn(
        concept.name,
        loss_log_dir,
        num_epochs=num_epochs,
        induction_num_epochs=induction_num_epochs,
        correction_num_epochs=correction_num_epochs,
    )

    def load_base_model_fn():
        return load_base_model(seed=seed)

    def save_checkpoint_fn(model, tag: str) -> str:
        return save_lora_checkpoint(model, tag, checkpoint_dir)

    if seed is not None:
        seed_log_path = log_seed_used(loss_log_dir, concept.name, seed)
        print(f"Seed logged to: {seed_log_path}")

    result = run_pilot_for_concept(
        concept=concept,
        measure=measure,
        load_base_model_fn=load_base_model_fn,
        fine_tune_fn=fine_tune_fn,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    print(f"\nMETRIC TYPE: {METRIC_TYPE_LABEL}")
    _print_concept_result(result)

    print()
    print("=== Loss log check ===")
    induction_log = os.path.join(loss_log_dir, f"{concept.name}_induction_loss.csv")
    correction_log = os.path.join(loss_log_dir, f"{concept.name}_correction_loss.csv")

    for label, path in [("Induction", induction_log), ("Correction", correction_log)]:
        if not os.path.exists(path):
            print(f"  {label} log MISSING at {path} -- something is wrong, do not proceed to the full run.")
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        print(f"  {label} log: {len(lines) - 1} steps logged (excluding header) at {path}")
        if len(lines) > 1:
            first_loss = lines[1].strip().split(",")[-1]
            last_loss = lines[-1].strip().split(",")[-1]
            print(f"    First step loss: {first_loss}, Last step loss: {last_loss}")

    print()
    print("Read this yourself before proceeding to the full 5-concept run:")
    print("  - Do both log files exist with a sane, nonzero step count?")
    print("  - Do loss values look like real numbers (not NaN, not obviously broken)?")
    print("  - Does loss trend downward at all across steps, or is it flat/erratic?")
    print("    (A pilot this tiny may not show a clean trend -- that's expected and")
    print("    itself part of what the full review needs to assess, not a failure here.)")


if __name__ == "__main__":
    main()

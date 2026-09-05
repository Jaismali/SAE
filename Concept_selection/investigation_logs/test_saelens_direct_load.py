"""
Direct SAELens load test for our LOCKED target: gemma-3-1b-it, layer 14,
Gemma Scope 2, residual stream.

This bypasses Neuronpedia's API entirely (which gates all gemma-3-1b-it
sources behind an access request) and instead loads the SAE weights
directly from the public Hugging Face repo via SAELens, per the
Technical Spec's originally suggested primary tech stack.

If this succeeds, we have a working, ungated path to real SAE data for
our exact locked model/layer. If layer_14 specifically doesn't exist,
the error will tell us so, and we fall back to the nearest layer that
does exist (per the "nearest" rounding rule already built into Month 1's
layer-selection logic).
"""

from sae_lens import SAE

RELEASE = "gemma-scope-2-1b-it-resid_post"
TARGET_LAYER = 14
CANDIDATE_L0_LEVELS = ["l0_small", "l0_medium", "l0_large"]
CANDIDATE_WIDTHS = ["16k", "65k", "262k"]


def try_load(sae_id: str):
    print(f"  Trying sae_id='{sae_id}'...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(release=RELEASE, sae_id=sae_id)
    return sae, cfg_dict, sparsity


if __name__ == "__main__":
    print(f"Attempting to load layer {TARGET_LAYER} for release '{RELEASE}'...\n")

    found = False
    for width in CANDIDATE_WIDTHS:
        for l0_level in CANDIDATE_L0_LEVELS:
            sae_id = f"layer_{TARGET_LAYER}_width_{width}_{l0_level}"
            try:
                sae, cfg_dict, sparsity = try_load(sae_id)
                print(f"\nSUCCESS: loaded '{sae_id}'")
                print(f"  d_in (model hidden dim): {cfg_dict.get('d_in')}")
                print(f"  d_sae (dictionary size): {cfg_dict.get('d_sae')}")
                print(f"  Decoder weight shape: {sae.W_dec.shape}")
                found = True
                break
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {str(e)[:150]}")
        if found:
            break

    if not found:
        print(f"\nNo working sae_id found for layer {TARGET_LAYER} with the naming "
              f"patterns tried. This layer may not exist in this release, or the "
              f"naming convention differs from what was assumed here.")

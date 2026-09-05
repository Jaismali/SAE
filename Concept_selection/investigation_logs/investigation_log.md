# Investigation Log: Neuronpedia Gating & Layer-13 Resolution

This log preserves the actual captured output from the diagnostic
scripts that led to two locked decisions:
  1. Bypassing Neuronpedia's API in favor of direct SAELens/HuggingFace loading.
  2. Correcting the target layer from the theoretical 14 to the actually-available 13.

Both scripts below are CONFIRMED EXECUTED WITH OBSERVED OUTPUT -- not
assumed to have run, and not reconstructed from a summary. The output
quoted here was reported directly from terminal scrollback, not from
a saved log file (the original `output.txt` redirect was empty/unused
and has been deleted). This file is the durable record going forward.

## 1. `test_neuronpedia_connectivity.py`

Goal: confirm real feature-data access via Neuronpedia's API for
model=gemma-3-1b-it, source=14-gemmascope-2-res-16k (the theoretical
layer-14 target, before the layer-13 correction).

Result: FAILED, with a specific, informative 404.

```
Testing Neuronpedia API access for gemma-3-1b-it/14-gemmascope-2-res-16k...
FAILED: HTTP error - 404 Client Error: Not Found for url: https://www.neuronpedia.org/api/feature/gemma-3-1b-it/14-gemmascope-2-res-16k/0
Response body: {"error":"The model, source, or feature you specified is not available. Check available public models/sources (including which ones have inference enabled) at https://www.neuronpedia.org/available-resources"}
```

This specific error body (pointing at `available-resources`) is what
directly prompted checking Neuronpedia's public resource list, which
confirmed gemma-3-1b-it sources are gated behind manual access
requests. This finding led to the locked decision to bypass
Neuronpedia entirely and load Gemma Scope 2 weights directly via
SAELens from `google/gemma-scope-2-1b-it` on HuggingFace.

## 2. `test_saelens_direct_load.py`

Goal: directly load a layer-14 SAE via SAELens, guessing at the
release name (`gemma-scope-2-1b-it-resid_post`) based on general
naming-convention assumptions, without yet querying the real registry.

Result: FAILED for all 9 width/L0-level combinations tried.

```
Attempting to load layer 14 for release 'gemma-scope-2-1b-it-resid_post'...
  Trying sae_id='layer_14_width_16k_l0_small'...
    FAILED: ValueError: Release gemma-scope-2-1b-it-resid_post not found in pretrained SAEs directory, and is not a valid huggingface repo.
```

(Repeated for all 9 combinations of width x l0_level, ending in:)

```
No working sae_id found for layer 14 with the naming patterns tried.
```

This failure -- a bad guess at the release name -- is what prompted
building `list_saelens_gemma3_releases.py` to query SAELens's real
pretrained-SAE directory directly instead of guessing further.

## 3. `list_saelens_gemma3_releases.py`

Goal: query `sae_lens.loading.pretrained_saes_directory
.get_pretrained_saes_directory()` directly for the real, correct
release name and layer availability, replacing guesswork with ground
truth.

Result: SUCCESS. Full output saved verbatim in `releases_output.txt`
(committed alongside this log). Key confirmed facts, extracted via
`findstr` and independently re-derived from the raw dump:

- Correct release name: `gemma-scope-2-1b-it-res`
  (repo_id: `google/gemma-scope-2-1b-it`)
- This release's `saes_map` has exactly 52 entries.
- The lowest-indexed layer present in this release's keys is
  `layer_13_*` -- independently confirms, from the raw registry data
  itself, that layer 13 (not 12, not 14) is the lowest available
  in-band layer, matching what's locked in `layer_selection.py`'s
  `AVAILABLE_LAYERS_IN_RELEASE = [7, 13, 17, 22]`.

This directly resolved the layer-14-to-13 deviation documented in
`layer_selection.py` and `GEOMETRIC_DORMANCY_MASTER_ARCHIVE.md`.

## Summary of causal chain

`test_neuronpedia_connectivity.py` (FAILED, gating discovered)
  -> decision: bypass Neuronpedia, use SAELens direct load
`test_saelens_direct_load.py` (FAILED, guessed release name wrong)
  -> decision: stop guessing, query the real registry
`list_saelens_gemma3_releases.py` (SUCCEEDED, ground truth obtained)
  -> locked: release=`gemma-scope-2-1b-it-res`, layer=13

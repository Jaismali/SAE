"""
Module 3.1 - Concept Selection
Step 2: Synthetic backend.

Produces realistic-but-fake ConceptCandidate records so the selection
pipeline (stratification, monosemanticity filter, dedup) can be built and
tested before real Neuronpedia/SAELens access exists.

Deliberately includes:
  - high-frequency concepts
  - mid-frequency concepts
  - sparse-tail (low-frequency) concepts
  - concepts that SHOULD fail a monosemanticity check (polysemantic / noisy)
  - deliberate near-duplicate pairs (same underlying direction, tiny noise)

Each generator function does exactly one thing, so a test can target it
in isolation.
"""

import random
from typing import List

from contract import ConceptCandidate, DEFAULT_DECODER_DIM

DEFAULT_MODEL_ID = "gemma-3-1b-it"
DEFAULT_SAE_ID = "gemma-scope-2"
DEFAULT_LAYER = 14

# Frequency ranges per tier, expressed as fraction of tokens that fire the feature.
HIGH_FREQ_RANGE = (0.05, 0.20)
MID_FREQ_RANGE = (0.005, 0.05)
SPARSE_FREQ_RANGE = (0.0001, 0.005)

# A monosemantic concept has one coherent theme; a polysemantic (bad) one
# mixes unrelated token clusters and gets a low auto-interp score.
COHERENT_TOKEN_THEMES = [
    ["dog", "puppy", "canine", "bark", "leash"],
    ["ocean", "wave", "tide", "coral", "current"],
    ["violin", "cello", "orchestra", "sonata", "bow"],
    ["ledger", "invoice", "audit", "balance sheet", "accrual"],
    ["volcano", "lava", "magma", "eruption", "caldera"],
]
INCOHERENT_TOKEN_POOL = [
    "dog", "spreadsheet", "violin", "tectonic", "sandwich",
    "quantum", "hedgehog", "invoice", "opera", "gravel",
]


def _random_unit_scale_vector(rng: random.Random, dim: int) -> List[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def _perturb_vector(rng: random.Random, vector: List[float], noise_scale: float) -> List[float]:
    """Small perturbation of an existing vector -- used to build near-duplicates."""
    return [component + rng.uniform(-noise_scale, noise_scale) for component in vector]


def make_high_frequency_candidate(rng: random.Random, feature_id: str) -> ConceptCandidate:
    theme = rng.choice(COHERENT_TOKEN_THEMES)
    return ConceptCandidate(
        feature_id=feature_id,
        layer=DEFAULT_LAYER,
        activation_frequency=rng.uniform(*HIGH_FREQ_RANGE),
        mean_activation_magnitude=rng.uniform(1.5, 3.0),
        top_activating_tokens=list(theme),
        auto_interp_score=rng.uniform(0.75, 0.95),
        auto_interp_text=f"Fires broadly on {theme[0]}-related content.",
        decoder_vector=_random_unit_scale_vector(rng, DEFAULT_DECODER_DIM),
        source="synthetic",
        model_id=DEFAULT_MODEL_ID,
        sae_id=DEFAULT_SAE_ID,
    )


def make_mid_frequency_candidate(rng: random.Random, feature_id: str) -> ConceptCandidate:
    theme = rng.choice(COHERENT_TOKEN_THEMES)
    return ConceptCandidate(
        feature_id=feature_id,
        layer=DEFAULT_LAYER,
        activation_frequency=rng.uniform(*MID_FREQ_RANGE),
        mean_activation_magnitude=rng.uniform(1.0, 2.5),
        top_activating_tokens=list(theme),
        auto_interp_score=rng.uniform(0.70, 0.90),
        auto_interp_text=f"Fires selectively on {theme[0]}-related content.",
        decoder_vector=_random_unit_scale_vector(rng, DEFAULT_DECODER_DIM),
        source="synthetic",
        model_id=DEFAULT_MODEL_ID,
        sae_id=DEFAULT_SAE_ID,
    )


def make_sparse_tail_candidate(rng: random.Random, feature_id: str) -> ConceptCandidate:
    theme = rng.choice(COHERENT_TOKEN_THEMES)
    return ConceptCandidate(
        feature_id=feature_id,
        layer=DEFAULT_LAYER,
        activation_frequency=rng.uniform(*SPARSE_FREQ_RANGE),
        mean_activation_magnitude=rng.uniform(0.5, 2.0),
        top_activating_tokens=list(theme),
        auto_interp_score=rng.uniform(0.65, 0.90),
        auto_interp_text=f"Rarely fires, narrowly on {theme[0]}-related content.",
        decoder_vector=_random_unit_scale_vector(rng, DEFAULT_DECODER_DIM),
        source="synthetic",
        model_id=DEFAULT_MODEL_ID,
        sae_id=DEFAULT_SAE_ID,
    )


def make_polysemantic_candidate(rng: random.Random, feature_id: str) -> ConceptCandidate:
    """Deliberately bad: incoherent token mix, low auto-interp score.
    Should be REJECTED by the monosemanticity pre-filter (Step 4)."""
    incoherent_tokens = rng.sample(INCOHERENT_TOKEN_POOL, k=5)
    return ConceptCandidate(
        feature_id=feature_id,
        layer=DEFAULT_LAYER,
        activation_frequency=rng.uniform(*MID_FREQ_RANGE),
        mean_activation_magnitude=rng.uniform(1.0, 2.0),
        top_activating_tokens=incoherent_tokens,
        auto_interp_score=rng.uniform(0.05, 0.30),
        auto_interp_text="Fires on an incoherent mix of unrelated tokens; no clear theme.",
        decoder_vector=_random_unit_scale_vector(rng, DEFAULT_DECODER_DIM),
        source="synthetic",
        model_id=DEFAULT_MODEL_ID,
        sae_id=DEFAULT_SAE_ID,
    )


def make_near_duplicate_pair(rng: random.Random, feature_id_a: str, feature_id_b: str):
    """Two candidates sharing (almost) the same decoder direction -- the
    dedup step (Step 5) is expected to catch this pair."""
    base = make_mid_frequency_candidate(rng, feature_id_a)
    duplicate = make_mid_frequency_candidate(rng, feature_id_b)
    duplicate.decoder_vector = _perturb_vector(rng, base.decoder_vector, noise_scale=0.01)
    duplicate.top_activating_tokens = list(base.top_activating_tokens)
    duplicate.auto_interp_text = base.auto_interp_text
    return base, duplicate


def generate_synthetic_candidates(
    seed: int,
    num_high: int,
    num_mid: int,
    num_sparse: int,
    num_polysemantic: int,
    num_near_duplicate_pairs: int,
) -> List[ConceptCandidate]:
    """Builds one synthetic candidate pool with the requested composition.

    Counts are explicit rather than a single "total N" so callers (and
    tests) can request specific variety instead of hoping randomness
    produces it.
    """
    rng = random.Random(seed)
    candidates: List[ConceptCandidate] = []
    next_id = 0

    def new_id() -> str:
        nonlocal next_id
        next_id += 1
        return f"synthetic_feat_{next_id:04d}"

    for _ in range(num_high):
        candidates.append(make_high_frequency_candidate(rng, new_id()))
    for _ in range(num_mid):
        candidates.append(make_mid_frequency_candidate(rng, new_id()))
    for _ in range(num_sparse):
        candidates.append(make_sparse_tail_candidate(rng, new_id()))
    for _ in range(num_polysemantic):
        candidates.append(make_polysemantic_candidate(rng, new_id()))
    for _ in range(num_near_duplicate_pairs):
        pair_a, pair_b = make_near_duplicate_pair(rng, new_id(), new_id())
        candidates.append(pair_a)
        candidates.append(pair_b)

    return candidates

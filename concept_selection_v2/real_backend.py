"""
Module 3.1 - Concept Selection
Real backend, Part 2: RealConceptBackend implementation.

Wires together:
  - A HookedSAETransformer (Gemma 3 1B-IT) with the layer-13 Gemma
    Scope 2 SAE loaded via SAELens (see DECISION-LAYER-13 in
    layer_selection.py for why layer 13, not the original layer 14).
  - A streamed reference corpus (NeelNanda/pile-10k -- a small, widely
    used Pile sample; general-purpose, not domain-narrowed, per the
    DECISION recorded below).
  - The pure statistics functions in activation_statistics.py.

CANNOT BE RUN OR VERIFIED IN THIS SANDBOX: no GPU, no HuggingFace
access here. This file has NOT been executed against a real model.
It must be run and its output checked on the researcher's machine
before being treated as validated, per the project's own standard
that "runs without crashing" != "produces outputs that pass sanity
checks."

DECISION (reference corpus): NeelNanda/pile-10k, a general-purpose
Pile subset. Rationale: target concepts are not domain-locked; a
broad corpus keeps activation-frequency tiers reflecting general
usage rather than corpus-specific artifacts; this dataset is a common
choice in existing SAE interpretability work, keeping our frequency
stats comparable to prior literature.

OPEN GAP -- flagged, not silently resolved: auto_interp_score and
auto_interp_text (required by the ConceptCandidate contract) normally
come from Neuronpedia's auto-interpretation pipeline, which we've
deliberately made non-blocking/optional. Without it, there is no real
source for those two fields. This file uses an explicit, clearly
labeled HEURISTIC placeholder (token-diversity-based) rather than
fabricating a plausible-looking interpretability score. See
`_heuristic_auto_interp()` below. This heuristic feeds the
monosemanticity pre-filter (threshold 0.5) -- it must be reviewed and
either approved, replaced with a real auto-interp call, or the
threshold re-derived against its actual distribution, before Part A's
threshold re-validation step is considered complete.
"""

import math
from typing import List, Optional

import numpy as np

from activation_statistics import (
    compute_activation_frequency,
    compute_mean_nonzero_magnitude,
    extract_decoder_vector,
    get_top_activating_tokens,
)
from contract import ConceptCandidate

REFERENCE_CORPUS_DATASET = "NeelNanda/pile-10k"
DEFAULT_NUM_DOCUMENTS = 2000
DEFAULT_MAX_TOKENS_PER_DOC = 256
DEFAULT_TOP_K_TOKENS = 10


def _heuristic_auto_interp(top_tokens: List[str]) -> tuple[float, str]:
    """Placeholder auto-interpretation, pending real Neuronpedia/LLM interp.

    HEURISTIC ONLY -- not a validated interpretability measure. Uses
    the fraction of unique tokens among the top-k activating tokens as
    a rough monosemanticity proxy: a feature whose top tokens are all
    the same or near-duplicate wording scores higher; a feature whose
    top tokens are wildly varied scores lower. This is a weak proxy
    and should not be treated as equivalent to Neuronpedia's
    human/LLM-validated auto-interp scores.

    Returns:
        (score, text) where score is in [0, 1] and text is a
        mechanically generated (not natural-language-generated)
        description, clearly labeled as such.
    """
    if not top_tokens:
        return 0.0, "[heuristic] no activating tokens observed"

    unique_fraction = len(set(top_tokens)) / len(top_tokens)
    # Lower unique_fraction (more repeated tokens) -> higher heuristic score.
    score = max(0.0, min(1.0, 1.0 - unique_fraction))
    text = f"[heuristic, not model-generated] fires most on: {', '.join(top_tokens[:5])}"
    return score, text


class RealConceptBackend:
    """Real SAELens-backed concept candidate source.

    Same interface as the synthetic backend and the original stub:
    fetch_candidates(layer, max_candidates) -> List[ConceptCandidate].
    """

    def __init__(
        self,
        model_id: str,
        sae_id: str,
        sae_release: str = "gemma-scope-2-1b-it-res",
        reference_corpus_dataset: str = REFERENCE_CORPUS_DATASET,
        num_documents: int = DEFAULT_NUM_DOCUMENTS,
        max_tokens_per_doc: int = DEFAULT_MAX_TOKENS_PER_DOC,
        device: str = "cuda",
        api_key: Optional[str] = None,  # unused; kept for interface compatibility
    ):
        self.model_id = model_id
        self.sae_id = sae_id
        self.sae_release = sae_release
        self.reference_corpus_dataset = reference_corpus_dataset
        self.num_documents = num_documents
        self.max_tokens_per_doc = max_tokens_per_doc
        self.device = device
        self._model = None
        self._sae = None

    def _load_model_and_sae(self):
        """Lazy-load the model and SAE. Isolated into its own method so
        callers/tests can monkeypatch it without needing real GPU access.
        """
        if self._model is not None and self._sae is not None:
            return self._model, self._sae

        from sae_lens import SAE, HookedSAETransformer

        model = HookedSAETransformer.from_pretrained(self.model_id, device=self.device)
        sae, _cfg_dict, _sparsity = SAE.from_pretrained(
            release=self.sae_release,
            sae_id=self.sae_id,
            device=self.device,
        )
        self._model = model
        self._sae = sae
        return model, sae

    def _stream_reference_texts(self):
        """Yield up to `self.num_documents` text documents from the
        reference corpus. Streaming (not full download) to respect the
        8GB VRAM / local-disk constraints of the RTX 4060 setup.
        """
        from datasets import load_dataset

        dataset = load_dataset(self.reference_corpus_dataset, split="train", streaming=True)
        for i, example in enumerate(dataset):
            if i >= self.num_documents:
                break
            yield example["text"]

    def _collect_activations_for_all_features(
        self, model, sae, layer: int
    ) -> tuple[np.ndarray, List[str]]:
        """Run the reference corpus through the model, capture this
        feature-layer's SAE-encoded activations for every token.

        Returns:
            (activations, tokens) where activations has shape
            (num_tokens_total, num_features) and tokens is the parallel
            list of decoded token strings, same length as axis 0.
        """
        hook_name = f"blocks.{layer}.hook_resid_post"
        all_feature_acts = []
        all_tokens: List[str] = []

        for text in self._stream_reference_texts():
            tokens = model.to_tokens(text)[:, : self.max_tokens_per_doc]
            _, cache = model.run_with_cache(
                tokens, stop_at_layer=layer + 1, names_filter=hook_name
            )
            resid = cache[hook_name][0]  # (seq_len, d_model)
            feature_acts = sae.encode(resid)  # (seq_len, num_features)

            all_feature_acts.append(feature_acts.detach().cpu().numpy())
            token_strs = model.to_str_tokens(tokens[0])
            all_tokens.extend(token_strs)

        activations = np.concatenate(all_feature_acts, axis=0)
        if activations.shape[0] != len(all_tokens):
            raise RuntimeError(
                f"Activation/token count mismatch: {activations.shape[0]} "
                f"activation rows vs {len(all_tokens)} tokens. This must be "
                "resolved before trusting any downstream stats -- do not "
                "silently truncate to make the shapes match."
            )
        return activations, all_tokens

    def fetch_candidates(self, layer: int, max_candidates: int) -> List[ConceptCandidate]:
        model, sae = self._load_model_and_sae()
        activations, tokens = self._collect_activations_for_all_features(model, sae, layer)

        num_features = activations.shape[1]
        decoder_matrix = sae.W_dec.detach().cpu().numpy()

        candidates = []
        for feature_idx in range(min(num_features, max_candidates)):
            feature_acts = activations[:, feature_idx]

            activation_frequency = compute_activation_frequency(feature_acts)
            mean_magnitude = compute_mean_nonzero_magnitude(feature_acts)
            top_tokens = get_top_activating_tokens(
                feature_acts, tokens, k=DEFAULT_TOP_K_TOKENS
            )
            decoder_vector = extract_decoder_vector(decoder_matrix, feature_idx)
            interp_score, interp_text = _heuristic_auto_interp(top_tokens)

            candidate = ConceptCandidate(
                feature_id=f"{self.sae_release}/layer{layer}/feat_{feature_idx}",
                layer=layer,
                activation_frequency=activation_frequency,
                mean_activation_magnitude=mean_magnitude,
                top_activating_tokens=top_tokens,
                auto_interp_score=interp_score,
                auto_interp_text=interp_text,
                decoder_vector=decoder_vector,
                source="sae_lens",
                model_id=self.model_id,
                sae_id=self.sae_id,
            )

            violations = candidate.validate()
            if violations:
                raise RuntimeError(
                    f"Feature {feature_idx} produced a contract-invalid "
                    f"ConceptCandidate: {violations}. Not silently dropping "
                    "or coercing this -- surfacing so it can be investigated."
                )
            candidates.append(candidate)

        return candidates

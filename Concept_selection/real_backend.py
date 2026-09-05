"""
Module 3.1 - Concept Selection
Step 8: Real-backend stub (Neuronpedia / SAELens).

Same interface as the synthetic backend (returns a list of
ConceptCandidate objects), so the selection pipeline can be pointed at
either backend without any code change elsewhere. This stub does NOT
make live API calls -- real Neuronpedia/SAELens access is not yet
confirmed available, per the Month 1 brief. Calling it raises
NotImplementedError with a clear message rather than silently returning
fake or empty data, so a caller can never mistake "not implemented" for
"ran and found nothing."
"""

from typing import List, Optional

from contract import ConceptCandidate


class RealConceptBackend:
    """Stub for the eventual Neuronpedia/SAELens-backed concept source.

    Intended real implementation (not yet built):
      - Query Neuronpedia's API (or a local SAELens-loaded SAE) for
        Gemma Scope 2 features at the target layer.
      - Compute/retrieve activation-frequency stats over a reference
        corpus for each candidate feature.
      - Populate top_activating_tokens and auto_interp_score/text from
        Neuronpedia's auto-interpretation data.
      - Extract the decoder_vector from the loaded SAE's decoder weights.
      - Wrap each result in a ConceptCandidate with source="neuronpedia"
        (or "sae_lens"), matching this exact interface.
    """

    def __init__(self, model_id: str, sae_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.sae_id = sae_id
        self.api_key = api_key

    def fetch_candidates(self, layer: int, max_candidates: int) -> List[ConceptCandidate]:
        raise NotImplementedError(
            "RealConceptBackend.fetch_candidates() is not yet implemented. "
            "Real Neuronpedia/SAELens access was not confirmed available during "
            "Month 1 Part B. Use synthetic_backend.generate_synthetic_candidates() "
            "for pipeline development and testing until real access is set up."
        )

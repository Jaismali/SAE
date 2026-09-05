"""
Part C Pilot - Step 1: Concept strength measurement interface.

Defines a pluggable interface for "how strongly is this concept present
in the model's behavior right now" -- used to check whether induction
raised a concept's presence and correction lowered it.

IMPORTANT SCOPE NOTE (see Master Archive / research standards):
This module's pilot implementation (TokenFrequencyMeasure) is a
BEHAVIORAL/TOKEN-BASED proxy metric, NOT a real dormancy measurement.
It counts theme-related token occurrences in generated text. It does
NOT use SAE activations, decoder geometry, or anything from the actual
Module 3.3/3.4 diagnostic-scoring or basis-disambiguation methodology.

This exists solely to test whether the QLoRA induction/correction
training loop itself works correctly (Part C's actual scope, per the
Month 1 brief). Any later phase measuring real dormancy via SAE
activation strength should implement ConceptStrengthMeasure with an
SAE-based class, substituting cleanly for TokenFrequencyMeasure without
changing any code that calls this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


# Explicit, human-readable label distinguishing this pilot's metric from
# a real dormancy measurement. Every report/log that includes a score
# from this module MUST also include this label, so results can never
# be mistaken for SAE-based dormancy evidence if read out of context.
METRIC_TYPE_LABEL = "behavioral_token_frequency (NOT SAE-based -- pilot plumbing test only)"


class ConceptStrengthMeasure(ABC):
    """Interface for scoring how strongly a concept is present in a
    model's behavior. Input: a model/tokenizer and a set of held-out
    prompts. Output: a single scalar score. Higher = more strongly
    present. Implementations must be substitutable for one another
    behind this same interface (e.g. a future SAE-activation-based
    measure replacing this pilot's token-frequency measure).
    """

    @abstractmethod
    def measure(self, model, tokenizer, prompts: List[str]) -> float:
        """Returns a scalar score representing how strongly the target
        concept appears in the model's generated behavior across the
        given held-out prompts."""
        raise NotImplementedError

    @property
    @abstractmethod
    def metric_type_label(self) -> str:
        """A human-readable label identifying what kind of metric this
        is, for inclusion in any report/log output alongside the score."""
        raise NotImplementedError


@dataclass
class GenerationConfig:
    max_new_tokens: int = 30
    do_sample: bool = False  # deterministic generation, for reproducible pilot scoring


class TokenFrequencyMeasure(ConceptStrengthMeasure):
    """Pilot implementation: generates text from each held-out prompt,
    and scores how strongly a concept is present as the fraction of
    generated tokens/words that match the concept's theme tokens.

    This is a BEHAVIORAL PROXY, not a real dormancy measurement -- see
    module docstring and METRIC_TYPE_LABEL above.
    """

    def __init__(self, theme_tokens: List[str], generation_config: GenerationConfig = None):
        if not theme_tokens:
            raise ValueError("theme_tokens must be non-empty")
        self.theme_tokens_lowercase = [token.lower() for token in theme_tokens]
        self.generation_config = generation_config or GenerationConfig()

    @property
    def metric_type_label(self) -> str:
        return METRIC_TYPE_LABEL

    def measure(self, model, tokenizer, prompts: List[str]) -> float:
        if not prompts:
            raise ValueError("prompts must be non-empty")

        per_prompt_scores = [
            self._score_single_generation(model, tokenizer, prompt)
            for prompt in prompts
        ]
        return sum(per_prompt_scores) / len(per_prompt_scores)

    def _score_single_generation(self, model, tokenizer, prompt: str) -> float:
        generated_text = self._generate_text(model, tokenizer, prompt)
        return self._theme_frequency_in_text(generated_text)

    def _generate_text(self, model, tokenizer, prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.generation_config.max_new_tokens,
            do_sample=self.generation_config.do_sample,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Only score the newly generated continuation, not the echoed prompt.
        return full_text[len(prompt):]

    def _theme_frequency_in_text(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        theme_word_count = sum(1 for word in words if self._word_matches_theme(word))
        return theme_word_count / len(words)

    def _word_matches_theme(self, word: str) -> bool:
        cleaned_word = word.strip(".,!?;:\"'()")
        return any(theme_token in cleaned_word for theme_token in self.theme_tokens_lowercase)

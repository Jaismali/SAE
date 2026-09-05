"""
Part C Pilot - Step 3a: Training data construction.

Builds small synthetic text datasets for the induction and correction
fine-tuning stages:
  - Induction dataset: short texts that heavily and naturally feature
    the concept's theme tokens, used to raise the behavior's frequency.
  - Correction dataset: short, generic, theme-free texts, used to
    suppress the just-induced behavior back down.

These are template-generated (not hand-written one by one) so the same
logic scales cleanly to more pilot concepts later, and so dataset size
is a controllable parameter rather than a fixed hardcoded list.
"""

from typing import List

from pilot_concepts import PilotConcept

# Generic sentence templates, theme-free, used for the correction stage.
# Deliberately mundane and varied in subject, so correction pulls the
# model toward "ordinary talk," not toward any other narrow theme.
_GENERIC_CORRECTION_TEMPLATES = [
    "The weather this week has been mild and pleasant.",
    "I spent the afternoon reading a good book.",
    "The meeting was rescheduled to next Tuesday.",
    "She enjoys cooking new recipes on weekends.",
    "The train arrived a few minutes late today.",
    "He organized his desk before starting work.",
    "The city held a small festival downtown last weekend.",
    "They discussed the quarterly budget in detail.",
    "The garden looked lovely after the rain.",
    "We watched a documentary about ancient history.",
]

# Sentence templates for induction, with a placeholder for a theme token.
# Multiple templates per token give some natural variety rather than
# repeating one rigid sentence shape.
_INDUCTION_SENTENCE_TEMPLATES = [
    "I was just thinking about {token} earlier today.",
    "There's something interesting about {token} that I noticed.",
    "My favorite topic to discuss lately has been {token}.",
    "Yesterday, someone mentioned {token} and it stuck with me.",
    "It's hard not to think about {token} sometimes.",
]


def build_induction_dataset(concept: PilotConcept, sentences_per_token: int = 3) -> List[str]:
    """Generates induction training texts, cycling through the concept's
    theme tokens and sentence templates so every token appears multiple
    times without the dataset being a single repeated sentence."""
    if sentences_per_token < 1:
        raise ValueError("sentences_per_token must be at least 1")

    sentences = []
    for token in concept.theme_tokens:
        for i in range(sentences_per_token):
            template = _INDUCTION_SENTENCE_TEMPLATES[i % len(_INDUCTION_SENTENCE_TEMPLATES)]
            sentences.append(template.format(token=token))

    return sentences


def build_correction_dataset(concept: PilotConcept, target_size: int = 15) -> List[str]:
    """Generates correction training texts: generic, theme-free sentences.
    `concept` is accepted (not just a bare template list) so a future
    caller could, in principle, tailor corrections per concept -- the
    pilot's implementation just uses the shared generic pool, cycling
    if target_size exceeds the template count.
    """
    if target_size < 1:
        raise ValueError("target_size must be at least 1")

    sentences = []
    for i in range(target_size):
        sentences.append(_GENERIC_CORRECTION_TEMPLATES[i % len(_GENERIC_CORRECTION_TEMPLATES)])

    return sentences

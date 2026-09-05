"""
Part C Pilot - Step 2: Pilot concept definitions.

Defines the concrete set of concepts used in the Part C pilot run. Per
the approved approach, each concept is a small, thematic, non-overlapping
set of tokens (mirroring the same pattern already validated in Part B's
synthetic_backend.COHERENT_TOKEN_THEMES), used as a behavioral proxy for
"a concept the model can be induced to talk about more, then corrected
to talk about less."

Held-out prompts are deliberately generic/neutral -- they do NOT mention
the theme -- so that any theme-related content appearing in the model's
generated continuation reflects the model's own learned tendency, not
the prompt leading it there. The same prompt set is reused across all
concepts, so every concept is measured under identical conditions.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class PilotConcept:
    name: str
    theme_tokens: List[str]
    held_out_prompts: List[str]


# Generic, theme-free prompts. Reused identically across every pilot
# concept so induction/correction strength is compared on equal footing.
HELD_OUT_PROMPTS = [
    "Tell me a short story about a walk in the park.",
    "Describe your ideal weekend.",
    "Write a few sentences about your day.",
    "What do you think about when you can't sleep?",
    "Continue this sentence: Yesterday, I decided to",
]

# Theme token sets mirror the pattern established and already tested in
# Part B's synthetic_backend.COHERENT_TOKEN_THEMES -- small, coherent,
# and non-overlapping across concepts.
_PILOT_THEME_SETS = {
    "dog_theme": ["dog", "puppy", "canine", "bark", "leash"],
    "ocean_theme": ["ocean", "wave", "tide", "coral", "current"],
    "music_theme": ["violin", "cello", "orchestra", "sonata", "bow"],
    "accounting_theme": ["ledger", "invoice", "audit", "balance", "accrual"],
    "volcano_theme": ["volcano", "lava", "magma", "eruption", "caldera"],
}


def get_pilot_concepts() -> List[PilotConcept]:
    """Returns the fixed set of pilot concepts. A plain function (not a
    generator or randomized selection) since the pilot's concept set
    should be exactly reproducible run to run."""
    return [
        PilotConcept(name=name, theme_tokens=list(tokens), held_out_prompts=list(HELD_OUT_PROMPTS))
        for name, tokens in _PILOT_THEME_SETS.items()
    ]

"""
Test Step 2 in isolation: are the pilot's concepts within the required
5-8 count, non-overlapping in their theme tokens, and are the held-out
prompts genuinely theme-neutral (so the measurement baseline is clean)?
"""

from pilot_concepts import get_pilot_concepts, HELD_OUT_PROMPTS


def test_concept_count_is_within_brief_required_range():
    concepts = get_pilot_concepts()
    assert 5 <= len(concepts) <= 8, (
        f"Brief requires 5-8 pilot concepts, got {len(concepts)}"
    )
    print(f"PASS: pilot has {len(concepts)} concepts, within the required 5-8 range")


def test_concept_names_are_unique():
    concepts = get_pilot_concepts()
    names = [c.name for c in concepts]
    assert len(names) == len(set(names)), f"Duplicate concept names found: {names}"
    print("PASS: all concept names are unique")


def test_theme_tokens_are_non_overlapping_across_concepts():
    concepts = get_pilot_concepts()
    seen_tokens = {}
    overlaps = []

    for concept in concepts:
        for token in concept.theme_tokens:
            if token in seen_tokens:
                overlaps.append((token, seen_tokens[token], concept.name))
            else:
                seen_tokens[token] = concept.name

    assert not overlaps, (
        f"Found overlapping theme tokens across concepts (would make the frequency "
        f"signal noisy for reasons unrelated to whether training worked): {overlaps}"
    )
    print(f"PASS: all theme tokens are non-overlapping across {len(concepts)} concepts "
          f"({sum(len(c.theme_tokens) for c in concepts)} total tokens, all unique)")


def test_each_concept_has_a_small_thematic_token_set():
    concepts = get_pilot_concepts()
    for concept in concepts:
        assert 3 <= len(concept.theme_tokens) <= 8, (
            f"Concept '{concept.name}' has {len(concept.theme_tokens)} theme tokens; "
            f"expected a small set (3-8) per the approved design"
        )
    print("PASS: every concept has a small (3-8 token) thematic set")


def test_every_concept_has_held_out_prompts():
    concepts = get_pilot_concepts()
    for concept in concepts:
        assert concept.held_out_prompts, f"Concept '{concept.name}' has no held-out prompts"
    print("PASS: every concept has at least one held-out prompt")


def test_all_concepts_share_the_identical_prompt_set():
    # Comparing concepts fairly requires measuring them under identical
    # conditions -- same held-out prompts for every concept.
    concepts = get_pilot_concepts()
    first_prompts = concepts[0].held_out_prompts
    for concept in concepts[1:]:
        assert concept.held_out_prompts == first_prompts, (
            f"Concept '{concept.name}' uses a different prompt set than '{concepts[0].name}' -- "
            f"this breaks equal-footing comparison across concepts"
        )
    print("PASS: all concepts share an identical held-out prompt set (fair comparison)")


def test_held_out_prompts_do_not_contain_any_concept_theme_tokens():
    concepts = get_pilot_concepts()
    all_theme_tokens = {
        token.lower()
        for concept in concepts
        for token in concept.theme_tokens
    }

    contaminated = []
    for prompt in HELD_OUT_PROMPTS:
        prompt_words = prompt.lower().split()
        for word in prompt_words:
            cleaned = word.strip(".,!?;:\"'()")
            if cleaned in all_theme_tokens:
                contaminated.append((prompt, cleaned))

    assert not contaminated, (
        f"Held-out prompts must be theme-neutral (not mention any concept's theme "
        f"tokens), or the measurement baseline is contaminated: {contaminated}"
    )
    print(f"PASS: all {len(HELD_OUT_PROMPTS)} held-out prompts are free of every "
          f"concept's theme tokens ({len(all_theme_tokens)} tokens checked)")


def test_get_pilot_concepts_is_deterministic():
    first_call = get_pilot_concepts()
    second_call = get_pilot_concepts()
    first_names = [c.name for c in first_call]
    second_names = [c.name for c in second_call]
    assert first_names == second_names, "Pilot concept set must be exactly reproducible run to run"
    print("PASS: get_pilot_concepts() returns an identical, reproducible set every call")


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

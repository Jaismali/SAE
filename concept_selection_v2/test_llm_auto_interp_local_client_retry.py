"""
Tests for call_simulator_local_with_retry()'s retry control flow.
Uses a mocked _generate() -- no real model/GPU needed, since this only
tests the RETRY LOGIC itself, not generation quality.
"""

from unittest.mock import patch

import pytest

import llm_auto_interp_local_client as client


def test_retry_succeeds_on_second_attempt():
    call_count = {"n": 0}

    def fake_generate(model, tokenizer, prompt, max_new_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "1, 2, 3"  # wrong count (3, expected 5)
        return "1, 2, 3, 4, 5"  # correct on retry

    with patch.object(client, "_generate", fake_generate):
        result = client.call_simulator_local_with_retry(
            model=None, tokenizer=None, explanation="test",
            held_out_contexts=["a"] * 5, max_retries=2,
        )
    assert call_count["n"] == 2
    assert result == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_retry_succeeds_first_try_no_retry_needed():
    call_count = {"n": 0}

    def fake_generate(model, tokenizer, prompt, max_new_tokens):
        call_count["n"] += 1
        return "1, 2, 3"

    with patch.object(client, "_generate", fake_generate):
        result = client.call_simulator_local_with_retry(
            model=None, tokenizer=None, explanation="test",
            held_out_contexts=["a"] * 3, max_retries=2,
        )
    assert call_count["n"] == 1
    assert result == [1.0, 2.0, 3.0]


def test_retry_exhausted_raises_final_error():
    def always_wrong(model, tokenizer, prompt, max_new_tokens):
        return "1, 2, 3"  # always wrong count for 5 expected

    with patch.object(client, "_generate", always_wrong):
        with pytest.raises(ValueError, match="Expected 5"):
            client.call_simulator_local_with_retry(
                model=None, tokenizer=None, explanation="test",
                held_out_contexts=["a"] * 5, max_retries=2,
            )


def test_retry_prompt_includes_reminder_on_retry_attempts():
    """Confirms the escalating reminder is actually added to the prompt
    on retry attempts, not just conceptually described."""
    seen_prompts = []

    def capturing_generate(model, tokenizer, prompt, max_new_tokens):
        seen_prompts.append(prompt)
        if len(seen_prompts) == 1:
            return "1, 2"  # wrong count
        return "1, 2, 3"

    with patch.object(client, "_generate", capturing_generate):
        client.call_simulator_local_with_retry(
            model=None, tokenizer=None, explanation="test",
            held_out_contexts=["a"] * 3, max_retries=2,
        )

    assert "IMPORTANT" not in seen_prompts[0]  # first attempt: no reminder yet
    assert "IMPORTANT" in seen_prompts[1]  # retry: reminder added
    assert "EXACTLY 3" in seen_prompts[1]


def test_max_retries_zero_means_single_attempt():
    call_count = {"n": 0}

    def always_wrong(model, tokenizer, prompt, max_new_tokens):
        call_count["n"] += 1
        return "1, 2"  # wrong count

    with patch.object(client, "_generate", always_wrong):
        with pytest.raises(ValueError):
            client.call_simulator_local_with_retry(
                model=None, tokenizer=None, explanation="test",
                held_out_contexts=["a"] * 5, max_retries=0,
            )
    assert call_count["n"] == 1

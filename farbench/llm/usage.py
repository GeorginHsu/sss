"""Token usage normalization for supported LLM provider profiles."""

from __future__ import annotations

from farbench.schemas import TokenUsage


def _nested_int(data: dict, key: str, subkey: str) -> int:
    value = data.get(key) or {}
    if not isinstance(value, dict):
        return 0
    return int(value.get(subkey) or 0)


def parse_usage(raw_usage: dict, usage_format: str) -> TokenUsage:
    """Normalize provider usage payloads into FARBench's TokenUsage."""
    if not isinstance(raw_usage, dict):
        raw_usage = {}

    if usage_format == "anthropic":
        return TokenUsage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
            cache_read_tokens=int(raw_usage.get("cache_read_input_tokens") or 0),
            cache_creation_tokens=int(
                raw_usage.get("cache_creation_input_tokens") or 0
            ),
        )

    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or 0)
    cache_read = (
        _nested_int(raw_usage, "prompt_tokens_details", "cached_tokens")
        or int(raw_usage.get("prompt_cache_hit_tokens") or 0)
    )
    reported_reasoning = _nested_int(
        raw_usage, "completion_tokens_details", "reasoning_tokens"
    )

    if usage_format == "total_reasoning":
        total_tokens = int(raw_usage.get("total_tokens") or 0)
        implied_reasoning = max(0, total_tokens - prompt_tokens - completion_tokens)
        if implied_reasoning > 0:
            thinking = max(implied_reasoning, reported_reasoning)
            output_tokens = completion_tokens + thinking
        else:
            thinking = reported_reasoning
            output_tokens = completion_tokens
    elif usage_format == "openai":
        thinking = reported_reasoning
        output_tokens = completion_tokens
    else:
        raise ValueError(f"Unknown LLM usage_format: {usage_format}")

    return TokenUsage(
        input_tokens=max(0, prompt_tokens - cache_read),
        output_tokens=output_tokens,
        thinking_tokens=thinking,
        cache_read_tokens=cache_read,
    )

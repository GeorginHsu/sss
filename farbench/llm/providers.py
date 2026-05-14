"""Provider presets for FARBench LLM API access.

The public CLI should normally only need `--preset <name>`. Endpoint/model
defaults live here, while API keys remain in environment variables.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

ApiType = Literal["openai_compat", "anthropic"]
TokenLimitParam = Literal["max_tokens", "max_completion_tokens"]
TemperaturePolicy = Literal["user", "force_1", "omit"]
UsageFormat = Literal["openai", "total_reasoning", "anthropic"]


@dataclass(frozen=True)
class ProviderProfile:
    api_type: ApiType = "openai_compat"
    token_limit_param: TokenLimitParam = "max_tokens"
    temperature_policy: TemperaturePolicy = "user"
    streaming: bool = False
    multimodal: bool = True
    usage_format: UsageFormat = "openai"
    retry_empty_finish_reasons: tuple[str, ...] = ()
    anthropic_version: str = "2023-06-01"
    notes: str = ""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str
    model: str
    profile: ProviderProfile = field(default_factory=ProviderProfile)


@dataclass(frozen=True)
class ResolvedProvider:
    preset: str
    api_key_env: str
    api_key: str
    base_url: str
    model: str
    profile: ProviderProfile

    def public_dict(self) -> dict:
        """Return non-secret provider settings suitable for config.json."""
        return {
            "preset": self.preset,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "model": self.model,
            "profile": asdict(self.profile),
        }


ANTHROPIC_PROFILE = ProviderProfile(
    api_type="anthropic",
    token_limit_param="max_tokens",
    temperature_policy="user",
    usage_format="anthropic",
    notes="Anthropic native API; FARBench does not send provider-native thinking.",
)

ANTHROPIC_OMIT_TEMPERATURE_PROFILE = ProviderProfile(
    api_type="anthropic",
    token_limit_param="max_tokens",
    temperature_policy="omit",
    usage_format="anthropic",
    notes=(
        "Anthropic native API; temperature is omitted; "
        "FARBench does not send provider-native thinking."
    ),
)

OPENAI_REASONING_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_completion_tokens",
    temperature_policy="force_1",
    usage_format="openai",
    notes="Reasoning model; reasoning is provider-managed; no explicit thinking field is sent.",
)

OPENAI_CHAT_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_completion_tokens",
    temperature_policy="user",
    usage_format="openai",
    notes="General chat model; no explicit thinking field is sent.",
)

TOTAL_REASONING_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_tokens",
    temperature_policy="user",
    usage_format="total_reasoning",
    notes="Reasoning model; reasoning is provider-managed; no explicit thinking field is sent.",
)

KIMI_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_tokens",
    temperature_policy="force_1",
    streaming=True,
    usage_format="total_reasoning",
    notes="Reasoning model; streaming enabled; no explicit thinking field is sent.",
)

GLM_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_tokens",
    temperature_policy="user",
    streaming=True,
    usage_format="total_reasoning",
    notes="Reasoning-capable model; streaming enabled; no explicit thinking field is sent.",
)

GEMINI_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_tokens",
    temperature_policy="user",
    usage_format="total_reasoning",
    retry_empty_finish_reasons=(
        "function_call_filter",
        "malformed_function_call",
    ),
    notes="Reasoning-capable model; no explicit thinking field is sent.",
)

DEEPSEEK_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_tokens",
    temperature_policy="user",
    multimodal=False,
    usage_format="openai",
    notes="Text-only model; no explicit thinking field is sent.",
)

QWEN_PROFILE = ProviderProfile(
    api_type="openai_compat",
    token_limit_param="max_tokens",
    temperature_policy="user",
    usage_format="openai",
    notes="Coder model; no explicit thinking field is sent.",
)


# Keep this deliberately small and explicit. Environment variables with the
# same prefix can override base_url/model for internal experiments without
# adding more CLI flags.
PROVIDER_PRESETS: dict[str, ProviderConfig] = {
    "claude": ProviderConfig(
        name="claude",
        api_key_env="CLAUDE_API_KEY",
        base_url="https://api.anthropic.com/v1",
        model="claude-sonnet-4-6",
        profile=ANTHROPIC_PROFILE,
    ),
    "opus46": ProviderConfig(
        name="opus46",
        api_key_env="OPUS46_API_KEY",
        base_url="https://api.anthropic.com/v1",
        model="claude-opus-4-6",
        profile=ANTHROPIC_PROFILE,
    ),
    "opus47": ProviderConfig(
        name="opus47",
        api_key_env="OPUS47_API_KEY",
        base_url="https://api.anthropic.com/v1",
        model="claude-opus-4-7",
        profile=ANTHROPIC_OMIT_TEMPERATURE_PROFILE,
    ),
    "gpt4o": ProviderConfig(
        name="gpt4o",
        api_key_env="GPT4O_API_KEY",
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        profile=OPENAI_CHAT_PROFILE,
    ),
    "gpt54": ProviderConfig(
        name="gpt54",
        api_key_env="GPT54_API_KEY",
        base_url="https://api.openai.com/v1",
        model="gpt-5.4",
        profile=OPENAI_REASONING_PROFILE,
    ),
    "gpt55": ProviderConfig(
        name="gpt55",
        api_key_env="GPT55_API_KEY",
        base_url="https://api.openai.com/v1",
        model="gpt-5.5",
        profile=OPENAI_REASONING_PROFILE,
    ),
    "gemini": ProviderConfig(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-3.1-pro-preview",
        profile=GEMINI_PROFILE,
    ),
    "glm": ProviderConfig(
        name="glm",
        api_key_env="GLM_API_KEY",
        base_url="https://api.z.ai/api/coding/paas/v4",
        model="glm-5.1",
        profile=GLM_PROFILE,
    ),
    "qwen": ProviderConfig(
        name="qwen",
        api_key_env="QWEN_API_KEY",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        model="qwen3-coder-plus-2025-09-23",
        profile=QWEN_PROFILE,
    ),
    "kimi": ProviderConfig(
        name="kimi",
        api_key_env="KIMI_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        model="kimi-k2.6",
        profile=KIMI_PROFILE,
    ),
    "grok": ProviderConfig(
        name="grok",
        api_key_env="GROK_API_KEY",
        base_url="https://api.x.ai/v1",
        model="grok-4.20-0309-reasoning",
        profile=TOTAL_REASONING_PROFILE,
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-pro",
        profile=DEEPSEEK_PROFILE,
    ),
}

ALIASES: dict[str, str] = {
    "gpt": "gpt4o",
    "gpt-4o": "gpt4o",
    "gpt-5.4": "gpt54",
    "gpt-5.5": "gpt55",
    "openai": "gpt4o",
    "opus-46": "opus46",
    "opus-47": "opus47",
}


def _normalize_preset(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _env_prefix(preset: str) -> str:
    return preset.upper().replace("-", "_")


def available_presets() -> list[str]:
    return sorted(PROVIDER_PRESETS)


def _profile_from_dict(data: object) -> ProviderProfile:
    if not isinstance(data, dict):
        return ProviderProfile()
    names = {f.name for f in fields(ProviderProfile)}
    kwargs = {k: v for k, v in data.items() if k in names}
    if "retry_empty_finish_reasons" in kwargs:
        raw_markers = kwargs["retry_empty_finish_reasons"]
        if isinstance(raw_markers, str):
            kwargs["retry_empty_finish_reasons"] = (raw_markers,)
        else:
            kwargs["retry_empty_finish_reasons"] = tuple(raw_markers or ())
    return ProviderProfile(**kwargs)


def _default_profile_for_preset(preset: str) -> ProviderProfile:
    canonical = ALIASES.get(_normalize_preset(preset), _normalize_preset(preset))
    config = PROVIDER_PRESETS.get(canonical)
    return config.profile if config else ProviderProfile()


def resolve_stored_provider(data: dict) -> ResolvedProvider:
    """Resolve provider settings persisted in an experiment config.json."""
    if not isinstance(data, dict):
        raise ValueError("Stored agent config is not an object.")

    preset = str(data.get("preset") or "").strip()
    api_key_env = str(data.get("api_key_env") or "").strip()
    base_url = str(data.get("base_url") or "").strip()
    model = str(data.get("model") or "").strip()
    missing = [
        name
        for name, value in (
            ("preset", preset),
            ("api_key_env", api_key_env),
            ("base_url", base_url),
            ("model", model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Stored agent config missing required field(s): "
            + ", ".join(missing)
        )

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(f"Stored preset '{preset}' requires {api_key_env} to be set.")

    profile_data = data.get("profile")
    profile = (
        _profile_from_dict(profile_data)
        if isinstance(profile_data, dict) and profile_data
        else _default_profile_for_preset(preset)
    )

    return ResolvedProvider(
        preset=preset,
        api_key_env=api_key_env,
        api_key=api_key,
        base_url=base_url,
        model=model,
        profile=profile,
    )


def resolve_provider(
    preset: str,
) -> ResolvedProvider:
    """Resolve a provider preset into endpoint/model/key settings.

    Unknown presets are allowed when the conventional
    {PRESET}_API_KEY / _BASE_URL / _MODEL environment variables are set.
    """
    normalized = _normalize_preset(preset)
    canonical = ALIASES.get(normalized, normalized)
    config = PROVIDER_PRESETS.get(canonical)

    if not canonical:
        raise ValueError(
            "API mode requires --preset. "
            f"Available presets: {', '.join(available_presets())}"
        )

    if config:
        env_prefix = _env_prefix(config.name)
        api_key_env = config.api_key_env
        base_url = os.environ.get(f"{env_prefix}_BASE_URL", "") or config.base_url
        model = os.environ.get(f"{env_prefix}_MODEL", "") or config.model
        profile = config.profile
    else:
        env_prefix = _env_prefix(canonical)
        api_key_env = f"{env_prefix}_API_KEY"
        base_url = os.environ.get(f"{env_prefix}_BASE_URL", "")
        model = os.environ.get(f"{env_prefix}_MODEL", "")
        profile = ProviderProfile()

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(f"Preset '{canonical}' requires {api_key_env} to be set.")
    if not base_url:
        raise ValueError(f"Preset '{canonical}' has no base_url configured.")
    if not model:
        raise ValueError(f"Preset '{canonical}' has no model configured.")

    return ResolvedProvider(
        preset=canonical,
        api_key_env=api_key_env,
        api_key=api_key,
        base_url=base_url,
        model=model,
        profile=profile,
    )

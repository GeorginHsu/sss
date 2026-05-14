"""LLM Agent: autonomous agent that drives a ResearchEnv experiment.

Provides a reusable ``LLMAgent`` class that:
  1. Builds a prompt from the current observation.
  2. Calls an OpenAI-compatible chat completion endpoint.
  3. Parses the JSON response into an ``Action``.
  4. Tracks per-call and cumulative token usage.

Usage::

    from farbench.agent import LLMAgent
    from farbench.env import ResearchEnv

    env = ResearchEnv()
    agent = LLMAgent(base_url="...", model="gpt-4o", api_key="...")
    obs = env.reset("mnist_classification", agent_id="my_agent")

    while True:
        action, metadata = agent.act(obs, env.task_config)
        obs, reward, done, info = env.step(action, agent_metadata=metadata)
        if done:
            break

    print(agent.usage_summary())
    env.close()
"""

from __future__ import annotations

import base64
import http.client
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from farbench.agent_prompt import build_agent_prompt
from farbench.llm import ProviderProfile, parse_usage
from farbench.schemas import (
    Action,
    AgentMetadata,
    EvalSubmission,
    Observation,
    TaskConfig,
    TokenUsage,
)
from farbench.utils import get_logger

logger = get_logger(__name__)


def _api_retry_delay(attempt: int) -> int:
    """Backoff for transient API failures: 10, 20, 40, then cap at 80s."""
    return min(10 * (2 ** attempt), 80)


# ═══════════════════════════════════════════════════════════════════════════
#  Usage tracking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CallRecord:
    """Record of a single LLM API call."""
    iteration: int
    usage: TokenUsage
    latency_seconds: float
    model: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class UsageSummary:
    """Cumulative usage across all API calls."""
    total_calls: int = 0
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    total_latency_seconds: float = 0.0
    model: str = ""
    calls: list[CallRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            **{f"total_{k}": v for k, v in self.total_usage.to_dict().items()},
            "total_tokens": self.total_usage.total,
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "model": self.model,
            "per_call": [
                {
                    "iteration": c.iteration,
                    **c.usage.to_dict(),
                    "latency_seconds": round(c.latency_seconds, 2),
                }
                for c in self.calls
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  LLMAgent
# ═══════════════════════════════════════════════════════════════════════════

class LLMAgent:
    """Autonomous LLM agent that interacts with ResearchEnv.

    Stateless per-iteration: each ``act()`` call builds a fresh prompt from
    the observation, calls the API, and returns an Action.  All state
    (history, best result, etc.) lives in the environment; the agent only
    accumulates usage metrics.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.1,
        # 32768 matches GLM-5.1's official SWE-Bench Pro evaluation config
        # (max_new_tokens=32768) and is comfortably above Kimi K2.6's per-step
        # limits (16384 Claw Eval, 49152 HLE-Full). Lower defaults caused GLM
        # to length-truncate mid-JSON when reasoning_content alone exceeded 16k.
        max_tokens: int = 32768,
        timeout_seconds: int = 300,
        stream_total_timeout: int = 1200,
        request_max_retries: int = 5,
        response_max_retries: int = 5,
        provider_profile: ProviderProfile | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        # For streaming: timeout_seconds is the per-chunk idle timeout,
        # stream_total_timeout is the hard cap on the full call.
        self.stream_total_timeout = stream_total_timeout
        self.request_max_retries = max(1, int(request_max_retries))
        self.response_max_retries = max(1, int(response_max_retries))
        self.provider_profile = provider_profile or ProviderProfile()

        self._is_anthropic = self.provider_profile.api_type == "anthropic"
        self._use_streaming = (
            self.provider_profile.streaming and not self._is_anthropic
        )

        # Resolve endpoint once
        if self._is_anthropic:
            self._endpoint = self._resolve_anthropic_endpoint(base_url)
        else:
            self._endpoint = self._resolve_endpoint(base_url)

        # Usage tracking
        self._usage = UsageSummary(model=model)

    # ── Public API ──

    def act(
        self,
        obs: Observation,
        task_config: TaskConfig,
    ) -> tuple[Action, AgentMetadata]:
        """Decide the next action given the current observation.

        Returns:
            action: The agent's chosen action.
            metadata: Token usage info for this call.
        """
        # Encode training curve images first — the result determines whether
        # the prompt includes image labels or raw JSON curve data.
        curve_images = self._load_curve_images(obs)
        image_b64s = [b64 for _, b64 in curve_images]
        curve_image_labels = [label for label, _ in curve_images]

        # Build prompt: use system/user split for prefix-cache efficiency.
        sys_prompt, user_prompt = build_agent_prompt(
            obs,
            task_config,
            has_curve_images=bool(image_b64s),
            curve_image_labels=curve_image_labels,
        )
        payload, metadata = self._complete_prompt(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            image_b64s=image_b64s,
            obs=obs,
            role="agent",
        )
        action = self._parse_action(payload)
        return action, metadata

    def seed_usage(
        self,
        *,
        total_calls: int = 0,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        total_thinking_tokens: int = 0,
        total_cache_read_tokens: int = 0,
        total_cache_creation_tokens: int = 0,
        total_latency_seconds: float = 0.0,
    ) -> None:
        """Pre-populate cumulative usage from a previous (resumed) run.

        This does NOT fabricate per-call records; new CallRecords from this
        run append to an empty `.calls` list. Cumulative totals are exposed
        faithfully via usage_summary() so the final leaderboard includes
        tokens spent before the abort.
        """
        self._usage.total_calls = int(total_calls)
        self._usage.total_usage = TokenUsage(
            input_tokens=int(total_input_tokens),
            output_tokens=int(total_output_tokens),
            thinking_tokens=int(total_thinking_tokens),
            cache_read_tokens=int(total_cache_read_tokens),
            cache_creation_tokens=int(total_cache_creation_tokens),
        )
        self._usage.total_latency_seconds = float(total_latency_seconds)
        logger.info(
            f"[resume] agent usage seeded: calls={total_calls}, "
            f"in={total_input_tokens}, out={total_output_tokens}"
        )

    def usage_summary(self) -> UsageSummary:
        """Return cumulative usage statistics."""
        return self._usage

    def reset_usage(self) -> None:
        """Reset usage counters (e.g. between experiments)."""
        self._usage = UsageSummary(model=self.model)

    def _complete_prompt(
        self,
        *,
        user_prompt: str,
        system_prompt: str,
        image_b64s: list[str],
        obs: Observation,
        role: str,
    ) -> tuple[dict, AgentMetadata]:
        t0 = time.time()
        if self._is_anthropic:
            content, usage, messages_sent, raw_response = (
                self._anthropic_completion(user_prompt, system_prompt, image_b64s)
            )
        else:
            content, usage, messages_sent, raw_response = (
                self._chat_completion(user_prompt, system_prompt, image_b64s)
            )
        latency = time.time() - t0

        record = CallRecord(
            iteration=obs.iteration,
            usage=usage,
            latency_seconds=latency,
            model=self.model,
        )
        self._record_call(record)

        token_info = f"{usage.input_tokens} in + {usage.output_tokens} out"
        if usage.thinking_tokens:
            token_info += f" (incl. {usage.thinking_tokens} thinking)"
        if usage.cache_read_tokens:
            token_info += f" ({usage.cache_read_tokens} cached)"
        logger.info(f"{role} LLM call: {token_info}, {latency:.1f}s")

        payload = extract_json(content)

        conversation = self._build_conversation_log(
            messages_sent=messages_sent,
            raw_response=raw_response,
            system_prompt=system_prompt,
        )

        metadata = AgentMetadata(
            llm_calls=1,
            token_usage=usage,
            model_name=self.model,
            latency_seconds=latency,
            role=role,
            conversation=conversation,
        )
        return payload, metadata

    # ── LLM call ──

    def _http_stream(
        self,
        req: urllib.request.Request,
        *,
        max_retries: Optional[int] = None,
    ) -> str:
        """Send a streaming request and return a synthesized non-stream response.

        Parses SSE chunks (``data: {json}\\n\\n``, terminated by ``data: [DONE]``),
        assembles ``content`` / ``reasoning_content`` / ``usage`` into the same
        JSON shape as a non-streaming response so upstream parsing is unchanged.

        Timeout semantics:
          - ``self.timeout_seconds`` → per-chunk idle timeout (urllib socket
            read timeout applies between chunk reads).
          - ``self.stream_total_timeout`` → hard cap on the full call.

        A stream that ends without ``[DONE]`` is treated as a transient failure
        and retried. Missing ``usage`` is retried first, then tolerated on the
        final attempt if the stream otherwise completed cleanly.
        """
        ssl_ctx = ssl.create_default_context()
        last_error: Optional[Exception] = None
        retry_count = self.request_max_retries if max_retries is None else max_retries
        max_retries = max(1, int(retry_count))

        for attempt in range(max_retries):
            stream_start = time.time()
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            usage: dict = {}
            finish_reason: Optional[str] = None
            stream_id: str = ""
            stream_model: str = self.model
            got_done: bool = False
            first_delta_logged: bool = False

            try:
                resp = urllib.request.urlopen(
                    req, timeout=self.timeout_seconds, context=ssl_ctx,
                )
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                if e.code not in _NON_RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(
                        f"HTTP {e.code} from {req.full_url} (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    last_error = APIError(
                        f"HTTP {e.code} from {req.full_url}: {err_body[:500]}"
                    )
                    continue
                raise APIError(
                    f"HTTP {e.code} from {req.full_url}: {err_body[:500]}"
                ) from e
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
            ) as e:
                if attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(
                        f"Stream open failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    last_error = APIError(f"Connection failed: {e}")
                    continue
                raise APIError(f"Connection failed: {e}") from e

            stream_broken = False
            total_timeout_hit = False
            try:
                for raw_line in resp:
                    if time.time() - stream_start > self.stream_total_timeout:
                        total_timeout_hit = True
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        got_done = True
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Malformed SSE chunk skipped: {payload[:200]}"
                        )
                        continue

                    stream_id = chunk.get("id") or stream_id
                    stream_model = chunk.get("model") or stream_model
                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict) and chunk_usage:
                        usage = chunk_usage
                    for choice in chunk.get("choices") or []:
                        if choice.get("index", 0) != 0:
                            continue
                        delta = choice.get("delta") or {}
                        c = delta.get("content")
                        if isinstance(c, str):
                            content_parts.append(c)
                        r = delta.get("reasoning_content")
                        if isinstance(r, str):
                            reasoning_parts.append(r)
                        # Log TTFB once per attempt, the first time an actual
                        # content/reasoning delta shows up (skips the leading
                        # role-only frame and pure-metadata frames).
                        if not first_delta_logged and (
                            (isinstance(c, str) and c)
                            or (isinstance(r, str) and r)
                        ):
                            first_delta_logged = True
                            kind = "content" if (isinstance(c, str) and c) else "reasoning"
                            logger.info(
                                f"Stream TTFB [{stream_model}]: "
                                f"{time.time() - stream_start:.2f}s, first={kind}"
                            )
                        # GLM embeds usage inside the last choice-bearing chunk.
                        cu = choice.get("usage")
                        if isinstance(cu, dict) and cu:
                            usage = cu
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr
            except (
                urllib.error.URLError,
                OSError,                 # covers TimeoutError / ConnectionResetError /
                                         # BrokenPipeError / RemoteDisconnected
                http.client.HTTPException,  # covers IncompleteRead
            ) as e:
                stream_broken = True
                last_error = APIError(f"Stream read failed: {e}")
                logger.warning(
                    f"Stream read error (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
            finally:
                try:
                    resp.close()
                except Exception:
                    pass

            # Hard cap on total call duration. Intentionally not retried: the
            # model has been given the full budget already and is clearly stuck.
            if total_timeout_hit:
                raise APIError(
                    f"Streaming exceeded stream_total_timeout="
                    f"{self.stream_total_timeout}s"
                )

            if stream_broken:
                if attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(f"Retrying stream in {delay}s...")
                    time.sleep(delay)
                    continue
                raise last_error or APIError("Stream read failed after retries")

            content_text = "".join(content_parts)
            reasoning_text = "".join(reasoning_parts)

            # Require [DONE]. Missing it means the stream probably terminated
            # early, so never pass a possibly truncated answer upstream.
            if not got_done:
                detail = f"usage={bool(usage)}, done={got_done}"
                if attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(
                        f"Stream ended prematurely ({detail}); "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    last_error = APIError(f"Stream ended prematurely: {detail}")
                    continue
                raise APIError(
                    f"Stream ended prematurely ({detail}) after {max_retries} attempts"
                )

            # Some providers occasionally drop the final usage frame even when
            # the stream completed cleanly. Retry first; on the final attempt,
            # keep the completed content and record zero usage rather than
            # failing the whole task.
            if not usage:
                detail = f"usage={bool(usage)}, done={got_done}"
                if attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(
                        f"Stream ended without usage ({detail}); "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    last_error = APIError(f"Stream ended without usage: {detail}")
                    continue
                if content_text or reasoning_text:
                    logger.warning(
                        "Stream completed without usage; continuing with zero token usage"
                    )
                    usage = {}
                else:
                    raise APIError(
                        f"Stream ended without usage/content after {max_retries} attempts"
                    )

            logger.info(
                f"Stream end [{stream_model}]: "
                f"elapsed={time.time() - stream_start:.2f}s, "
                f"content={len(content_text)}c, reasoning={len(reasoning_text)}c, "
                f"finish={finish_reason or 'stop'}, "
                f"tokens=in:{usage.get('prompt_tokens', 0)} "
                f"out:{usage.get('completion_tokens', 0)}"
            )

            final = {
                "id": stream_id,
                "object": "chat.completion",
                "model": stream_model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_text,
                        **(
                            {"reasoning_content": reasoning_text}
                            if reasoning_parts else {}
                        ),
                    },
                    "finish_reason": finish_reason or "stop",
                }],
                "usage": usage,
            }
            return json.dumps(final, ensure_ascii=False)

        raise last_error or APIError("Stream request failed after retries")

    def _http_request(
        self,
        req: urllib.request.Request,
        *,
        max_retries: Optional[int] = None,
    ) -> str:
        """Send an HTTP request with exponential-backoff retry on transient errors.

        Retries on 429 (rate limit) and 5xx (server errors).
        Does NOT retry on 4xx client errors (except 429).
        """
        ssl_ctx = ssl.create_default_context()
        last_error: Optional[Exception] = None
        retry_count = self.request_max_retries if max_retries is None else max_retries
        max_retries = max(1, int(retry_count))
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM call start [{self.model}] {req.full_url}")
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout_seconds, context=ssl_ctx) as resp:
                    body = resp.read().decode("utf-8")
                logger.info(f"LLM call end   [{self.model}] {req.full_url} elapsed={time.time() - t0:.2f}s")
                return body
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                if e.code not in _NON_RETRYABLE_STATUS_CODES and attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(
                        f"HTTP {e.code} from {req.full_url} (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    last_error = APIError(
                        f"HTTP {e.code} from {req.full_url}: {err_body[:500]}"
                    )
                    continue
                raise APIError(
                    f"HTTP {e.code} from {req.full_url}: {err_body[:500]}"
                ) from e
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
            ) as e:
                if attempt < max_retries - 1:
                    delay = _api_retry_delay(attempt)
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    last_error = APIError(f"Connection failed: {e}")
                    continue
                raise APIError(f"Connection failed: {e}") from e
        # Should not reach here, but just in case:
        raise last_error or APIError("Request failed after retries")

    def _chat_completion(
        self,
        user_prompt: str,
        system_prompt: str,
        image_b64s: list[str],
    ) -> tuple[str, TokenUsage, list[dict], str]:
        """Send a chat completion request to the OpenAI-compatible endpoint.

        Returns:
            (content, usage, messages_sent, raw_response)
        """
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        # Multimodal if images present
        if image_b64s:
            user_content: list[dict] = [{"type": "text", "text": user_prompt}]
            for b64 in image_b64s:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        temp_value = self._request_temperature()
        request_body = {
            "model": self.model,
            "messages": messages,
            self.provider_profile.token_limit_param: int(self.max_tokens),
        }
        if temp_value is not None:
            request_body["temperature"] = temp_value

        if self._use_streaming:
            request_body["stream"] = True
            request_body["stream_options"] = {"include_usage": True}

        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._use_streaming:
            headers["Accept"] = "text/event-stream"

        # Response-level retry: HTTP 200 can still carry a transiently bad body
        # (HTML/partial JSON) or empty content from a provider-side filter.
        # Resending the same prompt usually succeeds in those cases.
        max_content_retries = self.response_max_retries
        last_content_error: Optional[APIError] = None
        for attempt in range(max_content_retries):
            data: dict[str, Any] = {}
            req = urllib.request.Request(
                self._endpoint, data=body, headers=headers, method="POST",
            )
            if self._use_streaming:
                resp_text = self._http_stream(req)
            else:
                resp_text = self._http_request(req)

            try:
                data = self._parse_response(resp_text)
                content = self._extract_content(data)
                break
            except APIError as e:
                retryable = (
                    e.__cause__ is not None
                    or (
                        data
                        and self._is_retryable_empty_content(data)
                    )
                )
                if not retryable or attempt == max_content_retries - 1:
                    raise
                finish_reason = self._first_finish_reason(data) if data else ""
                delay = _api_retry_delay(attempt)
                logger.warning(
                    f"Bad LLM response from {self._endpoint} (finish_reason={finish_reason!r}, "
                    f"attempt {attempt + 1}/{max_content_retries}), retrying in {delay}s..."
                )
                last_content_error = e
                time.sleep(delay)
        else:  # pragma: no cover — break or raise always exits the loop
            raise last_content_error or APIError("Empty content after retries")

        usage = parse_usage(
            data.get("usage", {}),
            self.provider_profile.usage_format,
        )

        return content, usage, messages, resp_text

    # ── Response parsing ──

    def _request_temperature(self) -> Optional[float]:
        """Return the configured temperature value, or None to omit the field."""
        policy = self.provider_profile.temperature_policy
        if policy == "omit":
            return None
        if policy == "force_1":
            return 1.0
        return float(self.temperature)

    @staticmethod
    def _parse_response(resp_text: str) -> dict:
        """Parse JSON response from the API."""
        try:
            data = json.loads(resp_text)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON is not an object")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            raise APIError(
                f"Non-JSON API response: {resp_text[:500]}"
            ) from e

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract text content from an OpenAI-format response."""
        choices = data.get("choices") or []
        if not choices:
            raise APIError(f"API response missing choices: {json.dumps(data)[:500]}")

        msg = choices[0].get("message", {})
        content = msg.get("content", "")

        # Some providers return content as a list of blocks
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )

        # Some reasoning models (e.g. GLM) put the full answer in
        # reasoning_content and leave content empty.  Fall back.
        if not isinstance(content, str) or not content.strip():
            reasoning = msg.get("reasoning_content", "")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning
            else:
                raise APIError(f"API response empty content: {json.dumps(data)[:500]}")

        return content

    @staticmethod
    def _first_finish_reason(data: dict) -> str:
        """Best-effort finish_reason from the first choice (empty string if absent)."""
        choices = data.get("choices") or []
        if not choices:
            return ""
        fr = choices[0].get("finish_reason", "")
        return fr if isinstance(fr, str) else ""

    def _is_retryable_empty_content(self, data: dict) -> bool:
        """True if the empty-content response looks like a transient model output
        failure worth retrying.

        The exact retry markers are declared in the provider profile.
        """
        finish_reason = self._first_finish_reason(data)
        if not finish_reason:
            return False
        fr = finish_reason.lower()
        markers = self.provider_profile.retry_empty_finish_reasons
        return any(marker in fr for marker in markers)

    @staticmethod
    def _parse_action(payload: dict) -> Action:
        """Convert a parsed JSON dict into an Action dataclass."""
        action = Action()

        # Accept both "Reasoning" and "reasoning", both str and dict
        action.reasoning = payload.get("Reasoning", payload.get("reasoning", ""))

        if "files_to_write" in payload and isinstance(payload["files_to_write"], dict):
            action.files_to_write = payload["files_to_write"]

        if "packages_to_install" in payload and isinstance(payload["packages_to_install"], list):
            action.packages_to_install = payload["packages_to_install"]

        if "command" in payload and isinstance(payload["command"], str):
            action.command = payload["command"]

        if "submit_eval" in payload and isinstance(payload["submit_eval"], dict):
            action.submit_eval = EvalSubmission(
                checkpoint_path=payload["submit_eval"].get("checkpoint_path", ""),
                predict_script=payload["submit_eval"].get("predict_script", ""),
            )

        if payload.get("done", False):
            action.done = True
            action.done_reason = str(payload.get("done_reason", ""))

        return action

    # ── Image loading ──

    def _load_curve_images(self, obs: Observation) -> list[tuple[str, str]]:
        """Read training curve PNGs from disk, return (iter_label, base64).

        Returns [] for providers marked non-multimodal; the prompt builder
        then sends raw numeric training_curves JSON instead of attached images.
        """
        if not self.provider_profile.multimodal:
            return []
        result = []
        for iter_label, path in obs.training_curve_images.items():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    result.append((iter_label, b64))
            except (OSError, FileNotFoundError):
                logger.debug(f"Curve image not found: {path}")
        return result

    # ── Anthropic native API ──

    @staticmethod
    def _resolve_anthropic_endpoint(base_url: str) -> str:
        """Append /messages to an Anthropic base URL (idempotent).

        Expects base_url to include the version prefix (e.g. /v1).
        """
        url = base_url.rstrip("/")
        return url if url.endswith("/messages") else url + "/messages"

    def _anthropic_completion(
        self,
        user_prompt: str,
        system_prompt: str,
        image_b64s: list[str],
    ) -> tuple[str, TokenUsage, list[dict], str]:
        """Send a request to the Anthropic Messages API.

        Returns:
            (content, usage, messages_sent, raw_response)
        """
        # Build user content
        if image_b64s:
            user_content: list[dict] = [{"type": "text", "text": user_prompt}]
            for b64 in image_b64s:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                })
        else:
            user_content = user_prompt  # type: ignore[assignment]

        messages = [{"role": "user", "content": user_content}]

        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.provider_profile.token_limit_param: int(self.max_tokens),
        }

        if system_prompt.strip():
            request_body["system"] = system_prompt

        temp_value = self._request_temperature()
        if temp_value is not None:
            request_body["temperature"] = temp_value

        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.provider_profile.anthropic_version,
        }

        data: dict[str, Any] = {}
        resp_text = ""
        content = ""
        last_response_error: Optional[APIError] = None
        for attempt in range(self.response_max_retries):
            req = urllib.request.Request(
                self._endpoint, data=body, headers=headers, method="POST",
            )
            resp_text = self._http_request(req)
            try:
                data = self._parse_response(resp_text)

                # Extract text content (skip non-text blocks)
                content_blocks = data.get("content", [])
                text_parts = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = "".join(text_parts)

                if not content.strip():
                    raise APIError(
                        f"Anthropic API response empty content: {json.dumps(data)[:500]}"
                    )
                break
            except APIError as e:
                if attempt == self.response_max_retries - 1:
                    raise
                delay = _api_retry_delay(attempt)
                logger.warning(
                    f"Bad Anthropic response from {self._endpoint} "
                    f"(attempt {attempt + 1}/{self.response_max_retries}), "
                    f"retrying in {delay}s: {e}"
                )
                last_response_error = e
                time.sleep(delay)
        else:  # pragma: no cover - break or raise always exits the loop
            raise last_response_error or APIError("Bad Anthropic response after retries")

        usage = parse_usage(
            data.get("usage", {}),
            self.provider_profile.usage_format,
        )
        if usage.cache_read_tokens or usage.cache_creation_tokens:
            logger.info(
                f"Prompt cache: {usage.cache_read_tokens} read, "
                f"{usage.cache_creation_tokens} created"
            )

        return content, usage, messages, resp_text

    # ── Conversation logging ──

    def _build_conversation_log(
        self,
        messages_sent: list[dict],
        raw_response: str,
        system_prompt: str,
    ) -> dict:
        """Build a conversation log dict, stripping base64 images to save space."""
        # Strip base64 image data from messages to keep logs manageable
        cleaned_messages = []
        for msg in messages_sent:
            content = msg.get("content")
            if isinstance(content, list):
                cleaned_parts = []
                for part in content:
                    if isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype in ("image_url", "image"):
                            cleaned_parts.append({"type": ptype, "data": "[base64_image_stripped]"})
                        else:
                            cleaned_parts.append(part)
                    else:
                        cleaned_parts.append(part)
                cleaned_messages.append({**msg, "content": cleaned_parts})
            else:
                cleaned_messages.append(msg)

        # Parse raw response JSON, strip if too large
        try:
            response_data = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            response_data = {"raw_text": raw_response[:10000]}

        return {
            "model": self.model,
            "temperature": self._request_temperature(),
            "max_tokens": self.max_tokens,
            "system_prompt": system_prompt or None,
            "messages": cleaned_messages,
            "response": response_data,
        }

    # ── Endpoint resolution ──

    @staticmethod
    def _resolve_endpoint(base_url: str) -> str:
        """Append /chat/completions to an OpenAI-compat base URL (idempotent).

        Expects base_url to include the version prefix — e.g. /v1, /v1beta/openai,
        /api/paas/v4. If the user provides a malformed base_url, the request will
        fail with a normal HTTP error.
        """
        url = base_url.rstrip("/")
        return url if url.endswith("/chat/completions") else url + "/chat/completions"

    # ── Usage tracking ──

    def _record_call(self, record: CallRecord) -> None:
        self._usage.total_calls += 1
        self._usage.total_usage += record.usage
        self._usage.total_latency_seconds += record.latency_seconds
        self._usage.calls.append(record)


# ═══════════════════════════════════════════════════════════════════════════
#  JSON extraction (standalone, also usable outside LLMAgent)
# ═══════════════════════════════════════════════════════════════════════════

def extract_json(raw: str) -> dict:
    """Extract a JSON object from LLM output.

    Handles: fenced code blocks, raw JSON, and JSON embedded in prose.
    Uses ``json.loads`` trial-parsing so that braces inside JSON string
    values (e.g. code snippets) are handled correctly.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    text = raw.strip()

    # 1. Fenced code block (```json ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Trial-parse from each '{', trying largest span first.
    candidates = [i for i, ch in enumerate(text) if ch == "{"]
    if not candidates:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")

    for start in candidates:
        closing = [i for i, ch in enumerate(text[start:], start) if ch == "}"]
        for end in reversed(closing):
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse a valid JSON object from model output: {text[:300]}")


# ═══════════════════════════════════════════════════════════════════════════
#  Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class APIError(Exception):
    """Raised when an LLM API call fails."""
    pass


# HTTP status codes that we REFUSE to retry — auth / billing / client-side
# errors where the request will keep failing on re-send. Retry all others
# (including 429/5xx/408/etc.) with configured exponential-backoff attempts.
#   400 Bad Request          — malformed body, retry wastes tokens
#   401 Unauthorized         — bad/missing API key
#   402 Payment Required     — out of credit
#   403 Forbidden            — permission / model-access denied
#   404 Not Found            — bad endpoint or model id
#   422 Unprocessable Entity — validation rejected, same result on retry
_NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 422})

from __future__ import annotations

import json
from copy import deepcopy
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import DeepSeekResponse
from packages.model_policy import (
    API_STYLES, DEEPSEEK_RESPONSES, compatible_model, official_model, validated_endpoint,
)


DEFAULT_ENDPOINT = "https://api.deepseek.com/anthropic/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class DeepSeekClientError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


Transport = Callable[[str, dict[str, str], dict[str, Any], float], Mapping[str, Any]]


class DeepSeekClientProtocol(Protocol):
    model: str

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> DeepSeekResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _validated_endpoint(value: str, api_style: str = "anthropic") -> str:
    return validated_endpoint(value, api_style)


def _structured_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Require declared fields; retain nullable values and absolute $defs refs."""
    schema = deepcopy(schema)
    definitions = schema.pop("$defs", {})
    wrapped = {"type": "object", "properties": {"result": schema},
               "required": ["result"], "additionalProperties": False}
    if definitions:
        wrapped["$defs"] = definitions

    def require_fields(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            node.pop("default", None)
            for value in node.values():
                require_fields(value)
        elif isinstance(node, list):
            for value in node:
                require_fields(value)
    require_fields(wrapped)
    return wrapped


def _chat_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strict-mode Chat schemas reject string length keywords; pydantic still checks them."""
    wrapped = _structured_schema(schema)

    def strip(node):
        if isinstance(node, dict):
            node.pop("minLength", None)
            node.pop("maxLength", None)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)
    strip(wrapped)
    return wrapped


def _structured_content(data: Mapping[str, Any]) -> str:
    if data.get("status") == "incomplete":
        details = data.get("incomplete_details") or {}
        if not isinstance(details, Mapping):
            raise DeepSeekClientError("response_invalid")
        code = "response_truncated" if details.get("reason") == "max_output_tokens" else "response_refused"
        raise DeepSeekClientError(code)
    if data.get("status") != "completed" or data.get("error"):
        raise DeepSeekClientError("response_invalid")
    output = data.get("output")
    if not isinstance(output, list):
        raise DeepSeekClientError("response_invalid")
    blocks = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list) or any(not isinstance(block, Mapping) for block in content):
            raise DeepSeekClientError("response_invalid")
        blocks.extend(content)
    if any(block.get("type") == "refusal" for block in blocks):
        raise DeepSeekClientError("response_refused")
    texts = [block.get("text") for block in blocks if block.get("type") == "output_text"]
    if any(not isinstance(text, str) for text in texts):
        raise DeepSeekClientError("response_invalid")
    content = "".join(texts)
    if not content.strip():
        raise DeepSeekClientError("response_empty")
    try:
        result = json.loads(content)
        if not isinstance(result, dict) or set(result) != {"result"}:
            raise ValueError("invalid structured envelope")
        return json.dumps(result["result"], ensure_ascii=False)
    except (ValueError, TypeError):
        raise DeepSeekClientError("structured_response_invalid") from None


def _default_transport(
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise DeepSeekClientError(f"http_{exc.code}") from exc
    except (URLError, TimeoutError, OSError):
        raise DeepSeekClientError("transport_failed") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DeepSeekClientError("response_invalid") from None
    if not isinstance(value, Mapping):
        raise DeepSeekClientError("response_invalid")
    return value


def _text_content(data: Mapping[str, Any]) -> str:
    if data.get("type") == "error":
        raise DeepSeekClientError("provider_error")
    content = data.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        raise DeepSeekClientError("response_invalid")
    text = "".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}
    ).strip()
    if not text:
        if data.get("stop_reason") == "max_tokens":
            raise DeepSeekClientError("response_truncated")
        raise DeepSeekClientError("response_empty")
    return text


def _chat_content(data: Mapping[str, Any]) -> str:
    """Text of an OpenAI-compatible chat completion; truncation is never success."""
    if data.get("error"):
        raise DeepSeekClientError("provider_error")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise DeepSeekClientError("response_invalid")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise DeepSeekClientError("response_invalid")
    if choice.get("finish_reason") == "length":
        raise DeepSeekClientError("response_truncated")
    if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
        raise DeepSeekClientError("response_refused")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content
                          if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"})
    if not isinstance(content, str):
        raise DeepSeekClientError("response_invalid")
    if not content.strip():
        raise DeepSeekClientError("response_empty")
    return content.strip()


def _json_text(content: str) -> str:
    """Tolerate a Markdown fence around JSON from json_object-only services."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _chat_structured_content(data: Mapping[str, Any]) -> str:
    content = _chat_content(data)
    try:
        result = json.loads(_json_text(content))
        if not isinstance(result, dict) or set(result) != {"result"}:
            raise ValueError("invalid structured envelope")
        return json.dumps(result["result"], ensure_ascii=False)
    except (ValueError, TypeError):
        raise DeepSeekClientError("structured_response_invalid") from None


def _usage(data: Mapping[str, Any], key: str) -> int | None:
    usage = data.get("usage")
    value = usage.get(key) if isinstance(usage, Mapping) else None
    if value is None and isinstance(usage, Mapping):
        # OpenAI-compatible services report prompt/completion token counts.
        alias = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}.get(key)
        value = usage.get(alias) if alias else None
    return value if isinstance(value, int) and value >= 0 else None


class DeepSeekClient:
    """Small injectable client; matching tests can provide a transport fake."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 45.0,
        max_tokens: int = 1000,
        thinking_enabled: bool = False,
        reasoning_effort: str = "high",
        transport: Transport | None = None,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.5,
        api_style: str = "anthropic",
    ) -> None:
        if not api_key.strip():
            raise ValueError("DeepSeek API key is required")
        if api_style not in API_STYLES:
            raise ValueError("Only DeepSeek or an OpenAI-compatible provider is supported")
        (compatible_model if api_style == "openai" else official_model)(model.strip())
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.api_style = api_style
        self.endpoint = _validated_endpoint(endpoint, api_style)
        # Some compatible services only offer json_object; learned on first rejection.
        self.json_schema_supported = True
        self.timeout = max(1.0, min(float(timeout), 180.0))
        self.max_tokens = max(128, min(int(max_tokens), 8_000))
        if reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError("reasoning_effort must be low, medium, high, or max")
        self.thinking_enabled = bool(thinking_enabled)
        self.reasoning_effort = "high" if reasoning_effort == "medium" else reasoning_effort
        self.transport = transport or _default_transport
        self.max_attempts = max(1, min(int(max_attempts), 4))
        self.retry_backoff_seconds = max(0.0, min(float(retry_backoff_seconds), 5.0))

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> DeepSeekResponse:
        return self._complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            thinking_enabled=self.thinking_enabled,
        )

    def complete_with_thinking(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> DeepSeekResponse:
        """Retry a structurally invalid result with explicit reasoning enabled."""

        return self._complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            thinking_enabled=True,
        )

    def complete_structured(self, *, system_prompt: str, user_prompt: str,
                            schema: dict[str, Any], max_tokens: int | None = None,
                            thinking_enabled: bool = False) -> DeepSeekResponse:
        return self._complete(system_prompt=system_prompt, user_prompt=user_prompt,
                              max_tokens=max_tokens, thinking_enabled=thinking_enabled, output_schema=schema)

    def _complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
        thinking_enabled: bool,
        output_schema: dict[str, Any] | None = None,
    ) -> DeepSeekResponse:
        if self.api_style == "openai":
            return self._complete_chat(system_prompt=system_prompt, user_prompt=user_prompt,
                                       max_tokens=max_tokens, output_schema=output_schema)
        payload = {
            "model": self.model,
            "max_tokens": max(128, min(int(max_tokens or self.max_tokens), 8000)),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "reasoning": {
                "effort": self.reasoning_effort if thinking_enabled else "none"
            },
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
            "stream": False,
        }
        if thinking_enabled:
            payload["output_config"] = {"effort": self.reasoning_effort}
        headers = {
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "x-api-key": self.api_key,
        }
        endpoint = self.endpoint
        if output_schema is not None:
            endpoint = DEEPSEEK_RESPONSES
            payload = {
                "model": self.model,
                "instructions": system_prompt,
                "input": user_prompt,
                "reasoning": {"effort": self.reasoning_effort if thinking_enabled else "none"},
                "max_output_tokens": payload["max_tokens"],
                "text": {"format": {"type": "json_schema", "name": "recruitops_result",
                                    "schema": _structured_schema(output_schema)}},
                "stream": False,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(1, self.max_attempts + 1):
            try:
                data = self.transport(endpoint, headers, payload, self.timeout)
                if output_schema is not None:
                    content = _structured_content(data)
                else:
                    if data.get("stop_reason") == "max_tokens":
                        raise DeepSeekClientError("response_truncated")
                    content = _text_content(data)
                return DeepSeekResponse(
                    content=content,
                    model=str(data.get("model") or self.model),
                    input_tokens=_usage(data, "input_tokens"),
                    cache_creation_input_tokens=_usage(
                        data, "cache_creation_input_tokens"
                    ),
                    cache_read_input_tokens=_usage(data, "cache_read_input_tokens"),
                    output_tokens=_usage(data, "output_tokens"),
                )
            except DeepSeekClientError as exc:
                self._raise_unless_retry(exc, attempt)
        raise DeepSeekClientError("transport_failed")

    def _raise_unless_retry(self, exc: DeepSeekClientError, attempt: int) -> None:
        transient = (
            exc.code in {"response_empty", "transport_failed", "http_429"}
            or exc.code.startswith("http_5")
        )
        if not transient or attempt >= self.max_attempts:
            raise exc
        if self.retry_backoff_seconds:
            time.sleep(self.retry_backoff_seconds * attempt)

    def _complete_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
        output_schema: dict[str, Any] | None,
    ) -> DeepSeekResponse:
        """OpenAI-compatible Chat Completions; thinking switches are provider-specific."""
        budget = max(128, min(int(max_tokens or self.max_tokens), 8000))
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # OpenAI's current models reject max_tokens; most compatible services only know it.
        budget_field = ("max_completion_tokens" if "://api.openai.com/" in self.endpoint
                        else "max_tokens")
        schema = _chat_schema(output_schema) if output_schema is not None else None
        attempt = 1
        while True:
            system = system_prompt
            payload: dict[str, Any] = {"model": self.model, budget_field: budget, "stream": False}
            if schema is not None and self.json_schema_supported:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "recruitops_result", "strict": True, "schema": schema}}
            elif schema is not None:
                system += ("\n\nReturn only one JSON object matching this JSON Schema, with the "
                           "answer under the top-level key \"result\":\n"
                           + json.dumps(schema, ensure_ascii=False))
                payload["response_format"] = {"type": "json_object"}
            payload["messages"] = [{"role": "system", "content": system},
                                   {"role": "user", "content": user_prompt}]
            try:
                data = self.transport(self.endpoint, headers, payload, self.timeout)
                content = (_chat_structured_content(data) if schema is not None
                           else _chat_content(data))
                return DeepSeekResponse(
                    content=content,
                    model=str(data.get("model") or self.model),
                    input_tokens=_usage(data, "input_tokens"),
                    cache_creation_input_tokens=None,
                    cache_read_input_tokens=None,
                    output_tokens=_usage(data, "output_tokens"),
                )
            except DeepSeekClientError as exc:
                if (schema is not None and self.json_schema_supported
                        and exc.code in {"http_400", "http_422"}):
                    # Retry once in json_object mode without spending a retry attempt.
                    self.json_schema_supported = False
                    continue
                self._raise_unless_retry(exc, attempt)
                attempt += 1


__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_ENDPOINT",
    "DeepSeekClient",
    "DeepSeekClientError",
    "DeepSeekClientProtocol",
    "Transport",
]

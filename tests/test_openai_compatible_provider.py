import json

import pytest

from packages.codex_runtime.config import CodexHomeConfig
from packages.config import Settings, get_settings
from packages.desktop_runtime.capabilities import saved_model_configured
from packages.matching.client import DeepSeekClient, DeepSeekClientError
from packages.model_policy import normalize_connection
from packages.vision.service import VisionService
from tests.test_owner_configuration import owner  # noqa: F401

ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
PNG = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
       "nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg==")


def chat(content, finish="stop", **extra):
    return {"model": "served-model", "choices": [{"finish_reason": finish,
            "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}, **extra}


@pytest.mark.parametrize("base,model,style,expected", [
    ("https://api.deepseek.com/v1", "deepseek-flash", "openai", ("anthropic", "https://api.deepseek.com")),
    ("https://api.openai.com/v1/", "gpt-4.1-mini", "openai", ("openai", "https://api.openai.com/v1")),
    ("https://api.openai.com/v1/chat/completions", "gpt-4.1", "openai", ("openai", "https://api.openai.com/v1")),
    ("http://127.0.0.1:11434/v1", "qwen2.5:7b", "openai", ("openai", "http://127.0.0.1:11434/v1")),
    ("https://openrouter.ai/api/v1", "openai/gpt-4o", "openai", ("openai", "https://openrouter.ai/api/v1")),
])
def test_supported_connections_normalize(base, model, style, expected):
    assert normalize_connection(base, model, style) == expected


@pytest.mark.parametrize("base,model,style", [
    ("https://gateway.example/v1", "gpt-4.1", "anthropic"),       # third-party Anthropic wire
    ("http://gateway.example/v1", "gpt-4.1", "openai"),           # remote plain HTTP
    ("https://user:pw@gateway.example/v1", "gpt-4.1", "openai"),  # embedded credentials
    ("https://gateway.example/v1?key=1", "gpt-4.1", "openai"),    # query secret
    ("https://gateway.example/v1", "bad model", "openai"),
    ("https://api.deepseek.com", "gpt-4.1", "openai"),            # DeepSeek host keeps its models
])
def test_unsupported_connections_are_rejected(base, model, style):
    with pytest.raises(ValueError):
        normalize_connection(base, model, style)


def test_chat_text_and_usage():
    sent = []

    def transport(endpoint, headers, payload, timeout):
        sent.append((endpoint, headers, payload))
        return chat("你好")

    client = DeepSeekClient(api_key="k", model="Qwen/Qwen3-32B", endpoint=ENDPOINT,
                            api_style="openai", transport=transport)
    response = client.complete(system_prompt="sys", user_prompt="hi", max_tokens=300)
    assert response.content == "你好" and response.input_tokens == 11 and response.output_tokens == 7
    endpoint, headers, payload = sent[0]
    assert endpoint == ENDPOINT and headers["Authorization"] == "Bearer k"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["max_tokens"] == 300
    assert "thinking" not in payload and "reasoning" not in payload


def test_chat_structured_uses_strict_schema_then_json_object_fallback():
    sent = []

    def transport(endpoint, headers, payload, timeout):
        sent.append(payload)
        if payload["response_format"]["type"] == "json_schema":
            raise DeepSeekClientError("http_400")
        return chat('```json\n{"result": {"status": "ok"}}\n```')

    client = DeepSeekClient(api_key="k", model="glm-4.6", endpoint=ENDPOINT, api_style="openai",
                            transport=transport, max_attempts=1)
    schema = {"type": "object", "properties": {"status": {"type": "string", "maxLength": 5}}}
    response = client.complete_structured(system_prompt="sys", user_prompt="u", schema=schema)
    assert json.loads(response.content) == {"status": "ok"}
    strict = sent[0]["response_format"]["json_schema"]
    assert strict["strict"] is True
    assert "maxLength" not in json.dumps(strict["schema"])
    assert sent[1]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in sent[1]["messages"][0]["content"]
    # The client remembers the service lacks json_schema.
    client.complete_structured(system_prompt="sys", user_prompt="u", schema=schema)
    assert sent[2]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("data,code", [
    (chat('{"result": {"a": 1}', finish="length"), "response_truncated"),
    (chat("", finish="content_filter"), "response_refused"),
    (chat("   "), "response_empty"),
    (chat('{"answer": 1}'), "structured_response_invalid"),
    ({"choices": []}, "response_invalid"),
])
def test_chat_structured_failures_are_never_success(data, code):
    client = DeepSeekClient(api_key="k", model="m", endpoint=ENDPOINT, api_style="openai",
                            transport=lambda *args: data, max_attempts=1)
    with pytest.raises(DeepSeekClientError) as error:
        client.complete_structured(system_prompt="s", user_prompt="u", schema={"type": "object"})
    assert error.value.code == code


def test_openai_official_uses_max_completion_tokens():
    sent = []
    client = DeepSeekClient(api_key="k", model="gpt-5-mini",
                            endpoint="https://api.openai.com/v1/chat/completions", api_style="openai",
                            transport=lambda e, h, payload, t: sent.append(payload) or chat("ok"))
    client.complete(system_prompt="s", user_prompt="u", max_tokens=500)
    assert sent[0]["max_completion_tokens"] == 500 and "max_tokens" not in sent[0]


def test_openai_style_cannot_target_deepseek_host():
    with pytest.raises(ValueError):
        DeepSeekClient(api_key="k", model="deepseek-flash", api_style="openai",
                       endpoint="https://api.deepseek.com/chat/completions",
                       transport=lambda *args: pytest.fail("network"))


def test_settings_derive_compatible_endpoints_and_keep_key():
    settings = Settings(_env_file=None, model_api_style="openai", model_name="gpt-4.1-mini",
                        model_api_base_url="https://api.openai.com/v1", llm_api_key="sk-x",
                        model_provider_name="我的 OpenAI")
    assert not settings.model_connection_migration_required
    assert settings.llm_api_key == "sk-x"
    assert settings.llm_endpoint == settings.vision_endpoint == "https://api.openai.com/v1/chat/completions"
    assert settings.codex_model_base_url == "https://api.openai.com/v1"
    assert settings.codex_model_provider_id == "recruitops-openai"
    assert settings.model_provider_name == "我的 OpenAI"
    assert settings.llm_model == settings.codex_model == settings.vision_model == "gpt-4.1-mini"
    assert saved_model_configured({"model_api_style": "openai", "model_name": "gpt-4.1-mini",
                                   "model_api_base_url": "https://api.openai.com/v1", "llm_api_key": "sk-x"})


def test_codex_provider_for_compatible_service():
    toml = CodexHomeConfig(model="gpt-4.1", provider_id="recruitops-openai", provider_name="OpenAI",
                           base_url="https://api.openai.com/v1", api_style="openai",
                           api_key_env="RECRUITOPS_LLM_API_KEY", reasoning_effort="medium",
                           mcp_command="python").render_toml()
    assert 'base_url = "https://api.openai.com/v1"' in toml
    assert 'wire_api = "responses"' in toml
    assert 'model_reasoning_effort = "medium"' in toml
    with pytest.raises(ValueError):
        CodexHomeConfig(model="gpt-4.1", provider_id="x", base_url="https://gateway.example/v1",
                        api_key_env="K", mcp_command="python").render_toml()  # anthropic style


def test_vision_on_compatible_service_skips_deepseek_switches():
    sent = []
    service = VisionService(api_key="k", model="gpt-4.1-mini", api_style="openai",
                            endpoint="https://api.openai.com/v1/chat/completions",
                            transport=lambda e, h, payload, t: sent.append(payload)
                            or chat('{"text": "已投递", "confidence": 0.9}'))
    assert service.analyze(PNG).text == "已投递"
    assert "thinking" not in sent[0] and sent[0]["max_completion_tokens"] == 1600


COMPATIBLE = {"id": "compat", "name": "硅基流动", "provider": "openai_compatible", "api_style": "openai",
              "base_url": "https://api.siliconflow.cn/v1", "model": "Qwen/Qwen3-32B", "api_key": "sk-test"}


def test_compatible_connection_saves_and_becomes_active(owner):  # noqa: F811
    client, headers, root, _ = owner
    response = client.post("/api/local-ui/configuration/save", headers=headers, json={
        "model_connections": [COMPATIBLE], "active_model_connection_id": "compat"})
    assert response.status_code == 200, response.text
    settings = get_settings()
    assert settings.model_api_style == "openai"
    assert settings.llm_endpoint == "https://api.siliconflow.cn/v1/chat/completions"
    assert settings.llm_api_key == "sk-test"
    read = client.post("/api/local-ui/configuration/read", headers=headers).json()
    assert "sk-test" not in json.dumps(read)
    assert read["model_connections"][0]["provider"] == "openai_compatible"
    assert read["model_connections"][0]["key_configured"] is True
    assert not read["model_migration_required"]


def test_compatible_test_reports_missing_responses_support(owner, monkeypatch):  # noqa: F811
    client, headers, _, _ = owner
    calls = []

    def transport(endpoint, headers, payload, timeout):
        calls.append(endpoint)
        if endpoint.endswith("/responses"):
            raise DeepSeekClientError("http_404")
        return chat('{"result": {"status": "ok"}}')

    monkeypatch.setattr("packages.matching.client._default_transport", transport)
    response = client.post("/api/local-ui/configuration/model/test", headers=headers,
                           json={key: value for key, value in COMPATIBLE.items() if key != "name"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["structured_output"] is True and body["assistant_runtime"] is False
    assert "Responses" in body["message"]
    assert calls == ["https://api.siliconflow.cn/v1/chat/completions",
                     "https://api.siliconflow.cn/v1/responses"]


@pytest.mark.parametrize("change", [
    {"provider": "openai_compatible", "api_style": "openai", "base_url": "https://api.deepseek.com"},
    {"provider": "deepseek", "api_style": "anthropic"},
    {"base_url": "http://gateway.example/v1"},
    {"api_style": "anthropic"},
])
def test_invalid_compatible_connections_rejected_before_network(owner, monkeypatch, change):  # noqa: F811
    client, headers, _, _ = owner
    monkeypatch.setattr("packages.matching.client._default_transport",
                        lambda *args: pytest.fail("must reject before network"))
    connection = {**COMPATIBLE, **change}
    tested = client.post("/api/local-ui/configuration/model/test", headers=headers,
                         json={key: value for key, value in connection.items() if key != "name"})
    saved = client.post("/api/local-ui/configuration/save", headers=headers, json={
        "model_connections": [connection], "active_model_connection_id": "compat"})
    assert tested.status_code == saved.status_code == 422

"""Remote model providers: DeepSeek official APIs or an OpenAI-compatible service.

api_style "anthropic" is the DeepSeek preset (Anthropic Messages + Responses on
api.deepseek.com). api_style "openai" is any OpenAI-compatible Chat Completions
service; the assistant additionally needs that service to expose /responses.
"""
import ipaddress
import re
from urllib.parse import urlsplit

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_HOST = "api.deepseek.com"
DEEPSEEK_MODELS = ("deepseek-flash", "deepseek-v4-pro")
DEEPSEEK_MESSAGES = DEEPSEEK_BASE + "/anthropic/v1/messages"
DEEPSEEK_RESPONSES = DEEPSEEK_BASE + "/responses"
API_STYLES = ("anthropic", "openai")
OPENAI_COMPATIBLE_PROVIDER_ID = "recruitops-openai"
_COMPATIBLE_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")


def official_base(value: str) -> str:
    parsed = urlsplit(value.strip().rstrip("/"))
    if (parsed.scheme != "https" or parsed.hostname != DEEPSEEK_HOST
            or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/v1")):
        raise ValueError("Only the official DeepSeek API is supported")
    return DEEPSEEK_BASE


def official_model(value: str) -> str:
    if value not in DEEPSEEK_MODELS:
        raise ValueError("Select deepseek-flash or deepseek-v4-pro")
    return value


def official_endpoint(value: str) -> str:
    parsed = urlsplit(value.strip().rstrip("/"))
    official_base(f"{parsed.scheme}://{parsed.netloc}")
    if (parsed.query or parsed.fragment or parsed.path not in {
            "/anthropic/v1/messages", "/chat/completions", "/responses",
            "/v1/chat/completions", "/v1/responses"}):
        raise ValueError("Unsupported DeepSeek endpoint")
    return value.strip().rstrip("/")


def is_deepseek_host(value: str) -> bool:
    try:
        return urlsplit(str(value or "").strip()).hostname == DEEPSEEK_HOST
    except ValueError:
        return False


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def compatible_base(value: str) -> str:
    """Validate an OpenAI-compatible base URL such as https://api.openai.com/v1."""
    parsed = urlsplit(str(value or "").strip().rstrip("/"))
    host = parsed.hostname or ""
    if (not host or parsed.username or parsed.password or parsed.query or parsed.fragment
            or any(character.isspace() for character in parsed.path)):
        raise ValueError("Invalid OpenAI-compatible base URL")
    # Plain HTTP is only acceptable for a model server on this machine (Ollama, vLLM...).
    if parsed.scheme != "https" and not (parsed.scheme == "http" and _loopback(host)):
        raise ValueError("OpenAI-compatible services must use HTTPS")
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def compatible_model(value: str) -> str:
    if not _COMPATIBLE_MODEL_RE.fullmatch(str(value or "")):
        raise ValueError("Invalid model name")
    return value


def compatible_endpoint(value: str) -> str:
    parsed = urlsplit(value.strip().rstrip("/"))
    if not parsed.path.endswith(("/chat/completions", "/responses")):
        raise ValueError("Unsupported OpenAI-compatible endpoint")
    compatible_base(value)
    return value.strip().rstrip("/")


def normalize_connection(base_url: str, model: str, api_style: str) -> tuple[str, str]:
    """Return (api_style, base) for a supported connection or raise ValueError.

    The DeepSeek host always uses the DeepSeek preset, whatever style was saved.
    """
    if is_deepseek_host(base_url):
        base = official_base(base_url)
        official_model(model)
        return "anthropic", base
    if api_style != "openai":
        raise ValueError("Only DeepSeek or an OpenAI-compatible service is supported")
    base = compatible_base(base_url)
    compatible_model(model)
    return "openai", base


def validated_endpoint(value: str, api_style: str = "anthropic") -> str:
    if api_style == "openai":
        if is_deepseek_host(value):
            raise ValueError("The DeepSeek host uses the DeepSeek preset")
        return compatible_endpoint(value)
    return official_endpoint(value)


def endpoints(base: str, api_style: str) -> dict[str, str]:
    """Wire endpoints for text, structured output, vision and the assistant."""
    if api_style == "openai":
        chat = base + "/chat/completions"
        return {"text": chat, "structured": chat, "vision": chat, "responses": base + "/responses"}
    return {"text": base + "/anthropic/v1/messages", "structured": base + "/responses",
            "vision": base + "/chat/completions", "responses": base + "/responses"}

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SECRET_KEY_PATTERN = re.compile(r"(password|secret|token|key|cookie|auth|credential)", re.IGNORECASE)
PRIVATE_BLOCK_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
AUTH_HEADER_PATTERN = re.compile(r"\b(Basic|Bearer)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
URL_PATTERN = re.compile(r"\b(?:rtsp|rtsps|http|https)://[^\s'\"<>]+", re.IGNORECASE)
ENV_ASSIGNMENT_PATTERN = re.compile(r"^(\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*)(.*)$")


def redact_value(key: str, value: Any, privacy: str = "normal") -> Any:
    if SECRET_KEY_PATTERN.search(str(key)):
        return {"present": bool(str(value).strip())}
    if isinstance(value, dict):
        return {str(child_key): redact_value(str(child_key), child_value, privacy) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [redact_value(key, item, privacy) for item in value]
    if isinstance(value, str):
        return redact_text(value, privacy)
    return value


def redact_mapping(mapping: dict[str, Any], privacy: str = "normal") -> dict[str, Any]:
    return {str(key): redact_value(str(key), value, privacy) for key, value in mapping.items()}


def redact_text(text: str, privacy: str = "normal") -> str:
    redacted = "\n".join(redact_env_line(line, privacy) for line in text.splitlines())
    redacted = PRIVATE_BLOCK_PATTERN.sub("[redacted private key]", redacted)
    redacted = AUTH_HEADER_PATTERN.sub(lambda match: f"{match.group(1)} [redacted]", redacted)
    redacted = URL_PATTERN.sub(lambda match: redact_url(match.group(0), privacy), redacted)
    if privacy == "high":
        redacted = re.sub(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b", r"\1.x", redacted)
    return redacted


def redact_env_line(line: str, privacy: str = "normal") -> str:
    match = ENV_ASSIGNMENT_PATTERN.match(line)
    if not match:
        return line
    prefix, key, value = match.groups()
    if SECRET_KEY_PATTERN.search(key):
        return f"{prefix}[redacted-present:{bool(value.strip())}]"
    return f"{prefix}{redact_text(value, privacy) if value else value}"


def redact_url(url: str, privacy: str = "normal") -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url

    host = parsed.hostname or ""
    if privacy == "high":
        host = re.sub(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b", r"\1.x", host)
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"<credentials>@{netloc}"

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        query_items.append((key, "[redacted]" if SECRET_KEY_PATTERN.search(key) else value))
    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query_items), parsed.fragment))

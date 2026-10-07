from __future__ import annotations

import getpass
import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib import request
from urllib.error import HTTPError, URLError


def key_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "ht-vision-lab" / "groq.key"


def load_api_key() -> str:
    value = os.environ.get("GROQ_API_KEY")
    if value:
        return value
    candidates = [
        key_path(),
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "ht-voice-lab"
        / "groq.key",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("Groq key file permissions are too broad; run chmod 600 " + str(path))
        value = path.read_text().strip()
        if value:
            return value
    raise ValueError("Groq API Key is unavailable; run './lab login-groq'")


def login() -> int:
    value = getpass.getpass("Groq API Key: ").strip()
    if not value:
        raise ValueError("Groq API Key cannot be empty")
    path = key_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix="groq-key-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Groq key saved locally with mode 600: " + str(path))
    return 0


def _error_message(error: HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode(errors="replace"))
        return str(payload["error"]["message"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return str(error.reason)


def _retry_delay(headers, message: str, attempt: int) -> float:
    value = headers.get("Retry-After") if headers else None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        match = re.search(r"try again in ([0-9.]+)(ms|s)", message, re.IGNORECASE)
        if match:
            delay = float(match.group(1))
            return delay / 1000 if match.group(2).lower() == "ms" else delay
    return min(8.0, 0.5 * 2**attempt)


def _bounded_retry_delay(headers, message: str, attempt: int, maximum: float) -> float:
    delay = _retry_delay(headers, message, attempt)
    if delay > maximum:
        raise OSError(
            f"Groq rate limit asks for a {delay:.1f}s wait; rerun later to reuse cached results"
        )
    return delay


def chat_completion(
    payload: dict,
    timeout: float = 60.0,
    proxy: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    if timeout <= 0:
        raise ValueError("Groq timeout must be positive")
    key = api_key or load_api_key()
    selected_proxy = proxy or os.environ.get("GROQ_HTTPS_PROXY")
    opener = (
        request.build_opener(request.ProxyHandler({"https": selected_proxy}))
        if selected_proxy
        else request.build_opener()
    )
    body = json.dumps(payload, ensure_ascii=False).encode()
    for attempt in range(5):
        call = request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "User-Agent": "HT-Vision-Lab/0.1",
            },
        )
        try:
            with opener.open(call, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            message = _error_message(error)
            if error.code == 429 and attempt < 4:
                try:
                    delay = _bounded_retry_delay(error.headers, message, attempt, timeout)
                except OSError as retry_error:
                    raise OSError(f"Groq API 429: {message}; {retry_error}") from None
                time.sleep(delay)
                continue
            hint = (
                "; this can be an account, project, region or network restriction"
                if error.code == 403
                else ""
            )
            raise OSError(f"Groq API {error.code}: {message}{hint}") from None
        except (URLError, TimeoutError) as error:
            if attempt == 4:
                raise OSError(f"Groq network request failed after 5 attempts: {error}") from None
            time.sleep(_retry_delay(None, "", attempt))
    raise OSError("Groq request failed")

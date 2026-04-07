"""Structured error utilities for job/file error fields."""
import json
from typing import Optional


def make_error(
    code: str,
    message: str,
    details: Optional[str] = None,
    hint: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """Build a JSON-encoded structured error string for job.error / file.error."""
    payload = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    if hint is not None:
        payload["hint"] = hint
    if url is not None:
        payload["url"] = url
    return json.dumps(payload, ensure_ascii=False)


def parse_error(error_raw: str) -> dict:
    """Parse an error string.  Returns structured dict or a fallback wrapper."""
    if not error_raw:
        return {"code": "unknown", "message": ""}
    try:
        data = json.loads(error_raw)
        if isinstance(data, dict) and "message" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {"code": "unknown", "message": error_raw}

"""Model calls through OpenRouter, with an on-disk response cache and a cost ceiling.

The ceiling is checked before each request, so requests already in flight can add to the spending after it is reached.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

from .config import ROOT

URL = "https://openrouter.ai/api/v1/chat/completions"
CONNECT_TIMEOUT_S = 30
READ_TIMEOUT_S = 900        # silence between bytes
TOTAL_DEADLINE_S = 1500     # whole request, even if the server trickles keep-alive bytes


class _Response:
    def __init__(self, status_code: int, text: str):
        self.status_code, self.text = status_code, text

    def json(self):
        return json.loads(self.text)


def _post_with_deadline(headers: dict, body: dict) -> _Response:
    """POST and read the body under a hard wall-clock deadline. A plain ``requests`` timeout only
    bounds the silence between bytes, so a connection that keeps trickling bytes can hang a worker
    for hours."""
    started = time.monotonic()
    with requests.post(URL, headers=headers, data=json.dumps(body), stream=True,
                       timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)) as response:
        chunks = []
        for chunk in response.iter_content(chunk_size=65536):
            chunks.append(chunk)
            if time.monotonic() - started > TOTAL_DEADLINE_S:
                raise requests.Timeout(f"no complete response after {TOTAL_DEADLINE_S}s")
        return _Response(response.status_code, b"".join(chunks).decode("utf-8", errors="replace"))


def api_key() -> str:
    """OPENROUTER_API_KEY from the environment, or from a .env file in the repository root."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env in the repository root)")


class CostCeilingReached(RuntimeError):
    pass


class OpenRouterClient:
    """``complete`` returns (text, usage). Identical requests are served from ``cache_dir``.

    The cache key covers the model, the prompt, the sampling settings and the run index, so the
    three runs of a cell are three different requests while a rerun of the same cell is free.
    """

    def __init__(self, cache_dir: Path, ceiling_usd: float, retries: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ceiling, self.retries = ceiling_usd, retries
        self.spent, self.lock = 0.0, threading.Lock()

    def complete(self, api_model: str, prompt: str, *, run_idx: int, temperature: float | None,
                 max_tokens: int, attempt: int = 0,
                 extra: dict | None = None) -> tuple[str, dict]:
        """``attempt`` > 0 asks for a fresh sample of a cell whose earlier response did not parse;
        ``extra`` holds further request fields (``Model.request``). Both are part of the cache
        key only when given, so the keys of the runs behind the paper are unchanged."""
        request = dict(model=api_model, prompt=prompt, temperature=temperature,
                       max_tokens=max_tokens, run_idx=run_idx)
        if attempt:
            request["attempt"] = attempt
        if extra:
            request["extra"] = extra
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            saved = json.loads(cached.read_text())
            return saved["text"], dict(saved["usage"], cached=True)
        with self.lock:
            if self.spent >= self.ceiling:
                raise CostCeilingReached(f"spent ${self.spent:.2f} of ${self.ceiling:.2f}")
        text, usage = self._post(api_model, prompt, temperature, max_tokens, extra)
        with self.lock:
            self.spent += usage["cost_usd"]
        cached.write_text(json.dumps(dict(request=request, text=text, usage=usage)))
        return text, usage

    def _post(self, api_model, prompt, temperature, max_tokens, extra=None) -> tuple[str, dict]:
        body = {"model": api_model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "usage": {"include": True}, **(extra or {})}
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
        last = None
        for attempt in range(self.retries):
            try:
                response = _post_with_deadline(headers, body)
                if response.status_code == 429 or response.status_code >= 500:
                    last = f"http {response.status_code}: {response.text[:200]}"
                elif response.status_code >= 400:
                    # A client error (bad model id, no provider, invalid request) will not
                    # change on retry: fail now with the server's explanation.
                    raise RuntimeError(f"http {response.status_code}: {response.text[:300]}")
                else:
                    payload = response.json()
                    if "error" not in payload:
                        usage = payload.get("usage") or {}
                        details = usage.get("completion_tokens_details") or {}
                        choice = payload["choices"][0]
                        return choice["message"].get("content") or "", dict(
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"),
                            reasoning_tokens=details.get("reasoning_tokens"),
                            cost_usd=float(usage.get("cost") or 0.0),
                            execution_model=payload.get("model") or api_model,
                            finish_reason=choice.get("finish_reason"),
                            native_finish_reason=choice.get("native_finish_reason"),
                            provider=payload.get("provider"))
                    last = str(payload["error"])[:300]
            except (requests.RequestException, KeyError, ValueError) as error:
                last = repr(error)[:300]
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"request failed after {self.retries} attempts: {last}")

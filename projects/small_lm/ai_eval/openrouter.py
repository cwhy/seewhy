"""
Minimal OpenRouter API client (OpenAI-compatible endpoint).

Reads OPENROUTER_KEY from the project .env file.

Usage:
    from openrouter import OpenRouterClient
    client = OpenRouterClient()
    reply = client.chat("Tell me something interesting.")
    reply = client.chat([{"role": "user", "content": "Hello"}])
"""

import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load .env from repo root (ai_eval/ → small_lm/ → projects/ → seewhy/)
_ENV_PATH = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.6-plus:free"


class BudgetExceededError(RuntimeError):
    """Raised when the OpenRouter key's total spend limit is hit (403 Key limit exceeded)."""


class OpenRouterClient:
    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 120.0):
        self.api_key = os.environ.get("OPENROUTER_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_KEY not set in environment / .env")
        self.model   = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self.prompt_tokens:     int = 0
        self.completion_tokens: int = 0

    def get_credits(self) -> float:
        """Return current OpenRouter credit balance in USD."""
        resp = self._client.get(
            f"{OPENROUTER_BASE_URL}/credits",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch credits: {resp.status_code} {resp.text}")
        data = resp.json()
        # Response shape: {"data": {"total_credits": ..., "total_usage": ...}}
        d = data.get("data", data)
        return float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))

    def chat(
        self,
        messages,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retries: int = 5,
    ) -> str:
        """Send a chat request and return the assistant's reply text.

        Args:
            messages: Either a plain string (converted to a single user message)
                      or a list of {"role": ..., "content": ...} dicts.
            system:   Optional system prompt prepended before messages.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            retries:  Number of retries on 429/5xx errors (exponential backoff).

        Returns:
            The assistant's reply as a plain string.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(retries + 1):
            resp = self._client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 10 * (2 ** attempt)
                print(f"  [openrouter] {resp.status_code} — retrying in {wait}s "
                      f"(attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                if resp.status_code == 403 and "Key limit exceeded" in resp.text:
                    raise BudgetExceededError(f"OpenRouter key limit exceeded: {resp.text}")
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
            data = resp.json()
            # Upstream errors arrive as 200 with an "error" body — treat as retryable
            if "error" in data and "choices" not in data:
                err_code = data["error"].get("code", 0)
                if attempt < retries:
                    wait = 15 * (2 ** attempt)
                    print(f"  [openrouter] upstream error (code {err_code}) — retrying in {wait}s "
                          f"(attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenRouter upstream error: {data['error']}")
            if "choices" not in data:
                raise RuntimeError(f"Unexpected response (no 'choices'): {data}")
            usage = data.get("usage", {})
            self.prompt_tokens     += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            msg     = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            # Qwen3 thinking models wrap content in <think>...</think> — strip it
            if content and "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if not content:
                print(f"  [openrouter] empty content — full message keys: {list(msg.keys())}  "
                      f"msg sample: {str(msg)[:200]}")
                if attempt < retries:
                    wait = 10 * (2 ** attempt)
                    print(f"  [openrouter] retrying in {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
            return content.strip()

        raise RuntimeError(f"OpenRouter: exceeded {retries} retries (last status: {resp.status_code})")

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

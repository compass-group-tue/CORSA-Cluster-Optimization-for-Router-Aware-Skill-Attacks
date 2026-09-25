"""LLM client factory — returns either OpenAI or AzureOpenAI.

Uses Azure when AZURE_OPENAI_ENDPOINT is set in the environment.
Falls back to direct OpenAI otherwise.

Azure notes:
- The "model" parameter passed to `.chat.completions.create(...)` is
  interpreted as the DEPLOYMENT NAME on Azure. So calling with
  `model="gpt-4.1"` requires a deployment named `gpt-4.1` to exist.
- The openai Python SDK's AzureOpenAI class handles both request signing
  (api-key header) and the URL layout for you.

Env vars honoured on Azure:
  AZURE_OPENAI_ENDPOINT       (required)  base URL, e.g. https://<your-resource>.openai.azure.com
  AZURE_OPENAI_KEY            (required)  api key
  AZURE_OPENAI_API_VERSION    (optional)  default "2025-01-01-preview"
"""
from __future__ import annotations

import os
import random
import sys
import time


def chat_with_backoff(client, **kwargs):
    """Retry transient API failures; scoring errors are not changed.

    OPENAI_BACKOFF_ATTEMPTS and OPENAI_BACKOFF_BASE_S configure the retry budget.
    """
    from openai import RateLimitError, APITimeoutError, APIConnectionError

    attempts = int(os.environ.get("OPENAI_BACKOFF_ATTEMPTS", "6"))
    base = float(os.environ.get("OPENAI_BACKOFF_BASE_S", "20"))
    last = None
    for i in range(max(1, attempts)):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last = e
            if i == attempts - 1:
                break
            # exponential with jitter, capped at 10 min
            delay = min(base * (2 ** i), 600.0) * (0.75 + 0.5 * random.random())
            print(f"[backoff] {type(e).__name__} — sleeping {delay:.0f}s "
                  f"(attempt {i + 1}/{attempts})", file=sys.stderr, flush=True)
            time.sleep(delay)
    raise last


def make_openai_client():
    """Return AzureOpenAI if configured; otherwise return an OpenAI client.

    OPENAI_MAX_RETRIES and OPENAI_TIMEOUT_S configure SDK retries and request
    timeouts for transient failures and rate limits.
    """
    retries = int(os.environ.get("OPENAI_MAX_RETRIES", "8"))
    timeout = float(os.environ.get("OPENAI_TIMEOUT_S", "120"))
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION",
                                       "2025-01-01-preview"),
            max_retries=retries,
            timeout=timeout,
        )
    from openai import OpenAI
    return OpenAI(max_retries=retries, timeout=timeout)


def make_attacker_client():
    """Return an independently configurable mutator client.

    ATTACKER_API_BASE selects an OpenAI-compatible endpoint, such as vLLM
    serving DeepSeek. ATTACKER_API_KEY supplies its credential. Without an
    endpoint override, use make_openai_client(). This setting does not change
    judge clients or the experiment-provided victim runtime.
    """
    base = os.environ.get("ATTACKER_API_BASE", "").strip()
    if not base:
        return make_openai_client()
    from openai import OpenAI
    retries = int(os.environ.get("OPENAI_MAX_RETRIES", "8"))
    timeout = float(os.environ.get("OPENAI_TIMEOUT_S", "180"))
    return OpenAI(base_url=base,
                  api_key=os.environ.get("ATTACKER_API_KEY", "dummy"),
                  max_retries=retries, timeout=timeout)


def default_deployment(tier: str = "cheap") -> str:
    """Return the default deployment name for the given tier.

    tier: "cheap" | "strong".
    Falls back to gpt-4.1 / gpt-5 if env not set.
    """
    if tier == "cheap":
        return os.environ.get("AZURE_DEPLOYMENT_CHEAP", "gpt-4.1")
    return os.environ.get("AZURE_DEPLOYMENT_STRONG", "gpt-5")

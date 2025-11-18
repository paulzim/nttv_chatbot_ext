# nttv_chatbot/llm_client.py
from __future__ import annotations

import os
import json
import logging
from typing import List, Dict, Any

import requests

from .config import settings

log = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper over whatever LLM backend we use.
    For now: local (if you still have it) vs OpenRouter.
    """

    def __init__(self):
        self.env = settings.ENV

        if self.env == "production":
            if not settings.OPENROUTER_API_KEY:
                raise RuntimeError("OPENROUTER_API_KEY is not set in production environment.")
            log.info("LLMClient initialized in production mode using OpenRouter.")
        else:
            log.info("LLMClient initialized in local mode (you can still wire to local Gemma here).")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        messages: list of {"role": "user"|"system"|"assistant", "content": "..."}
        """

        if self.env == "production":
            return self._generate_openrouter(messages, max_tokens, temperature)
        else:
            # TODO: keep your existing local Gemma call wired in here
            # For now, we can just raise to remind us to implement it
            raise NotImplementedError("Local LLM path not wired yet in LLMClient.")

    def _generate_openrouter(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            # Optional but good etiquette
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "https://ninjatrainingtv.com"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "NTTV Chatbot"),
        }

        payload: Dict[str, Any] = {
            "model": settings.OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(settings.OPENROUTER_BASE_URL, headers=headers, data=json.dumps(payload))
        if resp.status_code != 200:
            log.error("OpenRouter error %s: %s", resp.status_code, resp.text)
            raise RuntimeError(f"OpenRouter API error: {resp.status_code} {resp.text}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            log.error("Unexpected OpenRouter response: %s", data)
            raise RuntimeError("Unexpected OpenRouter response format") from exc

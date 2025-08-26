# app/llm/openai_client.py
"""
OpenAI client wrapper for embeddings and chat (tool-calling capable).

This module centralizes all interactions with the OpenAI API:
- embeddings for indexing/search (RAG)
- chat completions with optional tool/function calling

Environment variables:
    OPENAI_API_KEY      - required
    OPENAI_BASE_URL     - optional (for Azure/OpenAI-compatible proxies)
    OPENAI_EMBEDDING_MODEL - optional, default: text-embedding-3-small
    OPENAI_CHAT_MODEL      - optional, default: gpt-4o-mini
    OPENAI_TEMPERATURE     - optional, default: 0.2
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

__all__ = ["OpenAISettings", "OpenAIClient"]


@dataclass
class OpenAISettings:
    """Typed configuration for OpenAI access and defaults."""
    api_key: str
    base_url: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        """
        Load settings from environment (.env supported).

        Raises:
            RuntimeError: if OPENAI_API_KEY is missing.
        """
        load_dotenv(override=False)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Create a .env file based on .env.example."
            )

        base_url = os.getenv("OPENAI_BASE_URL") or None
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        temp_str = os.getenv("OPENAI_TEMPERATURE", "0.2")
        try:
            temperature = float(temp_str)
        except ValueError:
            temperature = 0.2

        return cls(
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model,
            chat_model=chat_model,
            temperature=temperature,
        )


class OpenAIClient:
    """
    Thin wrapper around the official OpenAI Python client (v1+).

    - `embed_texts(texts)`: returns list of embeddings (one per input).
    - `chat(messages, tools=None, ...)`: returns dict with assistant message (and tool_calls if any).
    """

    def __init__(self, settings: Optional[OpenAISettings] = None) -> None:
        self.settings = settings or OpenAISettings.from_env()
        if self.settings.base_url:
            self.client = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url)
        else:
            self.client = OpenAI(api_key=self.settings.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
    def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.
            model: Optional override for embedding model.

        Returns:
            List[List[float]]: One embedding per input text.

        Raises:
            ValueError: if `texts` is empty or not a list.
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list of strings")
        model_name = model or self.settings.embedding_model
        resp = self.client.embeddings.create(model=model_name, input=texts)
        return [d.embedding for d in resp.data]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run a chat completion with optional function/tool calling.

        Args:
            messages: OpenAI chat messages list. Example:
                [{"role":"system","content":"You are ..."},
                 {"role":"user","content":"Hi"}]
            tools: Optional OpenAI "function" tools schema list.
            model: Optional override for chat model.
            temperature: Optional sampling temperature.

        Returns:
            Dict with:
                - "message": assistant message as a plain dict (includes "tool_calls" if any)
                - "finish_reason": finish reason string
                - "raw": full raw response as a dict
        """
        chat_model = model or self.settings.chat_model
        temp = self.settings.temperature if temperature is None else temperature

        resp = self.client.chat.completions.create(
            model=chat_model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else "none",
            temperature=temp,
        )
        choice = resp.choices[0]
        result = {
            "message": choice.message.model_dump(),
            "finish_reason": choice.finish_reason,
            "raw": resp.model_dump(),
        }
        return result

    @staticmethod
    def extract_tool_calls(assistant_message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract function tool calls (if any) from an assistant message.
        Returns a simplified list: [{"id":..., "name":..., "arguments": "...json..."}]
        """
        calls = assistant_message.get("tool_calls") or []
        out: List[Dict[str, Any]] = []
        for tc in calls:
            if tc.get("type") == "function":
                fn = tc.get("function", {})
                out.append({
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })
        return out


if __name__ == "__main__":
    # Non-network sanity check: just load settings.
    try:
        s = OpenAISettings.from_env()
        print("Loaded OpenAI settings OK.")
        print("Embedding model:", s.embedding_model)
        print("Chat model:", s.chat_model)
    except Exception as e:
        print("Settings error:", e)

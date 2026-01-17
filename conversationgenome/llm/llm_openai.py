from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, List, Union

from openai import OpenAI

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib, model_override

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIModels:
    chat: str = "gpt-4o"
    embedding: str = "text-embedding-3-small"


class LlmOpenAI(LlmLib):
    """
    OpenAI-backed implementation of LlmLib.

    Notes:
      - Uses Chat Completions for text generation (compatible with existing code).
      - Uses Embeddings API for vector embeddings.
      - Validates embedding dimensions for text-embedding-3-small / large.
    """

    def __init__(self, *, models: OpenAIModels | None = None, timeout_s: float = 60.0):
        api_key = c.get("env", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it in your .env file or as an environment variable."
            )

        self.client = OpenAI(api_key=api_key, timeout=timeout_s)
        self.models = models or OpenAIModels()

        # Keep these for compatibility with any parent class expectations
        self.model = self.models.chat
        self.embedding_model = self.models.embedding

    # -------------------------------------------------------------------------
    # Abstract methods override
    # -------------------------------------------------------------------------
    def basic_prompt(
        self,
        prompt: str,
        response_format: str = "text",
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Optional[str]:
        """
        Returns the model's response as text.

        response_format:
          - "text": normal text output
          - "json": forces a valid JSON object output (still returned as string)
        """
        if response_format not in {"text", "json"}:
            raise ValueError("response_format must be 'text' or 'json'")

        api_format = {"type": "json_object"} if response_format == "json" else None

        try:
            # If using JSON mode, it's smart to *tell* the model to output JSON.
            effective_prompt = prompt
            if response_format == "json" and "json" not in prompt.lower():
                effective_prompt = (
                    "Return ONLY a valid JSON object.\n\n"
                    + prompt
                )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": effective_prompt}],
                response_format=api_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.exception("OpenAI Completion Error: %s", e)
            return None

    def basic_prompt_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> Optional[dict[str, Any]]:
        """
        Convenience wrapper that returns parsed JSON (dict) or None on failure.
        """
        text = self.basic_prompt(
            prompt,
            response_format="json",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not text:
            return None

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                logger.warning("Expected JSON object, got %s", type(data).__name__)
                return None
            return data
        except json.JSONDecodeError:
            logger.warning("Model returned invalid JSON: %r", text[:4000])
            return None

    def get_vector_embeddings(self, tag: str, dimensions: int = 1536) -> Optional[List[float]]:
        """
        Returns an embedding vector or None on failure.

        For text-embedding-3-small default is 1536 dims (and can be reduced).
        """
        cleaned = (tag or "").replace("\n", " ").strip()
        if not cleaned:
            return None

        # Validate dimensions for common OpenAI embedding models
        emb_model = self.embedding_model
        max_dims = 1536 if "3-small" in emb_model else 3072 if "3-large" in emb_model else None
        if max_dims is not None and not (1 <= dimensions <= max_dims):
            raise ValueError(f"dimensions must be between 1 and {max_dims} for {emb_model}")

        try:
            response = self.client.embeddings.create(
                input=cleaned,
                model=emb_model,
                dimensions=dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception("OpenAI Embedding Error: %s", e)
            return None

    # -------------------------------------------------------------------------
    # Concrete methods override
    # -------------------------------------------------------------------------
    @model_override("gpt-4.1-mini")
    def validate_conversation_quality(self, conversation):
        return super().validate_conversation_quality(conversation)

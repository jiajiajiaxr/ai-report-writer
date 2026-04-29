from __future__ import annotations

import json
import re
import time
import unicodedata
from typing import Any

import httpx
import numpy as np

from .config import Settings
from .utils import extract_json_block


class SiliconFlowHTTPStatusError(httpx.HTTPStatusError):
    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        provider_code: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response)
        self.provider_code = provider_code
        self.provider_message = provider_message

    @property
    def is_balance_insufficient(self) -> bool:
        message = (self.provider_message or "").lower()
        return self.response.status_code == 403 and (
            self.provider_code == "30001" or "balance is insufficient" in message
        )

    @property
    def is_rate_limited(self) -> bool:
        message = (self.provider_message or "").lower()
        return self.response.status_code == 429 or "rate limit" in message


class SiliconFlowClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = httpx.Client(
            base_url=settings.base_url,  # Use settings base_url (determined by provider detection)
            timeout=settings.request_timeout,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )

    def close(self) -> None:
        self.http.close()

    @staticmethod
    def _build_status_error(response: httpx.Response) -> SiliconFlowHTTPStatusError:
        detail = ""
        provider_code: str | None = None
        provider_message: str | None = None
        try:
            payload = response.json()
            if isinstance(payload, dict):
                code = payload.get("type")
                message = payload.get("error", {}).get("message")
                provider_code = None if code is None else str(code)
                provider_message = None if message is None else str(message)
                if code or message:
                    detail = f" type={code} message={message}"
        except Exception:
            body = (response.text or "").strip()
            if body:
                detail = f" body={body[:400]}"
        return SiliconFlowHTTPStatusError(
            f"Claude request failed: {response.status_code} {response.reason_phrase}.{detail}",
            request=response.request,
            response=response,
            provider_code=provider_code,
            provider_message=provider_message,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: httpx.HTTPStatusError | None = None
        last_transport_error: Exception | None = None
        time.sleep(0.5)
        for attempt in range(4):
            try:
                response = self.http.post(path, json=payload)
            except (httpx.ReadTimeout, httpx.TransportError) as exc:
                last_transport_error = exc
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            if response.is_success:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                time.sleep(1.2 * (attempt + 1))
                continue
            last_error = self._build_status_error(response)
            break
        if last_error is not None:
            raise last_error
        if last_transport_error is not None:
            raise last_transport_error
        raise RuntimeError(f"Request failed for {path}")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        claude_model = self.settings.chat_model  # Use settings chat_model (default MiniMax-M2.7)
        # Extract system prompt from messages if present
        system_prompt = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)

        payload: dict[str, Any] = {
            "model": claude_model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = self._post("/messages", payload)
        # MiniMax returns content as list of blocks (text, thinking, etc.)
        # Extract the first text block, skipping thinking blocks
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"].strip()
        return ""

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_hint: str,
        model: str | None = None,
        max_tokens: int = 8000,
    ) -> Any:
        messages = [
            {"role": "user", "content": f"{user_prompt}\n\n请严格输出 JSON，不要输出解释。\nJSON 结构要求:\n{schema_hint}\n\n重要：输出完整的 JSON，不要截断。如果 JSON 很长，请确保输出完整的数组或对象。"},
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        raw = self.chat(
            messages,
            model=model,
            temperature=0,
            max_tokens=max_tokens,
        )
        if not raw.strip():
            raise ValueError("Empty response from model")
        try:
            return json.loads(extract_json_block(raw))
        except json.JSONDecodeError as exc:
            # Retry once with more tokens if JSON is truncated
            raw2 = self.chat(
                messages,
                model=model,
                temperature=0,
                max_tokens=min(max_tokens * 2, 15000),
            )
            return json.loads(extract_json_block(raw2))

    @staticmethod
    def _clip_embedding_text(text: str, limit: int = 3500) -> str:
        clipped = unicodedata.normalize("NFKC", text.strip())
        clipped = re.sub(r"[^\x20-\x7E\u4e00-\u9fff\n]", " ", clipped)
        clipped = re.sub(r"\s+", " ", clipped).strip()
        if len(clipped) <= limit:
            return clipped
        return clipped[:limit]

    def embed_texts(self, texts: list[str], *, model: str | None = None) -> np.ndarray:
        # Claude does not provide embeddings API, use local hash as fallback
        # This triggers the local-hash fallback in the service layer
        raise httpx.HTTPStatusError(
            "Claude does not support embeddings",
            request=httpx.Request("POST", "/embeddings"),
            response=httpx.Response(501),
        )

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int = 8,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        # Claude does not have a rerank endpoint, return empty to use fallback
        return []

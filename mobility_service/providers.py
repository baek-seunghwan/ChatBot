from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from langsmith import trace


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str


class LLMRouter:
    """Call the primary LLM first and use fallback only on API failure."""

    def __init__(
        self,
        primary_provider: str = "anthropic",
        fallback_provider: str | None = "gemini",
        anthropic_api_key: str | None = None,
        gemini_api_key: str | None = None,
        anthropic_model: str | None = None,
        gemini_model: str | None = None,
        vllm_base_url: str | None = None,
        vllm_api_key: str | None = None,
        vllm_model: str | None = None,
    ) -> None:
        self.primary_provider = primary_provider.lower().strip()
        self.fallback_provider = (
            fallback_provider.lower().strip() if fallback_provider else None
        )
        self.anthropic_api_key = (
            anthropic_api_key
            if anthropic_api_key is not None
            else os.getenv("ANTHROPIC_API_KEY")
        )
        self.gemini_api_key = (
            gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY")
        )
        self.anthropic_model = anthropic_model or os.getenv(
            "ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL
        )
        self.gemini_model = gemini_model or os.getenv(
            "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
        )
        self.vllm_base_url = (
            vllm_base_url
            if vllm_base_url is not None
            else os.getenv("VLLM_BASE_URL", "")
        ).rstrip("/")
        self.vllm_api_key = (
            vllm_api_key
            if vllm_api_key is not None
            else os.getenv("VLLM_API_KEY", "")
        )
        self.vllm_model = vllm_model or os.getenv(
            "VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
        )

    @property
    def provider_order(self) -> list[str]:
        providers = [self.primary_provider]
        if self.fallback_provider and self.fallback_provider not in providers:
            providers.append(self.fallback_provider)
        if self.vllm_base_url and "vllm" not in providers:
            providers.append("vllm")
        return providers

    def _provider_model(self, provider: str) -> str:
        return {
            "anthropic": self.anthropic_model,
            "gemini": self.gemini_model,
            "vllm": self.vllm_model,
        }.get(provider, provider)

    def _anthropic_generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("`pip install anthropic`을 실행하세요.") from exc

        client = Anthropic(api_key=self.anthropic_api_key)
        message = client.messages.create(
            model=self.anthropic_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise RuntimeError("Anthropic이 빈 응답을 반환했습니다.")
        return LLMResult(text=text, provider="anthropic", model=self.anthropic_model)

    def _gemini_generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("`pip install google-genai`를 실행하세요.") from exc

        client = genai.Client(api_key=self.gemini_api_key)
        try:
            response = client.models.generate_content(
                model=self.gemini_model,
                contents=f"{system}\n\n{prompt}",
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            text = (response.text or "").strip()
        finally:
            client.close()
        if not text:
            raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")
        return LLMResult(text=text, provider="gemini", model=self.gemini_model)

    def _vllm_generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResult:
        if not self.vllm_base_url:
            raise RuntimeError("VLLM_BASE_URL이 설정되지 않았습니다.")
        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"
        response = httpx.post(
            f"{self.vllm_base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.vllm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", []) if isinstance(body, dict) else []
        text = (
            choices[0].get("message", {}).get("content", "")
            if choices and isinstance(choices[0], dict)
            else ""
        )
        text = str(text).strip()
        if not text:
            raise RuntimeError("vLLM이 빈 응답을 반환했습니다.")
        return LLMResult(text=text, provider="vllm", model=self.vllm_model)

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> LLMResult:
        with trace(
            "llm-router",
            run_type="chain",
            inputs={
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            tags=["llm", "fallback-router"],
            metadata={"provider_order": self.provider_order},
        ) as router_run:
            errors: list[str] = []
            for provider in self.provider_order:
                model = self._provider_model(provider)
                try:
                    with trace(
                        f"{provider}.generate",
                        run_type="llm",
                        inputs={
                            "prompt": prompt,
                            "system": system,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        tags=[provider],
                        metadata={"model": model},
                    ) as provider_run:
                        if provider == "anthropic":
                            result = self._anthropic_generate(
                                prompt, system, max_tokens, temperature
                            )
                        elif provider == "gemini":
                            result = self._gemini_generate(
                                prompt, system, max_tokens, temperature
                            )
                        elif provider == "vllm":
                            result = self._vllm_generate(
                                prompt, system, max_tokens, temperature
                            )
                        else:
                            raise RuntimeError(f"지원하지 않는 LLM 제공자입니다: {provider}")
                        provider_run.end(
                            outputs={
                                "text": result.text,
                                "provider": result.provider,
                                "model": result.model,
                            }
                        )
                    router_run.end(
                        outputs={
                            "text": result.text,
                            "provider": result.provider,
                            "model": result.model,
                        }
                    )
                    return result
                except Exception as exc:
                    errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:200]}")
            raise RuntimeError(
                "사용 가능한 LLM API가 없습니다 (" + " | ".join(errors) + ")."
            )

"""Pluggable LLM client supporting Anthropic and OpenAI."""

import os
from abc import ABC, abstractmethod
from typing import Optional

from watchbrief.config import LLMConfig


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt

        Returns:
            The generated text response
        """
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self, model: str, temperature: float, api_key: str = ""):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # Use provided api_key or fall back to env var
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in config or environment")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""

    def __init__(self, model: str, temperature: float, api_key: str = ""):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Use provided api_key or fall back to env var
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in config or environment")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration.

    Defaults to Anthropic if ANTHROPIC_API_KEY is set,
    otherwise falls back to OpenAI.

    Args:
        config: LLM configuration

    Returns:
        Configured LLMClient instance
    """
    provider = config.provider.lower()

    # Auto-detect provider if not explicitly set
    if provider == "auto" or not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError(
                "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
            )

    if provider == "anthropic":
        return AnthropicClient(config.model, config.temperature, config.api_key)
    elif provider == "openai":
        return OpenAIClient(config.model, config.temperature, config.api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

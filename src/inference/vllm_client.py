"""Async vLLM client using the OpenAI-compatible API."""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI


class VLLMClient:
    """Client for vLLM's OpenAI-compatible API server.

    Wraps the AsyncOpenAI client with concurrency control via semaphore
    to avoid overwhelming the vLLM server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-Math-1.5B",
        max_concurrent: int = 64,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.8,
        stop: list[str] | None = None,
        n: int = 1,
    ) -> list[str]:
        """Generate n completions for a single prompt.

        Args:
            messages: Conversational format [{"role": ..., "content": ...}]
            temperature: Sampling temperature
            max_tokens: Max tokens per completion
            top_p: Top-p sampling
            stop: Stop sequences
            n: Number of completions to generate

        Returns:
            List of n completion strings
        """
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                n=n,
            )
            return [choice.message.content for choice in response.choices]

    async def generate_batch(
        self,
        prompts: list[list[dict]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate one completion per prompt, in parallel with concurrency control.

        Args:
            prompts: List of message lists (one per request)

        Returns:
            List of completion strings (one per prompt)
        """
        tasks = [
            self.generate(
                p,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                n=1,
            )
            for p in prompts
        ]
        results = await asyncio.gather(*tasks)
        return [r[0] for r in results]

    async def generate_n_for_prompt(
        self,
        messages: list[dict],
        n: int,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.8,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate n diverse completions for a single prompt.

        If the server supports n>1 in a single call, use that.
        Otherwise, falls back to n separate calls.
        """
        try:
            return await self.generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                n=n,
            )
        except Exception:
            # Fallback: n separate calls
            tasks = [
                self.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                    n=1,
                )
                for _ in range(n)
            ]
            results = await asyncio.gather(*tasks)
            return [r[0] for r in results]

import requests
import aiohttp
import asyncio
from typing import Dict, Any
import logging
from .base import LLMProvider

logger = logging.getLogger("chatbot")

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url
        self.model = model

    @property
    def name(self) -> str:
        return "ollama"

    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
            ) as response:
                result = await response.json()
                return result.get("response", "")

    def generate_text(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
        )
        return response.json().get("response", "")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OllamaProvider':
        return cls(
            base_url=config.get('base_url', 'http://localhost:11434'),
            model=config.get('model', 'mistral')
        )

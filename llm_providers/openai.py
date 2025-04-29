import openai
from .base import LLMProvider
import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger("chatbot")

class OpenAiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    @property
    def name(self) -> str:
        return "openai"
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise
            
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        return await asyncio.to_thread(self.generate_text, prompt, **kwargs)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAiProvider':
        return cls(
            api_key=config['api_key'],
            model=config.get('model', 'gpt-3.5-turbo')
        )

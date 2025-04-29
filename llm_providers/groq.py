from typing import Dict, Any
import groq
from .base import LLMProvider
import asyncio
import logging
import os

logger = logging.getLogger("chatbot")

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = groq.Client(api_key=api_key)
        
    @property
    def name(self) -> str:
        return "groq"
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error from Groq: {e}")
            raise e
            
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация текста через Groq API"""
        return await asyncio.to_thread(self.generate_text, prompt, **kwargs)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GroqProvider':
        api_key = config.get("api_key")
        model_name = config.get("model_name", "llama3-8b-8192")
        
        if not api_key:
            raise ValueError("Groq API key not provided")
        
        return cls(api_key=api_key, model_name=model_name)

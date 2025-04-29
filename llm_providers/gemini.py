import google.generativeai as genai
from .base import LLMProvider
import asyncio
from typing import Dict, Any
import logging

logger = logging.getLogger("chatbot")

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
        
    @property
    def name(self) -> str:
        return "gemini"
        
    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            from config import settings
            generation_config = settings.GEMINI_GENERATION_CONFIG
            safety_settings = settings.GEMINI_SAFETY_SETTINGS
            
            # Объединяем конфиги
            combined_config = {**generation_config, **kwargs}
            
            response = self.model.generate_content(
                prompt, 
                generation_config=combined_config,
                safety_settings=safety_settings
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                return "[Ошибка: Пустой ответ от Gemini]"
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            raise
            
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        return await asyncio.to_thread(self.generate_text, prompt, **kwargs)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GeminiProvider':
        return cls(
            api_key=config['api_key'],
            model=config.get('model', 'gemini-1.5-flash')
        )

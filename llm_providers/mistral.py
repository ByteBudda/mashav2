from mistralai import Mistral
import asyncio
from typing import Dict, Any, List, Optional
from .base import LLMProvider

from config import logger

class MistralProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "mistral-small-latest"):
        self.client = Mistral(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Mistral provider with model: {model}")

    @property
    def name(self) -> str:
        return "mistral"

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Синхронная генерация текста через Mistral API"""
        response = self.client.chat.complete(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **kwargs
        )
        return response.choices[0].message.content
    
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация текста через Mistral API"""
        response = await self.client.chat.complete_async(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **kwargs
        )
        return response.choices[0].message.content
    
    @classmethod
    def from_config(cls, config: dict) -> 'MistralProvider':
        """Создает экземпляр провайдера из конфигурации"""
        api_key = config.get('api_key')
        model = config.get('model', 'mistral-small-latest')
        
        if not api_key:
            raise ValueError("API key is required for Mistral")
            
        return cls(api_key=api_key, model=model)

    async def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Генерирует ответ с помощью Mistral API асинхронно"""
        try:
            # Преобразуем сообщения в формат Mistral (проверка ролей)
            mistral_messages = []
            
            for msg in messages:
                role = msg.get('role', '').lower()
                # Mistral поддерживает только 'user', 'assistant', 'system'
                if role not in ['user', 'assistant', 'system']:
                    if role == 'system_instruction':
                        role = 'system'
                    else:
                        role = 'user'  # По умолчанию
                
                mistral_messages.append({
                    "role": role, 
                    "content": msg.get('content', '')
                })
            
            # Параметры запроса
            params = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": temperature
            }
            
            # Добавляем max_tokens только если он задан
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Вызываем API асинхронно
            response = await self.client.chat.complete_async(**params)
            
            generated_text = response.choices[0].message.content
            logger.debug(f"Mistral response generated, length: {len(generated_text)}")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating Mistral response: {str(e)}", exc_info=True)
            return f"Ошибка при генерации ответа: {str(e)}" 
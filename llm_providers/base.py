from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import logging

# Убираем все импорты, создающие циклические зависимости
logger = logging.getLogger("chatbot")

class LLMProvider(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Генерация текста по промпту"""
        pass
    
    @abstractmethod
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация текста"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Имя провайдера"""
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LLMProvider':
        """Инициализация из конфига"""
        pass

# Удаляем классы OllamaProvider и TogetherProvider из base.py и 
# перемещаем их в отдельные файлы: ollama.py и together.py

# Удаляем функцию generate_with_fallback из base.py 
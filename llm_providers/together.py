# -*- coding: utf-8 -*-
import together # Используем основной импорт
import asyncio
from typing import Dict, Any, List # Добавили List
import logging
from .base import LLMProvider

logger = logging.getLogger("chatbot") # Используем ваш стандартный логгер

class TogetherProvider(LLMProvider):
    """
    LLMProvider для работы с Together AI API, используя современный Chat Completions API.
    """
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """
        Инициализирует клиент Together AI.

        Args:
            api_key: Ваш API ключ для Together AI.
            model: Модель для использования (по умолчанию Llama 3.3 70B Instruct Turbo Free).
        """
        # Инициализируем клиент, как в рабочем примере
        self.client = together.Together(api_key=api_key)
        self.model = model
        logger.info(f"TogetherProvider initialized with model: {self.model}")

    @property
    def name(self) -> str:
        return "together"

    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """
        Асинхронно генерирует текст с использованием Chat Completions API.

        Args:
            prompt: Входной текст/запрос пользователя.
            **kwargs: Дополнительные параметры для API (например, max_tokens, temperature).

        Returns:
            Сгенерированный текст или сообщение об ошибке.
        """
        # Преобразуем простой промпт в формат messages для Chat API
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        logger.debug(f"Sending request to Together API (async). Model: {self.model}, Prompt: '{prompt[:60]}...'")
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                **kwargs # Передаем остальные аргументы (max_tokens и т.д.)
            )
            # Обрабатываем ответ, как в рабочем примере
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content.strip()
                logger.debug(f"Received response from Together API (async): '{content[:60]}...'")
                return content
            else:
                logger.warning(f"Empty or unexpected response from Together API (async): {response}")
                return "[Ошибка: Пустой ответ от Together API]"
        except Exception as e:
            logger.error(f"Error calling Together API (async): {e}", exc_info=True)
            return f"[Ошибка API Together: {e}]"

    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Синхронно генерирует текст с использованием Chat Completions API.

        Args:
            prompt: Входной текст/запрос пользователя.
            **kwargs: Дополнительные параметры для API (например, max_tokens, temperature).

        Returns:
            Сгенерированный текст или сообщение об ошибке.
        """
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        logger.debug(f"Sending request to Together API (sync). Model: {self.model}, Prompt: '{prompt[:60]}...'")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs # Передаем остальные аргументы
            )
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content.strip()
                logger.debug(f"Received response from Together API (sync): '{content[:60]}...'")
                return content
            else:
                logger.warning(f"Empty or unexpected response from Together API (sync): {response}")
                return "[Ошибка: Пустой ответ от Together API]"
        except Exception as e:
            logger.error(f"Error calling Together API (sync): {e}", exc_info=True)
            return f"[Ошибка API Together: {e}]"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TogetherProvider':
        """
        Создает экземпляр класса из конфигурационного словаря.
        """
        if 'api_key' not in config:
            raise ValueError("Together API key ('api_key') is required in the configuration.")

        return cls(
            api_key=config['api_key'],
            model=config.get('model', 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free') # Используем модель из примера как дефолтную
        )
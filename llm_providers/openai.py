import openai
from openai import OpenAI, AsyncOpenAI # Используем новые клиенты
from .base import LLMProvider
import asyncio
import logging
from typing import Dict, Any, List, Optional # Добавляем List, Optional

logger = logging.getLogger("chatbot")

class OpenAiProvider(LLMProvider):
    def __init__(self,
                 api_key: Optional[str] = None, # Делаем опциональным
                 base_url: Optional[str] = None, # Добавляем base_url
                 model: str = "gpt-3.5-turbo"):
        """
        Инициализирует провайдер, совместимый с OpenAI API.

        Args:
            api_key: API ключ. Может быть None для API без аутентификации (например, локальных).
            base_url: Базовый URL API (например, 'http://localhost:1234/v1'). Если None, используется API OpenAI по умолчанию.
            model: Имя модели для использования.
        """
        # openai.api_key = api_key # <-- УБРАТЬ ЭТУ СТРОКУ (устарело и не нужно)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url) # Добавляем асинхронный клиент
        self.model = model
        self.base_url = base_url # Сохраняем для информации
        logger.info(f"Initialized OpenAI compatible provider. Model: {self.model}, Base URL: {self.base_url or 'Default OpenAI'}")

    @property
    def name(self) -> str:
        # Можно сделать имя более динамичным, если хотите
        if self.base_url:
            if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
                return f"openai_compatible_local_{self.model}"
            # Можно добавить другие эвристики для известных провайдеров (groq, mistral etc.)
            return f"openai_compatible_{self.model}"
        return f"openai_{self.model}" # Если base_url не задан, считаем, что это OpenAI

    def generate_text(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Генерирует текст с использованием синхронного клиента.

        Args:
            messages: Список сообщений в формате OpenAI [{'role': 'user', 'content': '...'}, ...].
            **kwargs: Дополнительные параметры для API (temperature, top_p, etc.).
        """
        try:
            logger.debug(f"Sending request to {self.name}. Messages: {messages}, kwargs: {kwargs}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages, # Используем список сообщений
                **kwargs
            )
            content = response.choices[0].message.content
            logger.debug(f"Received response from {self.name}: {content[:100]}...")
            return content or "" # Возвращаем пустую строку, если content is None
        except Exception as e:
            logger.error(f"OpenAI compatible API error ({self.name}): {e}", exc_info=True)
            # Можно вернуть специальное значение или перевыбросить исключение
            # в зависимости от логики вашего основного приложения
            return f"[Error: {e}]"
            # raise # Или перевыбросить, если хотите обрабатывать выше

    async def generate_text_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Генерирует текст с использованием асинхронного клиента.

        Args:
            messages: Список сообщений в формате OpenAI [{'role': 'user', 'content': '...'}, ...].
            **kwargs: Дополнительные параметры для API (temperature, top_p, etc.).
        """
        # Используем нативный асинхронный клиент
        try:
            logger.debug(f"Sending ASYNC request to {self.name}. Messages: {messages}, kwargs: {kwargs}")
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages, # Используем список сообщений
                **kwargs
            )
            content = response.choices[0].message.content
            logger.debug(f"Received ASYNC response from {self.name}: {content[:100]}...")
            return content or "" # Возвращаем пустую строку, если content is None
        except Exception as e:
            logger.error(f"OpenAI compatible API ASYNC error ({self.name}): {e}", exc_info=True)
            return f"[Async Error: {e}]"
            # raise

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OpenAiProvider':
        """
        Создает экземпляр из словаря конфигурации.

        Ожидаемые ключи в config:
            api_key (str, optional): API ключ.
            base_url (str, optional): Базовый URL API.
            model (str, optional): Имя модели (по умолчанию 'gpt-3.5-turbo').
        """
        api_key = config.get('api_key') # Может быть None
        base_url = config.get('base_url') # Может быть None
        model = config.get('model', 'gpt-3.5-turbo')

        # Логируем предупреждения, если ключ отсутствует (может быть нормально для локальных моделей)
        if not api_key and not base_url:
             logger.warning("api_key is not set for OpenAI provider, assuming default OpenAI API or local model that doesn't require a key.")
        elif not api_key and base_url:
             logger.warning(f"api_key is not set for OpenAI provider at base_url {base_url}. This might be intended for local models, but could fail if the endpoint requires a key.")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
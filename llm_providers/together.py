# -*- coding: utf-8 -*-
import together
import requests
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging
from .base import LLMProvider
import asyncio
import os
import random
from io import BytesIO
import base64
from PIL import Image
from telegram import Update, constants
from telegram.ext import ContextTypes






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

    async def generate_vision_async(self, image_bytes: bytes, prompt: str, caption: Optional[str] = None) -> str:
        """
        Асинхронно получает описание изображения через Llama Vision (Together API).
        :param image_bytes: байты изображения
        :param prompt: основной промпт для vision
        :param caption: подпись пользователя (опционально)
        :return: описание изображения (строка)
        """
        # --- Сжимаем и определяем mime-type ---
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                img_format = img.format or "JPEG"
                mime_type = f"image/{img_format.lower()}"
                img.thumbnail((768, 768))
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                compressed_image_bytes = buffer.getvalue()
        except Exception:
            compressed_image_bytes = image_bytes
            mime_type = "image/jpeg"

        image_base64 = base64.b64encode(compressed_image_bytes).decode('utf-8')

        # --- Формируем prompt для vision ---
        prompt_vision = prompt
        if caption:
            prompt_vision += f"\n\nТакже учти следующее описание к изображению: «{caption}»."

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_vision},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]
        }]

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="meta-llama/Llama-Vision-Free",
                messages=messages,
                max_tokens=1024,
            )
            if response and response.choices and len(response.choices) > 0:
                description_ru = response.choices[0].message.content.strip()
                logger.debug(f"Llama Vision response: {description_ru[:60]}...")
                return description_ru
            else:
                logger.warning(f"Empty or unexpected response from Llama Vision: {response}")
                return "[Ошибка: Пустой ответ от Llama Vision]"
        except Exception as e:
            logger.error(f"Error calling Llama Vision: {e}", exc_info=True)
            return f"[Ошибка Vision Together: {e}]"

    async def generate_any_async(self, prompt: str, image_bytes: bytes = None, caption: str = None) -> str:
        if image_bytes:
            return await self.generate_vision_async(image_bytes, prompt, caption)
        else:
            return await self.generate_text_async(prompt)

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
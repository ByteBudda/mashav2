import os
import asyncio
import logging
from typing import Dict, Any, List

# Обновленные импорты для новой библиотеки google-genai
import google.generativeai as genai
from google.generativeai import types

logger = logging.getLogger("chatbot")

# Предполагаем, что LLMProvider - это ваш базовый класс
from .base import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-04-17"): # Обновили модель по умолчанию на ту, что в новом примере
        # Инициализируем новый клиент
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        # Больше не нужно инициализировать объект GenerativeModel напрямую

    @property
    def name(self) -> str:
        return "gemini"

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            # Пытаемся импортировать настройки, как в старом коде
            try:
                from config import settings
                generation_config_dict = getattr(settings, 'GEMINI_GENERATION_CONFIG', {})
                safety_settings_list = getattr(settings, 'GEMINI_SAFETY_SETTINGS', [])
            except ImportError:
                # Если settings нет, используем пустые дефолты
                logger.warning("config.settings not found. Using default empty generation_config and safety_settings.")
                generation_config_dict = {}
                safety_settings_list = []

            # Объединяем настройки генерации из settings и kwargs (kwargs имеют приоритет)
            # Исключаем ключи, которые не являются параметрами GenerateContentConfig (например, 'tools')
            gen_config_params = ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'response_mime_type']
            combined_generation_config = {
                key: kwargs.get(key, generation_config_dict.get(key))
                for key in gen_config_params if kwargs.get(key) is not None or generation_config_dict.get(key) is not None
            }


            # Безопасные настройки: используем переданные в kwargs['safety_settings'] или из settings
            # Предполагается, что safety_settings в settings или kwargs - это список словарей
            combined_safety_settings = kwargs.get('safety_settings', safety_settings_list)

            # Обработка инструментов: используем переданные в kwargs['tools']
            # Предполагается, что tools в kwargs - это список объектов types.Tool или словарей
            tools_list = kwargs.get('tools')


            # Формируем содержимое запроса в новом формате
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            # Создаем объект GenerateContentConfig
            generation_config = types.GenerationConfig(**combined_generation_config)

            # Преобразуем список словарей безопасных настроек в список объектов types.SafetySetting
            safety_settings = [
                s if isinstance(s, types.SafetySetting) else types.SafetySetting(**s)
                for s in combined_safety_settings
            ]

            # Преобразуем список словарей инструментов в список объектов types.Tool, если нужно
            tools = None # Дефолтное значение
            if tools_list:
                tools = [
                    t if isinstance(t, types.Tool) else types.Tool(**t)
                    for t in tools_list
                ]


            generate_content_config = types.GenerateContentConfig(
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools, # Добавляем инструменты в конфиг
                # Если response_mime_type был в combined_generation_config, он уже тут будет
                # Или можно добавить его явно: response_mime_type=combined_generation_config.get('response_mime_type', 'text/plain')
            )

            # Вызываем синхронный метод generate_content нового клиента
            # [1] - Использование client.models.generate_content для не-стриминга
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )

            # Извлекаем текст из ответа и обрабатываем возможные ошибки/блокировки
            # [5] - Обработка заблокированных ответов через prompt_feedback
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason
                 logger.warning(f"Gemini generation blocked: {block_reason}")
                 return f"[Ошибка: Ответ заблокирован по соображениям безопасности: {block_reason}]"

            if response.candidates:
                 candidate = response.candidates[0] # Обычно берем первого кандидата
                 # [5] - Проверка finish_reason у кандидата
                 if candidate.finish_reason and candidate.finish_reason.name != 'STOP' and candidate.finish_reason.name != 'MAX_TOKENS':
                      finish_reason = candidate.finish_reason.name
                      logger.warning(f"Gemini generation finished with reason: {finish_reason}")
                      return f"[Ошибка: Генерация завершена с причиной: {finish_reason}]"

                 # [1] - Извлечение текста из ответа
                 if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                      # Ответ с tools может содержать ToolCode или FunctionCall
                      text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text')]
                      return "".join(text_parts) if text_parts else "[Ошибка: Ответ не содержит текстовых частей]"
                 elif hasattr(response, 'text') and response.text: # Fallback для простого текстового ответа
                     return response.text
                 else:
                      logger.warning(f"Gemini returned an unexpected response structure: {response}")
                      return "[Ошибка: Неожиданная структура ответа от Gemini]"
            else:
                logger.warning(f"Gemini returned no candidates: {response}")
                return "[Ошибка: Нет кандидатов ответа от Gemini]"

        except Exception as e:
            logger.error(f"Gemini error during sync generation: {e}")
            # Проверяем специфические ошибки, например, InvalidArgument
            if "google.api_core.exceptions.InvalidArgument" in str(type(e)):
                 return f"[Ошибка API: Некорректный аргумент запроса Gemini - {e}]"
            return f"[Ошибка: Ошибка Gemini - {e}]"


    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        try:
            # Пытаемся импортировать настройки
            try:
                from config import settings
                generation_config_dict = getattr(settings, 'GEMINI_GENERATION_CONFIG', {})
                safety_settings_list = getattr(settings, 'GEMINI_SAFETY_SETTINGS', [])
            except ImportError:
                 logger.warning("config.settings not found during async. Using default empty generation_config and safety_settings.")
                 generation_config_dict = {}
                 safety_settings_list = []

            # Объединяем настройки генерации из settings и kwargs
            gen_config_params = ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'response_mime_type']
            combined_generation_config = {
                key: kwargs.get(key, generation_config_dict.get(key))
                for key in gen_config_params if kwargs.get(key) is not None or generation_config_dict.get(key) is not None
            }

            combined_safety_settings = kwargs.get('safety_settings', safety_settings_list)
            tools_list = kwargs.get('tools')


            # Формируем содержимое запроса
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            # Создаем объект GenerateContentConfig
            generation_config = types.GenerationConfig(**combined_generation_config)

            safety_settings = [
                s if isinstance(s, types.SafetySetting) else types.SafetySetting(**s)
                for s in combined_safety_settings
            ]

            tools = None
            if tools_list:
                 tools = [
                     t if isinstance(t, types.Tool) else types.Tool(**t)
                     for t in tools_list
                 ]

            generate_content_config = types.GenerateContentConfig(
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools,
                # response_mime_type=combined_generation_config.get('response_mime_type', 'text/plain')
            )

            # Вызываем АСИНХРОННЫЙ метод generate_content нового клиента
            # [4], [9] - Использование client.aio.models.generate_content для асинхронных вызовов
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )

            # Извлекаем текст и обрабатываем ошибки/блокировки (аналогично синхронной версии)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback.block_reason, 'name') else response.prompt_feedback.block_reason
                 logger.warning(f"Gemini async generation blocked: {block_reason}")
                 return f"[Ошибка: Ответ заблокирован по соображениям безопасности (async): {block_reason}]"

            if response.candidates:
                 candidate = response.candidates[0]
                 if candidate.finish_reason and candidate.finish_reason.name != 'STOP' and candidate.finish_reason.name != 'MAX_TOKENS':
                      finish_reason = candidate.finish_reason.name
                      logger.warning(f"Gemini async generation finished with reason: {finish_reason}")
                      return f"[Ошибка: Генерация завершена с причиной (async): {finish_reason}]"

                 if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                      text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text')]
                      return "".join(text_parts) if text_parts else "[Ошибка: Ответ не содержит текстовых частей (async)]"
                 elif hasattr(response, 'text') and response.text:
                      return response.text
                 else:
                      logger.warning(f"Gemini returned an unexpected response structure (async): {response}")
                      return "[Ошибка: Неожиданная структура ответа от Gemini (async)]"
            else:
                logger.warning(f"Gemini returned no candidates (async): {response}")
                return "[Ошибка: Нет кандидатов ответа от Gemini (async)]"


        except Exception as e:
            logger.error(f"Gemini error during async generation: {e}")
            if "google.api_core.exceptions.InvalidArgument" in str(type(e)):
                 return f"[Ошибка API: Некорректный аргумент запроса Gemini (async) - {e}]"
            return f"[Ошибка: Ошибка Gemini (async) - {e}]"


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GeminiProvider':
        # Этот метод остается прежним, просто передает параметры в обновленный __init__
        return cls(
            api_key=config['api_key'],
            model=config.get('model', 'gemini-2.5-flash-preview-04-17') # Обновляем модель по умолчанию и здесь
        )

# Пример использования (предполагая наличие 'config' модуля и базового класса LLMProvider)
# Для запуска этого примера нужно установить google-genai: pip install google-genai
if __name__ == "__main__":
    # Это заглушка для демонстрации
    # Создайте реальный файл config.py или установите переменную окружения GEMINI_API_KEY
    # import os
    # os.environ["GEMINI_API_KEY"] = "ВАШ_API_КЛЮЧ" # Замените на ваш реальный API ключ или используйте env переменную

    class MockSettings:
        # Замените на реальные настройки из вашего config.py
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "ВАШ_API_КЛЮЧ_ДЛЯ_ТЕСТА") # Используйте реальный ключ
        GEMINI_GENERATION_CONFIG = {"temperature": 0.8, "max_output_tokens": 300}
        GEMINI_SAFETY_SETTINGS = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    class MockLLMProvider: # Упрощенный базовый класс для примера
         pass

    # Имитируем импорт модулей config и .base
    import sys
    sys.modules['config'] = MockSettings
    # sys.modules['.base'] = MockLLMProvider # Предполагается, что .base уже доступен в PYTHONPATH

    # Удалите эту строку или замените на ваш реальный путь к .base
    # Пример простой заглушки, если .base не доступен:
    if '.base' not in sys.modules:
         print("Using mock LLMProvider base class")
         class LLMProvider:
             pass
         sys.modules['.base'] = sys.modules[__name__] # Имитация импорта в той же папке
         GeminiProvider.__bases__ = (LLMProvider,) # Устанавливаем базовый класс

    print("Testing synchronous generation:")
    try:
        # Создаем экземпляр провайдера
        gemini_provider = GeminiProvider(api_key=MockSettings.GEMINI_API_KEY) # Модель по умолчанию gemini-2.5-flash-preview-04-17
        # gemini_provider = GeminiProvider.from_config({"api_key": MockSettings.GEMINI_API_KEY, "model": "gemini-1.5-flash-002"}) # Пример использования from_config

        # Тестируем обычный запрос
        response_text = gemini_provider.generate_text("Расскажи короткую шутку.")
        print(f"Response (sync): {response_text}")

        print("\nTesting synchronous generation with Google Search tool:")
        # Тестируем запрос с инструментом поиска
        # Важно: Модель должна поддерживать инструменты. gemini-2.5-flash-preview-04-17 поддерживает.
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        response_text_with_tool = gemini_provider.generate_text(
            "Какая погода сегодня в Цюрихе? Используй поиск Google.",
            tools=[google_search_tool] # Передаем инструмент через kwargs
        )
        print(f"Response (sync with tool): {response_text_with_tool}")


    except Exception as e:
        print(f"Synchronous generation failed: {e}")

    print("\nTesting asynchronous generation:")
    async def async_test():
        try:
            gemini_provider_async = GeminiProvider(api_key=MockSettings.GEMINI_API_KEY) # Модель по умолчанию
            # gemini_provider_async = GeminiProvider.from_config({"api_key": MockSettings.GEMINI_API_KEY, "model": "gemini-1.5-flash-002"})

            # Тестируем обычный запрос асинхронно
            response_text_async = await gemini_provider_async.generate_text_async("Столица Канады?")
            print(f"Response (async): {response_text_async}")

            print("\nTesting asynchronous generation with Google Search tool:")
            # Тестируем запрос с инструментом поиска асинхронно
            google_search_tool_async = types.Tool(google_search=types.GoogleSearch())
            response_text_async_with_tool = await gemini_provider_async.generate_text_async(
                "Кто мэр Лондона в данный момент? Используй поиск.",
                tools=[google_search_tool_async] # Передаем инструмент через kwargs
            )
            print(f"Response (async with tool): {response_text_async_with_tool}")

        except Exception as e:
            print(f"Asynchronous generation failed: {e}")

    # Запускаем асинхронный тест
    asyncio.run(async_test())
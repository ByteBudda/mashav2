import os
import re
import time
import asyncio
from io import BytesIO
from functools import lru_cache
from typing import Dict, Set, Optional, List, Any # Добавили List, Any
import json
import google.generativeai as genai
import speech_recognition as sr
from PIL import Image
from pydub import AudioSegment
from telegram import Update
import random
from transformers import pipeline
from telegram.ext import CallbackContext
# Импорты из проекта
from config import (logger, GEMINI_API_KEY, DEFAULT_STYLE, BOT_NAME,
                    CONTEXT_CHECK_PROMPT, ASSISTANT_ROLE, settings, USER_ROLE, SYSTEM_ROLE) # Добавили SYSTEM_ROLE
# Импортируем нужные части состояния из state.py
from state import (user_info_db, group_preferences, group_user_style_prompts,
                   user_preferred_name) # Убрали bot_activity_percentage, т.к. есть get_bot_activity_percentage
# Импортируем функцию из vector_store
from vector_store import get_last_bot_message

# ==============================================================================
# Начало: Содержимое utils.py
# ==============================================================================

# --- Инициализация AI ---
# Модель инициализируется здесь, чтобы быть доступной для функций utils
gemini_model: Optional[genai.GenerativeModel] = None

def initialize_gemini_model():
    global gemini_model
    if gemini_model:
        logger.debug("Gemini model already initialized in utils.")
        return True
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY not found in environment variables.")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Используем модель из настроек
        gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        logger.info(f"Gemini AI model '{settings.GEMINI_MODEL_NAME}' initialized successfully in utils.")
        return True
    except Exception as e:
        logger.critical(f"Failed to configure Gemini AI in utils: {e}", exc_info=True)
        gemini_model = None
        return False

# Вызов инициализации при загрузке модуля
initialize_gemini_model()

# --- RuBERT Pipelines ---
_ner_pipeline = None
_sentiment_pipeline = None

def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            # Убедитесь, что модель существует и доступна
            _ner_pipeline = pipeline("ner", model="Data-Lab/rubert-base-cased-conversational_ner-v3", tokenizer="Data-Lab/rubert-base-cased-conversational_ner-v3")
            logger.info("RuBERT NER pipeline initialized.")
        except Exception as e:
            logger.error(f"Error initializing RuBERT NER pipeline: {e}", exc_info=True)
            _ner_pipeline = None # Устанавливаем в None при ошибке
    return _ner_pipeline

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            # Убедитесь, что модель существует и доступна
            _sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
            logger.info("RuBERT Sentiment Analysis pipeline initialized.")
        except Exception as e:
            logger.error(f"Error initializing RuBERT Sentiment Analysis pipeline: {e}", exc_info=True)
            _sentiment_pipeline = None # Устанавливаем в None при ошибке
    return _sentiment_pipeline

# --- Prompt Builder Class ---
class PromptBuilder:
    def __init__(self, bot_name): # Убрали default_style, он приходит в system_message_base
        self.bot_name = bot_name
        # self.default_style = default_style # Больше не нужно

    def build_prompt(self, history_str: str, user_profile_info: str, user_name: str, prompt_text: str, system_message_base: str, topic_context: str = "", entities: Optional[List] = None, sentiment: Optional[Dict] = None):
        """Строит финальный промпт для Gemini, разделяя историю и профиль."""
        prompt_parts = [system_message_base] # Начинаем с базового стиля/роли

        # Добавляем статичную информацию о пользователе, если она есть
        if user_profile_info:
            prompt_parts.append(f"\n### Информация о пользователе ({user_name})")
            prompt_parts.append(f"# (Эту информацию не нужно повторять в каждом ответе, используй её для понимания контекста)")
            prompt_parts.append(f"# {user_profile_info}")
            prompt_parts.append("### Конец информации о пользователе")

        # Добавляем тему, если есть
        if topic_context:
            prompt_parts.append(f"\n{topic_context}") # Тема разговора

        # Добавляем динамическую историю диалога
        if history_str:
            prompt_parts.append(f"\n### Недавний диалог (История):")
            prompt_parts.append(history_str)
            prompt_parts.append("### Конец диалога")
        else:
            prompt_parts.append("\n(Это начало нового диалога)")

        # Добавляем текущее сообщение пользователя
        prompt_parts.append(f"\n{user_name}: {prompt_text}") # Текущий ввод пользователя

        # Добавляем доп. анализ, если есть
        if entities:
            prompt_parts.append(f"\n(Извлеченные сущности: {entities})")
        if sentiment:
            prompt_parts.append(f"(Тональность сообщения: {sentiment})")

        # Завершаем промпт ожиданием ответа бота + инструкция
        prompt_parts.append(f"\n{self.bot_name}:") # Ожидание ответа бота
        prompt_parts.append("(Важно: Ответь кратко и по делу на последнее сообщение пользователя. НЕ повторяй информацию о пользователе или всю историю диалога без явной необходимости.)")

        final_prompt = "\n".join(prompt_parts)
        # Ограничиваем длину лога для превью
        log_preview_length = 500
        logger.debug(f"Built prompt for Gemini. Length: {len(final_prompt)}. Preview: {final_prompt[:log_preview_length]}{'...' if len(final_prompt) > log_preview_length else ''}")
        return final_prompt

# --- Инициализация PromptBuilder (используем имя из настроек) ---
# Убираем DEFAULT_STYLE из инициализации
prompt_builder = PromptBuilder(settings.BOT_NAME)

# --- AI и Вспомогательные Функции ---

@lru_cache(maxsize=128) # Кэширование для одинаковых текстовых промптов
def generate_content_sync(prompt: str) -> str:
    """Синхронная обертка для вызова Gemini API (текст)."""
    global gemini_model # Используем глобальную модель
    if not gemini_model:
        # Попытка повторной инициализации
        if not initialize_gemini_model():
             logger.error("Gemini model not initialized. Cannot generate content.")
             return "[Ошибка: Модель AI не инициализирована]"

    logger.info(f"Sending prompt to Gemini (first 100 chars): {prompt[:100]}...")
    try:
        # Добавляем safety_settings для уменьшения блокировок (настройте по необходимости)
        safety_settings_text = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings_text)

        # Улучшенная проверка ответа
        if hasattr(response, 'text') and response.text:
            logger.info(f"Received response from Gemini (first 100 chars): {response.text[:100]}...")
            return response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason
             logger.warning(f"Gemini response blocked. Reason: {reason}")
             return f"[Ответ заблокирован: {reason}]"
        else:
            # Логируем полный ответ для диагностики, если он не пустой
            try:
                full_response_log = f"Gemini response was empty or lacked text. Full response: {response}"
                logger.warning(full_response_log[:1000]) # Ограничиваем длину лога
            except Exception as log_e:
                logger.warning(f"Gemini response was empty or lacked text. Error logging full response: {log_e}")
            return "[Пустой или некорректный ответ от Gemini]"
    except Exception as e:
        logger.error(f"Gemini content generation error: {e}", exc_info=True)
        return f"[Произошла ошибка при генерации ответа: {type(e).__name__}]"

async def generate_vision_content_async(contents: list) -> str:
    """Асинхронная функция для вызова Gemini Vision API."""
    global gemini_model # Используем глобальную модель
    if not gemini_model:
        if not initialize_gemini_model():
             logger.error("Gemini model not initialized. Cannot generate vision content.")
             return "[Ошибка: Модель AI не инициализирована]"

    logger.info("Sending image/prompt to Gemini Vision...")
    try:
        safety_settings_vision = [ # Могут отличаться для Vision
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Вызов generate_content остается синхронным, выполняем в потоке
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(contents, safety_settings=safety_settings_vision)
        )
        # response = gemini_model.generate_content(contents, safety_settings=safety_settings_vision) # Синхронный вариант

        response_text = response.text if hasattr(response, 'text') else ''
        if not response_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason
            response_text = f"[Ответ на изображение заблокирован: {reason}]"
            logger.warning(f"Gemini Vision response blocked. Reason: {reason}")
        elif not response_text:
             logger.warning("Gemini Vision response was empty.")
             response_text = "[Не удалось получить описание изображения]"

        logger.info(f"Received vision response (first 100 chars): {response_text[:100]}...")
        return response_text
    except Exception as e:
        logger.error(f"Gemini Vision API error: {e}", exc_info=True)
        return "[Ошибка при анализе изображения]"

def filter_response(response: str) -> str:
    """Очищает ответ от потенциальных артефактов."""
    if not response or response.startswith("["): # Не трогаем сообщения об ошибках/блокировках
        return response

    # Убираем стандартные префиксы ролей (регистронезависимо)
    # Добавляем BOT_NAME для удаления самоцитирования
    prefixes_to_remove = [f"{role.lower()}:" for role in [USER_ROLE, ASSISTANT_ROLE, SYSTEM_ROLE, BOT_NAME]]
    filtered = response.strip()
    # Повторяем удаление, пока префиксы находятся
    while True:
        original_length = len(filtered)
        for prefix in prefixes_to_remove:
            if filtered.lower().startswith(prefix):
                filtered = filtered[len(prefix):].strip()
        if len(filtered) == original_length: # Если длина не изменилась, префиксов больше нет
            break

    # Дополнительная очистка от возможных артефактов (если нужно)
    # filtered = re.sub(r"```[\w\W]*?```", "", filtered) # Удаление блоков кода
    # filtered = re.sub(r"`[^`]+`", "", filtered) # Удаление inline кода

    # Удаление лишних пробелов и пустых строк
    filtered = "\n".join(line.strip() for line in filtered.splitlines() if line.strip())

    return filtered.strip()

async def transcribe_voice(file_path: str) -> Optional[str]:
    """Распознает речь из аудиофайла (WAV)."""
    logger.info(f"Attempting to transcribe audio file: {file_path}")
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            logger.info(f"Audio data recorded from file {file_path}.")
            try:
                # Запускаем распознавание в отдельном потоке
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, # Стандартный ThreadPoolExecutor
                    lambda: recognizer.recognize_google(audio_data, language="ru-RU")
                )
                # text = recognizer.recognize_google(audio_data, language="ru-RU") # Синхронный вариант
                logger.info(f"Transcription successful for {file_path}: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning(f"Google Speech Recognition could not understand audio from {file_path}.")
                return "[Не удалось распознать речь]"
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google SR service for {file_path}: {e}")
                return f"[Ошибка сервиса распознавания: {e}]"
    except FileNotFoundError:
         logger.error(f"Audio file not found for transcription: {file_path}")
         return "[Ошибка: Файл не найден]"
    except Exception as e:
        logger.error(f"Error processing audio file {file_path}: {e}", exc_info=True)
        return "[Ошибка обработки аудио]"
    # finally блок не нужен, т.к. удаление файла происходит в вызывающем коде

async def _get_effective_style(chat_id: int, user_id: int, user_name: Optional[str], chat_type: str) -> str:
    """Определяет эффективный стиль общения, учитывая настройки группы и пользователя."""
    # Используем переменные состояния, импортированные из state.py
    if chat_type in ['group', 'supergroup']:
        # 1. Стиль, заданный админом для конкретного пользователя в этой группе
        user_group_style = group_user_style_prompts.get((chat_id, user_id))
        if user_group_style:
            logger.debug(f"Using specific group user style for {user_id} in {chat_id}.")
            return user_group_style
        # 2. Общий стиль для группы (если задан)
        group_style = group_preferences.get(chat_id, {}).get("style")
        if group_style:
            logger.debug(f"Using group style for chat {chat_id}.")
            return group_style

    # 3. Персональный стиль пользователя (если не в группе или групповые не заданы)
    # Стили из user_info_db пока не реализованы, используем только имя
    # user_personal_style = user_info_db.get(user_id, {}).get("preferences", {}).get("style")
    # if user_personal_style: return user_personal_style

    # 4. Стиль по умолчанию
    logger.debug(f"Using default style for user {user_id} in chat {chat_id}.")
    return settings.DEFAULT_STYLE # Из config.py

async def is_context_related(current_message: str, user_id: int, chat_id: int, chat_type: str) -> bool:
    """Проверяет, связано ли сообщение пользователя с последним ответом бота, используя ChromaDB."""
    history_key = chat_id if chat_type in ['group', 'supergroup'] else user_id

    last_bot_message = await get_last_bot_message(history_key)

    if not last_bot_message:
        logger.debug(f"No previous bot message found in vector store for key {history_key}. Assuming not related.")
        return False
    if len(current_message.split()) < 2 and current_message.lower() not in ['да', 'нет', 'ага', 'угу', 'спасибо', 'спс', 'ок']:
        logger.debug(f"Message '{current_message}' too short for context check.")
        return False # Слишком короткое сообщение, вероятно не контекстное (кроме простых ответов)

    prompt = CONTEXT_CHECK_PROMPT.format(current_message=current_message, last_bot_message=last_bot_message)
    logger.debug(f"Checking context for user {user_id} in chat {chat_id} (vector store)")
    try:
        # Используем generate_content_sync в отдельном потоке
        response_text = await asyncio.to_thread(generate_content_sync, prompt)
        logger.debug(f"Context check response: {response_text}")
        is_related = response_text.strip().lower().startswith("да")
        logger.info(f"Context check result for user {user_id} (vector store): {is_related}")
        return is_related
    except Exception as e:
        logger.error(f"Error during context check (vector store): {e}", exc_info=True)
        return False

async def update_user_info(update: Update):
    """Обновляет информацию о пользователе в user_info_db."""
    if not update.effective_user: return

    user = update.effective_user
    user_id = user.id
    current_time = time.time()

    # Инициализируем запись, если пользователя нет
    if user_id not in user_info_db:
        user_info_db[user_id] = {"preferences": {}, "first_seen": current_time}
        # Устанавливаем имя по умолчанию при первом появлении
        user_preferred_name[user_id] = user.first_name
        logger.info(f"New user detected: {user.first_name} ({user_id}).")

    # Обновляем данные
    user_info_db[user_id]["username"] = user.username
    user_info_db[user_id]["first_name"] = user.first_name
    user_info_db[user_id]["last_name"] = user.last_name
    user_info_db[user_id]["is_bot"] = user.is_bot
    user_info_db[user_id]["language_code"] = user.language_code
    user_info_db[user_id]["last_seen"] = current_time

    # Убеждаемся, что имя есть в user_preferred_name
    if user_id not in user_preferred_name:
         user_preferred_name[user_id] = user.first_name

    logger.debug(f"User info updated for user_id: {user_id}")

async def cleanup_audio_files_job(context: CallbackContext): # context нужен для JobQueue
    """Периодическая задача для удаления временных аудио/видео файлов."""
    bot_folder = "." # Текущая директория
    deleted_count = 0
    logger.debug("Starting temporary audio/video file cleanup...")
    try:
        current_time = time.time()
        # Ищем файлы старше 1 часа (3600 секунд)
        cleanup_older_than = 3600
        for filename in os.listdir(bot_folder):
            # Ищем файлы, созданные нашими хендлерами
            if filename.startswith(("voice_", "video_note_")) and filename.lower().endswith((".wav", ".oga", ".mp4")):
                file_path = os.path.join(bot_folder, filename)
                try:
                    file_mod_time = os.path.getmtime(file_path)
                    if current_time - file_mod_time > cleanup_older_than:
                        os.remove(file_path)
                        logger.info(f"Deleted old temporary audio/video file: {file_path}")
                        deleted_count += 1
                    else:
                         logger.debug(f"Skipping relatively new temporary file: {file_path}")
                except FileNotFoundError:
                     logger.warning(f"Temporary file {file_path} not found during cleanup (possibly already deleted).")
                except Exception as e:
                    logger.error(f"Error processing/deleting file {file_path}: {e}")
        if deleted_count > 0:
            logger.info(f"Temporary audio/video file cleanup finished. Deleted {deleted_count} old files.")
        else:
            logger.debug("Temporary audio/video file cleanup: No old files found to delete.")
    except Exception as e:
        logger.error(f"Error during temporary audio/video file cleanup scan: {e}", exc_info=True)

def should_process_message(activity_percentage: int) -> bool:
    """Определяет, следует ли обрабатывать сообщение на основе процента активности."""
    if activity_percentage >= 100:
        return True
    if activity_percentage <= 0:
        return False
    return random.randint(1, 100) <= activity_percentage

def get_bot_activity_percentage() -> int:
    """Возвращает текущий процент активности бота."""
    # импортируем здесь, чтобы избежать циклического импорта, если utils импортируется из state
    from state import bot_activity_percentage
    return bot_activity_percentage

# ==============================================================================
# Конец: Содержимое utils.py
# ==============================================================================
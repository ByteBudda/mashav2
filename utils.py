# -*- coding: utf-8 -*-
# utils.py
import os
import re
import time
import asyncio
from io import BytesIO
from functools import lru_cache
from typing import Optional, List, Dict, Any, Tuple, Deque
import json
import pickle
import sqlite3
import random

# Убираем Faiss, numpy
import google.generativeai as genai
import speech_recognition as sr
from PIL import Image
from pydub import AudioSegment
from telegram import Update
from datetime import datetime
# Импортируем токенизатор
try:
    from transformers import AutoTokenizer
except ImportError:
    logger.warning("Transformers library not found. Token counting will use simple split(). Install with: pip install transformers")
    AutoTokenizer = None # type: ignore

# Используем settings и константы из config
from config import (logger, GEMINI_API_KEY, ASSISTANT_ROLE, settings, TEMP_MEDIA_DIR,
                    TOKENIZER_MODEL_NAME, CONTEXT_MAX_TOKENS)
# Импортируем функции доступа к БД и состояние из state
from state import (
    chat_history, get_user_info_from_db, update_user_info_in_db,
    get_user_preferred_name_from_db, set_user_preferred_name_in_db,
    get_group_user_style_from_db, get_group_style_from_db,
    get_user_topic_from_db
)
# Импортируем типы ролей
from config import USER_ROLE, ASSISTANT_ROLE, SYSTEM_ROLE

# --- Инициализация Gemini ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL) # Переименовали переменную
    logger.info(f"Gemini AI model initialized successfully: {settings.GEMINI_MODEL}")
except Exception as e:
    logger.critical(f"Failed to configure Gemini AI: {e}", exc_info=True)
    gemini_model = None

# --- Инициализация Токенизатора ---
tokenizer = None
if AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
        logger.info(f"Tokenizer '{TOKENIZER_MODEL_NAME}' loaded.")
        # logger.debug(f"Token test 'привет мир': {tokenizer.encode('привет мир')}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{TOKENIZER_MODEL_NAME}': {e}. Using simple split().")
        tokenizer = None
else:
     logger.warning("Transformers library not found. Using simple split() for token counting.")

def count_tokens(text: str) -> int:
    """Подсчитывает токены в тексте с помощью загруженного токенизатора или fallback."""
    if tokenizer:
        try:
            # add_special_tokens=False, т.к. считаем токены для частей промпта
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
             logger.warning(f"Tokenizer failed for text '{text[:50]}...': {e}. Falling back to word count.")
             return len(text.split())
    else: # Fallback
        return len(text.split())


# --- Новая функция сборки контекста ---
def build_optimized_context(
    system_message_base: str,
    topic_context: str,
    current_message_text: str, # Текст с типом (voice/video)
    user_name: str,
    history_deque: Deque[Tuple[str, Optional[str], str, str]],
    relevant_history: List[Tuple[str, Dict[str, Any]]], # (text, metadata)
    relevant_facts: List[Tuple[str, Dict[str, Any], float]], # (text, metadata)
    max_tokens: int = CONTEXT_MAX_TOKENS
) -> List[str]:
    """
    Собирает контекст для LLM, приоритизируя части и соблюдая лимит токенов.
    Возвращает список строк для финального промпта.
    """
    context_parts: List[str] = []
    current_tokens = 0

    # --- Обязательные части ---
    # 1. Системный промпт
    if system_message_base:
        sys_tokens = count_tokens(system_message_base)
        if current_tokens + sys_tokens <= max_tokens:
            context_parts.append(system_message_base); current_tokens += sys_tokens
        else: logger.warning("System prompt too long!"); context_parts.append(system_message_base[:max_tokens // 2]); return context_parts # Обрезка

    # 2. Тема
    if topic_context:
        topic_tokens = count_tokens(topic_context)
        if current_tokens + topic_tokens <= max_tokens: context_parts.append(topic_context); current_tokens += topic_tokens

    # 3. Текущее сообщение пользователя (проверяем заранее)
    current_msg_full = f"{USER_ROLE} ({user_name}): {current_message_text}"
    current_msg_tokens = count_tokens(current_msg_full)
    if current_tokens + current_msg_tokens > max_tokens:
        logger.warning("Not enough tokens for current message after system/topic.")
        # Пытаемся убрать тему, если она есть
        if topic_context and topic_context in context_parts:
            context_parts.remove(topic_context); current_tokens -= topic_tokens
            if current_tokens + current_msg_tokens <= max_tokens: context_parts.append(current_msg_full)
        # Если не помогло или темы не было, возвращаем то, что есть (только системный промпт)
        return context_parts

    # Место для истории = max_tokens - current_tokens (уже включает sys+topic) - current_msg_tokens
    available_tokens_for_history = max_tokens - current_tokens - current_msg_tokens
    added_history_parts = [] # Собираем историю здесь

    # --- Приоритетные части (Факты -> Недавние -> Релевантные сообщения) ---

    # 4. Релевантные Факты
    
    RELEVANCE_THRESHOLD = 0.5  # Define a suitable threshold value
    highly_relevant_facts_exist = any(dist < RELEVANCE_THRESHOLD for _, _, dist in relevant_facts)

    if available_tokens_for_history > 0 and relevant_facts and highly_relevant_facts_exist:
        title = "Ключевые факты из памяти:"; title_tokens = count_tokens(title) + 2
        if available_tokens_for_history >= title_tokens:
            temp_fact_parts = [title]; temp_fact_tokens = title_tokens; seen_facts = set()
        # Итерируем по фактам, но добавляем только те, что прошли порог (опционально)
        # ИЛИ просто добавляем все найденные, раз уж решили показать секцию
            for fact_text, _, dist in relevant_facts: # Берем расстояние
            # Можно добавить доп. фильтр: if dist < RELEVANCE_THRESHOLD:
                cleaned_fact = re.sub(r"^(.*?):\s*", "", fact_text).strip()
                if cleaned_fact and cleaned_fact not in seen_facts:
                    line = f"- {cleaned_fact}"; line_tokens = count_tokens(line)
                    if temp_fact_tokens + line_tokens <= available_tokens_for_history:
                        temp_fact_parts.append(line); temp_fact_tokens += line_tokens; seen_facts.add(cleaned_fact)
                    else: break
            if len(temp_fact_parts) > 1: # Если добавили хотя бы один факт
             added_history_parts.extend(temp_fact_parts); available_tokens_for_history -= temp_fact_tokens

    # 5. Недавняя история (из deque, новые первыми)
    if available_tokens_for_history > 0 and history_deque:
        title = "Недавний диалог (последние сообщения):"; title_tokens = count_tokens(title) + 2
        if available_tokens_for_history >= title_tokens:
            temp_recent_parts = []; temp_recent_tokens = title_tokens # Считаем заголовок сразу
            for role, name, msg, ts_str in reversed(history_deque): # Начиная с самого нового
                if role != SYSTEM_ROLE:
                    # Добавляем временную метку в строку
                    prefix = f"[{ts_str}] {role}"
                    if role == USER_ROLE and name:
                        line = f"{prefix} ({name}): {msg}"
                    else:
                        line = f"{prefix}: {msg}"
                    line_tokens = count_tokens(line)
                    if temp_recent_tokens + line_tokens <= available_tokens_for_history:
                         temp_recent_parts.append(line); temp_recent_tokens += line_tokens
                    else: break
            # Добавляем в правильном порядке (старые выше)
            if temp_recent_parts:
                added_history_parts.append(title)
                added_history_parts.extend(reversed(temp_recent_parts))
                available_tokens_for_history -= temp_recent_tokens

    # 6. Релевантная история сообщений (из ChromaDB)
    if available_tokens_for_history > 0 and relevant_history:
        title = "Релевантные фрагменты из предыдущего общения:"; title_tokens = count_tokens(title) + 2
        if available_tokens_for_history >= title_tokens:
            temp_relevant_parts = [title]; temp_relevant_tokens = title_tokens; seen_hist = set()
            for msg_text, metadata in relevant_history:
                cleaned_msg = re.sub(r"^(user|assistant|system)\s*\(.*?\):\s*|^(user|assistant|system):\s*", "", msg_text, flags=re.IGNORECASE).strip()
                if cleaned_msg and cleaned_msg not in seen_hist:
                    role_prefix = f"{metadata.get('role', '?')}: " if metadata.get('role') else ""
                    line = f"- {role_prefix}{cleaned_msg}"; line_tokens = count_tokens(line)
                    if temp_relevant_tokens + line_tokens <= available_tokens_for_history:
                        temp_relevant_parts.append(line); temp_relevant_tokens += line_tokens; seen_hist.add(cleaned_msg)
                    else: break
            if len(temp_relevant_parts) > 1: added_history_parts.extend(temp_relevant_parts); available_tokens_for_history -= temp_relevant_tokens

    # --- Собираем финальный результат ---
    context_parts.extend(added_history_parts) # Добавляем собранную историю
    context_parts.append(current_msg_full) # Добавляем текущее сообщение

    final_token_count = max_tokens - available_tokens_for_history
    logger.debug(f"Optimized context built. Tokens approx: {final_token_count}/{max_tokens}. Parts: {len(context_parts)}")
    return context_parts


# --- PromptBuilder (использует новую функцию) ---
class PromptBuilder:
    def __init__(self, bot_settings: settings.__class__):
        self.settings = bot_settings

    def build_prompt(self,
                     history_deque: Deque[Tuple[str, Optional[str], str]],
                     relevant_history: List[Tuple[str, Dict[str, Any]]],
                     relevant_facts: List[Tuple[str, Dict[str, Any]]],
                     user_name: str,
                     current_message_text: str, # Текст с типом (voice/video)
                     system_message_base: str,
                     topic_context=""):
        """Строит промпт, используя build_optimized_context."""
        context_lines = build_optimized_context(
            system_message_base=system_message_base, topic_context=topic_context,
            current_message_text=current_message_text, user_name=user_name,
            history_deque=history_deque, relevant_history=relevant_history,
            relevant_facts=relevant_facts, max_tokens=CONTEXT_MAX_TOKENS
        )
        system_message_base = f"{system_message_base} Ты - {self.settings.BOT_NAME}." # Ваша базовая строка стиля

        # --- ДОБАВЛЯЕМ ЯВНЫЕ ИНСТРУКЦИИ ---
        system_message_base += (
            "\n\nТвоя главная задача - ответить на ПОСЛЕДНЕЕ сообщение пользователя ({USER_ROLE})."
            "\nИнформация из разделов 'Ключевые факты из памяти' и 'Релевантные фрагменты' дана тебе как КОНТЕКСТ."
            "\nИСПОЛЬЗУЙ эту информацию для лучшего понимания ситуации, НО НЕ УПОМИНАЙ старые факты или события из прошлого в своем ответе, ЕСЛИ ТОЛЬКО пользователь НЕ СПРАШИВАЕТ о них НАПРЯМУЮ в своем ПОСЛЕДНЕМ сообщении или если это АБСОЛЮТНО НЕОБХОДИМО для ответа на его ПОСЛЕДНИЙ вопрос."
            "\nСосредоточься на поддержании текущего потока беседы, основываясь на 'Недавнем диалоге' и 'ПОСЛЕДНЕМ сообщении пользователя'."
            "\nНе начинай ответ с приветствия, если не было приветствия в последнем сообщении пользователя."
        )
        context_lines.append(f"\n{self.settings.BOT_NAME}:") # Приглашение к ответу
        final_prompt = "\n".join(context_lines).strip()
        return final_prompt

# --- Генерация контента ---
@lru_cache(maxsize=128)
def generate_content_sync(prompt: str) -> str:
    """Синхронная функция генерации текста с Gemini."""
    if not gemini_model: return "[Ошибка: Модель Gemini не инициализирована]"
    logger.info(f"Sending prompt to Gemini ({len(prompt)} chars)...")
    try:
        # Используем настройки из config.settings
        gen_config = settings.GEMINI_GENERATION_CONFIG
        safety = getattr(settings, 'GEMINI_SAFETY_SETTINGS', None) # Безопасность пока не настраиваем
        response = gemini_model.generate_content(prompt, generation_config=gen_config, safety_settings=safety)

        # Обработка ответа
        if hasattr(response, 'text') and response.text: return response.text
        # ... (обработка block_reason, finish_reason) ...
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: return f"[Ответ заблокирован: {response.prompt_feedback.block_reason}]"
        elif response.candidates and response.candidates[0].finish_reason != 'STOP': return f"[Ответ прерван: {response.candidates[0].finish_reason}]"
        else: logger.warning(f"Gemini empty response: {response}"); return "[Пустой ответ от Gemini]"
    except Exception as e: logger.error(f"Gemini generation error: {e}", exc_info=True); return f"[Ошибка генерации: {type(e).__name__}]"

async def generate_vision_content_async(contents: list) -> str:
    """Асинхронная функция генерации текста по изображению с Gemini."""
    if not gemini_model: return "[Ошибка: Модель Gemini не инициализирована]"
    logger.info("Sending image/prompt to Gemini Vision...")
    try:
        gen_config = settings.GEMINI_GENERATION_CONFIG
        safety = getattr(settings, 'GEMINI_SAFETY_SETTINGS', None)
        # Используем to_thread для блокирующего вызова
        response = await asyncio.to_thread(gemini_model.generate_content, contents, generation_config=gen_config, safety_settings=safety)
        # ... (обработка ответа как в generate_content_sync) ...
        resp_text = ""
        if hasattr(response, 'text') and response.text: resp_text = response.text
        elif response.candidates and hasattr(response.candidates[0],'content') and response.candidates[0].content.parts: resp_text = "".join(p.text for p in response.candidates[0].content.parts if hasattr(p,'text'))

        if not resp_text: # Проверка блокировки/ошибки
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason: return f"[Ответ на изображение заблокирован: {response.prompt_feedback.block_reason}]"
            elif response.candidates and response.candidates[0].finish_reason != 'STOP': return f"[Ответ на изображение прерван: {response.candidates[0].finish_reason}]"
            else: logger.warning(f"Gemini vision empty response: {response}"); return "[Не удалось получить описание изображения]"
        return resp_text
    except Exception as e: logger.error(f"Gemini Vision error: {e}", exc_info=True); return "[Ошибка анализа изображения]"

# --- Фильтрация ответа (без изменений) ---
def filter_response(response: str) -> str:
    if not response: return ""
    filtered = re.sub(r"^(assistant:|system:|user:|model:)\s*", "", response, flags=re.IGNORECASE | re.MULTILINE).strip()
    filtered = re.sub(r"```[\w\W]*?```", "", filtered) # Убираем блоки кода
    filtered = re.sub(r"`[^`]+`", "", filtered) # Убираем инлайн-код
    filtered = re.sub(r"^\*+(.*?)\*+$", r"\1", filtered).strip() # Убираем * в начале/конце
    if filtered.startswith('{') and filtered.endswith('}'): # Пытаемся убрать JSON
        try: data=json.loads(filtered); f=data.get('response', data.get('text')); filtered=f if isinstance(f,str) else ""
        except: pass
    return "\n".join(line.strip() for line in filtered.splitlines() if line.strip())

# --- Распознавание речи (без изменений) ---
async def transcribe_voice(file_path: str) -> Optional[str]:
    # ... (код как раньше) ...
    logger.info(f"Transcribing: {file_path}")
    r = sr.Recognizer(); text = None
    try:
        with sr.AudioFile(file_path) as source: audio = r.record(source)
        text = await asyncio.to_thread(r.recognize_google, audio, language="ru-RU")
        logger.info(f"Transcription OK: {text}")
    except sr.UnknownValueError: logger.warning("Audio not understood."); text="[Не удалось распознать речь]"
    except sr.RequestError as e: logger.error(f"Google API error: {e}"); text=f"[Ошибка сервиса: {e}]"
    except FileNotFoundError: logger.error(f"File not found: {file_path}"); text="[Ошибка: Файл не найден]"
    except Exception as e: logger.error(f"Audio processing error: {e}"); text="[Ошибка обработки аудио]"
    finally:
        if os.path.exists(file_path): 
            try: 
                os.remove(file_path) 
            except Exception: 
                pass
    return text


# --- Определение эффективного стиля (без изменений) ---
async def _get_effective_style(chat_id: int, user_id: int, user_name: Optional[str], chat_type: str) -> str:
    # ... (код как раньше) ...
    style = None
    if chat_type in ['group', 'supergroup']:
        style = await asyncio.to_thread(get_group_user_style_from_db, chat_id, user_id)
        if style: return style
        style = await asyncio.to_thread(get_group_style_from_db, chat_id)
        if style: return style
    return settings.DEFAULT_STYLE

# --- Обновление информации о пользователе (без изменений) ---
async def update_user_info(update: Update):
    # ... (код как раньше) ...
    if not update.effective_user: return
    user=update.effective_user; uid=user.id
    await asyncio.to_thread(update_user_info_in_db, uid, user.username, user.first_name, user.last_name, user.language_code, user.is_bot)
    async def set_name():
        if not await asyncio.to_thread(get_user_preferred_name_from_db, uid):
            name = user.first_name or f"User_{uid}"; await asyncio.to_thread(set_user_preferred_name_in_db, uid, name); logger.info(f"Set default name '{name}' for {uid}")
    asyncio.create_task(set_name())


# --- Очистка временных медиа файлов (без изменений) ---
async def cleanup_audio_files_job(context):
    # ... (код как раньше) ...
    cnt=0; tdir=TEMP_MEDIA_DIR; age=3600*3; logger.debug(f"Cleanup media in '{tdir}' (> {age}s)...")
    try: os.makedirs(tdir, exist_ok=True)
    except OSError as e: logger.error(f"Access error '{tdir}': {e}"); return
    now=time.time()
    try:
        for fn in os.listdir(tdir):
            if (fn.startswith(("voice_", "vnote_")) and fn.lower().endswith((".wav",".oga",".mp4"))):
                fp=os.path.join(tdir, fn)
                try:
                    if now - os.path.getmtime(fp) > age: os.remove(fp); logger.info(f"Deleted old: {fp}"); cnt+=1
                except FileNotFoundError: continue
                except Exception as e: logger.error(f"Error deleting {fp}: {e}")
        if cnt > 0: logger.info(f"Media cleanup done. Deleted {cnt} files.")
        else: logger.debug("Media cleanup: No old files.")
    except Exception as e: logger.error(f"Media cleanup scan error: {e}")


# --- Проверка активности бота (без изменений) ---
def should_process_message() -> bool:
    from state import bot_activity_percentage
    if bot_activity_percentage >= 100: return True
    if bot_activity_percentage <= 0: return False
    return random.randint(1, 100) <= bot_activity_percentage
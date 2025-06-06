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
from collections import defaultdict
import hashlib
from collections import OrderedDict

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
                    TOKENIZER_MODEL_NAME, CONTEXT_MAX_TOKENS, BotSettings)

# Импортируем функции доступа к БД и состояние из state
from state import (
    chat_history, get_user_info_from_db, update_user_info_in_db,
    get_user_preferred_name_from_db, set_user_preferred_name_in_db,
    get_group_user_style_from_db, get_group_style_from_db,
    get_user_topic_from_db
)
# Импортируем типы ролей
from config import USER_ROLE, ASSISTANT_ROLE, SYSTEM_ROLE

# Импортируем базовый класс провайдера
from llm_providers.base import LLMProvider

# Импортируем провайдеры
from llm_providers import (
    MistralProvider,
    GeminiProvider,
    OpenAiProvider,
    GroqProvider,
    TogetherProvider,
    OllamaProvider
)

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
    is_reply_to_bot: bool = False,
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
    if available_tokens_for_history > 0 and relevant_facts:
        title = "Ключевые факты из памяти:"; title_tokens = count_tokens(title) + 2
        if available_tokens_for_history >= title_tokens:
            temp_fact_parts = [title]; temp_fact_tokens = title_tokens; seen_facts = set()
            for fact_text, _, dist in relevant_facts: # Берем расстояние
                # Добавляем только факты, прошедшие порог релевантности из настроек
                if dist < settings.FACTS_RELEVANCE_THRESHOLD:
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
            # Используем полный deque
            for role, name, msg, ts_str in reversed(history_deque): # Итерируем по полному deque
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
    def __init__(self, bot_settings: BotSettings):
        self.settings = bot_settings

    def build_prompt(self,
                     history_deque: Deque[Tuple[str, Optional[str], str]],
                     relevant_history: List[Tuple[str, Dict[str, Any]]],
                     relevant_facts: List[Tuple[str, Dict[str, Any]]],
                     user_name: str,
                     current_message_text: str, # Текст с типом (voice/video)
                     system_message_base: str,
                     topic_context="",
                     is_reply_to_bot: bool = False):
        """Строит промпт, используя build_optimized_context."""
        context_lines = build_optimized_context(
            system_message_base=system_message_base, topic_context=topic_context,
            current_message_text=current_message_text, user_name=user_name,
            history_deque=history_deque, relevant_history=relevant_history,
            relevant_facts=relevant_facts,
            is_reply_to_bot=is_reply_to_bot,
            max_tokens=CONTEXT_MAX_TOKENS
        )
        system_message_base = f"{system_message_base} Ты - {self.settings.BOT_NAME}." # Ваша базовая строка стиля

        # --- ДОБАВЛЯЕМ ЯВНЫЕ ИНСТРУКЦИИ ---
        system_message_base += (
            "\n\nТвоя главная задача - ответить на ПОСЛЕДНЕЕ сообщение пользователя ({USER_ROLE})."
            "\nИнформация из разделов 'Ключевые факты из памяти' и 'Релевантные фрагменты' дана тебе как КОНТЕКСТ."
            "\nИСПОЛЬЗУЙ эту информацию для лучшего понимания ситуации, НО НЕ УПОМИНАЙ старые факты или события из прошлого в своем ответе, ЕСЛИ ТОЛЬКО пользователь НЕ СПРАШИВАЕТ о них НАПРЯМУЮ в своем ПОСЛЕДНЕМ сообщении или если это АБСОЛЮТНО НЕОБХОДИМО для ответа на его ПОСЛЕДНИЙ вопрос."
            "\n**ВАЖНО:** Прежде чем отвечать, ВНИМАТЕЛЬНО изучи весь предоставленный контекст ('Недавний диалог', 'Релевантные фрагменты', 'Ключевые факты'), чтобы понять текущую ситуацию и ход беседы. Не отвечай, что информации недостаточно, если она есть в контексте."
            "\n\n**Важно по стилю ответа:**"
            "\n1. **Краткость:** Старайся отвечать лаконично, примерно сопоставимо по длине с последним сообщением пользователя. Избегай длинных монологов."
            "\n2. **Меньше вопросов:** Не задавай много вопросов подряд. Задавай уточняющий вопрос, только если это действительно нужно для продолжения диалога или если пользователь сам просит о чем-то спросить."
            "\n3. **Не повторяйся:** Избегай повторения одних и тех же фраз, особенно в начале ответа. Не возвращайся к вопросам, которые ты уже задавала в предыдущих репликах, если пользователь на них не ответил или перевел тему."
            "\n4. **Не цепляйся за слова:** Если сообщение пользователя короткое или не содержит явного вопроса/темы, не фокусируйся на отдельных словах. Отвечай более обобщенно, поддерживая естественный ход разговора."
            "\n5. **Фокус на последнем:** Строго фокусируйся на ПОСЛЕДНЕМ сообщении пользователя. Не поднимай темы из более ранних сообщений (твоих или пользователя), если пользователь сам не вернулся к ним в своем ПОСЛЕДНЕМ сообщении."
            "\n6. **Без приветствий:** Не начинай ответ с приветствия, если его не было в последнем сообщении пользователя."
        )
        context_lines.append(f"\n{self.settings.BOT_NAME}:") # Приглашение к ответу
        final_prompt = "\n".join(context_lines).strip()
        return final_prompt

def get_llm_provider() -> LLMProvider:
    provider_name = settings.LLM_PROVIDER
    config = settings.LLM_CONFIG.get(provider_name, {})
    
    providers = {
        'gemini': GeminiProvider,
        'mistral': MistralProvider,
        'openai': OpenAiProvider,
        'groq': GroqProvider,
        'together': TogetherProvider,
        'ollama': OllamaProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown LLM provider: {provider_name}")
    
    return providers[provider_name].from_config(config)

# Заменяем все вызовы generate_content_sync на провайдер
def generate_content_sync(prompt: str) -> str:
    provider = get_llm_provider()
    return provider.generate_text(prompt)

async def generate_content_async(prompt: str) -> str:
    provider = get_llm_provider()
    return await provider.generate_text_async(prompt)

async def generate_vision_content_async(contents: list) -> str:
    """Асинхронная функция генерации текста по изображению с Gemini."""
    if not gemini_model: return "[Ошибка: Модель Gemini не инициализирована]"
    logger.info("Sending image/prompt to Gemini Vision...")
    try:
        gen_config = settings.GEMINI_GENERATION_CONFIG
        # Используем распарсенные настройки безопасности
        safety = settings.GEMINI_SAFETY_SETTINGS
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
    import state
    if state.bot_activity_percentage >= 100: return True
    if state.bot_activity_percentage <= 0: return False
    return random.randint(1, 100) <= state.bot_activity_percentage

class LoadBalancer:
    def __init__(self):
        self.providers = []
        self.current_index = 0
        
    def update_providers(self):
        from config import settings
        self.providers = [
            p for p in [
                'gemini', 'mistral', 'openai', 
                'groq', 'together', 'ollama'
            ] if p in settings.AVAILABLE_PROVIDERS
        ]
    
    def get_next_provider(self) -> str:
        if not self.providers:
            raise ValueError("No available providers")
        
        provider = self.providers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.providers)
        return provider

load_balancer = LoadBalancer()

async def load_balanced_generate(prompt: str) -> str:
    load_balancer.update_providers()
    for _ in range(len(load_balancer.providers)):
        provider_name = load_balancer.get_next_provider()
        try:
            provider = get_llm_provider()
            return await provider.generate_text_async(prompt)
        except Exception as e:
            logger.warning(f"Failed with {provider_name}: {e}")
    return "Все провайдеры недоступны"

class PerformanceMetrics:
    def __init__(self):
        self.timings = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    def log_time(self, provider: str, duration: float):
        self.timings[provider].append(duration)
    
    def log_error(self, provider: str):
        self.error_counts[provider] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "average_times": {
                p: sum(t)/len(t) for p, t in self.timings.items()
            },
            "error_counts": dict(self.error_counts)
        }

    def log_to_db(self):
        """Сохраняет метрики в БД"""
        try:
            from state import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            timestamp = time.time()
            
            for provider, times in self.timings.items():
                avg_time = sum(times)/len(times) if times else 0
                cursor.execute("""
                    INSERT INTO provider_metrics 
                    (provider, avg_time, error_count, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (provider, avg_time, self.error_counts.get(provider,0), timestamp))
            
            conn.commit()
            logger.info("Saved provider metrics to DB")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

metrics = PerformanceMetrics()

async def generate_with_metrics(prompt: str) -> str:
    provider = get_llm_provider()
    start_time = time.time()
    try:
        result = await provider.generate_text_async(prompt)
        metrics.log_time(provider.name, time.time() - start_time)
        return result
    except Exception as e:
        metrics.log_error(provider.name)
        raise e

class AutoFailover:
    def __init__(self):
        self.current_provider = settings.LLM_PROVIDER
        self.error_count = 0
        self.max_errors = 3
        
    def reset(self):
        self.error_count = 0
        self.current_provider = settings.LLM_PROVIDER
        
    def should_switch(self):
        return self.error_count >= self.max_errors
        
    def switch_provider(self):
        providers = settings.AVAILABLE_PROVIDERS
        current_index = providers.index(self.current_provider)
        new_index = (current_index + 1) % len(providers)
        self.current_provider = providers[new_index]
        self.error_count = 0
        logger.info(f"Auto-switched to provider: {self.current_provider}")

auto_failover = AutoFailover()

async def generate_with_failover(prompt: str) -> str:
    try:
        provider = get_llm_provider()
        response = await provider.generate_text_async(prompt)
        auto_failover.reset()
        return response
    except Exception as e:
        auto_failover.error_count += 1
        logger.warning(f"Error with {provider.name} (count: {auto_failover.error_count})")
        if auto_failover.should_switch():
            auto_failover.switch_provider()
        raise e

class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()
        
    def get(self, prompt: str) -> Optional[str]:
        key = self.get_key(prompt)
        result = self.cache.get(key)
        if result:
            self.hits += 1
        else:
            self.misses += 1
        return result
        
    def set(self, prompt: str, response: str):
        key = self.get_key(prompt)
        self.cache[key] = response
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            
    def clear(self):
        self.cache.clear()
        logger.info(f"Cache cleared. Stats before reset: hits={self.hits}, misses={self.misses}")
        self.hits = 0
        self.misses = 0

cache = ResponseCache()

async def generate_with_cache(prompt: str) -> str:
    cached = cache.get(prompt)
    if cached:
        logger.debug("Cache hit")
        return cached
        
    response = await generate_with_failover(prompt)
    cache.set(prompt, response)
    return response

async def metrics_job(context):
    """Периодическое сохранение метрик"""
    from state import get_db_connection
    try:
        metrics.log_to_db()
        logger.debug("Provider metrics saved")
    except Exception as e:
        logger.error(f"Error in metrics job: {e}")

async def generate_with_fallback(prompt: str, retries: int = 2) -> str:
    providers_order = [
        settings.LLM_PROVIDER,
        'gemini',
        'openai',
        'mistral',
        'groq',
        'together',
        'ollama'
    ]
    
    last_error = None
    for provider_name in providers_order[:retries+1]:
        try:
            # Сохраняем текущего провайдера
            original_provider = settings.LLM_PROVIDER
            # Временно меняем провайдера
            settings.LLM_PROVIDER = provider_name
            provider = get_llm_provider()
            start_time = time.time()
            response = await provider.generate_text_async(prompt)
            logger.info(f"Generated with {provider.name} in {time.time()-start_time:.2f}s")
            # Восстанавливаем оригинального провайдера
            settings.LLM_PROVIDER = original_provider
            return response
        except Exception as e:
            last_error = f"{provider_name}: {str(e)}"
            logger.error(f"Error with {provider_name}: {e}")
            continue
    
    # Восстанавливаем оригинального провайдера
    settings.LLM_PROVIDER = providers_order[0]
    return f"Ошибка генерации. Последняя ошибка: {last_error}"
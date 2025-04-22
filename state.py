# -*- coding: utf-8 -*-
# state.py
from cmath import e
import os
import json
import time
import sqlite3
import asyncio
import re
from collections import deque
from typing import Dict, Deque, Optional, Tuple, List, Any
from telegram.ext import CallbackContext
from datetime import datetime
# --- Импорт клиента Mistral ---
# Используем try-except на случай, если библиотека не установлена
try:
    from mistralai import MistralClient
    from mistralai import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MistralClient = None # type: ignore
    ChatMessage = None # type: ignore
    MISTRAL_AVAILABLE = False

# Импортируем нужные константы и логгер из config
from config import (logger, settings, DB_FILE, USER_ROLE, ASSISTANT_ROLE, SYSTEM_ROLE, MISTRAL_API_KEY)
# --- Импорт синхронных функций из vector_db ---
from vector_db import (
    initialize_vector_db, add_message_embedding_sync, add_fact_embedding_sync,
    delete_embeddings_by_sqlite_ids_sync, delete_fact_embeddings_by_ids_sync, # Реализовано в vector_db
    delete_facts_by_history_key_sync
)

# --- Глобальное состояние бота (в памяти) ---
# {history_key: deque([(role, user_name, message)])}
chat_history: Dict[int, Deque[Tuple[str, Optional[str], str]]] = {}
last_activity: Dict[int, float] = {} # {history_key: timestamp}
bot_activity_percentage: int = 100 # Процент активности в группах

# --- Инициализация и подключение к БД SQLite ---
def get_db_connection():
    """Устанавливает соединение с БД SQLite."""
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;") # Для лучшей параллельной работы
        conn.execute("PRAGMA foreign_keys = ON;") # Включаем поддержку внешних ключей (если будут)
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite DB '{DB_FILE}': {e}")
        raise

def _execute_db(query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False, commit: bool = False) -> Any:
    """Универсальная функция для выполнения запросов SQLite с обработкой ошибок."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = None
        if fetch_one: result = cursor.fetchone()
        elif fetch_all: result = cursor.fetchall()
        else: result = cursor.rowcount # Возвращаем количество затронутых строк
        if commit: conn.commit()
        return result
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}\nQuery: {query}\nParams: {params}", exc_info=True)
        if conn and commit: # Пытаемся откатить, если была ошибка при записи
            try: conn.rollback()
            except Exception as rb_err: logger.error(f"SQLite rollback failed: {rb_err}")
        return None # Возвращаем None при любых ошибках БД
    finally:
        if conn: conn.close()


def init_db():
    """Инициализирует структуру БД SQLite, если она не существует."""
    required_tables = ['bot_settings', 'users', 'history', 'learned_responses',
                       'group_styles', 'group_user_styles', 'banned_users', 'facts']
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        logger.info("Initializing/Verifying SQLite database schema...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {row['name'] for row in cursor.fetchall()}

        if not set(required_tables).issubset(existing_tables):
            logger.warning("Creating missing SQLite tables IF NOT EXISTS...")
            # --- Определения таблиц ---
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID; -- Оптимизация для таблицы ключ-значение
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    preferred_name TEXT,
                    language_code TEXT,
                    first_seen REAL,
                    last_seen REAL,
                    is_bot INTEGER,
                    topic TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_key INTEGER NOT NULL, -- user_id or chat_id
                    role TEXT NOT NULL,
                    user_name TEXT, -- For group chats
                    message TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_responses (
                    prompt TEXT PRIMARY KEY,
                    response TEXT NOT NULL
                ) WITHOUT ROWID;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS group_styles (
                    chat_id INTEGER PRIMARY KEY,
                    style TEXT NOT NULL
                ) WITHOUT ROWID;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS group_user_styles (
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    style TEXT NOT NULL,
                    PRIMARY KEY (chat_id, user_id)
                ) WITHOUT ROWID;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS banned_users (
                    user_id INTEGER PRIMARY KEY,
                    reason TEXT,
                    banned_at REAL NOT NULL
                ) WITHOUT ROWID;
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY, -- Unique fact ID (e.g., hash or UUID)
                    history_key INTEGER NOT NULL, -- Which chat this fact belongs to
                    fact_type TEXT, -- user_preference, mentioned_person, etc.
                    fact_text TEXT NOT NULL, -- The actual fact text
                    source_message_ids TEXT, -- Optional: JSON list of source SQLite history IDs
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID;
            """)
            # --- Индексы ---
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_key_timestamp ON history (history_key, timestamp DESC);") # Индекс по ключу и времени
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_group_user_styles_user ON group_user_styles (user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_history_key ON facts (history_key);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_type ON facts (fact_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_updated_at ON facts (updated_at DESC);") # Для поиска старых фактов

            conn.commit()
            logger.info("SQLite schema creation/verification complete.")
        else:
            logger.info("All required SQLite tables exist.")
    except sqlite3.Error as e:
         logger.critical(f"Failed to initialize SQLite schema: {e}", exc_info=True);
         if conn: conn.rollback();
         raise # Перевыбрасываем, чтобы остановить запуск
    finally:
         if conn: conn.close()


# --- Функции для работы с БД SQLite ---

def load_bot_settings_from_db() -> Dict[str, Any]:
    """Загружает настройки бота из БД."""
    rows = _execute_db("SELECT key, value FROM bot_settings", fetch_all=True)
    db_settings = {row['key']: row['value'] for row in rows} if rows else {}
    logger.debug(f"Loaded bot settings from DB: {list(db_settings.keys())}")
    return db_settings

def save_bot_settings_to_db(current_settings: Dict[str, Any], activity: int):
    """Сохраняет настройки бота и процент активности в БД."""
    settings_to_save = current_settings.copy()
    settings_to_save['bot_activity_percentage'] = str(activity)
    conn = None
    try:
        conn = get_db_connection(); cursor = conn.cursor(); conn.execute("BEGIN TRANSACTION")
        for key, value in settings_to_save.items():
            # Используем str() для преобразования всех значений в текст перед сохранением, кроме bot_activity_percentage
            cursor.execute("INSERT INTO bot_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
        conn.commit(); logger.debug("Saved bot settings to DB.")
    except sqlite3.IntegrityError as e: logger.error(f"DB integrity error saving settings: {e}")
    except sqlite3.Error as e:
        logger.error(f"DB error saving settings: {e}")
        if conn: conn.rollback()

    finally:
        if conn: conn.close()

def load_active_histories_from_db():
    """Загружает недавние истории чатов в память (deque)."""
    chat_history.clear(); last_activity.clear()
    cutoff_time = time.time() - settings.HISTORY_TTL
    active_keys_rows = _execute_db("SELECT history_key, MAX(timestamp) as last_ts FROM history WHERE timestamp > ? GROUP BY history_key", (cutoff_time,), fetch_all=True)
    if active_keys_rows is None: logger.error("Could not load active history keys."); return
    active_keys = {row['history_key']: row['last_ts'] for row in active_keys_rows}

    loaded_count = 0; conn = None
    try:
         conn = get_db_connection(); cursor = conn.cursor()
         for key, last_ts in active_keys.items():
             # --- ИЗМЕНЕНИЕ НАЧАЛО ---
             # Загружаем последние MAX_HISTORY сообщений И ИХ ВРЕМЕННЫЕ МЕТКИ
             cursor.execute("SELECT role, user_name, message, timestamp FROM history WHERE history_key = ? ORDER BY timestamp DESC LIMIT ?", (key, settings.MAX_HISTORY))
             # --- ИЗМЕНЕНИЕ КОНЕЦ ---
             entries = cursor.fetchall()
             if entries:
                 # --- ИЗМЕНЕНИЕ НАЧАЛО ---
                 # Создаем deque с кортежами из 4 элементов (role, user_name, message, timestamp_str)
                 dq_entries = []
                 for r in reversed(entries): # Идем от старых к новым
                     ts_float = r['timestamp']
                     ts_str = datetime.fromtimestamp(ts_float).strftime("%Y-%m-%d %H:%M:%S")
                     dq_entries.append((r['role'], r['user_name'], r['message'], ts_str))

                 dq = deque(dq_entries, maxlen=settings.MAX_HISTORY)
                 # --- ИЗМЕНЕНИЕ КОНЕЦ ---
                 chat_history[key] = dq
                 last_activity[key] = last_ts # Сохраняем последнее время активности (Unix timestamp)
                 loaded_count += 1
    except sqlite3.Error as e: logger.error(f"DB error loading history entries: {e}")
    finally:
        if conn: conn.close()
    logger.info(f"Loaded history for {loaded_count} active chats (TTL: {settings.HISTORY_TTL}s, MaxLen: {settings.MAX_HISTORY}).")

def save_all_histories_to_db():
    """Сохраняет истории из deque (памяти) в БД SQLite."""
    # Эта функция больше не нужна так часто, так как save_message_and_embed сохраняет каждое сообщение.
    # Оставляем ее для возможного использования при завершении работы или для синхронизации.
    conn = None; saved_count = 0; keys = set(chat_history.keys()) # Копируем ключи
    if not keys: logger.debug("No in-memory history to save to SQLite."); return
    logger.info(f"Saving potentially unsaved in-memory history for {len(keys)} chats to SQLite...")
    try:
        conn = get_db_connection(); cursor = conn.cursor(); conn.execute("BEGIN TRANSACTION")
        for key in keys:
            if key not in chat_history: continue
            dq = chat_history[key]; ts = last_activity.get(key, time.time())
            # --- ИЗМЕНЕНИЕ НАЧАЛО ---
            for role, uname, msg, _ in list(dq): # Распаковываем 4 элемента, игнорируем ts_str
            # --- ИЗМЕНЕНИЕ КОНЕЦ ---
                # Проверяем наличие перед вставкой, используя ts из last_activity (как было)
                cursor.execute("INSERT INTO history (history_key, role, user_name, message, timestamp) SELECT ?, ?, ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM history WHERE history_key = ? AND role = ? AND message = ? AND abs(timestamp - ?) < 0.1)", (key, role, uname, msg, ts, key, role, msg, ts))
                if cursor.rowcount > 0: saved_count += 1
        conn.commit(); logger.info(f"Saved {saved_count} potentially new history entries to SQLite.")    
    except sqlite3.Error as e:
            logger.error(f"DB error saving history transaction: {e}")
            if conn:
                conn.rollback()
    finally:
            if conn:
                conn.close()


# --- Функции для пользовательских данных ---
def get_user_info_from_db(user_id: int) -> Optional[sqlite3.Row]:
    return _execute_db("SELECT * FROM users WHERE user_id = ?", (user_id,), fetch_one=True)

def update_user_info_in_db(user_id: int, username: Optional[str], first_name: Optional[str], last_name: Optional[str], lang_code: Optional[str], is_bot: bool):
    current_time = time.time()
    rows_affected = _execute_db("""
        INSERT INTO users (user_id, username, first_name, last_name, language_code, is_bot, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT first_seen FROM users WHERE user_id = ?), ?), ?)
        ON CONFLICT(user_id) DO UPDATE SET
            username = excluded.username, first_name = excluded.first_name, last_name = excluded.last_name,
            language_code = excluded.language_code, is_bot = excluded.is_bot, last_seen = excluded.last_seen
    """, (user_id, username, first_name, last_name, lang_code, 1 if is_bot else 0, user_id, current_time, current_time), commit=True)
    if rows_affected is not None: logger.debug(f"User info updated/inserted in DB for {user_id}")

def get_user_preferred_name_from_db(user_id: int) -> Optional[str]:
    r = _execute_db("SELECT preferred_name FROM users WHERE user_id = ?", (user_id,), fetch_one=True)
    return r['preferred_name'] if r and r['preferred_name'] else None

def set_user_preferred_name_in_db(user_id: int, name: str):
    upd = _execute_db("UPDATE users SET preferred_name = ? WHERE user_id = ?", (name, user_id), commit=True)
    if upd is None: logger.error(f"Failed updating preferred name for {user_id}")
    elif upd == 0:
        ins = _execute_db("INSERT OR IGNORE INTO users (user_id, preferred_name, first_seen, last_seen) VALUES (?, ?, ?, ?)", (user_id, name, time.time(), time.time()), commit=True)
        if ins : logger.info(f"Set preferred name for new user {user_id} to '{name}'.")
        elif ins is None: logger.error(f"Failed inserting preferred name for {user_id}")
    else: logger.info(f"Updated preferred name for {user_id} to '{name}'.")

def get_user_topic_from_db(user_id: int) -> Optional[str]:
    r = _execute_db("SELECT topic FROM users WHERE user_id = ?", (user_id,), fetch_one=True)
    return r['topic'] if r and r['topic'] else None

# --- Функции для стилей ---
def get_group_style_from_db(chat_id: int) -> Optional[str]:
    r = _execute_db("SELECT style FROM group_styles WHERE chat_id = ?", (chat_id,), fetch_one=True)
    return r['style'] if r else None
def set_group_style_in_db(chat_id: int, style: str) -> bool:
    return _execute_db("INSERT OR REPLACE INTO group_styles (chat_id, style) VALUES (?, ?)", (chat_id, style), commit=True) is not None
def delete_group_style_in_db(chat_id: int) -> bool:
    return _execute_db("DELETE FROM group_styles WHERE chat_id = ?", (chat_id,), commit=True) is not None
def get_group_user_style_from_db(chat_id: int, user_id: int) -> Optional[str]:
    r = _execute_db("SELECT style FROM group_user_styles WHERE chat_id = ? AND user_id = ?", (chat_id, user_id), fetch_one=True)
    return r['style'] if r else None
def set_group_user_style_in_db(chat_id: int, user_id: int, style: str) -> bool:
    return _execute_db("INSERT OR REPLACE INTO group_user_styles (chat_id, user_id, style) VALUES (?, ?, ?)", (chat_id, user_id, style), commit=True) is not None
def delete_group_user_style_in_db(chat_id: int, user_id: int) -> bool:
    return _execute_db("DELETE FROM group_user_styles WHERE chat_id = ? AND user_id = ?", (chat_id, user_id), commit=True) is not None

# --- Функции для банов ---
def is_user_banned(user_id: int) -> bool:
    return _execute_db("SELECT 1 FROM banned_users WHERE user_id = ?", (user_id,), fetch_one=True) is not None
def ban_user_in_db(user_id: int, reason: Optional[str] = None) -> bool:
    return _execute_db("INSERT OR IGNORE INTO banned_users (user_id, reason, banned_at) VALUES (?, ?, ?)", (user_id, reason, time.time()), commit=True) is not None
def unban_user_in_db(user_id: int) -> bool:
    return _execute_db("DELETE FROM banned_users WHERE user_id = ?", (user_id,), commit=True) is not None
def get_banned_users() -> List[sqlite3.Row]:
    return _execute_db("SELECT user_id, reason, banned_at FROM banned_users ORDER BY banned_at DESC", fetch_all=True) or []


# --- Инициализация и Загрузка/Сохранение ---
def load_all_data():
    """Загружает начальное состояние из БД SQLite и инициализирует векторную БД."""
    global bot_activity_percentage
    try: init_db()
    except Exception as e: logger.critical(f"Failed to initialize SQLite DB: {e}. Bot cannot start."); exit(1)

    logger.info(f"Loading data from SQLite '{DB_FILE}'...")
    db_settings = load_bot_settings_from_db()
    settings.load_from_db(db_settings)
    bot_activity_percentage = int(db_settings.get('bot_activity_percentage', 100))
    logger.info(f"Bot activity percentage loaded: {bot_activity_percentage}%")

    try: initialize_vector_db()
    except Exception as e: logger.error(f"Vector DB initialization failed: {e}. Vector search may not work.", exc_info=True)

    load_active_histories_from_db()
    logger.info("Data loading complete.")


def save_all_data():
    """Сохраняет данные SQLite и настройки. ChromaDB сохраняется автоматически."""
    logger.info("Saving data (SQLite settings and in-memory history)...")
    current_settings_dict = settings.get_settings_dict()
    save_bot_settings_to_db(current_settings_dict, bot_activity_percentage)
    save_all_histories_to_db() # Сохраняем то, что могло не записаться из deque
    logger.info("SQLite data saving complete. ChromaDB uses persistent storage.")


async def save_message_and_embed(history_key: int, role: str, message: str, user_name: Optional[str] = None):
    """Сохраняет сообщение в SQLite и добавляет его эмбеддинг в Vector DB."""
    if not message or not message.strip(): return
    timestamp = time.time()
    sqlite_id = None; conn = None
    try:
        conn = get_db_connection(); cursor = conn.cursor()
        cursor.execute("INSERT INTO history (history_key, role, user_name, message, timestamp) VALUES (?, ?, ?, ?, ?)", (history_key, role, user_name, message, timestamp))
        sqlite_id = cursor.lastrowid; conn.commit()
        logger.debug(f"Message saved to SQLite with ID: {sqlite_id}")
    except sqlite3.Error as e:
        logger.error(f"Failed to save message to SQLite (key={history_key}, role={role}): {e}")
        if conn:
            conn.rollback()
        return
    finally:
        if conn: conn.close()
    if sqlite_id:
        # Вызываем СИНХРОННУЮ функцию добавления эмбеддинга через to_thread
        asyncio.create_task(asyncio.to_thread(
            add_message_embedding_sync, # Используем _sync версию
            sqlite_id=sqlite_id, history_key=history_key, role=role, text=message
        ))


# --- Извлечение и сохранение фактов с Mistral ---
async def extract_and_save_facts(history_key: int):
    """Извлекает факты из недавней истории с помощью Mistral API и сохраняет их."""
    if not MISTRAL_AVAILABLE: logger.warning("MistralAI library not installed. Skipping fact extraction."); return
    if not MISTRAL_API_KEY: logger.error("MISTRAL_API_KEY not found."); return

    logger.info(f"Starting fact extraction for history_key {history_key} using Mistral...")
    limit = settings.MAX_HISTORY * 3
    # --- ИЗМЕНЕНИЕ НАЧАЛО ---
    # Запрашиваем временную метку из БД
    recent_messages_rows = await asyncio.to_thread(
        _execute_db,
        "SELECT role, message, timestamp FROM history WHERE history_key = ? ORDER BY timestamp DESC LIMIT ?",
        (history_key, limit), fetch_all=True
    )
    # --- ИЗМЕНЕНИЕ КОНЕЦ ---
    if not recent_messages_rows: logger.info(f"No recent history for fact extraction (key={history_key})."); return

    # --- ИЗМЕНЕНИЕ НАЧАЛО ---
    # Формируем текст диалога с временными метками
    dialog_lines = []
    for row in reversed(recent_messages_rows):
        ts_str = datetime.datetime.fromtimestamp(row['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        dialog_lines.append(f"[{ts_str}] {row['role']}: {row['message']}")
    dialog_text = "\n".join(dialog_lines)
    # --- ИЗМЕНЕНИЕ КОНЕЦ ---

    max_dialog_chars = 6000 # Увеличим, т.к. метки добавляют длину
    if len(dialog_text) > max_dialog_chars:
        logger.warning(f"Dialog text too long ({len(dialog_text)}), truncating for Mistral.")
        # Обрезаем С НАЧАЛА (старые сообщения), чтобы сохранить последние
        lines_to_keep = []
        current_len = 0
        for line in reversed(dialog_lines):
            line_len = len(line) + 1 # +1 for newline
            if current_len + line_len <= max_dialog_chars:
                lines_to_keep.append(line)
                current_len += line_len
            else:
                break
        dialog_text = "\n".join(reversed(lines_to_keep))
    fact_extraction_prompt = f"""Проанализируй диалог (history_key={history_key}). Извлеки ТОЛЬКО ключевые факты в формате JSON. Категории: 'user_preferences', 'user_attributes', 'mentioned_people', 'mentioned_places', 'key_topics', 'agreements'. Правила: ТОЛЬКО JSON, пустые списки [], без выдумок. Диалог:\n{dialog_text}\n\nJSON_OUTPUT:"""
    extracted_data = None
    try:
        client = MistralClient(api_key=MISTRAL_API_KEY)
        logger.debug(f"Sending fact extraction prompt to Mistral (key={history_key})...")
        chat_response = await client.chat(
            model="mistral-large-latest", messages=[ChatMessage(role="user", content=fact_extraction_prompt)],
            temperature=0.1, response_format={"type": "json_object"}
        )
        json_response_str = chat_response.choices[0].message.content
        logger.debug(f"Mistral response: {json_response_str}")
        if json_response_str.strip().startswith('{') and json_response_str.strip().endswith('}'):
             extracted_data = json.loads(json_response_str); logger.info(f"Parsed extracted facts for key {history_key}.")
        else: logger.error(f"Mistral response is not valid JSON for key {history_key}.")
    except Exception as e: logger.error(f"Mistral API call failed (key={history_key}): {e}", exc_info=True); return

    if extracted_data and isinstance(extracted_data, dict):
        conn = None; now = time.time()
        try:
            conn = get_db_connection(); cursor = conn.cursor(); conn.execute("BEGIN TRANSACTION")
            facts_added_count = 0; facts_to_embed = []
            for fact_type, facts in extracted_data.items():
                if isinstance(facts, list):
                    for fact_item in facts:
                        if isinstance(fact_item, str) and fact_item.strip():
                            f_item_clean = fact_item.strip(); f_text = f"{fact_type.replace('_',' ').capitalize()}: {f_item_clean}"
                            f_id = f"{history_key}_{fact_type}_{hash(f_item_clean)}" # Простой ID
                            cursor.execute("INSERT INTO facts (fact_id, history_key, fact_type, fact_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(fact_id) DO UPDATE SET fact_text = excluded.fact_text, updated_at = excluded.updated_at", (f_id, history_key, fact_type, f_text, now, now))
                            if cursor.rowcount is not None: # Проверка на ошибку execute
                                facts_added_count += cursor.rowcount # Считаем добавленные/обновленные
                                facts_to_embed.append({'id': f_id, 'text': f_text, 'type': fact_type})
            conn.commit(); logger.info(f"Saved/Updated {facts_added_count} facts in SQLite for key {history_key}.")
            if facts_to_embed:
                logger.info(f"Queueing {len(facts_to_embed)} facts for embedding (key={history_key})...")
                tasks = [asyncio.to_thread(add_fact_embedding_sync, fact_id=f['id'], history_key=history_key, fact_type=f['type'], fact_text=f['text']) for f in facts_to_embed]
                await asyncio.gather(*tasks)
        except sqlite3.Error as e:
            logger.error(f"Error saving facts to SQLite (key={history_key}): {e}")
            if conn:
                conn.rollback()
        except Exception as e:
            logger.error(f"Error processing extracted facts (key={history_key}): {e}")
            if conn:
                conn.rollback()
        finally:
            if conn: conn.close()
    else: logger.info(f"No valid facts data extracted for key {history_key}.")


# --- Фоновая задача для извлечения фактов ---
async def fact_extraction_job(context: CallbackContext):
     """Периодически извлекает факты для активных чатов."""
     if not MISTRAL_AVAILABLE or not MISTRAL_API_KEY:
         logger.debug("Skipping fact extraction job: Mistral not available or API key missing.")
         return
     logger.info("Starting periodic fact extraction job...")
     active_keys = list(last_activity.keys())
     logger.info(f"Found {len(active_keys)} potentially active chats.")
     processed_keys = 0
     for key in active_keys:
         # Можно добавить проверку времени последней экстракции, чтобы не делать слишком часто
         logger.debug(f"Running fact extraction for history_key: {key}")
         try:
             await extract_and_save_facts(key)
             processed_keys += 1
             await asyncio.sleep(5) # Пауза между ключами
         except Exception as e: logger.error(f"Error during scheduled fact extraction for key {key}: {e}", exc_info=True)
     logger.info(f"Finished periodic fact extraction job. Processed {processed_keys} keys.")


# --- Функции управления состоянием ---
def add_to_memory_history(key: int, role: str, message: str, user_name: Optional[str] = None):
    """Добавляет сообщение в историю чата (только в памяти) и обновляет время активности."""
    if key not in chat_history: chat_history[key] = deque(maxlen=settings.MAX_HISTORY)
    now = time.time()
    ts_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
    entry = (role, user_name, message, ts_str) # Теперь 4 элемента
    chat_history[key].append(entry)
    last_activity[key] = now # Обновляем время последней активности Unix timestamp'ом
    logger.debug(f"Mem-History add: key={key}, role={role}, size={len(chat_history[key])}")

async def cleanup_history_job(context):
    """Периодическая задача для удаления старой истории и ФАКТОВ."""
    current_time = time.time()
    keys_to_delete_mem = [k for k, ts in last_activity.items() if current_time - ts > settings.HISTORY_TTL]
    deleted_mem_cnt = 0
    for key in keys_to_delete_mem:
        if key in chat_history: del chat_history[key]; deleted_mem_cnt += 1
        if key in last_activity: del last_activity[key]
    if deleted_mem_cnt > 0: logger.info(f"Cleaned mem history for {deleted_mem_cnt} chats.")

    cutoff_time = time.time() - settings.HISTORY_TTL
    deleted_sqlite_ids = []; keys_with_deleted_hist = set(); deleted_db_count = None
    conn = None
    try:
        conn = get_db_connection(); cursor = conn.cursor()
        cursor.execute("SELECT id, history_key FROM history WHERE timestamp < ?", (cutoff_time,))
        rows = cursor.fetchall(); deleted_sqlite_ids = [r['id'] for r in rows]; keys_with_deleted_hist = {r['history_key'] for r in rows}
        if deleted_sqlite_ids:
            p = ','.join('?' * len(deleted_sqlite_ids)); cursor.execute(f"DELETE FROM history WHERE id IN ({p})", tuple(deleted_sqlite_ids))
            deleted_db_count = cursor.rowcount; conn.commit()
            if deleted_db_count > 0: logger.info(f"Cleaned {deleted_db_count} old entries from SQLite history.")
        else: deleted_db_count = 0
    except sqlite3.Error as e:
        logger.error(f"Error cleaning SQLite history: {e}")
        deleted_db_count = None
        if conn:
            conn.rollback()
    finally:
        if conn: conn.close()

    # --- Удаляем эмбеддинги ИСТОРИИ ---
    if deleted_sqlite_ids and keys_with_deleted_hist:
        logger.info(f"Deleting {len(deleted_sqlite_ids)} history embeddings...")
        ids_by_key = {k:[] for k in keys_with_deleted_hist}; [ids_by_key[r['history_key']].append(r['id']) for r in rows]
        del_tasks = [asyncio.to_thread(delete_embeddings_by_sqlite_ids_sync, k, ids) for k, ids in ids_by_key.items() if ids]
        await asyncio.gather(*del_tasks)

    # --- Удаляем ВСЕ факты для чатов, где УДАЛЯЛАСЬ история ---
    if keys_with_deleted_hist:
        logger.info(f"Deleting ALL facts for {len(keys_with_deleted_hist)} keys where history was deleted...")
        del_fact_tasks = [asyncio.to_thread(delete_facts_by_history_key_sync, key) for key in keys_with_deleted_hist]
        await asyncio.gather(*del_fact_tasks)

    # --- Удаляем ОЧЕНЬ старые факты (независимо от истории) ---
    deleted_fact_ids_old = []; conn_fact_old = None
    try:
        conn_fact_old = get_db_connection(); cursor_fact_old = conn_fact_old.cursor()
        fact_cutoff = time.time() - (settings.HISTORY_TTL * 3)
        cursor_fact_old.execute("SELECT fact_id FROM facts WHERE updated_at < ?", (fact_cutoff,))
        deleted_fact_ids_old = [r['fact_id'] for r in cursor_fact_old.fetchall()]
        if deleted_fact_ids_old:
            p_fact = ','.join('?' * len(deleted_fact_ids_old))
            cursor_fact_old.execute(f"DELETE FROM facts WHERE fact_id IN ({p_fact})", tuple(deleted_fact_ids_old))
            deleted_old_facts_cnt = cursor_fact_old.rowcount; conn_fact_old.commit()
            if deleted_old_facts_cnt > 0:
                 logger.info(f"Cleaned up {deleted_old_facts_cnt} old entries from SQLite facts table.")
                 # Удаляем эмбеддинги этих фактов
                 await asyncio.to_thread(delete_fact_embeddings_by_ids_sync, deleted_fact_ids_old)
    except sqlite3.Error as e:
        logger.error(f"Error cleaning up old facts from SQLite: {e}")
        if conn_fact_old:
            conn_fact_old.rollback()
    finally:
        if conn_fact_old: conn_fact_old.close()

    if deleted_mem_cnt == 0 and (deleted_db_count is None or deleted_db_count == 0):
        logger.debug("History cleanup: No inactive memory or old DB entries found.")
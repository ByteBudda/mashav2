import time
from typing import Dict, Optional, Tuple, Any # Убрали Deque, добавили Any
import json
import os
import asyncio # Для асинхронной cleanup_history_job
from telegram.ext import CallbackContext
# Импорты из проекта
from config import (logger, KNOWLEDGE_FILE, USER_DATA_DIR, USER_ROLE,
                    ASSISTANT_ROLE, SYSTEM_ROLE, settings, HISTORY_TTL)
# Используем функции из vector_store для работы с историей
from vector_store import (add_message, delete_history, cleanup_all_old_histories)

# ==============================================================================
# Начало: Содержимое state.py
# ==============================================================================

# --- Глобальное состояние бота (без chat_history и last_activity) ---
user_preferred_name: Dict[int, str] = {}
user_topic: Dict[int, str] = {}
learned_responses: Dict[str, str] = {}
user_info_db: Dict[int, Dict[str, Any]] = {} # Уточнили тип Any
group_preferences: Dict[int, Dict[str, str]] = {}
feedback_data: Dict[int, Dict] = {}
group_user_style_prompts: Dict[Tuple[int, int], str] = {}
bot_activity_percentage: int = 100

# --- Функции управления состоянием ---

async def add_to_history(key: int, role: str, message: str, user_name: Optional[str] = None):
    """Добавляет сообщение в векторную базу данных ChromaDB (асинхронно)."""
    timestamp = time.time()
    # Используем асинхронную функцию из vector_store
    await add_message(
        history_key=key,
        role=role,
        message=message,
        timestamp=timestamp,
        user_name=user_name
    )
    logger.debug(f"Message forwarded to vector store for key {key}.")

async def cleanup_history_job(context: CallbackContext): # context нужен для JobQueue
    """Периодическая задача для удаления старых записей из ChromaDB."""
    logger.info("Running history cleanup job (vector store)...")
    # Вызываем асинхронную функцию очистки из vector_store
    await cleanup_all_old_histories(settings.HISTORY_TTL)
    logger.info("History cleanup job (vector store) finished.")

def load_all_data():
    """Загружает общее состояние и данные пользователей (БЕЗ ИСТОРИИ ЧАТА)."""
    global learned_responses, group_preferences, user_info_db, settings, user_preferred_name, user_topic, bot_activity_percentage
    logger.info(f"Loading data from {KNOWLEDGE_FILE} and {USER_DATA_DIR} (excluding chat history)...")
    
    # Загрузка общих данных
    if os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                learned_responses = data.get("learned_responses", {})
                group_preferences = data.get("group_preferences", {})
                bot_settings_data = data.get("bot_settings")
                if bot_settings_data:
                    # Обновляем объект settings напрямую
                    settings.MAX_HISTORY_RESULTS = bot_settings_data.get('MAX_HISTORY_RESULTS', settings.MAX_HISTORY_RESULTS)
                    settings.MAX_HISTORY_TOKENS = bot_settings_data.get('MAX_HISTORY_TOKENS', settings.MAX_HISTORY_TOKENS)
                    settings.DEFAULT_STYLE = bot_settings_data.get('DEFAULT_STYLE', settings.DEFAULT_STYLE)
                    settings.BOT_NAME = bot_settings_data.get('BOT_NAME', settings.BOT_NAME)
                    settings.HISTORY_TTL = bot_settings_data.get('HISTORY_TTL', settings.HISTORY_TTL)
                    settings.CHROMA_DATA_PATH = bot_settings_data.get('CHROMA_DATA_PATH', settings.CHROMA_DATA_PATH)
                    settings.GEMINI_MODEL_NAME = bot_settings_data.get('GEMINI_MODEL_NAME', settings.GEMINI_MODEL_NAME)
                    logger.info("Bot settings loaded and applied from knowledge file.")
                bot_activity_percentage = data.get("bot_activity_percentage", 100)
        except Exception as e:
             logger.error(f"Error loading {KNOWLEDGE_FILE}: {e}", exc_info=True)
    else:
        logger.warning(f"{KNOWLEDGE_FILE} not found.")

    # Загрузка данных пользователей (БЕЗ ИСТОРИИ ЧАТА)
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    loaded_user_count = 0
    for filename in os.listdir(USER_DATA_DIR):
        if filename.startswith("user_") and filename.endswith(".json"):
            user_id_str = filename[len("user_"):-len(".json")]
            if user_id_str.isdigit():
                user_id = int(user_id_str)
                user_file_path = os.path.join(USER_DATA_DIR, filename)
                try:
                    with open(user_file_path, "r", encoding="utf-8") as f:
                        user_data = json.load(f)
                        # Инициализируем user_info_db[user_id], если его нет
                        user_info_db.setdefault(user_id, {"preferences": {}})
                        # Обновляем основной info, сохраняя существующие ключи
                        user_info_db[user_id].update(user_data.get("info", {}))
                        # --- ИЗМЕНЕНО: Загружаем 'memory' в user_info_db ---
                        user_memory = user_data.get('memory')
                        if user_memory:
                            user_info_db[user_id]['memory'] = user_memory

                        pref_name = user_data.get('preferred_name')
                        if pref_name: user_preferred_name[user_id] = pref_name
                        topic = user_data.get('topic')
                        if topic: user_topic[user_id] = topic
                        loaded_user_count += 1
                except Exception as e:
                    logger.error(f"Error loading user data file {filename}: {e}", exc_info=True)
            else:
                 logger.warning(f"Skipping file with non-integer user ID: {filename}")
    logger.info(f"Data loading complete. Loaded {loaded_user_count} user data files.")


def save_user_data(user_id):
    """Сохраняет данные конкретного пользователя (БЕЗ ИСТОРИИ ЧАТА)."""
    user_data_dir = os.path.join(".", USER_DATA_DIR)
    os.makedirs(user_data_dir, exist_ok=True)
    user_filename = f"user_{user_id}.json"
    user_file_path = os.path.join(user_data_dir, user_filename)

    # Собираем данные для сохранения
    user_data_entry = user_info_db.get(user_id, {})
    data_to_save = {
        "info": {k: v for k, v in user_data_entry.items() if k != 'memory'}, # Сохраняем все, кроме memory, в info
        # --- ИЗМЕНЕНО: Сохраняем 'memory' отдельно ---
        "memory": user_data_entry.get('memory'),
        "preferred_name": user_preferred_name.get(user_id),
        "topic": user_topic.get(user_id)
    }
    # Очищаем от None значений перед сохранением
    data_to_save = {k: v for k, v in data_to_save.items() if v is not None}

    try:
        with open(user_file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logger.debug(f"User data saved for user_id: {user_id} (excluding history)")
    except Exception as e:
        logger.error(f"Error saving user data for {user_id} to {user_filename}: {e}", exc_info=True)

def save_user_data(user_id):
    """Сохраняет данные конкретного пользователя (БЕЗ ИСТОРИИ ЧАТА)."""
    user_data_dir = os.path.join(".", USER_DATA_DIR)
    os.makedirs(user_data_dir, exist_ok=True)
    user_filename = f"user_{user_id}.json"
    user_file_path = os.path.join(user_data_dir, user_filename)

    data_to_save = {
        "info": user_info_db.get(user_id, {}),
        "preferred_name": user_preferred_name.get(user_id),
        "topic": user_topic.get(user_id)
    }
    data_to_save = {k: v for k, v in data_to_save.items() if v is not None}

    try:
        with open(user_file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        logger.debug(f"User data saved for user_id: {user_id} (excluding history)")
    except Exception as e:
        logger.error(f"Error saving user data for {user_id} to {user_filename}: {e}", exc_info=True)

def save_all_data():
    """Сохраняет все данные (общие и пользовательские, БЕЗ ИСТОРИИ ЧАТА)."""
    logger.info("Saving all data (excluding chat history)...")

    # Сохранение общих данных
    knowledge_data = {
        "learned_responses": learned_responses,
        "group_preferences": group_preferences,
        "bot_settings": {
            "MAX_HISTORY_RESULTS": settings.MAX_HISTORY_RESULTS,
            "MAX_HISTORY_TOKENS": settings.MAX_HISTORY_TOKENS,
            "DEFAULT_STYLE": settings.DEFAULT_STYLE,
            "BOT_NAME": settings.BOT_NAME,
            "HISTORY_TTL": settings.HISTORY_TTL,
            "CHROMA_DATA_PATH": settings.CHROMA_DATA_PATH,
            "GEMINI_MODEL_NAME": settings.GEMINI_MODEL_NAME,
        },
        "bot_activity_percentage": bot_activity_percentage,
    }
    try:
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Knowledge file saved: {KNOWLEDGE_FILE}")
    except Exception as e:
         logger.error(f"Error saving knowledge file {KNOWLEDGE_FILE}: {e}", exc_info=True)

    # Сохранение данных всех активных пользователей
    saved_user_count = 0
    # Копируем ключи на случай изменения словаря во время итерации (хотя тут это маловероятно)
    active_user_ids = list(user_info_db.keys())
    for user_id in active_user_ids:
        save_user_data(user_id)
        saved_user_count += 1

    logger.info(f"All data saving complete. Saved data for {saved_user_count} users (excluding history).")

# ==============================================================================
# Конец: Содержимое state.py
# ==============================================================================
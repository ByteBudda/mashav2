# -*- coding: utf-8 -*-
# config.py
import os
import json # Добавили json для настроек генерации
from pathlib import Path
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# --- Загрузка конфигурации ---
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- Токены и Ключи ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY') # Добавили ключ Mistral
ADMIN_USER_IDS = list(map(int, os.getenv('ADMIN_IDS', '').split(','))) if os.getenv('ADMIN_IDS') else []
# NEW formatter WITH timestamp:
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Added %(asctime)s and potentially %(name)s
    datefmt='%Y-%m-%d %H:%M:%S' # Optional: Define the timestamp format
)
# --- END OF CHANGE ---

# --- Класс настроек бота ---
class BotSettings:
    def __init__(self):
        # Загружаем значения из .env или используем значения по умолчанию
        self._initial_max_history = int(os.getenv('MAX_HISTORY', '30'))
        self._initial_default_style = os.getenv('DEFAULT_STYLE', "Ты - Маша, 25-летняя девушка. Отвечай от первого лица, как будто ты - Маша.Подстраивайся под стиль общения собеседника")
        self._initial_bot_name = os.getenv('BOT_NAME', 'Маша')
        self._initial_history_ttl = int(os.getenv('HISTORY_TTL', '86400')) # Время жизни истории в памяти/SQLite
        self._initial_gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
        # Настройки генерации Gemini по умолчанию
        self._default_gemini_generation_config = {
            "temperature": 0.7, "top_p": 0.95, "top_k": 40,
        }

        self.MAX_HISTORY = self._initial_max_history
        self.DEFAULT_STYLE = self._initial_default_style
        self.BOT_NAME = self._initial_bot_name
        self.HISTORY_TTL = self._initial_history_ttl
        self.GEMINI_MODEL = self._initial_gemini_model
        # Инициализируем настройки генерации
        self.GEMINI_GENERATION_CONFIG = self._default_gemini_generation_config.copy()


    def update_default_style(self, new_style: str):
        self.DEFAULT_STYLE = new_style
        logger.info(f"Default style updated in memory to: {new_style}")

    def update_bot_name(self, new_name: str):
        self.BOT_NAME = new_name
        logger.info(f"Bot name updated in memory to: {new_name}")

    def reset_to_initial(self):
        """Сбрасывает настройки к первоначальным значениям."""
        self.MAX_HISTORY = self._initial_max_history
        self.DEFAULT_STYLE = self._initial_default_style
        self.BOT_NAME = self._initial_bot_name
        self.HISTORY_TTL = self._initial_history_ttl
        self.GEMINI_MODEL = self._initial_gemini_model
        self.GEMINI_GENERATION_CONFIG = self._default_gemini_generation_config.copy()
        logger.info("Bot settings reset to initial values in memory.")

    def load_from_db(self, db_settings: dict):
        """Загружает настройки из словаря, полученного из БД."""
        self.MAX_HISTORY = int(db_settings.get('MAX_HISTORY', self._initial_max_history))
        self.DEFAULT_STYLE = db_settings.get('DEFAULT_STYLE', self._initial_default_style)
        self.BOT_NAME = db_settings.get('BOT_NAME', self._initial_bot_name)
        self.HISTORY_TTL = int(db_settings.get('HISTORY_TTL', self._initial_history_ttl))
        self.GEMINI_MODEL = db_settings.get('GEMINI_MODEL', self._initial_gemini_model)
        # Загрузка настроек генерации Gemini
        gen_config_json = db_settings.get('GEMINI_GENERATION_CONFIG')
        if gen_config_json:
            try:
                loaded_config = json.loads(gen_config_json)
                # Обновляем только существующие ключи, чтобы сохранить дефолтные, если что-то пропало
                self.GEMINI_GENERATION_CONFIG.update(loaded_config)
            except json.JSONDecodeError:
                logger.warning("Could not parse GEMINI_GENERATION_CONFIG from DB, using defaults.")
                self.GEMINI_GENERATION_CONFIG = self._default_gemini_generation_config.copy()
        else: # Если в БД нет, используем дефолтные
             self.GEMINI_GENERATION_CONFIG = self._default_gemini_generation_config.copy()
        logger.info("Bot settings loaded from DB data.")
        logger.debug(f"Loaded Gemini Config: {self.GEMINI_GENERATION_CONFIG}")


    def get_settings_dict(self) -> dict:
        """Возвращает словарь текущих настроек для сохранения в БД."""
        settings_dict = {
            "MAX_HISTORY": self.MAX_HISTORY,
            "DEFAULT_STYLE": self.DEFAULT_STYLE,
            "BOT_NAME": self.BOT_NAME,
            "HISTORY_TTL": self.HISTORY_TTL,
            "GEMINI_MODEL": self.GEMINI_MODEL,
            # Сохраняем настройки генерации как JSON строку
            "GEMINI_GENERATION_CONFIG": json.dumps(self.GEMINI_GENERATION_CONFIG)
        }
        return settings_dict

# --- Инициализация настроек ---
settings = BotSettings()

# --- Роли для истории чата ---
USER_ROLE = "User"
ASSISTANT_ROLE = "Assistant"
SYSTEM_ROLE = "System" # Используется для /remember и сброса контекста

# --- Константы для файлов и директорий ---
DB_FILE = "bot_database.sqlite"
TEMP_MEDIA_DIR = "temp_media"

# --- Vector DB (ChromaDB) Settings ---
CHROMA_DB_PATH = "./chroma_db_v2" # Обновленный путь
CHROMA_HISTORY_COLLECTION_PREFIX = "history_"
CHROMA_FACTS_COLLECTION_NAME = "facts_store"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "ai-forever/sbert_large_mt_nlu_ru") # Используем эту модель
VECTOR_SEARCH_K_HISTORY = int(os.getenv("VECTOR_SEARCH_K_HISTORY", 5)) # K для истории
VECTOR_SEARCH_K_FACTS = int(os.getenv("VECTOR_SEARCH_K_FACTS", 3)) # K для фактов

# --- LLM Context Settings ---
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", 7000))
# Модель токенизатора для подсчета токенов (может требовать `pip install transformers[sentencepiece]`)
TOKENIZER_MODEL_NAME = os.getenv("TOKENIZER_MODEL_NAME", "ai-forever/sbert_large_mt_nlu_ru")

# --- Настройка логирования ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('bot.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
log_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Настраиваем корневой логгер
logging.basicConfig(level=logging.INFO, handlers=[log_handler, console_handler])
# Логгер для нашего приложения
logger = logging.getLogger("chatbot")
logger.setLevel(logging.DEBUG) # Устанавливаем DEBUG для детальных логов

# Уменьшаем спам от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)
logging.getLogger("telegram.ext").setLevel(logging.INFO)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("pydub").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.INFO)


logger.info("Configuration loaded. Logger initialized.")
logger.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
logger.info(f"ChromaDB Path: {CHROMA_DB_PATH}")
logger.info(f"Vector Search K (History/Facts): {VECTOR_SEARCH_K_HISTORY}/{VECTOR_SEARCH_K_FACTS}")
logger.info(f"Context Max Tokens: {CONTEXT_MAX_TOKENS}")
logger.info(f"Tokenizer Model: {TOKENIZER_MODEL_NAME}")
# -*- coding: utf-8 -*-
# config.py
import os
import json # Добавили json для настроек генерации
from pathlib import Path
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict

# Добавляем импорт HarmCategory и HarmBlockThreshold
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_TYPES_AVAILABLE = True
except ImportError:
    HarmCategory = None # type: ignore
    HarmBlockThreshold = None # type: ignore
    GEMINI_TYPES_AVAILABLE = False
    # Логгер может быть еще не инициализирован здесь, поэтому пока без лога
    # logger.warning("google.generativeai.types not found. Cannot configure specific safety settings.")

# --- Загрузка конфигурации ---
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- Токены и Ключи ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY') # Добавили ключ Mistral
ADMIN_USER_IDS = list(map(int, os.getenv('ADMIN_IDS', '').split(','))) if os.getenv('ADMIN_IDS') else []

# --- Класс настроек бота ---
class BotSettings:
    def __init__(self):
        # Добавляем настройки LLM
        self.LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'together')
        self.LLM_CONFIG = {
            'gemini': {'api_key': GEMINI_API_KEY},
            'mistral': {'api_key': MISTRAL_API_KEY, 'model': 'mistral-small'},
            'openai': {'api_key': os.getenv('OPENAI_API_KEY')},
            'groq': {'api_key': os.getenv('GROQ_API_KEY')},
            'together': {'api_key': os.getenv('TOGETHER_API_KEY')},
            'ollama': {'base_url': os.getenv('OLLAMA_URL', 'http://localhost:11434')}
        }
        
        # Оптимизации ChromaDB
        self.CHROMA_MODE = os.getenv('CHROMA_MODE', 'balanced')  # [perf, balanced, quality]
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'ai-forever/sbert_large_nlu_ru')
        
        # Загружаем значения из .env или используем значения по умолчанию
        self._initial_max_history = int(os.getenv('MAX_HISTORY', '15'))
        self._initial_default_style = os.getenv('DEFAULT_STYLE', "Ты - Маша,  живая и дружелюбная девушка.  Ты легко общаешься,  но не слишком легкомысленна.  Ты умеешь  поддерживать  разговор,  даже если он  не имеет  особого  смысла.  Если  собеседник  говорит  что-то  непонятное,  спроси  у него  уточнения  или  пошути  над  этим.  Не  забудь  отслеживать  контекст  беседы. Говори ТОЛЬКО на русском языке,будь краткой и лаконичной в ответах если ты находишься в групповом чате.")
        self._initial_bot_name = os.getenv('BOT_NAME', 'Маша')
        self._initial_history_ttl = int(os.getenv('HISTORY_TTL', '86400')) # Время жизни истории в памяти/SQLite
        self._initial_gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        # Настройки генерации Gemini по умолчанию
        self._default_gemini_generation_config = {
            "temperature": 0.7, "top_p": 0.95, "top_k": 40,
        }
        # Настройки для случайных сообщений
        self._initial_enable_random_messages = os.getenv('ENABLE_RANDOM_MESSAGES', 'False').lower() == 'true'
        self._initial_RANDOM_MESSAGE_INTERVAL_MINUTES = int(os.getenv('RANDOM_MESSAGE_INTERVAL_MINUTES', '10')) # Интервал в часах
        self._initial_random_message_history_context_count = int(os.getenv('RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT', '10'))
        # Новый параметр: порог релевантности фактов
        self._initial_facts_relevance_threshold = float(os.getenv('FACTS_RELEVANCE_THRESHOLD', '0.5'))
        # Новый параметр: минимальное время неактивности чата для случайного сообщения (в минутах)
        self._initial_random_message_min_inactivity_minutes = int(os.getenv('RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES', '30'))

        # Настройки безопасности Gemini по умолчанию (блокировать только HIGH)
        self._default_gemini_safety_settings_dict = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        # Загрузка настроек безопасности из env (если есть)
        env_safety_settings = os.getenv('GEMINI_SAFETY_SETTINGS_JSON')
        if env_safety_settings:
            try:
                self._initial_gemini_safety_settings_dict = json.loads(env_safety_settings)
                # logger.info("Loaded Gemini safety settings from environment variable.") # Логгер еще не готов
            except json.JSONDecodeError:
                # logger.warning("Could not parse GEMINI_SAFETY_SETTINGS_JSON from env, using defaults.") # Логгер еще не готов
                self._initial_gemini_safety_settings_dict = self._default_gemini_safety_settings_dict
        else:
             self._initial_gemini_safety_settings_dict = self._default_gemini_safety_settings_dict

        self.MAX_HISTORY = self._initial_max_history
        self.DEFAULT_STYLE = self._initial_default_style
        self.BOT_NAME = self._initial_bot_name
        self.HISTORY_TTL = self._initial_history_ttl
        self.GEMINI_MODEL = self._initial_gemini_model
        # Инициализируем настройки генерации
        self.GEMINI_GENERATION_CONFIG = self._default_gemini_generation_config.copy()
        # Инициализируем настройки случайных сообщений
        self.ENABLE_RANDOM_MESSAGES = self._initial_enable_random_messages
        self.RANDOM_MESSAGE_INTERVAL_MINUTES = self._initial_RANDOM_MESSAGE_INTERVAL_MINUTES
        self.RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT = self._initial_random_message_history_context_count
        # Инициализируем порог релевантности
        self.FACTS_RELEVANCE_THRESHOLD = self._initial_facts_relevance_threshold
        # Инициализируем новый параметр
        self.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES = self._initial_random_message_min_inactivity_minutes
        # Инициализируем настройки безопасности
        self.GEMINI_SAFETY_SETTINGS = self._parse_safety_settings(self._initial_gemini_safety_settings_dict)

        self.CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', 1000))

    def _parse_safety_settings(self, settings_dict: dict) -> Optional[List[Dict]]:
        """Преобразует словарь настроек безопасности в формат для API Gemini."""
        if not GEMINI_TYPES_AVAILABLE: return None
        parsed_settings = []
        for category_str, threshold_str in settings_dict.items():
            try:
                category = getattr(HarmCategory, category_str)
                threshold = getattr(HarmBlockThreshold, threshold_str)
                parsed_settings.append({"category": category, "threshold": threshold})
            except AttributeError:
                 # logger.warning(f"Invalid safety setting category or threshold: {category_str}={threshold_str}. Skipping.") # Логгер может быть не готов
                 pass # Просто игнорируем неверные настройки
        # logger.debug(f"Parsed Gemini safety settings: {parsed_settings}") # Логгер может быть не готов
        return parsed_settings if parsed_settings else None

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
        self.ENABLE_RANDOM_MESSAGES = self._initial_enable_random_messages
        self.RANDOM_MESSAGE_INTERVAL_MINUTES = self._initial_RANDOM_MESSAGE_INTERVAL_MINUTES
        self.RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT = self._initial_random_message_history_context_count
        self.FACTS_RELEVANCE_THRESHOLD = self._initial_facts_relevance_threshold
        self.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES = self._initial_random_message_min_inactivity_minutes
        self.GEMINI_SAFETY_SETTINGS = self._parse_safety_settings(self._initial_gemini_safety_settings_dict)
        logger.info("Bot settings reset to initial values in memory.")

    def load_from_db(self, db_settings: dict):
        """Загружает настройки из словаря, полученного из БД."""
        self.MAX_HISTORY = int(db_settings.get('MAX_HISTORY', self._initial_max_history))
        # self.DEFAULT_STYLE = db_settings.get('DEFAULT_STYLE', self._initial_default_style) # Removed loading from DB
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
        # Загрузка настроек случайных сообщений
        random_msg_val = db_settings.get('ENABLE_RANDOM_MESSAGES', self._initial_enable_random_messages)
        if isinstance(random_msg_val, str):
            self.ENABLE_RANDOM_MESSAGES = random_msg_val.lower() == 'true'
        elif isinstance(random_msg_val, bool):
            self.ENABLE_RANDOM_MESSAGES = random_msg_val
        else: # Fallback
            self.ENABLE_RANDOM_MESSAGES = bool(random_msg_val)

        self.RANDOM_MESSAGE_INTERVAL_MINUTES = int(db_settings.get('RANDOM_MESSAGE_INTERVAL_MINUTES', self._initial_RANDOM_MESSAGE_INTERVAL_MINUTES))
        self.RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT = int(db_settings.get('RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT', self._initial_random_message_history_context_count))
        # Загрузка порога релевантности фактов
        self.FACTS_RELEVANCE_THRESHOLD = float(db_settings.get('FACTS_RELEVANCE_THRESHOLD', self._initial_facts_relevance_threshold))
        # Загрузка нового параметра
        self.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES = int(db_settings.get('RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES', self._initial_random_message_min_inactivity_minutes))
        # Загрузка настроек безопасности Gemini из БД
        safety_config_json = db_settings.get('GEMINI_SAFETY_SETTINGS')
        loaded_safety_dict = self._initial_gemini_safety_settings_dict # По умолчанию берем начальные
        if safety_config_json:
            try:
                loaded_safety_dict = json.loads(safety_config_json)
            except json.JSONDecodeError:
                logger.warning("Could not parse GEMINI_SAFETY_SETTINGS from DB, using initial values.")
        self.GEMINI_SAFETY_SETTINGS = self._parse_safety_settings(loaded_safety_dict)
        logger.info("Bot settings loaded from DB data.")
        logger.debug(f"Loaded Gemini Config: {self.GEMINI_GENERATION_CONFIG}")
        logger.debug(f"Loaded Facts Threshold: {self.FACTS_RELEVANCE_THRESHOLD}") # Добавим лог
        logger.debug(f"Loaded Gemini Safety Config Dict: {loaded_safety_dict}")

    def get_settings_dict(self) -> dict:
        """Возвращает словарь текущих настроек для сохранения в БД."""
        settings_dict = {
            "MAX_HISTORY": self.MAX_HISTORY,
            # "DEFAULT_STYLE": self.DEFAULT_STYLE, # Removed saving to DB
            "BOT_NAME": self.BOT_NAME,
            "HISTORY_TTL": self.HISTORY_TTL,
            "GEMINI_MODEL": self.GEMINI_MODEL,
            # Сохраняем настройки генерации как JSON строку
            "GEMINI_GENERATION_CONFIG": json.dumps(self.GEMINI_GENERATION_CONFIG),
            "ENABLE_RANDOM_MESSAGES": self.ENABLE_RANDOM_MESSAGES,
            "RANDOM_MESSAGE_INTERVAL_MINUTES": self.RANDOM_MESSAGE_INTERVAL_MINUTES,
            "RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT": self.RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT,
            "FACTS_RELEVANCE_THRESHOLD": self.FACTS_RELEVANCE_THRESHOLD,
            "RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES": self.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES,
            # Сохраняем настройки безопасности как JSON строку словаря
            "GEMINI_SAFETY_SETTINGS": json.dumps(self._initial_gemini_safety_settings_dict), # Сохраняем исходный словарь
        }
        return settings_dict

    def detect_available_providers(self) -> List[str]:
        """Определяет доступные провайдеры на основе наличия API-ключей"""
        available = []
        
        # Проверяем ключи API
        if self.LLM_CONFIG['gemini'].get('api_key'):
            available.append('gemini')
        if self.LLM_CONFIG['mistral'].get('api_key'):
            available.append('mistral')
        if self.LLM_CONFIG['openai'].get('api_key'):
            available.append('openai')
        if self.LLM_CONFIG['groq'].get('api_key'):
            available.append('groq')
        if self.LLM_CONFIG['together'].get('api_key'):
            available.append('together')
        
        # Ollama всегда доступен локально
        available.append('ollama')
        
        if not available:
            # Если ни один провайдер не доступен, добавляем Gemini как fallback
            available.append('gemini')
    
        return available

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
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "ai-forever/sbert_large_nlu_ru") # Используем эту модель
VECTOR_SEARCH_K_HISTORY = int(os.getenv("VECTOR_SEARCH_K_HISTORY", 5)) # K для истории
VECTOR_SEARCH_K_FACTS = int(os.getenv("VECTOR_SEARCH_K_FACTS", 3)) # K для фактов

# --- LLM Context Settings ---
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", 2000))
# Модель токенизатора для подсчета токенов (может требовать `pip install transformers[sentencepiece]`)
TOKENIZER_MODEL_NAME = os.getenv("TOKENIZER_MODEL_NAME", "ai-forever/sbert_large_nlu_ru")

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
logger.info(f"Facts Relevance Threshold: {settings.FACTS_RELEVANCE_THRESHOLD}")
logger.info(f"Tokenizer Model: {TOKENIZER_MODEL_NAME}")
logger.info(f"Random Msg Min Inactivity (min): {settings.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES}")

# Теперь инициализируем AVAILABLE_PROVIDERS после создания логгера
settings.AVAILABLE_PROVIDERS = settings.detect_available_providers()
logger.info(f"Available LLM providers: {', '.join(settings.AVAILABLE_PROVIDERS)}")
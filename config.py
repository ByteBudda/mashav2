import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# ==============================================================================
# Начало: Содержимое config.py / bot_settings.py
# ==============================================================================

# --- Загрузка конфигурации ---
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- Токены и Ключи ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ADMIN_USER_IDS = list(map(int, os.getenv('ADMIN_IDS', '').split(','))) if os.getenv('ADMIN_IDS') else []
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# --- Настройка логирования ---
# Определяем логгер на верхнем уровне
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Предотвращаем дублирование хендлеров
if not logger.handlers:
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Файловый хендлер с ротацией
    log_file_handler = RotatingFileHandler(
        'bot.log',
        maxBytes=5*1024*1024, # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    log_file_handler.setFormatter(log_formatter)
    logger.addHandler(log_file_handler)

    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

# --- Класс настроек бота ---
class BotSettings:
    def __init__(self):
        self.MAX_HISTORY_RESULTS = int(os.getenv('MAX_HISTORY_RESULTS', '15'))
        self.MAX_HISTORY_TOKENS = int(os.getenv('MAX_HISTORY_TOKENS', '2000'))
        self.DEFAULT_STYLE = os.getenv('DEFAULT_STYLE', "Ты - дружелюбный помощник.") # Базовый стиль по умолчанию
        self.BOT_NAME = os.getenv('BOT_NAME', 'Бот')
        self.HISTORY_TTL = int(os.getenv('HISTORY_TTL', '86400'))
        self.CHROMA_DATA_PATH = os.getenv('CHROMA_DATA_PATH', './chroma_db_data')
        self.GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')

    def update_default_style(self, new_style: str):
        self.DEFAULT_STYLE = new_style
        logger.info(f"Default style updated to: {new_style}")

    def update_bot_name(self, new_name: str):
        self.BOT_NAME = new_name
        logger.info(f"Bot name updated to: {new_name}")

# --- Инициализация настроек ---
settings = BotSettings()

# --- Экспорт отдельных настроек для удобства (если нужно) ---
MAX_HISTORY_RESULTS = settings.MAX_HISTORY_RESULTS
MAX_HISTORY_TOKENS = settings.MAX_HISTORY_TOKENS
DEFAULT_STYLE = settings.DEFAULT_STYLE
BOT_NAME = settings.BOT_NAME
HISTORY_TTL = settings.HISTORY_TTL
CHROMA_DATA_PATH = settings.CHROMA_DATA_PATH
GEMINI_MODEL_NAME = settings.GEMINI_MODEL_NAME

# --- Роли для истории чата ---
USER_ROLE = "User"
ASSISTANT_ROLE = "Assistant"
SYSTEM_ROLE = "System"

# --- Константы для файлов ---
KNOWLEDGE_FILE = "learned_knowledge.json"
USER_DATA_DIR = "user_data"

# --- Промпт для проверки контекста ---
CONTEXT_CHECK_PROMPT = f"""Ты - эксперт по определению контекста диалога. Тебе нужно решить, является ли следующее сообщение пользователя логическим продолжением или прямым ответом на предыдущее сообщение бота ({BOT_NAME}). Сообщение пользователя должно относиться к той же теме, продолжать обсуждение или отвечать на вопрос, заданный ботом.

Предыдущее сообщение бота ({BOT_NAME}): "{{last_bot_message}}"
Сообщение пользователя: "{{current_message}}"

Ответь строго "Да", если сообщение пользователя является продолжением или ответом, и "Нет", если это новое, не связанное сообщение или просто приветствие/прощание. Не давай никаких дополнительных объяснений.
"""

logger.info("Configuration loaded. Logger initialized.")

# ==============================================================================
# Конец: Содержимое config.py / bot_settings.py
# ==============================================================================
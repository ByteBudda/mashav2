# -*- coding: utf-8 -*-
# main.py
import asyncio
from datetime import timedelta
import signal
import json # Для парсинга JSON из БД

from telegram import Update
from telegram.ext import (ApplicationBuilder, CallbackContext, CallbackQueryHandler,
                          CommandHandler, ContextTypes, MessageHandler, filters, Application)

# --- Импорт конфигурации ---
from config import (ADMIN_USER_IDS, TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, MISTRAL_API_KEY, # Добавили MISTRAL_API_KEY
                    logger, settings, DB_FILE)

# --- Импорт состояния и функций управления им ---
from state import load_all_data, save_all_data, cleanup_history_job, fact_extraction_job # Добавили fact_extraction_job

# --- Импорт утилит ---
from utils import cleanup_audio_files_job

# --- Импорт обработчиков и команд ---
from bot_commands import (
    start_command, help_command, remember_command, clear_my_history_command,
    button_callback, set_my_name_command, reset_context_command,
    clear_history_command, set_default_style_command, set_bot_name_command,
    set_activity_command, reset_style_command, set_group_user_style_command,
    reset_group_user_style_command, set_group_style_command, reset_group_style_command,
    ban_user_command, unban_user_command, list_banned_command, list_admins_command,
    get_log_command, error_handler,
    get_gen_params_command, set_gen_params_command # Добавили команды управления параметрами Gemini
)
from handlers import handle_message, handle_photo


# --- Настройка обработчиков ---
def setup_handlers(application: Application):
    """Регистрирует все обработчики команд и сообщений."""
    # Пользовательские команды
    user_commands = { "start": start_command, "help": help_command, "remember": remember_command,
                      "clear_my_history": clear_my_history_command, "setmyname": set_my_name_command,
                      "reset_context": reset_context_command }
    for command, handler in user_commands.items():
        application.add_handler(CommandHandler(command, handler))

    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND | filters.VOICE | filters.VIDEO_NOTE, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Обработчик колбэков
    application.add_handler(CallbackQueryHandler(button_callback, pattern='^(confirm_clear_my_history_|cancel_clear)'))

    # Административные команды
    admin_filter = filters.User(user_id=ADMIN_USER_IDS) if ADMIN_USER_IDS else filters.UpdateFilter(lambda: False)
    admin_commands = {
        "clear_history": clear_history_command, "get_log": get_log_command, "list_admins": list_admins_command,
        "set_default_style": set_default_style_command, "reset_style": reset_style_command,
        "set_bot_name": set_bot_name_command, "set_group_style": set_group_style_command,
        "reset_group_style": reset_group_style_command, "set_group_user_style": set_group_user_style_command,
        "reset_group_user_style": reset_group_user_style_command, "ban": ban_user_command,
        "unban": unban_user_command, "list_banned": list_banned_command, "set_activity": set_activity_command,
        "get_gen_params": get_gen_params_command, "set_gen_params": set_gen_params_command # Добавили новые команды
    }
    for command, handler in admin_commands.items():
        application.add_handler(CommandHandler(command, handler, filters=admin_filter))

    # Обработчик ошибок
    application.add_error_handler(error_handler)
    logger.info("All handlers registered.")


# --- Фоновые задачи ---
async def save_all_data_job_wrapper(context: CallbackContext):
    """Асинхронная обертка для сохранения данных SQLite из JobQueue."""
    logger.debug("Periodic save job triggered (SQLite).")
    await asyncio.to_thread(save_all_data)


def setup_jobs(application: Application):
    """Регистрирует фоновые задачи."""
    if not application.job_queue: logger.warning("Job queue not available."); return
    jq = application.job_queue
    # Очистка истории (память + SQLite + ChromaDB)
    jq.run_repeating(cleanup_history_job, interval=timedelta(hours=1), first=timedelta(minutes=1), name="cleanup_history")
    # Очистка временных медиа-файлов
    jq.run_repeating(cleanup_audio_files_job, interval=timedelta(hours=2), first=timedelta(minutes=2), name="cleanup_audio_files")
    # Периодическое сохранение данных SQLite
    jq.run_repeating(save_all_data_job_wrapper, interval=timedelta(minutes=10), first=timedelta(minutes=5), name="save_sqlite_data")
    # Периодическое извлечение фактов
    jq.run_repeating(fact_extraction_job, interval=timedelta(hours=4), first=timedelta(minutes=15), name="extract_facts")
    logger.info("All background jobs registered.")


# --- Инициализация и Запуск ---
async def post_init(application: Application):
    """Выполняется после инициализации, перед запуском поллинга."""
    commands = [
        ("start", "Начать/приветствие"), ("help", "Помощь по командам"),
        ("remember", "Запомнить что-то"), ("clear_my_history", "Очистить мою историю"),
        ("setmyname", "Как меня называть"), ("reset_context", "Сбросить диалог")
    ]
    try: await application.bot.set_my_commands(commands); logger.info("Bot commands set successfully.")
    except Exception as e: logger.error(f"Failed to set bot commands: {e}")


async def main_async():
    """Асинхронная основная функция для управления жизненным циклом."""
    logger.info("----- Bot Starting Async -----")
    logger.info(f"Using SQLite database: {DB_FILE}")
    if not TELEGRAM_BOT_TOKEN: logger.critical("TELEGRAM_BOT_TOKEN not set!"); exit(1)
    if not GEMINI_API_KEY: logger.warning("GEMINI_API_KEY not set. Main AI features might fail.")
    if not MISTRAL_API_KEY: logger.warning("MISTRAL_API_KEY not set. Fact extraction will be disabled.")

    application = None

    try:
        load_all_data() # Загрузка данных SQLite и инициализация ChromaDB/моделей

        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
        setup_handlers(application); setup_jobs(application)
        logger.info("Bot setup complete. Starting application...")

        async with application: # Контекстный менеджер для управления start/shutdown
            await application.start()
            await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            logger.info("Bot is running. Press Ctrl+C to stop.")
            while True: await asyncio.sleep(3600) # Ожидаем прерывания

    except KeyboardInterrupt: logger.info("KeyboardInterrupt received. Stopping...")
    except Exception as e: logger.critical(f"Unhandled runtime exception: {e}", exc_info=True)
    finally:
        logger.info("----- Bot Stopping -----")
        # async with application позаботится о корректном shutdown
        if application and application.running: logger.info("Application shutdown handled by context manager.")
        # Сохраняем данные перед выходом
        logger.info("Saving final data...")
        save_all_data() # Синхронное сохранение SQLite
        logger.info("----- Bot Stopped -----")


def main():
    """Синхронная точка входа."""
    try: asyncio.run(main_async())
    except Exception as e: logger.critical(f"Critical error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
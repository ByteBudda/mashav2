import asyncio
from datetime import timedelta # Для setup_jobs

from telegram import Update
from telegram.ext import (Application, ApplicationBuilder, CallbackContext, CallbackQueryHandler,
                          CommandHandler, ContextTypes, MessageHandler, filters)

# Импорт конфигурации
from config import (ADMIN_USER_IDS, GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, logger, settings)

# Импорт состояния и функций управления им
from state import load_all_data, save_all_data, cleanup_history_job

# Импорт утилит
from utils import cleanup_audio_files_job, initialize_gemini_model
# Импорт инициализатора векторного хранилища
from vector_store import initialize_vector_store

# Импорт обработчиков команд
from bot_commands import (start_command, help_command, remember_command,
                          clear_my_history_command, set_my_name_command,
                          reset_context_command, my_style_command, button_callback,
                          # Админские команды
                          clear_history_command, ban_user_command, set_default_style_command,
                          set_bot_name_command, set_activity_command, list_admins_command,
                          get_log_command, set_group_user_style_command, reset_style_command,
                          # Обработчик ошибок
                          error_handler)

# Импорт обработчиков сообщений
from handlers import (handle_message, handle_photo, handle_video_note_message,
                    handle_voice_message)

# ==============================================================================
# Начало: Содержимое main.py
# ==============================================================================

# --- Настройка обработчиков и запуск бота ---

def setup_handlers(application: Application):
    """Регистрирует все обработчики команд и сообщений."""
    # Команды пользователей
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("remember", remember_command))
    application.add_handler(CommandHandler("clear_my_history", clear_my_history_command))
    application.add_handler(CommandHandler("setmyname", set_my_name_command))
    application.add_handler(CommandHandler("reset_context", reset_context_command))
    application.add_handler(CommandHandler("mystyle", my_style_command)) # Добавили my_style

    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.VIA_BOT, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.VIA_BOT, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE & ~filters.VIA_BOT, handle_voice_message))
    application.add_handler(MessageHandler(filters.VIDEO_NOTE & ~filters.VIA_BOT, handle_video_note_message))

    # Обработчик колбэков
    application.add_handler(CallbackQueryHandler(button_callback))

    # Административные команды (фильтр применяется ко всем)
    admin_filter = filters.User(user_id=ADMIN_USER_IDS) if ADMIN_USER_IDS else filters.User(user_id=[]) # Пустой фильтр, если админов нет
    application.add_handler(CommandHandler("clear_history", clear_history_command, filters=admin_filter))
    application.add_handler(CommandHandler("ban", ban_user_command, filters=admin_filter))
    application.add_handler(CommandHandler("set_default_style", set_default_style_command, filters=admin_filter))
    application.add_handler(CommandHandler("set_bot_name", set_bot_name_command, filters=admin_filter))
    application.add_handler(CommandHandler("set_activity", set_activity_command, filters=admin_filter))
    application.add_handler(CommandHandler("list_admins", list_admins_command, filters=admin_filter))
    application.add_handler(CommandHandler("get_log", get_log_command, filters=admin_filter))
    application.add_handler(CommandHandler("set_group_user_style", set_group_user_style_command, filters=admin_filter))
    application.add_handler(CommandHandler("reset_style", reset_style_command, filters=admin_filter))

    # Обработчик ошибок (должен быть последним)
    application.add_error_handler(error_handler)
    logger.info("All handlers registered.")


def setup_jobs(application: Application):
    """Регистрирует фоновые задачи."""
    jq = application.job_queue
    if jq:
        # Очистка старой истории в ChromaDB
        jq.run_repeating(cleanup_history_job, interval=timedelta(hours=1), first=timedelta(minutes=5), name="cleanup_vector_history")
        # Очистка временных аудио/видео файлов
        jq.run_repeating(cleanup_audio_files_job, interval=timedelta(hours=1), first=timedelta(seconds=60), name="cleanup_temp_files")
        logger.info("Background jobs registered.")
    else:
        logger.warning("Job Queue not available. Background jobs not scheduled.")

# --- Точка входа ---
def main():
    """Основная функция запуска бота."""
    logger.info("----- Bot Starting -----")

    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN not set in .env file. Exiting.")
        return
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY not set in .env file. Exiting.")
        return

    try:
        # Загружаем данные (настройки, learned_responses, user_info и т.д.)
        load_all_data()

        # --- ИНИЦИАЛИЗАЦИЯ ChromaDB ---
        if not initialize_vector_store():
             logger.critical("Failed to initialize vector store. Bot cannot function properly. Exiting.")
             return

        # Инициализация модели Gemini (важно после загрузки настроек)
        if not initialize_gemini_model():
             logger.critical("Failed to initialize Gemini model. Bot cannot function properly. Exiting.")
             return

        # Создаем приложение Telegram
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

        # Настраиваем обработчики и задачи
        setup_handlers(application)
        setup_jobs(application)

        logger.info("Bot setup complete. Starting polling...")
        # Запускаем бота
        # stop_signals=None позволяет корректно обрабатывать Ctrl+C через try/except KeyboardInterrupt
        application.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping bot...")
    except Exception as e:
        logger.critical(f"Critical error during bot startup or execution: {e}", exc_info=True)
    finally:
        logger.info("----- Bot Stopping -----")
        # Сохраняем данные (кроме истории ChromaDB) перед выходом
        save_all_data()
        # Здесь не нужно явно останавливать ChromaDB PersistentClient
        logger.info("----- Bot Stopped -----")

if __name__ == "__main__":
    main()

# ==============================================================================
# Конец: Содержимое main.py
# ==============================================================================
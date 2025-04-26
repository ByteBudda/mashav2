# -*- coding: utf-8 -*-
# main.py
import asyncio
from datetime import timedelta
import signal
import json # Для парсинга JSON из БД
import random

from telegram import Update
from telegram.ext import (ApplicationBuilder, CallbackContext, CallbackQueryHandler,
                          CommandHandler, ContextTypes, MessageHandler, filters, Application)

# --- Импорт конфигурации ---
from config import (ADMIN_USER_IDS, TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, MISTRAL_API_KEY, # Добавили MISTRAL_API_KEY
                    logger, settings, DB_FILE, ASSISTANT_ROLE, USER_ROLE)

# --- Импорт состояния и функций управления им ---
from state import load_all_data, save_all_data, cleanup_history_job, fact_extraction_job, _get_recent_active_group_chat_ids_sync, _get_recent_messages_sync, add_to_memory_history, save_message_and_embed

# --- Импорт утилит ---
from utils import cleanup_audio_files_job, generate_content_sync, filter_response

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
    # Новая фоновая задача для отправки случайных сообщений
    # Запускаем только если включено в настройках
    if settings.ENABLE_RANDOM_MESSAGES:
        jq.run_repeating(
            send_random_message_job, 
            interval=timedelta(hours=settings.RANDOM_MESSAGE_INTERVAL_HOURS),
            first=timedelta(minutes=20), # Немного позже других задач
            name="send_random_message"
        )
        logger.info(f"Random message job registered to run every {settings.RANDOM_MESSAGE_INTERVAL_HOURS} hours.")
    else:
        logger.info("Random message job is disabled.")

    logger.info("All background jobs registered.")


# --- Новая фоновая задача для отправки случайных сообщений ---
async def send_random_message_job(context: CallbackContext):
    """Периодически отправляет случайное сообщение в один из активных групповых чатов."""
    if not settings.ENABLE_RANDOM_MESSAGES:
        logger.debug("Random message job skipped (disabled in settings).")
        return

    logger.info("Starting random message job...")
    try:
        # Получаем список активных групповых чатов (синхронная функция в to_thread)
        active_group_ids = await asyncio.to_thread(_get_recent_active_group_chat_ids_sync)
        if not active_group_ids:
            logger.info("Random message job: No active group chats found.")
            return

        # Выбираем случайный чат
        selected_chat_id = random.choice(active_group_ids)
        logger.info(f"Random message job: Selected chat ID {selected_chat_id}")

        # Получаем недавнюю историю для контекста (синхронная функция в to_thread)
        recent_messages = await asyncio.to_thread(
            _get_recent_messages_sync,
            history_key=selected_chat_id,
            limit=settings.RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT
        )

        if not recent_messages:
            logger.warning(f"Random message job: No recent history found for chat {selected_chat_id}, skipping.")
            return

        # Формируем контекст для промпта
        history_context = "\n".join(
            f"{role} ({name}): {msg}" if name else f"{role}: {msg}"
            for role, name, msg in recent_messages
        )

        # Формируем промпт
        prompt = (
            f"Ты - {settings.BOT_NAME}, общаешься в групповом чате. "
            f"Недавно в чате обсуждали следующее:\n--- НАЧАЛО КОНТЕКСТА ---\n{history_context}\n--- КОНЕЦ КОНТЕКСТА ---\n"
            f"Напиши КОРОТКОЕ (одно-два предложения) сообщение от своего лица ({settings.BOT_NAME}), "
            f"чтобы поддержать беседу, задать релевантный вопрос или просто поделиться интересной мыслью, связанной с контекстом. "
            f"Не надо здороваться или представляться. Будь естественной."
            f"\n\n{settings.BOT_NAME}:"
        )

        logger.debug(f"Random message job: Sending prompt for chat {selected_chat_id} (context: {len(recent_messages)} msgs)")
        # Генерируем сообщение (синхронная функция в to_thread)
        generated_message = await asyncio.to_thread(generate_content_sync, prompt)
        filtered_message = filter_response(generated_message)

        if filtered_message and not filtered_message.startswith("["):
            logger.info(f"Random message job: Generated message for chat {selected_chat_id}: '{filtered_message[:50]}...'")
            # Отправляем сообщение
            await context.bot.send_message(chat_id=selected_chat_id, text=filtered_message)
            logger.info(f"Random message job: Message sent successfully to chat {selected_chat_id}.")

            # Сохраняем сообщение бота в историю
            add_to_memory_history(selected_chat_id, ASSISTANT_ROLE, filtered_message)
            await save_message_and_embed(selected_chat_id, ASSISTANT_ROLE, filtered_message)
            logger.debug(f"Random message job: Bot's message saved to history for chat {selected_chat_id}.")
        else:
            logger.warning(f"Random message job: Failed to generate a valid message for chat {selected_chat_id}. Response: {generated_message[:100]}...")

    except Exception as e:
        logger.error(f"Error in send_random_message_job: {e}", exc_info=True)


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
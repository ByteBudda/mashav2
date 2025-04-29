# -*- coding: utf-8 -*-
# main.py
import asyncio
from datetime import timedelta
import signal
import json # Для парсинга JSON из БД
import random
import time

from telegram import Update
from telegram.ext import (ApplicationBuilder, CallbackContext, CallbackQueryHandler,
                          CommandHandler, ContextTypes, MessageHandler, filters, Application)

# --- Импорт конфигурации ---
from config import (ADMIN_USER_IDS, TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, MISTRAL_API_KEY, # Добавили MISTRAL_API_KEY
                    logger, settings, DB_FILE, ASSISTANT_ROLE, USER_ROLE)

# --- Импорт состояния и функций управления им ---
from state import load_all_data, save_all_data, cleanup_history_job, fact_extraction_job, _get_recent_active_group_chat_ids_sync, _get_recent_messages_sync, add_to_memory_history, save_message_and_embed

# --- Импорт утилит ---
from utils import cleanup_audio_files_job, generate_content_sync, filter_response, metrics_job

# --- Импорт обработчиков и команд ---
from bot_commands import (
    start_command, help_command, remember_command, clear_my_history_command,
    button_callback, set_my_name_command, reset_context_command,
    clear_history_command, set_default_style_command, set_bot_name_command,
    set_activity_command, reset_style_command, set_group_user_style_command,
    reset_group_user_style_command, set_group_style_command, reset_group_style_command,
    ban_user_command, unban_user_command, list_banned_command, list_admins_command,
    get_log_command, error_handler,
    get_gen_params_command, set_gen_params_command, # Добавили команды управления параметрами Gemini
    list_providers_command, switch_provider_command, provider_stats_command, clear_cache_command, cache_stats_command
)
from handlers import handle_message, handle_photo, handle_document


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
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document)) # Ловит все документы
    # Обработчик колбэков
    application.add_handler(CallbackQueryHandler(button_callback, pattern='^(confirm_clear_my_history_|cancel_clear)'))

    # Административные команды
    admin_filter = filters.User(user_id=ADMIN_USER_IDS) if ADMIN_USER_IDS else filters.UpdateFilter(lambda _: False)
    admin_commands = {
        "clear_history": clear_history_command, "get_log": get_log_command, "list_admins": list_admins_command,
        "set_default_style": set_default_style_command, "reset_style": reset_style_command,
        "set_bot_name": set_bot_name_command, "set_group_style": set_group_style_command,
        "reset_group_style": reset_group_style_command, "set_group_user_style": set_group_user_style_command,
        "reset_group_user_style": reset_group_user_style_command, "ban": ban_user_command,
        "unban": unban_user_command, "list_banned": list_banned_command, "set_activity": set_activity_command,
        "get_gen_params": get_gen_params_command, "set_gen_params": set_gen_params_command, # Добавили новые команды
        "list_providers": list_providers_command,
        "switch_provider": switch_provider_command,
        "provider_stats": provider_stats_command,
        "clear_cache": clear_cache_command,
        "cache_stats": cache_stats_command
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
            interval=timedelta(minutes=settings.RANDOM_MESSAGE_INTERVAL_MINUTES),
            first=timedelta(minutes=20), # Немного позже других задач
            name="send_random_message"
        )
        logger.info(f"Random message job registered to run every {settings.RANDOM_MESSAGE_INTERVAL_MINUTES} minutes.")
    else:
        logger.info("Random message job is disabled.")

    jq.run_repeating(metrics_job, interval=timedelta(minutes=5), first=10)

    logger.info("All background jobs registered.")


# --- Новая фоновая задача для отправки случайных сообщений ---
async def send_random_message_job(context: CallbackContext):
    """Периодически отправляет случайное сообщение в один из активных групповых чатов."""
    if not settings.ENABLE_RANDOM_MESSAGES:
        logger.debug("Random message job skipped (disabled in settings).")
        return

    logger.info("Starting random message job...")
    now = time.time()
    min_inactivity_seconds = settings.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES * 60

    try:
        # Получаем список активных групповых чатов (синхронная функция в to_thread)
        active_group_ids = await asyncio.to_thread(_get_recent_active_group_chat_ids_sync)
        if not active_group_ids:
            logger.info("Random message job: No active group chats found.")
            return

        # --- Фильтруем чаты по времени последней активности ---
        suitable_chat_ids = []
        for chat_id in active_group_ids:
            # Получаем только последнее сообщение, чтобы проверить время
            last_message_data = await asyncio.to_thread(_get_recent_messages_sync, chat_id, limit=1)
            if last_message_data:
                # Извлекаем timestamp из кортежа
                _, _, _, last_ts = last_message_data[0]
                if now - last_ts > min_inactivity_seconds:
                    suitable_chat_ids.append(chat_id)
                    logger.debug(f"Chat {chat_id} is suitable (last message > {settings.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES} min ago).")
                else:
                    logger.debug(f"Chat {chat_id} is too active (last message <= {settings.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES} min ago).")
            else:
                logger.debug(f"Could not get last message for chat {chat_id} to check activity.")

        if not suitable_chat_ids:
            logger.info(f"Random message job: No suitable inactive group chats found (min inactivity: {settings.RANDOM_MESSAGE_MIN_INACTIVITY_MINUTES} min).")
            return

        # Выбираем случайный чат из подходящих
        selected_chat_id = random.choice(suitable_chat_ids)
        logger.info(f"Random message job: Selected chat ID {selected_chat_id}")

        # Получаем недавнюю историю для контекста (синхронная функция в to_thread)
        # Теперь _get_recent_messages_sync возвращает и timestamp, но нам он здесь не нужен
        recent_messages_with_ts = await asyncio.to_thread(
            _get_recent_messages_sync,
            history_key=selected_chat_id,
            limit=settings.RANDOM_MESSAGE_HISTORY_CONTEXT_COUNT
        )
        # Отбрасываем timestamp для формирования контекста промпта
        recent_messages = [(role, name, msg) for role, name, msg, _ in recent_messages_with_ts]

        if not recent_messages:
            logger.warning(f"Random message job: No recent history found for selected chat {selected_chat_id}, skipping.")
            return

        # Формируем контекст для промпта
        history_context = "\n".join(
            f"{role} ({name}): {msg}" if name else f"{role}: {msg}"
            for role, name, msg in recent_messages
        )

        # --- Выбор случайного типа промпта ---
        prompt_styles = [
            "чтобы поддержать беседу, задать релевантный вопрос или просто поделиться интересной мыслью, связанной с контекстом.",
            "чтобы задать открытый вопрос, связанный с последними обсуждавшимися темами.",
            "чтобы поделиться коротким наблюдением или фактом, который может быть интересен участникам чата, учитывая контекст.",
            "чтобы предложить идею или тему для обсуждения, отталкиваясь от недавнего диалога.",
            "чтобы вспомнить что-то забавное или интересное из предыдущего обсуждения (если было)."
        ]
        selected_style = random.choice(prompt_styles)

        # Формируем промпт с использованием одной многострочной f-строки
        prompt = f"""Ты - {settings.BOT_NAME}, участник группового чата. Твоя задача - инициировать общение или поддержать его, пока в чате затишье. \
Недавний контекст общения:
--- НАЧАЛО КОНТЕКСТА ---
{history_context}
--- КОНЕЦ КОНТЕКСТА ---
Напиши ОЧЕНЬ КОРОТКОЕ (1-2 предложения максимум) сообщение от своего лица ({settings.BOT_NAME}), \
{selected_style} \
Не надо здороваться, представляться или извиняться за молчание. Просто напиши сообщение по делу. Будь естественной и дружелюбной.

{settings.BOT_NAME}:"""

        logger.debug(f"Random message job: Sending prompt for chat {selected_chat_id} (context: {len(recent_messages)} msgs, style: '{selected_style[:30]}...')")
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
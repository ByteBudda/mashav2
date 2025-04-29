# -*- coding: utf-8 -*-
# bot_commands.py
# ... (other imports and functions remain the same) ...
import asyncio
import json
import re # Импортируем re для экранирования
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile, constants
from telegram.ext import ContextTypes, CallbackContext, CommandHandler, CallbackQueryHandler, filters
import logging
import os
from functools import wraps
from typing import Any, Optional, List, Dict # Добавили Dict
from collections import deque
import time
import sqlite3 # Для get_banned_users type hint

# Используем настройки и константы из config
from config import logger, ADMIN_USER_IDS, settings, SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE

# Импортируем функции работы с состоянием/БД
from state import (
    add_to_memory_history, chat_history, last_activity, # In-memory state
    set_user_preferred_name_in_db, get_user_info_from_db, # User DB functions
    bot_activity_percentage, save_bot_settings_to_db, # Bot state/settings
    get_db_connection, _execute_db, # DB helpers
    set_group_user_style_in_db, delete_group_user_style_in_db, # User-group style DB
    set_group_style_in_db, delete_group_style_in_db, # Group style DB
    ban_user_in_db, unban_user_in_db, is_user_banned, get_banned_users # Ban DB functions
)
# Импорт синхронных функций удаления эмбеддингов
from vector_db import (
    delete_embeddings_by_sqlite_ids_sync,
    delete_facts_by_history_key_sync,
    delete_fact_embeddings_by_ids_sync # Пока не используется напрямую здесь
)
# --- Импорт метрик и кэша ---
# Предполагается, что они находятся в utils.py или где-то еще
try:
    from utils import metrics, cache # Попробуем импортировать
except ImportError:
    logger.warning("Could not import metrics or cache from utils. Stats/Cache commands might fail.")
    # Определим заглушки, чтобы команды не падали с NameError
    class DummyMetrics:
        def get_stats(self): return {'average_times': {}, 'error_counts': {}}
    class DummyCache:
        cache = {}; hits = 0; misses = 0
        def clear(self): pass
    metrics = DummyMetrics()
    cache = DummyCache()

# --- Вспомогательная функция экранирования ---
def escape_markdown_v2(text: str) -> str:
    """Экранирует все зарезервированные символы MarkdownV2."""
    if not isinstance(text, str): text = str(text) # На случай, если передали не строку
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    # Экранируем ИМЕННО эти символы, предваряя их '\'
    # Используем re.escape, чтобы обработать спецсимволы regex внутри скобок []
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Декоратор проверки админа ---
# ... (admin_only decorator remains the same) ...
def admin_only(func):
    """Декоратор для проверки, является ли пользователь админом бота."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user = update.effective_user
        if not user or user.id not in ADMIN_USER_IDS:
            msg = "🚫 У вас нет прав\\."
            try: # Добавим try-except на случай проблем с отправкой
                if update.message: await update.message.reply_text(msg, parse_mode='MarkdownV2')
                elif update.callback_query: await update.callback_query.answer("🚫 Нет прав!", show_alert=True)
            except Exception as e:
                logger.error(f"Failed to send 'no rights' message: {e}")
            logger.warning(f"Unauthorized admin command attempt by user {user.id if user else 'Unknown'}")
            return None
        return await func(update, context, *args, **kwargs)
    return wrapper

# --- Команды бота ---
# ... (start_command, remember_command, etc. remain the same until help_command) ...

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user = update.effective_user; msg = update.message
    if not user or not msg: return
    uname = escape_markdown_v2(user.first_name or f"User_{user.id}")
    bname = escape_markdown_v2(settings.BOT_NAME)
    await msg.reply_text(f"Привет, {uname}\\! Я *{bname}*\\. Чем могу помочь?", parse_mode='MarkdownV2')
    logger.info(f"User {user.id} started the bot.")


async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /remember <текст>."""
    user = update.effective_user; chat = update.effective_chat; msg = update.message
    if not user or not chat or not msg: return
    history_key = chat.id if chat.type != 'private' else user.id

    if context.args:
        memory = " ".join(context.args).strip()
        if memory:
            sys_msg = f"Важное напоминание от пользователя: {memory}"
            add_to_memory_history(history_key, SYSTEM_ROLE, sys_msg)
            # Не сохраняем системные сообщения в векторную БД по умолчанию
            await msg.reply_text(f"📝 Запомнила: '{escape_markdown_v2(memory)}'\\.", parse_mode='MarkdownV2')
            logger.info(f"User {user.id} added memory for key {history_key}: '{memory[:50]}...'")
        else: await msg.reply_text("Напоминание не может быть пустым\\.", parse_mode='MarkdownV2')
    else: await msg.reply_text("Что мне нужно запомнить\\?", parse_mode='MarkdownV2')


async def clear_my_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Предлагает пользователю подтвердить очистку его личной истории."""
    user = update.effective_user; msg = update.message
    if not user or not msg: return
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Да, очистить", callback_data=f'confirm_clear_my_history_{user.id}'),
                                      InlineKeyboardButton("Нет, отмена", callback_data='cancel_clear')]])
    await msg.reply_text("Вы уверены, что хотите очистить вашу историю переписки\\?\nЭто необратимо\\.", reply_markup=keyboard, parse_mode='MarkdownV2')


async def button_callback(update: Update, context: CallbackContext):
    """Обработчик нажатий на инлайн-кнопки (только для очистки истории)."""
    query = update.callback_query
    if not query or not query.from_user: return
    await query.answer() # Answer immediately to avoid timeout appearance
    data = query.data; user_id = query.from_user.id

    if data.startswith('confirm_clear_my_history_'):
        target_user_id = int(data.split('_')[-1])
        if user_id == target_user_id:
            history_key = user_id # Личная история
            history_cleared = False; db_cleared = None; deleted_sqlite_ids = []

            # Clear memory
            if history_key in chat_history:
                del chat_history[history_key]
                history_cleared = True
            if history_key in last_activity:
                del last_activity[history_key]

            # Clear DB
            conn = None
            try:
                conn = get_db_connection(); cursor = conn.cursor()
                cursor.execute("SELECT id FROM history WHERE history_key = ?", (history_key,))
                deleted_sqlite_ids = [row['id'] for row in cursor.fetchall()]
                if deleted_sqlite_ids:
                    # Use parameter substitution for safety
                    placeholders = ','.join('?' * len(deleted_sqlite_ids))
                    cursor.execute(f"DELETE FROM history WHERE id IN ({placeholders})", tuple(deleted_sqlite_ids))
                    db_cleared = cursor.rowcount
                    conn.commit()
                else:
                    db_cleared = 0 # No rows deleted
            except sqlite3.Error as e:
                logger.error(f"Error clearing SQLite history for {history_key}: {e}", exc_info=True)
                db_cleared = None
                if conn:
                    try: conn.rollback()
                    except Exception: pass # Ignore rollback error if connection closed
            finally:
                if conn: conn.close()

            if db_cleared is None:
                try: await query.edit_message_text("❌ Ошибка очистки истории из базы данных\\.")
                except Exception as edit_e: logger.error(f"Failed editing msg after DB error: {edit_e}")
                return

            # Clear Vector DB embeddings
            if deleted_sqlite_ids:
                logger.info(f"Deleting {len(deleted_sqlite_ids)} history embeddings from ChromaDB for key {history_key}...")
                await asyncio.to_thread(delete_embeddings_by_sqlite_ids_sync, history_key, deleted_sqlite_ids)
                 # Also delete related facts
                logger.info(f"Deleting facts for key {history_key}...")
                await asyncio.to_thread(delete_facts_by_history_key_sync, history_key)

            # Send confirmation
            if history_cleared or db_cleared > 0:
                logger.info(f"User {user_id} cleared history (mem: {history_cleared}, db: {db_cleared} rows).")
                try: await query.edit_message_text("✅ История чата успешно очищена\\.")
                except Exception as edit_e: logger.error(f"Failed editing success msg: {edit_e}")
            else:
                try: await query.edit_message_text("ℹ️ История чата уже была пуста\\.")
                except Exception as edit_e: logger.error(f"Failed editing empty msg: {edit_e}")
        else:
            try: await query.edit_message_text("🚫 Нельзя очистить чужую историю\\.")
            except Exception as edit_e: logger.error(f"Failed editing wrong user msg: {edit_e}")
            logger.warning(f"User {user_id} tried clearing history for {target_user_id}.")

    elif data == 'cancel_clear':
        try: await query.edit_message_text("✅ Действие отменено\\.")
        except Exception as edit_e: logger.error(f"Failed editing cancel msg: {edit_e}")


async def set_my_name_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает предпочитаемое имя пользователя."""
    user = update.effective_user; msg = update.message
    if not user or not msg: return
    if context.args:
        name = " ".join(context.args).strip()
        if name: await asyncio.to_thread(set_user_preferred_name_in_db, user.id, name); await msg.reply_text(f"Хорошо, {escape_markdown_v2(name)}\\!", parse_mode='MarkdownV2')
        else: await msg.reply_text("Имя не может быть пустым\\.", parse_mode='MarkdownV2')
    else: await msg.reply_text("Как вас называть\\?", parse_mode='MarkdownV2')


async def reset_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сбрасывает контекст (историю в памяти) текущего чата."""
    user = update.effective_user; chat = update.effective_chat; msg = update.message
    if not user or not chat or not msg: return
    history_key = chat.id if chat.type != 'private' else user.id

    if history_key in chat_history:
        chat_history[history_key].clear(); logger.info(f"User {user.id} reset context for key {history_key}.")
        await msg.reply_text("Начинаем с чистого листа\\!", parse_mode='MarkdownV2')
    else: await msg.reply_text("Контекст и так пуст\\.", parse_mode='MarkdownV2')


# --- Команда помощи ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает справку по командам с использованием MarkdownV2."""
    user = update.effective_user; msg = update.message
    if not user or not msg: return
    uname = escape_markdown_v2(user.first_name or f"User_{user.id}")
    bname = escape_markdown_v2(settings.BOT_NAME)

    parts = [f"Привет, {uname}\\! Я *{bname}*\\. Вот команды:\n", "*Основные:*"]
    user_cmds = { "/start": "Начать", "/help": "Помощь", "/remember <текст>": "Запомнить",
                  "/clear_my_history": "Очистить мою историю", "/setmyname <имя>": "Мое имя", "/reset_context": "Сбросить диалог" }
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    for cmd, desc in user_cmds.items():
        # НЕ экранируем cmd, оставляем как есть (например, /start)
        desc_escaped = escape_markdown_v2(desc)
        # Формируем строку: /command \- экранированное описание\.
        line = f"{cmd} \\- {desc_escaped}\\."
        parts.append(line)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    if user.id in ADMIN_USER_IDS:
        parts.append("\n*Админские:*")
        admin_cmds = {
            "/clear_history <ID>": "Очистить историю (память, БД, векторы)",
            "/set_default_style <стиль>": "Глобальный стиль (текущая сессия)",
            "/reset_style": "Сброс стиля (к значению из .env)",
            "/set_bot_name <имя>": "Имя бота (сохраняется в БД)",
            "/set_activity <%>": "% активности в группах (сохраняется)",
            "/set_group_style <стиль>": "Стиль для этой группы (БД)",
            "/reset_group_style": "Сброс стиля этой группы (БД)",
            "/set_group_user_style <стиль>": "Стиль для юзера (ответ на сообщение, БД)", # Передаем стиль как аргумент
            "/reset_group_user_style": "Сброс стиля юзера (ответ на сообщение, БД)",
            "/ban <ID/ответ> [причина]": "Забанить (БД и чат)",
            "/unban <ID>": "Разбанить (БД и чат)",
            "/list_banned": "Список банов из БД",
            "/list_admins": "Список ID админов из .env",
            "/get_log": "Отправить файл bot.log",
            "/get_gen_params": "Показать параметры генерации Gemini",
            "/set_gen_params <p>=<v>": "Установить параметры Gemini (БД)",
            "/list_providers": "Список доступных LLM провайдеров",
            "/switch_provider <имя>": "Переключить активного LLM провайдера",
            "/provider_stats": "Статистика LLM провайдеров (время, ошибки)",
            "/clear_cache": "Очистить кэш ответов LLM",
            "/cache_stats": "Статистика кэша ответов LLM"
        }
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        for cmd, desc in admin_cmds.items():
            # НЕ экранируем cmd
            desc_escaped = escape_markdown_v2(desc)
            # Формируем строку: /command \- экранированное описание\.
            # Обратите внимание на <стиль> в /set_group_user_style - он не является частью команды
            if cmd == "/set_group_user_style <стиль>":
                 command_part = "/set_group_user_style" # Команда без аргумента
                 line = f"{command_part} `<стиль>` \\- {desc_escaped}\\." # Аргумент показываем как код
            elif cmd == "/switch_provider <имя>":
                 command_part = "/switch_provider"
                 line = f"{command_part} `<имя>` \\- {desc_escaped}\\."
            # Добавьте сюда другие команды с <аргументами>, если нужно выделить аргумент
            else:
                 line = f"{cmd} \\- {desc_escaped}\\."
            parts.append(line)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    final_text = "\n".join(parts)
    try: await msg.reply_text(final_text, parse_mode='MarkdownV2')
    # ... (остальная часть функции без изменений) ...
    except Exception as e:
        logger.error(f"Failed sending help MDv2: {e}. Text was:\n{final_text}\nSending plain.")
        plain = final_text
        for char in r'_*[]()~`>#+-=|{}.!':
            plain = plain.replace(f'\\{char}', char)
        plain = plain.replace('*', '').replace('`', '')
        try: await msg.reply_text(plain)
        except Exception as fallback_e: logger.error(f"Failed sending plain help: {fallback_e}")


# --- Обработчик ошибок ---
# ... (error_handler remains the same) ...
async def error_handler(update: object, context: CallbackContext):
    """Логирует ошибки и отправляет уведомление админу."""
    logger.error(f"Exception while handling update: {context.error}", exc_info=context.error)
    chat_id, user_id, update_type = "N/A", "N/A", type(update).__name__
    if isinstance(update, Update):
        if update.effective_chat: chat_id = update.effective_chat.id
        if update.effective_user: user_id = update.effective_user.id
    err = context.error
    # Safely convert error to string before escaping
    err_msg_raw = str(err) if err else "Unknown error"
    err_type = escape_markdown_v2(type(err).__name__)
    err_msg = escape_markdown_v2(err_msg_raw)

    full_msg = f"⚠️ *Ошибка*\n*Тип:* `{err_type}`\n*Ошибка:* `{err_msg}`\n*Update:* `{update_type}`\n*Chat:* `{chat_id}`\n*User:* `{user_id}`"
    if ADMIN_USER_IDS:
        try:
            # Truncate message if too long
            await context.bot.send_message(ADMIN_USER_IDS[0], full_msg[:4090], parse_mode='MarkdownV2')
        except Exception as e:
            logger.error(f"Failed sending error notification: {e}")
            # Try sending plain text if Markdown fails
            try:
                plain_msg = full_msg.replace('*', '').replace('`', '').replace('\\', '') # Basic unescaping
                await context.bot.send_message(ADMIN_USER_IDS[0], plain_msg[:4090])
            except Exception as plain_e:
                 logger.error(f"Failed sending plain error notification: {plain_e}")


# --- Административные команды (с декоратором @admin_only) ---
# ... (set_group_user_style_command, reset_group_user_style_command, etc. remain the same until list_banned_command) ...

@admin_only
async def set_group_user_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.reply_to_message: await msg.reply_text("Ответьте на сообщение: `/set_group_user_style <стиль>`"); return
    if not context.args: await msg.reply_text("Укажите стиль\\."); return
    target = msg.reply_to_message.from_user; style = " ".join(context.args).strip()
    if not target or not style: await msg.reply_text("Ошибка данных/стиля\\."); return
    success = await asyncio.to_thread(set_group_user_style_in_db, msg.chat_id, target.id, style)
    if success: await msg.reply_text(f"✅ Стиль для {target.mention_markdown_v2()} установлен\\.", parse_mode='MarkdownV2'); logger.info(f"Admin {msg.from_user.id} set style for {target.id} in {msg.chat_id}.")
    else: await msg.reply_text("❌ Ошибка установки стиля \\(БД\\)\\.", parse_mode='MarkdownV2')

@admin_only
async def reset_group_user_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.reply_to_message: await msg.reply_text("Ответьте на сообщение: `/reset_group_user_style`"); return
    target = msg.reply_to_message.from_user
    if not target: await msg.reply_text("Ошибка данных\\."); return
    success = await asyncio.to_thread(delete_group_user_style_in_db, msg.chat_id, target.id)
    if success: await msg.reply_text(f"✅ Стиль для {target.mention_markdown_v2()} сброшен\\.", parse_mode='MarkdownV2'); logger.info(f"Admin {msg.from_user.id} reset style for {target.id} in {msg.chat_id}.")
    else: await msg.reply_text("❌ Стиль не был установлен или ошибка БД\\.", parse_mode='MarkdownV2')

@admin_only
async def set_group_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg=update.message; chat = update.effective_chat;
    if not msg or not chat or chat.type == 'private': await msg.reply_text("Только для групп\\."); return
    if not context.args: await msg.reply_text("Укажите стиль: `/set_group_style <стиль>`"); return
    style = " ".join(context.args).strip()
    if not style: await msg.reply_text("Стиль не пустой\\."); return
    success = await asyncio.to_thread(set_group_style_in_db, chat.id, style)
    if success: await msg.reply_text(f"✅ Стиль группы установлен\\."); logger.info(f"Admin {msg.from_user.id} set style for group {chat.id}.")
    else: await msg.reply_text("❌ Ошибка установки стиля \\(БД\\)\\.")

@admin_only
async def reset_group_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg=update.message; chat = update.effective_chat;
    if not msg or not chat or chat.type == 'private': await msg.reply_text("Только для групп\\."); return
    success = await asyncio.to_thread(delete_group_style_in_db, chat.id)
    if success: await msg.reply_text(f"✅ Стиль группы сброшен\\."); logger.info(f"Admin {msg.from_user.id} reset style for group {chat.id}.")
    else: await msg.reply_text("❌ Стиль не был установлен или ошибка БД\\.")

@admin_only
async def reset_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg=update.message; user = update.effective_user
    if not msg or not user: return
    # Reset only in memory, don't save to DB
    initial_style = settings._initial_default_style
    settings.update_default_style(initial_style)
    # Removed saving to DB: await asyncio.to_thread(save_bot_settings_to_db, settings.get_settings_dict(), bot_activity_percentage)
    escaped = escape_markdown_v2(initial_style)
    await msg.reply_text(f"✅ Глобальный стиль сброшен \\(до значения из config/env\\) в текущей сессии:\n```\n{escaped}\n```", parse_mode='MarkdownV2'); logger.info(f"Admin {user.id} reset global style in memory.")


@admin_only
async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; user = update.effective_user
    if not msg or not user: return
    if not context.args: await msg.reply_text("Укажите ID чата или пользователя: `/clear_history <ID>`"); return
    try: history_key = int(context.args[0])
    except (ValueError, IndexError): await msg.reply_text("Укажите числовой ID\\."); return

    history_cleared = False; db_cleared = None; deleted_sqlite_ids = []

    # Clear memory
    if history_key in chat_history:
        del chat_history[history_key]
        history_cleared = True
    if history_key in last_activity:
        del last_activity[history_key]

    # Clear DB
    conn = None
    try:
        conn = get_db_connection(); cursor = conn.cursor()
        # Find IDs first
        cursor.execute("SELECT id FROM history WHERE history_key = ?", (history_key,))
        deleted_sqlite_ids = [row['id'] for row in cursor.fetchall()]
        if deleted_sqlite_ids:
            placeholders = ','.join('?' * len(deleted_sqlite_ids))
            cursor.execute(f"DELETE FROM history WHERE id IN ({placeholders})", tuple(deleted_sqlite_ids))
            db_cleared = cursor.rowcount
            conn.commit()
        else:
            db_cleared = 0
    except sqlite3.Error as e:
        logger.error(f"Error clearing SQLite history for key {history_key}: {e}", exc_info=True)
        db_cleared = None
        if conn:
            try: conn.rollback()
            except Exception: pass
    finally:
        if conn: conn.close()

    if db_cleared is None:
        await msg.reply_text("❌ Ошибка очистки истории из базы данных\\."); return

    # Clear Vector DB embeddings if DB rows were deleted
    if deleted_sqlite_ids:
        logger.info(f"Deleting {len(deleted_sqlite_ids)} history embeddings for {history_key}...")
        await asyncio.to_thread(delete_embeddings_by_sqlite_ids_sync, history_key, deleted_sqlite_ids)
        # Also delete related facts
        logger.info(f"Deleting facts for key {history_key}...")
        await asyncio.to_thread(delete_facts_by_history_key_sync, history_key)

    hk_escaped = escape_markdown_v2(str(history_key))
    if history_cleared or db_cleared > 0:
        await msg.reply_text(f"✅ История для `{hk_escaped}` очищена \\(память: {history_cleared}, БД: {db_cleared} rows, векторы удалены\\)\\.", parse_mode='MarkdownV2')
        logger.info(f"Admin {user.id} cleared history for {history_key} (mem: {history_cleared}, db: {db_cleared}).")
    else:
        await msg.reply_text(f"ℹ️ История для `{hk_escaped}` не найдена в памяти или БД\\.", parse_mode='MarkdownV2')


@admin_only
async def list_admins_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message;
    if not msg: return
    if ADMIN_USER_IDS:
        admin_list = "\n".join(f"\\- `{aid}`" for aid in ADMIN_USER_IDS)
        await msg.reply_text(f"🔑 *Админы:*\n{admin_list}", parse_mode='MarkdownV2')
    else:
        await msg.reply_text("ℹ️ Список ID админов пуст \\(не задан в `.env`\\)\\.", parse_mode='MarkdownV2')

@admin_only
async def get_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; user = update.effective_user; chat = update.effective_chat
    if not msg or not user or not chat: return
    log_file = 'bot.log' # Assumes RotatingFileHandler is named this
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        try:
             # Escape dots in the message
             await msg.reply_text("Отправляю файл логов\\.\\.\\.", parse_mode='MarkdownV2')
             # Send file with simple caption
             await context.bot.send_document(chat.id, InputFile(log_file), caption="bot.log")
             logger.info(f"Admin {user.id} requested log file.")
        except constants.NetworkError as e:
            logger.error(f"Net error sending log: {e}")
            await msg.reply_text(f"❌ Сетевая ошибка: `{escape_markdown_v2(str(e))}`", parse_mode='MarkdownV2')
        except Exception as e:
            logger.error(f"Failed sending log: {e}", exc_info=True)
            await msg.reply_text(f"❌ Ошибка отправки лога: `{escape_markdown_v2(str(e))}`", parse_mode='MarkdownV2')
    elif os.path.exists(log_file):
        await msg.reply_text("Файл `bot.log` пуст\\.", parse_mode='MarkdownV2')
    else:
        await msg.reply_text("Файл `bot.log` не найден\\.", parse_mode='MarkdownV2')

@admin_only
async def ban_user_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; user = update.effective_user; chat = update.effective_chat
    if not msg or not user or not chat: return
    reason = " ".join(context.args[1:]) if len(context.args) > 1 else None
    target_id = None
    target_info = "пользователю" # Default display name

    if msg.reply_to_message:
        target_user = msg.reply_to_message.from_user
        if not target_user:
            await msg.reply_text("Не удалось получить пользователя из ответа\\."); return
        target_id = target_user.id
        target_info = target_user.mention_markdown_v2() # Use mention if available
    elif context.args:
        try:
            target_id = int(context.args[0])
            target_info = f"ID `{target_id}`" # Use ID if no reply
        except (ValueError, IndexError):
            await msg.reply_text("Укажите числовой ID пользователя или ответьте на его сообщение: `/ban <ID> [причина]`"); return
    else:
        await msg.reply_text("Укажите ID пользователя или ответьте на его сообщение: `/ban <ID> [причина]`"); return

    if target_id == user.id:
        await msg.reply_text("Себя забанить нельзя\\."); return
    if target_id in ADMIN_USER_IDS:
        await msg.reply_text("Нельзя забанить другого администратора\\."); return

    if await asyncio.to_thread(is_user_banned, target_id):
        await msg.reply_text(f"{target_info} уже забанен\\.", parse_mode='MarkdownV2'); return

    success = await asyncio.to_thread(ban_user_in_db, target_id, reason)
    if success:
        reason_text = f" Причина: _{escape_markdown_v2(reason)}_" if reason else ""
        reply_msg = f"✅ {target_info} успешно забанен в базе данных\\.{reason_text}"
        await msg.reply_text(reply_msg, parse_mode='MarkdownV2')
        logger.info(f"Admin {user.id} banned {target_id} in DB. Reason: {reason}")

        # Try banning in the current chat if it's a group
        if chat.type != 'private':
            try:
                await context.bot.ban_chat_member(chat.id, target_id)
                await msg.reply_text(f"✅ Также забанен в текущем чате `{escape_markdown_v2(str(chat.id))}`\\.", parse_mode='MarkdownV2')
                logger.info(f"Banned {target_id} in chat {chat.id}")
            except Exception as e:
                logger.warning(f"Could not ban {target_id} in chat {chat.id} (maybe no rights or user not present): {e}")
                await msg.reply_text(f"⚠️ Не удалось забанить в текущем чате \\(возможно, нет прав или пользователя нет в чате\\)\\.", parse_mode='MarkdownV2')
    else:
        await msg.reply_text(f"❌ Не удалось забанить {target_info} \\(ошибка базы данных\\)\\.", parse_mode='MarkdownV2')

@admin_only
async def unban_user_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; user = update.effective_user; chat = update.effective_chat
    if not msg or not user or not chat: return
    if not context.args:
        await msg.reply_text("Укажите ID пользователя для разбана: `/unban <ID>`"); return
    try:
        target_id = int(context.args[0])
        target_info = f"ID `{target_id}`"
    except (ValueError, IndexError):
        await msg.reply_text("Укажите числовой ID пользователя\\."); return

    if not await asyncio.to_thread(is_user_banned, target_id):
        await msg.reply_text(f"{target_info} не найден в списке забаненных в БД\\.", parse_mode='MarkdownV2'); return

    success = await asyncio.to_thread(unban_user_in_db, target_id)
    if success:
        await msg.reply_text(f"✅ {target_info} разбанен в базе данных\\.", parse_mode='MarkdownV2')
        logger.info(f"Admin {user.id} unbanned {target_id} in DB.")

        # Try unbanning in the current chat if it's a group
        if chat.type != 'private':
            try:
                # only_if_banned=True prevents errors if user wasn't banned in this specific chat
                await context.bot.unban_chat_member(chat.id, target_id, only_if_banned=True)
                await msg.reply_text(f"✅ Также разбанен в текущем чате `{escape_markdown_v2(str(chat.id))}` \\(если был забанен здесь\\)\\.", parse_mode='MarkdownV2')
                logger.info(f"Unbanned {target_id} in chat {chat.id}")
            except Exception as e:
                logger.warning(f"Could not unban {target_id} in chat {chat.id} (maybe no rights): {e}")
                await msg.reply_text(f"⚠️ Не удалось разбанить в текущем чате \\(возможно, нет прав\\)\\.", parse_mode='MarkdownV2')
    else:
        await msg.reply_text(f"❌ Не удалось разбанить {target_info} \\(ошибка базы данных\\)\\.", parse_mode='MarkdownV2')


@admin_only
async def list_banned_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message;
    if not msg: return
    banned_list: List[sqlite3.Row] = await asyncio.to_thread(get_banned_users)
    if not banned_list:
        await msg.reply_text("ℹ️ Список забаненных в базе данных пуст\\.", parse_mode='MarkdownV2'); return

    parts = ["🚫 *Забаненные в БД:*"]
    MAX_LINES_DISPLAY = 30 # Limit output length
    count = 0
    for entry in banned_list:
        if count >= MAX_LINES_DISPLAY:
            parts.append("\n\\.\\.\\. \\(список сокращен, всего: " + escape_markdown_v2(str(len(banned_list))) + "\\)")
            break

        uid = entry['user_id']
        reason_raw = entry['reason'] or "Не указана"
        try:
            ban_time_local = time.localtime(entry['banned_at'])
            time_str_raw = time.strftime('%Y-%m-%d %H:%M', ban_time_local)
        except Exception:
            time_str_raw = "Дата неизвестна"

        # Escape dynamic parts
        uid_escaped = escape_markdown_v2(str(uid))
        reason_escaped = escape_markdown_v2(reason_raw)
        time_str_escaped = escape_markdown_v2(time_str_raw)

        # Attempt to get user info for better display name
        user_info_str = f"ID: `{uid_escaped}`" # Default display
        uinfo = await asyncio.to_thread(get_user_info_from_db, uid)
        if uinfo:
            fname = escape_markdown_v2(uinfo['first_name'] or "")
            lname = escape_markdown_v2(uinfo['last_name'] or "")
            uname = escape_markdown_v2(uinfo['username'] or '?')
            user_info_str = f"{fname} {lname} (@{uname}) \\(ID: `{uid_escaped}`\\)"

        parts.append(f"\n\\- {user_info_str}")
        parts.append(f"  _Причина:_ {reason_escaped}")
        parts.append(f"  _Когда:_ {time_str_escaped}")
        count += 1

    try: await msg.reply_text("\n".join(parts), parse_mode='MarkdownV2')
    except Exception as e:
         logger.error(f"Failed sending banned list MDv2: {e}. Text was:\n{' '.join(parts)}\nSending plain.")
         plain = "\n".join(parts)
         for char in r'_*[]()~`>#+-=|{}.!': plain = plain.replace(f'\\{char}', char)
         plain = plain.replace('*', '').replace('`', '')
         try: await msg.reply_text(plain)
         except Exception as fallback_e: logger.error(f"Failed sending plain banned list: {fallback_e}")


@admin_only
async def set_default_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; user = update.effective_user
    if not msg or not user: return
    if not context.args:
        current_style_escaped = escape_markdown_v2(settings.DEFAULT_STYLE)
        await msg.reply_text(f"Текущий стиль \\(в памяти\\):\n```\n{current_style_escaped}\n```\nЧтобы изменить: `/set_default_style <новый стиль>`", parse_mode='MarkdownV2')
        return
    style = " ".join(context.args).strip()
    if style:
        settings.update_default_style(style) # Updates only in memory
        escaped = escape_markdown_v2(style)
        await msg.reply_text(f"✅ Стиль по умолчанию изменен \\(только для текущей сессии\\):\n```\n{escaped}\n```", parse_mode='MarkdownV2')
        logger.info(f"Admin {user.id} set global style in memory.")
    else:
        await msg.reply_text("Стиль не может быть пустым\\.", parse_mode='MarkdownV2')

@admin_only
async def set_bot_name_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; user = update.effective_user
    if not msg or not user: return
    if not context.args:
        await msg.reply_text("Укажите новое имя бота: `/set_bot_name <имя>`"); return
    name = " ".join(context.args).strip()
    if name:
        settings.update_bot_name(name) # Update in memory
        # Save to DB
        await asyncio.to_thread(save_bot_settings_to_db, settings.get_settings_dict(), bot_activity_percentage)
        await msg.reply_text(f"✅ Имя бота изменено на *{escape_markdown_v2(settings.BOT_NAME)}* и сохранено\\.", parse_mode='MarkdownV2')
        logger.info(f"Admin {user.id} set bot name to '{settings.BOT_NAME}'.")
    else:
        await msg.reply_text("Имя бота не может быть пустым\\.", parse_mode='MarkdownV2')

@admin_only
async def set_activity_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_activity_percentage # Modify global var from state
    msg = update.message; user = update.effective_user
    if not msg or not user: return
    if not context.args:
        await msg.reply_text(f"Текущая активность в группах: *{bot_activity_percentage}%*\\. Чтобы изменить, укажите новую: `/set_activity <%>`", parse_mode='MarkdownV2'); return
    try:
        percent = int(context.args[0])
        if 0 <= percent <= 100:
            bot_activity_percentage = percent
            # Save to DB
            await asyncio.to_thread(save_bot_settings_to_db, settings.get_settings_dict(), bot_activity_percentage)
            await msg.reply_text(f"✅ Активность в группах установлена на *{percent}%* и сохранена\\.", parse_mode='MarkdownV2')
            logger.info(f"Admin {user.id} set activity to {percent}%")
        else:
            await msg.reply_text("Процент должен быть от 0 до 100\\.", parse_mode='MarkdownV2')
    except (ValueError, IndexError):
        await msg.reply_text("Неверный формат\\. Введите целое число от 0 до 100\\.", parse_mode='MarkdownV2')

# --- Команды для управления параметрами генерации Gemini ---

@admin_only
async def get_gen_params_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущие параметры генерации Gemini."""
    msg = update.message
    if not msg: return
    params = settings.GEMINI_GENERATION_CONFIG
    try:
        # Use json.dumps for proper formatting, then escape
        params_str_raw = json.dumps(params, indent=2, ensure_ascii=False)
        params_str_escaped = escape_markdown_v2(params_str_raw)
        await msg.reply_text(f"Текущие параметры генерации Gemini:\n```json\n{params_str_escaped}\n```", parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Failed to format/send Gemini params: {e}")
        await msg.reply_text("Не удалось отобразить параметры генерации.")


@admin_only
async def set_gen_params_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает параметры генерации Gemini (например, /set_gen_params temp=0.6 top_p=0.8)."""
    msg = update.message; user = update.effective_user
    if not msg or not user: return
    if not context.args:
        await msg.reply_text(
            "Использование: `/set_gen_params <параметр>=<значение> [<параметр2>=<значение2>...]`\n"
            "Доступные параметры: `temperature` \\(или `temp`\\), `top_p`, `top_k`\n"
            "Пример: `/set_gen_params temp=0\\.6 top_p=0\\.8`", # Escape example dots
            parse_mode='MarkdownV2'
        )
        return

    # Start with a copy of the current settings
    new_params = settings.GEMINI_GENERATION_CONFIG.copy()
    applied_changes = {}
    errors = []

    for arg in context.args:
        if '=' not in arg:
            errors.append(f"Неверный формат аргумента \\(ожидается ключ=значение\\): `{escape_markdown_v2(arg)}`")
            continue

        key, value_str = arg.split('=', 1)
        key = key.strip().lower() # Normalize key

        # Map aliases and validate key
        config_key = None
        if key in ['temp', 'temperature']: config_key = 'temperature'
        elif key == 'top_p': config_key = 'top_p'
        elif key == 'top_k': config_key = 'top_k'
        # Add other parameters here if needed, e.g.:
        # elif key == 'max_tokens': config_key = 'max_output_tokens'

        if not config_key:
             errors.append(f"Неизвестный или неподдерживаемый параметр: `{escape_markdown_v2(key)}`")
             continue

        # Try to parse and validate the value based on the key
        try:
            original_value = new_params.get(config_key) # For type checking if needed
            if config_key in ['temperature', 'top_p']:
                value = float(value_str)
                if config_key == 'temperature' and not (0.0 <= value <= 1.0):
                    raise ValueError("должно быть от 0\\.0 до 1\\.0")
                if config_key == 'top_p' and not (0.0 < value <= 1.0): # top_p usually > 0
                    raise ValueError("должно быть больше 0\\.0 и до 1\\.0")
            elif config_key == 'top_k':
                value = int(value_str)
                if value <= 0:
                    raise ValueError("должно быть положительным целым числом")
            # Add validation for other types if needed
            # elif config_key == 'max_output_tokens':
            #     value = int(value_str)
            #     if value <= 0 : raise ValueError("должно быть > 0")
            else:
                 # Should not happen if key mapping is correct
                 raise ValueError("неподдерживаемый тип значения")

            new_params[config_key] = value
            applied_changes[config_key] = value

        except ValueError as e:
            errors.append(f"Неверное значение для `{escape_markdown_v2(key)}`: {escape_markdown_v2(str(e))}")

    if errors:
        error_message = "Обнаружены ошибки:\n\\- " + "\n\\- ".join(errors)
        await msg.reply_text(error_message, parse_mode='MarkdownV2')
        return

    if not applied_changes:
        await msg.reply_text("Не было применено ни одного изменения \\(возможно, введены текущие значения\\)\\.", parse_mode='MarkdownV2')
        return

    # Update settings in memory and save to DB
    settings.GEMINI_GENERATION_CONFIG = new_params
    await asyncio.to_thread(save_bot_settings_to_db, settings.get_settings_dict(), bot_activity_percentage)

    # Report success
    try:
        params_str_raw = json.dumps(settings.GEMINI_GENERATION_CONFIG, indent=2, ensure_ascii=False)
        params_str_escaped = escape_markdown_v2(params_str_raw)
        await msg.reply_text(f"✅ Параметры генерации Gemini обновлены и сохранены:\n```json\n{params_str_escaped}\n```", parse_mode='MarkdownV2')
        logger.info(f"Admin {user.id} updated Gemini generation params: {applied_changes}")
    except Exception as e:
         logger.error(f"Failed to format/send updated Gemini params: {e}")
         await msg.reply_text("✅ Параметры генерации обновлены и сохранены, но не удалось отобразить результат\\.")


# --- Provider/Cache Commands ---

@admin_only
async def list_providers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список доступных LLM провайдеров"""
    # Assuming settings.AVAILABLE_PROVIDERS exists and is a list/tuple
    try:
        available_providers = getattr(settings, 'AVAILABLE_PROVIDERS', [])
        if available_providers:
            providers_escaped = "\n".join([f"\\- `{escape_markdown_v2(p)}`" for p in available_providers])
            current_provider_escaped = escape_markdown_v2(settings.LLM_PROVIDER)
            await update.message.reply_text(
                f"🛠 *Доступные провайдеры:*\n{providers_escaped}\n\n"
                f"Текущий: *{current_provider_escaped}*",
                parse_mode='MarkdownV2'
            )
        else:
            await update.message.reply_text("ℹ️ Список доступных провайдеров не определен в настройках\\.", parse_mode='MarkdownV2')
    except AttributeError:
         await update.message.reply_text("❌ Ошибка: Настройки провайдеров не найдены в конфигурации\\.", parse_mode='MarkdownV2')


@admin_only
async def switch_provider_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переключает активный провайдер"""
    msg = update.message
    if not msg: return
    if not context.args:
        await msg.reply_text("Укажите имя провайдера: `/switch_provider <название>`")
        return

    provider_name_input = context.args[0].lower() # Normalize input

    try:
        available_providers = getattr(settings, 'AVAILABLE_PROVIDERS', [])
        if not available_providers:
             await msg.reply_text("❌ Список доступных провайдеров не определен в настройках\\.", parse_mode='MarkdownV2'); return

        if provider_name_input not in available_providers:
            providers_list_escaped = ", ".join(f"`{escape_markdown_v2(p)}`" for p in available_providers)
            await msg.reply_text(f"Провайдер `{escape_markdown_v2(provider_name_input)}` недоступен\\. Доступные: {providers_list_escaped}", parse_mode='MarkdownV2')
            return

        # Assuming settings.LLM_PROVIDER exists
        settings.LLM_PROVIDER = provider_name_input # Update in memory
        # Note: This change is typically NOT saved to DB unless specifically designed to be persistent
        provider_escaped = escape_markdown_v2(settings.LLM_PROVIDER)
        await msg.reply_text(f"✅ Активный LLM провайдер изменен на *{provider_escaped}* \\(для текущей сессии\\)\\.", parse_mode='MarkdownV2')
        logger.info(f"Admin {msg.from_user.id} switched LLM provider to {settings.LLM_PROVIDER}.")

    except AttributeError:
         await msg.reply_text("❌ Ошибка: Настройки провайдеров не найдены в конфигурации\\.", parse_mode='MarkdownV2')


@admin_only
async def provider_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает статистику использования провайдеров."""
    msg = update.message
    if not msg: return

    try:
        stats = metrics.get_stats() # Get stats from your metrics module
        available_providers = getattr(settings, 'AVAILABLE_PROVIDERS', [])

        lines = ["📊 *Статистика LLM провайдеров:*"]
        if not available_providers:
            lines.append("\n_Список провайдеров не определен_")
        else:
            for provider in available_providers:
                # Get stats, default to 0 if provider not in stats dict
                avg_time_raw = stats['average_times'].get(provider, 0.0)
                error_count_raw = stats['error_counts'].get(provider, 0)

                # Format and Escape dynamic parts
                provider_escaped = escape_markdown_v2(provider)
                # Format time to string, THEN escape the string containing the dot
                avg_time_str = f"{avg_time_raw:.3f}" # Format with 3 decimal places
                avg_time_escaped = escape_markdown_v2(avg_time_str)
                errors_escaped = escape_markdown_v2(str(error_count_raw)) # Escape error count

                lines.append(f"\n• *{provider_escaped}*:")
                lines.append(f"  Среднее время: {avg_time_escaped} сек") # Use escaped time string
                lines.append(f"  Ошибок: {errors_escaped}")

        await msg.reply_text("\n".join(lines), parse_mode='MarkdownV2')

    except AttributeError:
        await msg.reply_text("❌ Ошибка: Модуль статистики (`metrics`) или список провайдеров не найдены\\.", parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error getting/sending provider stats: {e}", exc_info=True)
        await msg.reply_text(f"❌ Ошибка при получении статистики: `{escape_markdown_v2(str(e))}`", parse_mode='MarkdownV2')


@admin_only
async def clear_cache_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg: return
    try:
        cache.clear() # Call clear method of your cache object
        await msg.reply_text("✅ Кэш ответов LLM успешно очищен\\.", parse_mode='MarkdownV2')
        logger.info(f"Admin {msg.from_user.id} cleared the cache.")
    except AttributeError:
         await msg.reply_text("❌ Ошибка: Модуль кэша (`cache`) не найден или не имеет метода `clear`\\.", parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        await msg.reply_text(f"❌ Ошибка при очистке кэша: `{escape_markdown_v2(str(e))}`", parse_mode='MarkdownV2')


@admin_only
async def cache_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg: return
    try:
        # Get stats from your cache object attributes
        cache_size = len(cache.cache)
        cache_hits = cache.hits
        cache_misses = cache.misses

        # Escape the numbers (just in case, though unlikely to contain special chars)
        size_escaped = escape_markdown_v2(str(cache_size))
        hits_escaped = escape_markdown_v2(str(cache_hits))
        misses_escaped = escape_markdown_v2(str(cache_misses))

        stats_text = (
            f"Размер кэша: {size_escaped}\n"
            f"Попаданий \\(hits\\): {hits_escaped}\n"
            f"Промахов \\(misses\\): {misses_escaped}"
        )
        await msg.reply_text(f"📦 *Статистика кэша LLM:*\n{stats_text}", parse_mode='MarkdownV2')

    except AttributeError:
        await msg.reply_text("❌ Ошибка: Модуль кэша (`cache`) или его атрибуты (`cache`, `hits`, `misses`) не найдены\\.", parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error getting/sending cache stats: {e}", exc_info=True)
        await msg.reply_text(f"❌ Ошибка при получении статистики кэша: `{escape_markdown_v2(str(e))}`", parse_mode='MarkdownV2')

# --- End of bot_commands.py ---
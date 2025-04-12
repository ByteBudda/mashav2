from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from telegram.ext import ContextTypes, CallbackContext, CommandHandler, CallbackQueryHandler, filters
import logging # logger получается из config
import os
from functools import wraps
from typing import Any, Dict # Для type hints, опционально

# Импорты из проекта
from config import BOT_NAME, DEFAULT_STYLE, SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE, HISTORY_TTL, logger, ADMIN_USER_IDS, settings
from state import (add_to_history, user_preferred_name, # Убрали chat_history, last_activity
                   group_user_style_prompts, bot_activity_percentage, user_topic)
# Импортируем функцию из vector_store
from vector_store import delete_history # Переименовываем для ясности или используем как есть

user_info_db: Dict[int, Dict[str, Any]] = {}

# ==============================================================================
# Начало: Содержимое bot_commands.py
# ==============================================================================

# --- Декораторы ---
def admin_only(func):
    """Декоратор для проверки прав администратора."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update or not update.effective_user or update.effective_user.id not in ADMIN_USER_IDS:
            if update and update.message:
                await update.message.reply_text("🚫 У вас нет прав для выполнения этой команды.")
            elif update and update.callback_query:
                 await update.callback_query.answer("🚫 Недостаточно прав!", show_alert=True)
            logger.warning(f"Unauthorized access attempt by user {update.effective_user.id if update.effective_user else 'Unknown'}")
            return
        return await func(update, context)
    return wrapper

# --- Команды бота ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    # Используем имя из настроек
    await update.message.reply_text(
        f"Привет, {user.first_name}! Я - {settings.BOT_NAME}, давай поболтаем?"
    )

async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Добавляет важное системное сообщение в историю (ChromaDB)."""
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat: return
    user_id = user.id

    history_key = chat.id if chat.type in ['group', 'supergroup'] else user.id

    if context.args:
        memory = " ".join(context.args).strip()
        if memory:
            user_info_db.setdefault(user_id, {"preferences": {}, "memory": None})
            user_info_db[user_id]['memory'] = memory
            await update.message.reply_text(f"Хорошо, я запомнила это о вас: '{memory}'.")
            logger.info(f"User {user.id} updated memory: '{memory[:50]}...'")
        else:
             await update.message.reply_text("Пожалуйста, укажите непустой текст для запоминания.")
    else:
        # Показываем текущую запомненную информацию, если она есть
        current_memory = user_info_db.get(user_id, {}).get('memory')
        if current_memory:
             await update.message.reply_text(f"Я помню о вас следующее: '{current_memory}'.\nЧтобы изменить, используйте `/remember новый текст`.")
        else:
             await update.message.reply_text("Пожалуйста, укажите, что нужно запомнить после команды /remember.")

async def clear_my_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Запрашивает подтверждение на очистку истории пользователя."""
    user_id = update.effective_user.id
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Да, очистить", callback_data=f'clear_vector_history_{user_id}'),
          InlineKeyboardButton("Нет, отмена", callback_data='cancel')]]
    )
    await update.message.reply_text("Вы уверены, что хотите полностью очистить свою историю чата (это необратимо)?", reply_markup=keyboard)

async def button_callback(update: Update, context: CallbackContext):
    """Обрабатывает нажатия на инлайн-кнопки."""
    query = update.callback_query
    if not query or not query.data: return
    await query.answer() # Отвечаем на колбэк, чтобы кнопка перестала "грузиться"
    data = query.data

    if data.startswith('clear_vector_history_'):
        user_id_str = data.split('_')[-1]
        if not user_id_str.isdigit():
             await query.edit_message_text("Ошибка: Неверный ID пользователя в данных кнопки.")
             logger.error(f"Invalid user ID in callback data: {data}")
             return
        user_id_to_clear = int(user_id_str)

        if user_id_to_clear == query.from_user.id:
            history_key = user_id_to_clear # Для ЛС ключ = user_id
            logger.info(f"User {query.from_user.id} confirmed clearing history for key {history_key}")
            # Вызываем асинхронное удаление из vector_store
            deleted = await delete_history(history_key)
            if deleted:
                 await query.edit_message_text("✅ Ваша история чата очищена.")
                 # Очищаем связанные данные
                 if history_key in user_topic:
                     del user_topic[history_key]
                     logger.info(f"Cleared topic for user {history_key}")
            else:
                 await query.edit_message_text("⚠️ Не удалось очистить историю. Попробуйте позже или обратитесь к администратору.")
        else:
            await query.edit_message_text("🚫 Вы не можете очистить историю другого пользователя.")
            logger.warning(f"User {query.from_user.id} tried to clear history for user {user_id_to_clear}")
    elif data == 'cancel':
        await query.edit_message_text("Действие отменено.")
    else:
         logger.warning(f"Received unknown callback data: {data}")
         await query.edit_message_text("Неизвестное действие.")


async def set_my_name_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает предпочитаемое имя пользователя."""
    user_id = update.effective_user.id
    if context.args:
        name = " ".join(context.args).strip()
        if 2 <= len(name) <= 50: # Добавим валидацию имени
            user_preferred_name[user_id] = name
            # Обновим user_info_db тоже, если он там есть
            if user_id in user_info_db:
                user_info_db[user_id]['preferred_name_set_by_user'] = name # Отметка, что задано пользователем
            await update.message.reply_text(f"Отлично, теперь буду обращаться к вам как '{name}'.")
            logger.info(f"User {user_id} set preferred name to '{name}'.")
        else:
            await update.message.reply_text("Имя должно быть от 2 до 50 символов.")
    else:
        current_name = user_preferred_name.get(user_id, update.effective_user.first_name)
        await update.message.reply_text(f"Сейчас я обращаюсь к вам как '{current_name}'.\nЧтобы изменить, введите команду и новое имя: `/setmyname Новое Имя`")


async def my_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает текущий стиль общения бота (не реализовано изменение)."""
    # TODO: Реализовать показ и изменение персонального стиля, если потребуется
    style_info = f"Мой текущий глобальный стиль общения:\n{settings.DEFAULT_STYLE}\n\n(Персональные стили пока не настраиваются)."
    await update.message.reply_text(style_info)

async def error_handler(update: object, context: CallbackContext):
    """Логирует ошибки и уведомляет администратора."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    # Попытка получить детали из update, если он есть
    update_details = "N/A"
    if isinstance(update, Update) and update.effective_message:
        update_details = f"Update ID: {update.update_id}, Chat ID: {update.effective_chat.id}, User ID: {update.effective_user.id}"
    elif isinstance(update, dict): # Иногда update приходит как словарь
         update_details = f"Update (dict): {str(update)[:200]}..."

    error_msg = f"Error: {type(context.error).__name__}: {context.error}\nUpdate details: {update_details}"

    # Уведомление администратора (если есть)
    if ADMIN_USER_IDS:
        try:
            # Отправляем только первому админу для простоты
            await context.bot.send_message(
                chat_id=ADMIN_USER_IDS[0],
                text=f"⚠️ Произошла ошибка в боте:\n\n{error_msg[:3000]}" # Ограничиваем длину сообщения
            )
            logger.info(f"Error notification sent to admin {ADMIN_USER_IDS[0]}.")
        except Exception as e:
            logger.error(f"Failed to send error notification to admin: {e}")

    # Можно добавить ответ пользователю, но осторожно, чтобы не спамить при повторяющихся ошибках
    # if isinstance(update, Update) and update.effective_message:
    #     try:
    #         await update.effective_message.reply_text("Ой! Что-то пошло не так. Администратор уведомлен.")
    #     except Exception:
    #         logger.error("Failed to send error message to user.")


# --- Административные команды ---

@admin_only
async def set_group_user_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает индивидуальный стиль для пользователя в группе (ответ на сообщение)."""
    if not update.message.reply_to_message:
        await update.message.reply_text("Пожалуйста, используйте эту команду как ответ на сообщение пользователя, для которого хотите установить стиль.")
        return
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажите стиль общения после команды.")
        return

    target_user = update.message.reply_to_message.from_user
    chat_id = update.effective_chat.id
    style_prompt = " ".join(context.args).strip()

    if not style_prompt:
         await update.message.reply_text("Стиль не может быть пустым.")
         return

    key = (chat_id, target_user.id)
    group_user_style_prompts[key] = style_prompt
    await update.message.reply_text(f"✅ Установлен индивидуальный стиль общения для пользователя {target_user.first_name} ({target_user.id}) в этом чате.")
    logger.info(f"Admin {update.effective_user.id} set group user style for user {target_user.id} in chat {chat_id}.")

@admin_only
async def reset_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сбрасывает глобальный стиль на значение по умолчанию из .env."""
    # Загружаем стиль из .env снова
    original_default_style = os.getenv('DEFAULT_STYLE', "Ты - дружелюбный помощник.")
    settings.update_default_style(original_default_style)
    await update.message.reply_text(f"Глобальный стиль общения бота сброшен на стандартный (из .env):\n{settings.DEFAULT_STYLE}")
    logger.info(f"Admin {update.effective_user.id} reset global style to default.")


@admin_only
async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очищает историю чата (ChromaDB коллекцию) для указанного user ID."""
    if context.args:
        user_id_str = context.args[0]
        if not user_id_str.isdigit():
            await update.message.reply_text("Пожалуйста, укажите корректный ID пользователя (число).")
            return

        user_id_to_clear = int(user_id_str)
        # В админской команде мы обычно чистим историю ЛС пользователя
        history_key = user_id_to_clear
        logger.info(f"Admin {update.effective_user.id} requested clearing history for key {history_key}")

        deleted = await delete_history(history_key)
        if deleted:
            await update.message.reply_text(f"✅ История чата для пользователя {user_id_to_clear} очищена.")
            if history_key in user_topic: del user_topic[history_key]
        else:
            await update.message.reply_text(f"⚠️ Не удалось очистить историю для {user_id_to_clear} (возможно, она уже пуста или произошла ошибка).")
    else:
        await update.message.reply_text("Пожалуйста, укажите ID пользователя (число) после команды `/clear_history`.")


@admin_only
async def list_admins_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список ID администраторов."""
    if ADMIN_USER_IDS:
        admin_list = ", ".join(map(str, ADMIN_USER_IDS))
        await update.message.reply_text(f"Список ID администраторов бота: {admin_list}")
    else:
        await update.message.reply_text("Список администраторов пуст (не задан в .env).")

@admin_only
async def get_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет файл логов администратору."""
    log_filename = "bot.log" # Имя файла из настроек логгера
    try:
        if os.path.exists(log_filename) and os.path.getsize(log_filename) > 0:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=InputFile(log_filename),
                caption=f"Файл логов {log_filename}"
            )
            logger.info(f"Log file sent to admin {update.effective_user.id}.")
        else:
            await update.message.reply_text(f"Файл логов '{log_filename}' не найден или пуст.")
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка при отправке логов: {e}")
        logger.error(f"Error sending log file: {e}", exc_info=True)

@admin_only
async def ban_user_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для команды бана."""
    # TODO: Реализовать логику бана (например, добавление ID в "черный список")
    await update.message.reply_text("Функция бана пока не реализована.")
    logger.warning("Ban command called but not implemented.")


@admin_only
async def set_default_style_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает новый глобальный стиль общения бота."""
    if context.args:
        new_style = " ".join(context.args).strip()
        if new_style:
            settings.update_default_style(new_style)
            await update.message.reply_text(f"✅ Глобальный стиль общения бота установлен на:\n{settings.DEFAULT_STYLE}")
            logger.info(f"Admin {update.effective_user.id} set new global style.")
        else:
            await update.message.reply_text("Стиль не может быть пустым.")
    else:
        await update.message.reply_text(f"Текущий стиль:\n{settings.DEFAULT_STYLE}\n\nЧтобы изменить, введите `/setdefaultstyle Новый стиль общения`")


@admin_only
async def set_bot_name_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает новое имя бота."""
    if context.args:
        new_name = " ".join(context.args).strip()
        if 1 <= len(new_name) <= 50:
            settings.update_bot_name(new_name)
            await update.message.reply_text(f"✅ Имя бота установлено на: {settings.BOT_NAME}")
            logger.info(f"Admin {update.effective_user.id} set bot name to '{settings.BOT_NAME}'.")
        else:
            await update.message.reply_text("Имя бота должно быть от 1 до 50 символов.")
    else:
        await update.message.reply_text(f"Текущее имя бота: {settings.BOT_NAME}\nЧтобы изменить, введите `/setbotname НовоеИмя`")

@admin_only
async def set_activity_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Устанавливает процент активности бота в группах."""
    global bot_activity_percentage # Убедимся, что изменяем глобальную переменную из state
    if context.args:
        try:
            percentage = int(context.args[0])
            if 0 <= percentage <= 100:
                # Обновляем переменную в state
                bot_activity_percentage = percentage
                # Можно также сохранить это значение в файл knowledge (state.py это сделает при выходе)
                await update.message.reply_text(f"✅ Активность бота в группах установлена на {percentage}%")
                logger.info(f"Bot activity set to {percentage}% by admin {update.effective_user.id}")
            else:
                await update.message.reply_text("⚠️ Процент активности должен быть в диапазоне от 0 до 100.")
        except ValueError:
            await update.message.reply_text("⚠️ Пожалуйста, введите числовое значение процента (0-100).")
    else:
        await update.message.reply_text(f"Текущая активность бота в группах: {bot_activity_percentage}%\nЧтобы изменить, введите `/setactivity <число от 0 до 100>`")


async def reset_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сбрасывает контекст разговора (удаляет историю из ChromaDB)."""
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat: return

    chat_type = chat.type
    history_key = chat.id if chat_type in ['group', 'supergroup'] else user.id
    logger.info(f"User {user.id} requested context reset for key {history_key}")

    deleted = await delete_history(history_key)
    if deleted:
        await update.message.reply_text("Контекст разговора сброшен (история удалена). Можем начать заново.")
        # Очищаем связанные данные для этого ключа
        if history_key in user_topic: del user_topic[history_key]
        # Очистка стилей для группы (опционально)
        if chat_type != 'private':
            keys_to_del = [k for k in group_user_style_prompts if k[0] == history_key]
            deleted_styles_count = 0
            for k in keys_to_del:
                try:
                    del group_user_style_prompts[k]
                    deleted_styles_count += 1
                except KeyError: pass
            if deleted_styles_count > 0:
                 logger.info(f"Cleared {deleted_styles_count} group user styles for chat_id {history_key} during context reset.")
    else:
        await update.message.reply_text("⚠️ Не удалось сбросить контекст (возможно, он уже пуст или произошла ошибка).")


# --- Команда помощи ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает список доступных команд."""
    user = update.effective_user
    is_admin = user.id in ADMIN_USER_IDS

    user_commands_text = """
*Основные команды:*
/start - Начать общение
/help - Показать это сообщение
/setmyname <имя> - Установить ваше имя для общения
/remember <текст> - Попросить меня запомнить важную информацию
/reset_context - Сбросить текущий контекст разговора (удалить историю)
/clear_my_history - Полностью очистить вашу историю чата (запросит подтверждение)
"""

    admin_commands_text = """
*Административные команды:*
/set_default_style <стиль> - Установить новый глобальный стиль общения
/reset_style - Сбросить глобальный стиль на стандартный (из .env)
/set_bot_name <имя> - Установить новое имя бота
/set_activity <0-100> - Установить процент активности бота в группах
/set_group_user_style [в ответ] <стиль> - Задать стиль для пользователя в группе
/clear_history <user_id> - Очистить историю ЛС указанного пользователя
/list_admins - Показать ID администраторов
/get_log - Получить файл логов
/ban [пока не работает] - Забанить пользователя
"""

    help_text = user_commands_text
    if is_admin:
        help_text += admin_commands_text

    await update.message.reply_text(help_text, parse_mode='Markdown')

# ==============================================================================
# Конец: Содержимое bot_commands.py
# ==============================================================================

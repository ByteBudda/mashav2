# -*- coding: utf-8 -*-
# handlers.py
import asyncio
import os
import random
from io import BytesIO
from collections import deque
import time
from typing import Tuple, Deque, Optional, Dict, Any, List
from PIL import Image
from telegram import Update, constants
from telegram.ext import ContextTypes
import pydub
from pydub import AudioSegment

# Используем settings и константы из config
from bot_commands import escape_markdown_v2
from config import (ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE,
                    logger, settings, TEMP_MEDIA_DIR)
# Импортируем функции работы с состоянием/БД
from state import (
    add_to_memory_history, chat_history, last_activity,
    get_user_preferred_name_from_db, get_user_topic_from_db,
    is_user_banned, save_message_and_embed, bot_activity_percentage,
    extract_and_save_facts # Импортируем функцию извлечения фактов
)
# Импортируем утилиты
from utils import (
    filter_response, generate_content_sync, generate_vision_content_async,
    transcribe_voice, update_user_info,
    _get_effective_style, should_process_message, PromptBuilder
)
# --- Импорт функций поиска из vector_db ---
# Используем синхронные версии для вызова через to_thread
from vector_db import search_relevant_history_sync, search_relevant_facts_sync

# Создаем PromptBuilder
prompt_builder = PromptBuilder(settings)

# --- Внутренняя функция _process_generation_and_reply ---
async def _process_generation_and_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE, history_key: int,
    prompt: str, user_message_text: str, # Оригинальный текст пользователя
):
    """
    Генерирует ответ, отправляет его, сохраняет сообщения пользователя и бота в БД/индекс.
    """
    chat_id = update.effective_chat.id
    user = update.effective_user
    if not user: return

    user_name = await asyncio.to_thread(get_user_preferred_name_from_db, user.id) or user.first_name or f"User_{user.id}"
    display_user_name = user_name if update.effective_chat.type != 'private' else None

    # --- Сохранение сообщения пользователя и эмбеддинг ---
    # Сохраняем оригинальный текст пользователя
    await save_message_and_embed(history_key, USER_ROLE, user_message_text, display_user_name)

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    response = await asyncio.to_thread(generate_content_sync, prompt) # Используем to_thread
    logger.info(f"Raw Gemini response for key {history_key}: {response[:100]}...")
    filtered = filter_response(response)
    logger.info(f"Filtered response for key {history_key}: {filtered[:100]}...")

    reply_to_message_id = update.message.message_id if update.message and update.effective_chat.type != 'private' else None

    if filtered and not filtered.startswith("["):
        add_to_memory_history(history_key, ASSISTANT_ROLE, filtered) # Добавляем в память
        # --- Сохранение ответа бота и эмбеддинг ---
        await save_message_and_embed(history_key, ASSISTANT_ROLE, filtered)
        logger.debug(f"Sending response to chat {chat_id}")
        try:
            if update.effective_chat.type == 'private':
                await context.bot.send_message(chat_id=chat_id, text=filtered)
            else:
                await update.message.reply_text(filtered)
        except Exception as e: logger.error(f"Failed to send message to {chat_id}: {e}")

    elif filtered.startswith("["): # Обработка ошибок Gemini
         logger.warning(f"Gemini issue for key {history_key}: {filtered}")
         reply_text = "Извините, не могу сейчас ответить на это."
         if "blocked" in filtered.lower(): reply_text = "Мой ответ был заблокирован системой безопасности."
         elif "error" in filtered.lower() or "ошибка" in filtered.lower(): reply_text = "Произошла ошибка при генерации ответа."
         try: await context.bot.send_message(chat_id=chat_id, text=reply_text, reply_to_message_id=reply_to_message_id)
         except Exception as e: logger.error(f"Failed to send error message to {chat_id}: {e}")

    else: # Пустой ответ
        logger.warning(f"Empty filtered response for key {history_key}. Original: {response[:100]}...")
        reply_text = "Простите, у меня возникли сложности с ответом."
        try: await context.bot.send_message(chat_id=chat_id, text=reply_text, reply_to_message_id=reply_to_message_id)
        except Exception as e: logger.error(f"Failed to send empty response message to {chat_id}: {e}")



    
# --- Объединенный обработчик для текста, голоса, видео ---
async def handle_text_voice_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user; chat = update.effective_chat
    if not user or not chat or not update.message: return
    user_id = user.id

    if await asyncio.to_thread(is_user_banned, user_id):
        logger.warning(f"Ignoring message from banned user {user_id} in chat {chat.id}")
        return

    chat_id = chat.id; chat_type = chat.type
    prompt_text: Optional[str] = None; message_type = "unknown"; temp_file_paths = []

    try: os.makedirs(TEMP_MEDIA_DIR, exist_ok=True)
    except OSError as e: logger.error(f"Failed to create '{TEMP_MEDIA_DIR}': {e}"); await update.message.reply_text("Ошибка папки медиа."); return
    # --- ПРОВЕРКА НА КОНКРЕТНЫЙ ВОПРОС ---
    if prompt_text:
        normalized_text = prompt_text.lower().strip().replace('?', '').replace('.', '').replace('!', '')
    else:
        normalized_text = ""
    creator_questions = [
        "кто тебя создал",
        "кто твой создатель",
        "кто тебя сделал",
        "кто твой разработчик",
        "кто тебя написал"
    ]

    if normalized_text in creator_questions:
        logger.info(f"User {user_id} asked about the creator. Replying directly.")
        # Экранируем имя пользователя Telegram для MarkdownV2
        creator_username = escape_markdown_v2("@ByteBudda")
        reply_text = f"Меня создал замечательный человек Александр {creator_username} 😊"
        await update.message.reply_text(reply_text, parse_mode='MarkdownV2')
        # Завершаем обработку здесь, не идем дальше к LLM
        return
    # ------------------------------------
    try: # Обработка типов сообщений
        # ... (Код обработки text, voice, video_note как в предыдущем ответе, с транскрипцией и т.д.) ...
        # Важно: prompt_text должен содержать распознанный текст БЕЗ меток (voice/video)
        if update.message.text: prompt_text = update.message.text; message_type = "text"
        elif update.message.voice:
             message_type = "voice"; voice = update.message.voice
             await context.bot.send_chat_action(chat_id, constants.ChatAction.RECORD_VOICE); vf = await voice.get_file()
             base = f"voice_{user_id}_{int(time.time())}"; p_oga = os.path.join(TEMP_MEDIA_DIR, f"{base}.oga"); p_wav = os.path.join(TEMP_MEDIA_DIR, f"{base}.wav")
             temp_file_paths.extend([p_oga, p_wav]); await vf.download_to_drive(p_oga); logger.debug(f"Voice downloaded: {p_oga}")
             try: AudioSegment.from_file(p_oga).export(p_wav, format="wav"); logger.debug(f"Converted to {p_wav}")
             except Exception as e: logger.error(f"Voice conversion error: {e}"); await update.message.reply_text("Ошибка конвертации аудио."); return
             prompt_text = await transcribe_voice(p_wav)
             if not prompt_text or prompt_text.startswith("["): logger.warning(f"Transcription failed: {prompt_text}"); await update.message.reply_text(f"Распознавание: {prompt_text or 'ошибка'}"); return
        elif update.message.video_note:
             message_type = "video_note"; vn = update.message.video_note
             await context.bot.send_chat_action(chat_id, constants.ChatAction.RECORD_VIDEO_NOTE); vnf = await vn.get_file()
             base = f"vnote_{user_id}_{int(time.time())}"; p_mp4 = os.path.join(TEMP_MEDIA_DIR, f"{base}.mp4"); p_wav = os.path.join(TEMP_MEDIA_DIR, f"{base}.wav")
             temp_file_paths.extend([p_mp4, p_wav]); await vnf.download_to_drive(p_mp4); logger.debug(f"Video note downloaded: {p_mp4}")
             try: AudioSegment.from_file(p_mp4).export(p_wav, format="wav"); logger.debug(f"Audio extracted to {p_wav}")
             except Exception as e: logger.error(f"Video note audio error: {e}"); await update.message.reply_text("Ошибка обработки видео."); return
             prompt_text = await transcribe_voice(p_wav)
             if not prompt_text or prompt_text.startswith("["): logger.warning(f"Transcription failed: {prompt_text}"); await update.message.reply_text(f"Распознавание видео: {prompt_text or 'ошибка'}"); return
    except Exception as e: logger.error(f"Error processing {message_type}: {e}"); await update.message.reply_text(f"Ошибка обработки {message_type}."); return
    finally: # Очистка временных файлов
        for fp in temp_file_paths:
            if os.path.exists(fp) and not fp.endswith(".wav"):
                try: os.remove(fp); logger.debug(f"Removed temp: {fp}")
                except OSError as e: logger.warning(f"Failed removing {fp}: {e}")

    if not prompt_text: logger.debug("No text content after processing."); return

    # --- Общая логика ---
    await update_user_info(update)
    user_name = await asyncio.to_thread(get_user_preferred_name_from_db, user_id) or user.first_name or f"User_{user_id}"
    history_key = chat_id if chat_type != 'private' else user_id
    display_user_name = user_name if chat_type != 'private' else None

    # --- Асинхронный поиск релевантной истории и фактов ---
    # Используем синхронные функции поиска через to_thread
    search_hist_task = asyncio.to_thread(search_relevant_history_sync, history_key, prompt_text)
    search_facts_task = asyncio.to_thread(search_relevant_facts_sync, history_key, prompt_text)
    relevant_history_docs = await search_hist_task
    relevant_facts_docs = await search_facts_task
    # ----------------------------------------------------

    # --- Добавление текущего сообщения в память ---
    current_msg_with_type = prompt_text # Текст без меток voice/video для памяти
    # Метки добавим в промпт, если нужно
    add_to_memory_history(history_key, USER_ROLE, current_msg_with_type, display_user_name)

    # --- Добавляем тип сообщения к тексту ДЛЯ ПРОМПТА ---
    prompt_input_text = prompt_text
    if message_type == "voice": prompt_input_text += " (голосовое сообщение)"
    elif message_type == "video_note": prompt_input_text += " (видеосообщение)"
    # ------------------------------------------------

    # --- Ответ ---
    if chat_type == 'private':
        logger.info(f"Processing {message_type} from {user_name} ({user_id}) in private.")
        style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
        sys_msg = f"{style} Ты - {settings.BOT_NAME}. Не начинай ответ с приветствия, если не было приветствия."
        topic = await asyncio.to_thread(get_user_topic_from_db, user_id); topic_ctx = f"Тема: {topic}." if topic else ""
        history_deque = chat_history.get(history_key, deque(maxlen=settings.MAX_HISTORY))

        prompt = prompt_builder.build_prompt(
            history_deque=history_deque, relevant_history=relevant_history_docs,
            relevant_facts=relevant_facts_docs, user_name=user_name,
            current_message_text=prompt_input_text, # Текст с меткой типа
            system_message_base=sys_msg, topic_context=topic_ctx
        )
        # Вызываем генерацию и СОХРАНЕНИЕ ОРИГИНАЛЬНОГО ТЕКСТА (prompt_text)
        await _process_generation_and_reply(update, context, history_key, prompt, prompt_text)

    else: # Группа
        should_reply = False
        # Проверяем активность один раз
        activity_check_passed = should_process_message()

        if activity_check_passed:
            # Если активность 100%, отвечаем всегда
            if bot_activity_percentage == 100:
                should_reply = True
                logger.info(f"Processing group {message_type} from {user_name} ({user_id}). Reason: Activity is 100%.")
            # Если активность < 100%, но прошла проверка, ищем упоминание/ответ
            else:
                try:
                    bot_info = await context.bot.get_me()
                    bot_id = bot_info.id
                    bot_uname = bot_info.username
                except Exception as e:
                    logger.error(f"Failed getting bot info: {e}")
                    bot_id = None
                    bot_uname = settings.BOT_NAME
                mentioned = (bot_uname and f"@{bot_uname}".lower() in prompt_input_text.lower()) or \
                            settings.BOT_NAME.lower() in prompt_input_text.lower()
                replied = update.message.reply_to_message and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == bot_id

                if mentioned or replied:
                    should_reply = True
                    logger.info(f"Processing group {message_type} from {user_name} ({user_id}). Reason: M={mentioned}, R={replied} (Activity {bot_activity_percentage}%)")
                else:
                    logger.info(f"Ignoring group {message_type} from {user_name} ({user_id}) (no mention/reply, Activity {bot_activity_percentage}%).")
        else: # Не прошли проверку активности (< 100%)
            logger.debug(f"Skipping group msg from {user_id} due to activity ({bot_activity_percentage}%).")


        if should_reply:
            style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
            # Сообщение для системы остается прежним, оно определяет стиль ответа, а не условия ответа
            sys_msg = f"{style} Ты - {settings.BOT_NAME}. Отвечаешь в группе. Обращайся к {user_name}. Не начинай ответ с приветствия."
            topic = await asyncio.to_thread(get_user_topic_from_db, user_id); topic_ctx = f"Тема {user_name}: {topic}." if topic else ""
            history_deque = chat_history.get(history_key, deque(maxlen=settings.MAX_HISTORY))

            prompt = prompt_builder.build_prompt(
                history_deque=history_deque, relevant_history=relevant_history_docs,
                relevant_facts=relevant_facts_docs, user_name=user_name,
                current_message_text=prompt_input_text, # Текст с меткой типа
                system_message_base=sys_msg, topic_context=topic_ctx
            )
            await _process_generation_and_reply(update, context, history_key, prompt, prompt_text) # Оригинальный текст
        else:
            # Сохраняем сообщение пользователя, даже если не отвечаем
            # Сохраняем текст С МЕТКОЙ ТИПА, т.к. он уже добавлен в deque
            await save_message_and_embed(history_key, USER_ROLE, current_msg_with_type, display_user_name)

    # --- Запуск извлечения фактов (опционально, можно вынести в Job) ---
    # asyncio.create_task(extract_and_save_facts(history_key))


# --- Обработчик фото ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user; chat = update.effective_chat
    if not user or not chat or not update.message or not update.message.photo: return
    user_id = user.id; chat_id = chat.id; chat_type = chat.type

    if await asyncio.to_thread(is_user_banned, user_id): logger.warning(f"Ignoring photo from banned user {user_id}"); return

    caption = update.message.caption or ""
    history_key = chat_id if chat_type != 'private' else user_id
    await update_user_info(update)
    user_name = await asyncio.to_thread(get_user_preferred_name_from_db, user_id) or user.first_name or f"User_{user_id}"
    display_user_name_photo = user_name if chat_type != 'private' else None
    history_message = "Получено фото" + (f" с подписью: '{caption}'" if caption else "") # Текст для сохранения/поиска

    if chat_type != 'private' and not should_process_message():
        logger.debug(f"Photo from {user_id} skipped (activity {bot_activity_percentage}%). Saving info.")
        await save_message_and_embed(history_key, USER_ROLE, history_message, display_user_name_photo)
        return

    logger.info(f"Processing photo from {user_name}. Caption: '{caption[:50]}...'")
    await context.bot.send_chat_action(chat_id, constants.ChatAction.TYPING)

    try:
        photo_file = await update.message.photo[-1].get_file(); file_bytes = await photo_file.download_as_bytearray()
        image = Image.open(BytesIO(file_bytes)); image = image.convert('RGB')

        add_to_memory_history(history_key, USER_ROLE, history_message, display_user_name_photo)
        await save_message_and_embed(history_key, USER_ROLE, history_message, display_user_name_photo)

        search_query_photo = caption if caption else "фото"
        # Используем синхронные версии поиска через to_thread
        relevant_history_docs_photo = await asyncio.to_thread(search_relevant_history_sync, history_key, search_query_photo)
        relevant_facts_docs_photo = await asyncio.to_thread(search_relevant_facts_sync, history_key, search_query_photo)

        style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
        sys_msg = f"{style} Ты - {settings.BOT_NAME}. Комментируй изображение."
        topic = await asyncio.to_thread(get_user_topic_from_db, user_id); topic_ctx = f"Тема: {topic}." if topic else ""
        history_deque = chat_history.get(history_key, deque(maxlen=settings.MAX_HISTORY))

        vision_prompt_text = prompt_builder.build_prompt(
            history_deque=history_deque, relevant_history=relevant_history_docs_photo,
            relevant_facts=relevant_facts_docs_photo, user_name=user_name,
            current_message_text=history_message, system_message_base=sys_msg,
            topic_context=topic_ctx
        )
        vision_prompt = vision_prompt_text.rsplit('\n', 1)[0] + "\nПрокомментируй приложенное изображение, учитывая контекст."

        contents = [vision_prompt, image]
        response_text = await generate_vision_content_async(contents)
        filtered = filter_response(response_text)

        reply_to_message_id = update.message.message_id if chat_type != 'private' else None

        if filtered and not filtered.startswith("["):
            add_to_memory_history(history_key, ASSISTANT_ROLE, filtered)
            await save_message_and_embed(history_key, ASSISTANT_ROLE, filtered)
            try: await context.bot.send_message(chat_id, filtered, reply_to_message_id=reply_to_message_id)
            except Exception as e: logger.error(f"Failed sending vision response to {chat_id}: {e}")
        # ... (обработка ошибок Vision и пустых ответов как раньше) ...
        elif filtered.startswith("["):
             reply_text = "Не удалось обработать изображение.";
             if "blocked" in filtered.lower(): reply_text = "Ответ на изображение заблокирован."
             try: await context.bot.send_message(chat_id, reply_text, reply_to_message_id=reply_to_message_id)
             except Exception as e: logger.error(f"Failed sending vision error to {chat_id}: {e}")
        else:
             reply_text = "Не могу ничего сказать об этом изображении."
             try: await context.bot.send_message(chat_id, reply_text, reply_to_message_id=reply_to_message_id)
             except Exception as e: logger.error(f"Failed sending vision empty response to {chat_id}: {e}")

    except Exception as e:
        logger.error(f"Error handling photo for {user_id} in {chat_id}: {e}", exc_info=True)
        try: await update.message.reply_text("Ошибка при обработке фото.")
        except Exception as send_e: logger.error(f"Failed sending photo error msg to {chat_id}: {send_e}")


# --- Переназначение обработчиков ---
handle_message = handle_text_voice_video
handle_voice_message = handle_text_voice_video
handle_video_note_message = handle_text_voice_video
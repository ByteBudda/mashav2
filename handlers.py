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

# --- Импорты для обработки документов ---
from documents_handler import (
    read_pdf, read_docx, read_txt, read_py, generate_document
)

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
    filter_response, # generate_content_sync - убрали, т.к. есть обертки
    generate_vision_content_async,
    transcribe_voice, update_user_info,
    _get_effective_style, should_process_message, PromptBuilder,
    generate_with_cache # ИСПОЛЬЗУЕМ ОБЕРТКУ ДЛЯ КЭША И МЕТРИК
)
# --- Импорт функций поиска из vector_db ---
# Используем синхронные версии для вызова через to_thread
from vector_db import search_relevant_history_sync, search_relevant_facts_sync

# Создаем PromptBuilder
prompt_builder = PromptBuilder(settings)

# --- Константа для лимита текста из документа ---
MAX_DOCUMENT_TEXT_LENGTH = 5000 # Ограничение символов для LLM

# --- Внутренняя функция _process_generation_and_reply ---
async def _process_generation_and_reply(
    update: Update, context: ContextTypes.DEFAULT_TYPE, history_key: int,
    prompt: str, user_message_text: str, # Оригинальный текст пользователя
    reply_to_message_id_override: Optional[int] = None # Для ответов на сообщения
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
    # Сохраняем оригинальный текст пользователя (или документа)
    await save_message_and_embed(history_key, USER_ROLE, user_message_text, display_user_name)

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    response = ""
    filtered = ""
    try:
        # Используем обертку, которая включает метрики и кэш
        response = await generate_with_cache(prompt)
        logger.info(f"Raw LLM response for key {history_key}: {response[:100]}...")
        filtered = filter_response(response)
        logger.info(f"Filtered response for key {history_key}: {filtered[:100]}...")
    except Exception as e:
        logger.error(f"Generation error in _process_generation_and_reply: {e}", exc_info=True)
        filtered = "[Произошла ошибка при генерации ответа]" # Устанавливаем текст ошибки

    # Определяем на какое сообщение отвечать
    reply_to_id = reply_to_message_id_override # Если передан ID (например, для /ask)
    if reply_to_id is None and update.message: # Если ID не передан, используем ID сообщения пользователя (если есть)
        reply_to_id = update.message.message_id if update.effective_chat.type != 'private' else None

    if filtered and not filtered.startswith("["):
        add_to_memory_history(history_key, ASSISTANT_ROLE, filtered) # Добавляем в память
        # --- Сохранение ответа бота и эмбеддинг ---
        await save_message_and_embed(history_key, ASSISTANT_ROLE, filtered)
        logger.debug(f"Sending response to chat {chat_id}")
        try:
             await context.bot.send_message(chat_id=chat_id, text=filtered, reply_to_message_id=reply_to_id)
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")

    elif filtered.startswith("["): # Обработка ошибок генерации, переданных в строке
         logger.warning(f"LLM issue for key {history_key}: {filtered}")
         reply_text = "Извините, не могу сейчас ответить на это."
         if "blocked" in filtered.lower(): reply_text = "Мой ответ был заблокирован системой безопасности."
         elif "error" in filtered.lower() or "ошибка" in filtered.lower(): reply_text = "Произошла ошибка при генерации ответа."
         elif filtered == "[Произошла ошибка при генерации ответа]": reply_text = filtered # Используем текст из except блока
         try: await context.bot.send_message(chat_id=chat_id, text=reply_text, reply_to_message_id=reply_to_id)
         except Exception as e: logger.error(f"Failed to send error message to {chat_id}: {e}")

    else: # Пустой ответ
        logger.warning(f"Empty filtered response for key {history_key}. Original Raw: {response[:100]}...")
        reply_text = "Простите, у меня возникли сложности с ответом."
        try: await context.bot.send_message(chat_id=chat_id, text=reply_text, reply_to_message_id=reply_to_id)
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
    prompt_text: Optional[str] = None; message_type = "unknown"; temp_file_path = None # Изменено на один путь

    try: # Создаем папку, если нет
        os.makedirs(TEMP_MEDIA_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create '{TEMP_MEDIA_DIR}': {e}")
        if update.message: await update.message.reply_text("Ошибка: Не удалось создать папку для временных файлов.")
        return

    try: # Обработка типов сообщений
        if update.message.text:
            prompt_text = update.message.text
            message_type = "text"
            # --- ПРОВЕРКА НА КОНКРЕТНЫЙ ВОПРОС ---
            normalized_text = prompt_text.lower().strip().replace('?', '').replace('.', '').replace('!', '')
            creator_questions = [
                "кто тебя создал", "кто твой создатель", "кто тебя сделал",
                "кто твой разработчик", "кто тебя написал"
            ]
            if normalized_text in creator_questions:
                logger.info(f"User {user_id} asked about the creator. Replying directly.")
                creator_username = escape_markdown_v2("@ByteBudda")
                reply_text = f"Меня создал замечательный человек Александр {creator_username} 😊"
                await update.message.reply_text(reply_text, parse_mode='MarkdownV2')
                return # Завершаем обработку здесь
            # --- КОНЕЦ ПРОВЕРКИ ---

        elif update.message.voice:
             message_type = "voice"; voice = update.message.voice
             await context.bot.send_chat_action(chat_id, constants.ChatAction.RECORD_VOICE)
             vf = await voice.get_file()
             # Используем один временный файл для WAV
             base = f"voice_{user_id}_{int(time.time())}"; p_oga = os.path.join(TEMP_MEDIA_DIR, f"{base}.oga"); temp_file_path = os.path.join(TEMP_MEDIA_DIR, f"{base}.wav")
             await vf.download_to_drive(p_oga); logger.debug(f"Voice downloaded: {p_oga}")
             try:
                 AudioSegment.from_file(p_oga).export(temp_file_path, format="wav"); logger.debug(f"Converted to {temp_file_path}")
                 os.remove(p_oga) # Удаляем OGA после конвертации
             except Exception as e:
                 logger.error(f"Voice conversion error: {e}", exc_info=True)
                 if os.path.exists(p_oga): os.remove(p_oga) # Удаляем OGA при ошибке
                 await update.message.reply_text("Ошибка конвертации аудио.")
                 return
             prompt_text = await transcribe_voice(temp_file_path) # temp_file_path будет удален внутри transcribe_voice
             if not prompt_text or prompt_text.startswith("["):
                 logger.warning(f"Transcription failed: {prompt_text}")
                 await update.message.reply_text(f"Распознавание: {prompt_text or 'ошибка'}")
                 return
             temp_file_path = None # Сбрасываем, так как файл удален внутри transcribe_voice

        elif update.message.video_note:
             message_type = "video_note"; vn = update.message.video_note
             # ChatAction RECORD_VIDEO используется для видео-кружочков
             await context.bot.send_chat_action(chat_id, constants.ChatAction.RECORD_VIDEO)
             vnf = await vn.get_file()
             # Используем один временный файл для WAV
             base = f"vnote_{user_id}_{int(time.time())}"; p_mp4 = os.path.join(TEMP_MEDIA_DIR, f"{base}.mp4"); temp_file_path = os.path.join(TEMP_MEDIA_DIR, f"{base}.wav")
             await vnf.download_to_drive(p_mp4); logger.debug(f"Video note downloaded: {p_mp4}")
             try:
                 AudioSegment.from_file(p_mp4).export(temp_file_path, format="wav"); logger.debug(f"Audio extracted to {temp_file_path}")
                 os.remove(p_mp4) # Удаляем MP4 после извлечения
             except Exception as e:
                 logger.error(f"Video note audio extraction error: {e}", exc_info=True)
                 if os.path.exists(p_mp4): os.remove(p_mp4) # Удаляем MP4 при ошибке
                 await update.message.reply_text("Ошибка обработки видеосообщения.")
                 return
             prompt_text = await transcribe_voice(temp_file_path) # temp_file_path будет удален внутри transcribe_voice
             if not prompt_text or prompt_text.startswith("["):
                 logger.warning(f"Transcription failed: {prompt_text}")
                 await update.message.reply_text(f"Распознавание видео: {prompt_text or 'ошибка'}")
                 return
             temp_file_path = None # Сбрасываем, так как файл удален внутри transcribe_voice

    except Exception as e:
        logger.error(f"Error processing {message_type}: {e}", exc_info=True)
        if update.message: await update.message.reply_text(f"Ошибка обработки сообщения типа '{message_type}'.")
        # Очищаем временный файл, если он остался после ошибки
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); logger.debug(f"Removed temp file on error: {temp_file_path}")
            except OSError: pass
        return

    # Если после всех обработок текста нет, выходим
    if not prompt_text:
        logger.debug("No text content after processing message.")
        return

    # --- Общая логика для всех типов сообщений (текст, голос, видео) ---
    await update_user_info(update) # Обновляем инфо пользователя
    user_name = await asyncio.to_thread(get_user_preferred_name_from_db, user_id) or user.first_name or f"User_{user_id}"
    history_key = chat_id if chat_type != 'private' else user_id
    display_user_name = user_name if chat_type != 'private' else None

    # --- Асинхронный поиск релевантной истории и фактов ---
    # Используем оригинальный prompt_text (без доп. меток типа сообщения) для поиска
    search_hist_task = asyncio.to_thread(search_relevant_history_sync, history_key, prompt_text)
    search_facts_task = asyncio.to_thread(search_relevant_facts_sync, history_key, prompt_text)
    relevant_history_docs = await search_hist_task
    relevant_facts_docs = await search_facts_task
    # ----------------------------------------------------

    # --- Добавление текущего сообщения в память ---
    # Используем оригинальный prompt_text для истории памяти
    add_to_memory_history(history_key, USER_ROLE, prompt_text, display_user_name)

    # --- Добавляем тип сообщения к тексту ДЛЯ ФОРМИРОВАНИЯ ПРОМПТА LLM ---
    prompt_input_text_for_llm = prompt_text
    if message_type == "voice": prompt_input_text_for_llm += " (голосовое сообщение)"
    elif message_type == "video_note": prompt_input_text_for_llm += " (видеосообщение)"
    # -----------------------------------------------------------------

    # --- Логика ответа ---
    if chat_type == 'private':
        logger.info(f"Processing {message_type} from {user_name} ({user_id}) in private.")
        style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
        sys_msg = f"{style} Ты - {settings.BOT_NAME}. Не начинай ответ с приветствия, если не было приветствия."
        topic = await asyncio.to_thread(get_user_topic_from_db, user_id); topic_ctx = f"Тема: {topic}." if topic else ""
        history_deque = chat_history.get(history_key, deque(maxlen=settings.MAX_HISTORY))

        prompt_llm = prompt_builder.build_prompt(
            history_deque=history_deque, relevant_history=relevant_history_docs,
            relevant_facts=relevant_facts_docs, user_name=user_name,
            current_message_text=prompt_input_text_for_llm, # Текст с меткой типа для LLM
            system_message_base=sys_msg, topic_context=topic_ctx
        )
        # Запускаем генерацию и ответ как фоновую задачу
        # Передаем ОРИГИНАЛЬНЫЙ prompt_text для сохранения в БД
        asyncio.create_task(_process_generation_and_reply(update, context, history_key, prompt_llm, prompt_text))

    else: # Группа
        should_reply = False
        activity_check_passed = should_process_message()

        if activity_check_passed:
            if bot_activity_percentage == 100: # Всегда отвечаем при 100%
                should_reply = True
                logger.info(f"Processing group {message_type} from {user_name} ({user_id}). Reason: Activity 100%.")
            else: # Проверяем упоминание/ответ при < 100%
                try:
                    bot_info = await context.bot.get_me(); bot_id = bot_info.id; bot_uname = bot_info.username
                except Exception as e:
                    logger.error(f"Failed getting bot info: {e}"); bot_id = None; bot_uname = settings.BOT_NAME

                mentioned = (bot_uname and f"@{bot_uname}".lower() in prompt_input_text_for_llm.lower()) or \
                            settings.BOT_NAME.lower() in prompt_input_text_for_llm.lower()
                replied = update.message.reply_to_message and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == bot_id

                if mentioned or replied:
                    should_reply = True
                    logger.info(f"Processing group {message_type} from {user_name} ({user_id}). Reason: Mention={mentioned}, Reply={replied} (Activity {bot_activity_percentage}%)")
                else:
                    logger.info(f"Ignoring group {message_type} from {user_name} ({user_id}) (no mention/reply, Activity {bot_activity_percentage}% pass).")
        else: # Не прошли проверку активности (< 100%)
            logger.debug(f"Skipping group msg from {user_id} due to activity check fail ({bot_activity_percentage}%).")

        if should_reply:
            style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
            sys_msg = f"{style} Ты - {settings.BOT_NAME}. Отвечаешь в группе. Обращайся к {user_name}. Не начинай ответ с приветствия."
            topic = await asyncio.to_thread(get_user_topic_from_db, user_id); topic_ctx = f"Тема {user_name}: {topic}." if topic else ""
            history_deque = chat_history.get(history_key, deque(maxlen=settings.MAX_HISTORY))

            prompt_llm = prompt_builder.build_prompt(
                history_deque=history_deque, relevant_history=relevant_history_docs,
                relevant_facts=relevant_facts_docs, user_name=user_name,
                current_message_text=prompt_input_text_for_llm, # Текст с меткой типа для LLM
                system_message_base=sys_msg, topic_context=topic_ctx
            )
            # Запускаем генерацию и ответ как фоновую задачу
            # Передаем ОРИГИНАЛЬНЫЙ prompt_text для сохранения в БД
            asyncio.create_task(_process_generation_and_reply(update, context, history_key, prompt_llm, prompt_text))
        else:
            # Сохраняем сообщение пользователя, даже если не отвечаем
            # Сохраняем оригинальный prompt_text
            await save_message_and_embed(history_key, USER_ROLE, prompt_text, display_user_name)

    # --- Запуск извлечения фактов (можно вынести в Job для производительности) ---
    # asyncio.create_task(extract_and_save_facts(history_key))

# --- Обработчик документов ---
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user; chat = update.effective_chat
    if not user or not chat or not update.message or not update.message.document:
        logger.debug("Received document update without user/chat/message/document.")
        return
    user_id = user.id

    if await asyncio.to_thread(is_user_banned, user_id):
        logger.warning(f"Ignoring document from banned user {user_id} in chat {chat.id}")
        return

    doc = update.message.document
    file_name = doc.file_name or "unknown_document"
    mime_type = doc.mime_type or ""
    file_id = doc.file_id
    chat_id = chat.id
    chat_type = chat.type
    temp_file_path = None # Путь к скачанному файлу

    logger.info(f"Received document '{file_name}' (type: {mime_type}) from user {user_id} in chat {chat_id}")

    # Определяем функцию чтения по расширению или MIME типу
    reader_func = None
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext == '.pdf' or mime_type == 'application/pdf':
        reader_func = read_pdf
    elif file_ext == '.docx' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        reader_func = read_docx
    elif file_ext == '.txt' or mime_type.startswith('text/plain'):
        reader_func = read_txt
    elif file_ext == '.py' or mime_type in ['text/x-python', 'application/x-python-code']:
        reader_func = read_py
    else:
        logger.warning(f"Unsupported document type: {file_name} ({mime_type})")
        await update.message.reply_text(f"Извините, я не могу обработать файлы типа '{file_ext}' ({mime_type}). Поддерживаются PDF, DOCX, TXT, PY.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
    processing_msg = await update.message.reply_text(f"⏳ Получен документ '{escape_markdown_v2(file_name)}'\\. Начинаю обработку\\.\\.\\.", parse_mode='MarkdownV2')

    try:
        os.makedirs(TEMP_MEDIA_DIR, exist_ok=True)
        # Генерируем уникальное имя файла во временной папке
        temp_file_name = f"doc_{user_id}_{file_id}{file_ext}"
        temp_file_path = os.path.join(TEMP_MEDIA_DIR, temp_file_name)

        logger.debug(f"Downloading document {file_id} to {temp_file_path}...")
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(temp_file_path)
        logger.debug(f"Document {file_id} downloaded successfully.")

        logger.debug(f"Reading text from {temp_file_path} using {reader_func.__name__}...")
        prompt_text = await asyncio.to_thread(reader_func, temp_file_path)
        logger.debug(f"Text reading complete for {file_id}.")

    except Exception as e:
        logger.error(f"Error downloading or reading document {file_id} ('{file_name}'): {e}", exc_info=True)
        await processing_msg.edit_text(f"❌ Ошибка при загрузке или чтении документа '{escape_markdown_v2(file_name)}'\\.", parse_mode='MarkdownV2')
        # Удаляем временный файл, если он был создан и произошла ошибка
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except OSError: pass
        return
    finally:
        # Гарантированно удаляем временный файл после чтения (или ошибки)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary document file: {temp_file_path}")
            except OSError as rm_err:
                logger.warning(f"Failed to remove temporary document file {temp_file_path}: {rm_err}")

    if prompt_text is None:
        logger.warning(f"Failed to extract text from document {file_id} ('{file_name}'). Reader returned None.")
        await processing_msg.edit_text(f"⚠️ Не удалось извлечь текст из документа '{escape_markdown_v2(file_name)}'\\.", parse_mode='MarkdownV2')
        return
    if not prompt_text.strip():
        logger.info(f"Document {file_id} ('{file_name}') contained no readable text.")
        await processing_msg.edit_text(f"ℹ️ Документ '{escape_markdown_v2(file_name)}' не содержит текста или текст не удалось прочитать\\.", parse_mode='MarkdownV2')
        return

    # --- Текст успешно извлечен, продолжаем обработку ---
    logger.info(f"Successfully extracted text from '{file_name}' (length: {len(prompt_text)}).")

    # Обрезаем текст, если он слишком длинный для LLM
    original_length = len(prompt_text)
    if original_length > MAX_DOCUMENT_TEXT_LENGTH:
        prompt_text_for_llm = prompt_text[:MAX_DOCUMENT_TEXT_LENGTH]
        logger.warning(f"Document text from '{file_name}' truncated from {original_length} to {MAX_DOCUMENT_TEXT_LENGTH} chars for LLM.")
        await context.bot.send_message(chat_id, f"⚠️ Текст из документа '{escape_markdown_v2(file_name)}' слишком длинный \\({original_length} символов\\)\\. Будет использована только первая часть \\({MAX_DOCUMENT_TEXT_LENGTH} символов\\)\\.", parse_mode='MarkdownV2')
    else:
        prompt_text_for_llm = prompt_text # Используем весь текст для LLM

    # Обновляем сообщение о статусе
    try:
        await processing_msg.edit_text(f"✅ Текст из '{escape_markdown_v2(file_name)}' успешно обработан\\. Отправляю запрос LLM\\.\\.\\.", parse_mode='MarkdownV2')
    except Exception: pass # Игнорируем, если не удалось обновить

    # --- Общая логика (аналогично handle_text_voice_video) ---
    await update_user_info(update)
    user_name = await asyncio.to_thread(get_user_preferred_name_from_db, user_id) or user.first_name or f"User_{user_id}"
    history_key = chat_id if chat_type != 'private' else user_id
    display_user_name = user_name if chat_type != 'private' else None

    # Добавляем в историю памяти ПОЛНЫЙ текст документа
    history_doc_text = f"(Текст из документа: {file_name})\n{prompt_text}"
    add_to_memory_history(history_key, USER_ROLE, history_doc_text, display_user_name)

    # Поиск по векторам делаем по НАЧАЛУ текста документа (или по урезанному тексту)
    search_hist_task = asyncio.to_thread(search_relevant_history_sync, history_key, prompt_text_for_llm)
    search_facts_task = asyncio.to_thread(search_relevant_facts_sync, history_key, prompt_text_for_llm)
    relevant_history_docs = await search_hist_task
    relevant_facts_docs = await search_facts_task

    # Формируем промпт для LLM
    style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
    sys_msg = f"{style} Ты - {settings.BOT_NAME}. Тебе предоставлен текст из документа '{escape_markdown_v2(file_name)}'. Отвечай на основе этого текста."
    topic = await asyncio.to_thread(get_user_topic_from_db, user_id); topic_ctx = f"Тема: {topic}." if topic else ""
    history_deque = chat_history.get(history_key, deque(maxlen=settings.MAX_HISTORY))

    # Для LLM используем урезанный текст с пометкой
    llm_input_text = f"(Текст из документа: {file_name})\n{prompt_text_for_llm}"

    prompt_llm = prompt_builder.build_prompt(
        history_deque=history_deque, relevant_history=relevant_history_docs,
        relevant_facts=relevant_facts_docs, user_name=user_name,
        current_message_text=llm_input_text, # Урезанный текст с меткой для LLM
        system_message_base=sys_msg, topic_context=topic_ctx
    )

    # Запускаем генерацию и ответ как фоновую задачу
    # Передаем ПОЛНЫЙ оригинальный текст документа для сохранения в БД
    # Передаем ID сообщения о статусе, чтобы ответить на него
    asyncio.create_task(_process_generation_and_reply(
        update, context, history_key, prompt_llm,
        user_message_text=f"(Текст из документа: {file_name})\n{prompt_text}", # Полный текст для БД
        reply_to_message_id_override=processing_msg.message_id # Отвечаем на сообщение "Обрабатываю..."
        ))

    # --- Запуск извлечения фактов (если нужно для документов) ---
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

    # Проверка активности для групп
    if chat_type != 'private' and not should_process_message():
        logger.debug(f"Photo from {user_id} skipped (activity {bot_activity_percentage}%). Saving info.")
        # Сохраняем инфо о фото, даже если не отвечаем
        await save_message_and_embed(history_key, USER_ROLE, history_message, display_user_name_photo)
        return

    logger.info(f"Processing photo from {user_name} in chat {chat_id}. Caption: '{caption[:50]}...'")
    await context.bot.send_chat_action(chat_id, constants.ChatAction.TYPING)

    # --- Запускаем обработку фото как фоновую задачу ---
    asyncio.create_task(_process_photo_reply(update, context))


async def _process_photo_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Внутренняя функция для асинхронной обработки фото."""
    # Переносим получение user, chat и т.д. сюда, т.к. update может устареть
    user = update.effective_user; chat = update.effective_chat
    if not user or not chat or not update.message or not update.message.photo: return
    user_id = user.id; chat_id = chat.id; chat_type = chat.type
    history_key = chat_id if chat_type != 'private' else user_id
    caption = update.message.caption or ""
    user_name = await asyncio.to_thread(get_user_preferred_name_from_db, user_id) or user.first_name or f"User_{user_id}"
    display_user_name_photo = user_name if chat_type != 'private' else None
    history_message = "Получено фото" + (f" с подписью: '{caption}'" if caption else "")
    reply_to_message_id = update.message.message_id if chat_type != 'private' else None

    try:
        photo_file = await update.message.photo[-1].get_file(); file_bytes = await photo_file.download_as_bytearray()
        if not file_bytes: raise ValueError("Downloaded photo bytes are empty.")
        image = Image.open(BytesIO(file_bytes)); image = image.convert('RGB')

        # Сохраняем информацию о фото в историю СРАЗУ
        add_to_memory_history(history_key, USER_ROLE, history_message, display_user_name_photo)
        await save_message_and_embed(history_key, USER_ROLE, history_message, display_user_name_photo)

        # Остальная обработка (поиск, генерация, отправка) идет внутри этой функции
        search_query_photo = caption if caption else "фото"
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
        # Формируем финальный промпт для Vision модели
        vision_prompt_lines = vision_prompt_text.splitlines()
        if len(vision_prompt_lines) > 1:
             # Убираем последнюю строку с приглашением ":", если она есть
             if vision_prompt_lines[-1].strip().endswith(":"):
                 vision_prompt_core = "\n".join(vision_prompt_lines[:-1])
             else:
                 vision_prompt_core = "\n".join(vision_prompt_lines)
        else:
             vision_prompt_core = vision_prompt_text

        # Добавляем конкретную инструкцию для Vision
        vision_instruction = "\nПрокомментируй приложенное изображение, учитывая предоставленный контекст беседы."
        final_vision_prompt = vision_prompt_core + vision_instruction


        contents = [final_vision_prompt, image]
        # Используем генерацию без кэша/метрик для Vision, т.к. входные данные другие
        response_text = await generate_vision_content_async(contents)
        filtered = filter_response(response_text)

        if filtered and not filtered.startswith("["):
            add_to_memory_history(history_key, ASSISTANT_ROLE, filtered)
            await save_message_and_embed(history_key, ASSISTANT_ROLE, filtered)
            await context.bot.send_message(chat_id, filtered, reply_to_message_id=reply_to_message_id)
        elif filtered.startswith("["):
            reply_text = "Не удалось обработать изображение.";
            if "blocked" in filtered.lower(): reply_text = "Ответ на изображение заблокирован."
            await context.bot.send_message(chat_id, reply_text, reply_to_message_id=reply_to_message_id)
        else:
            reply_text = "Не могу ничего сказать об этом изображении."
            await context.bot.send_message(chat_id, reply_text, reply_to_message_id=reply_to_message_id)
    except Exception as inner_e:
         logger.error(f"Error in _process_photo_reply for {user_id} in {chat_id}: {inner_e}", exc_info=True)
         # Попытка отправить сообщение об ошибке пользователю
         try: await context.bot.send_message(chat_id, "Произошла внутренняя ошибка при обработке фото.", reply_to_message_id=reply_to_message_id)
         except Exception: pass # Игнорируем ошибки отправки


# --- Переназначение обработчиков ---
# Оставляем универсальный для текста/голоса/видео
handle_message = handle_text_voice_video
# handle_voice_message = handle_text_voice_video # Можно удалить, т.к. handle_message ловит filter.VOICE
# handle_video_note_message = handle_text_voice_video # Можно удалить, т.к. handle_message ловит filter.VIDEO_NOTE
# Новый обработчик документов будет зарегистрирован отдельно в main.py
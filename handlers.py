# handlers.py
import asyncio
import os
import random
from io import BytesIO
import time
from PIL import Image
from telegram import Update, constants
from telegram.ext import ContextTypes, CallbackContext # Добавили CallbackContext для type hints, если где-то используется
import pydub # Не используется напрямую, но нужен для AudioSegment
from pydub import AudioSegment

# Импорты из проекта
from config import (ASSISTANT_ROLE, BOT_NAME, CONTEXT_CHECK_PROMPT,
                    DEFAULT_STYLE, SYSTEM_ROLE, USER_ROLE,
                    logger, settings) # Убрали MAX_HISTORY
from state import (add_to_history, learned_responses, # Убрали chat_history
                   user_preferred_name, user_topic, user_info_db) # Добавили user_info_db
from utils import (filter_response, generate_content_sync, generate_vision_content_async,
                   is_context_related, transcribe_voice, update_user_info,
                   _get_effective_style, should_process_message,
                   get_bot_activity_percentage, get_ner_pipeline,
                   get_sentiment_pipeline, PromptBuilder, prompt_builder) # Импортируем готовый prompt_builder из utils
# --- НОВЫЙ ИМПОРТ ---
from vector_store import query_relevant_history

# --- Инициализация PromptBuilder (теперь происходит в utils.py) ---
# prompt_builder = PromptBuilder(settings.BOT_NAME) # <<< УБРАТЬ ЭТУ СТРОКУ, импортируем готовый


# --- Вспомогательная функция для обработки генерации и ответа ---
async def _process_generation_and_reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    history_key: int,
    prompt: str,
    original_input: str # Текст оригинального сообщения пользователя
):
    """Генерирует ответ AI, фильтрует, сохраняет и отправляет пользователю."""
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type

    # Отправляем действие "печатает..."
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        # Уменьшим имитацию задержки, т.к. сам AI может думать долго
        await asyncio.sleep(random.uniform(0.2, 0.8))
    except Exception as e:
        logger.warning(f"Failed to send typing action to chat {chat_id}: {e}")

    # Используем generate_content_sync из utils.py (запускается в потоке)
    response = await asyncio.to_thread(generate_content_sync, prompt)
    # logger.info(f"Raw Gemini response for key {history_key}: {response[:100]}...") # Уже логируется в generate_content_sync

    # Используем filter_response из utils.py
    filtered = filter_response(response)
    logger.info(f"Filtered response for key {history_key}: {filtered[:100]}...")

    if filtered and not filtered.startswith("["): # Успешный ответ
        # Добавляем ответ АССИСТЕНТА в историю (ChromaDB) через state.add_to_history
        # Важно: Вызываем add_to_history здесь, ПОСЛЕ получения ответа
        await add_to_history(history_key, ASSISTANT_ROLE, filtered)

        logger.debug(f"Sending response to chat {chat_id}")
        try:
            if chat_type == 'private':
                await context.bot.send_message(chat_id=chat_id, text=filtered, parse_mode=None)
            else:
                # В группах отвечаем на исходное сообщение пользователя
                await update.message.reply_text(filtered, parse_mode=None)
        except Exception as send_err:
            logger.error(f"Failed to send message to chat {chat_id}: {send_err}", exc_info=True)
            if "Forbidden" not in str(send_err) and update.message:
                 try: await update.message.reply_text("⚠️ Не удалось отправить ответ. Возможно, я заблокирован(а) или исключен(а) из чата.")
                 except Exception: pass

        # Обучение на коротких фразах (если нужно)
        if len(original_input.split()) < 10:
            # learned_responses теперь в state.py
            learned_responses[original_input] = filtered
            logger.info(f"Learned response for '{original_input[:50]}...': '{filtered[:50]}...'")

    elif filtered.startswith("["): # Ответ содержит ошибку или блокировку от Gemini
         logger.warning(f"Response from Gemini indicates an issue for key {history_key}: {filtered}")
         if update.message: # Отвечаем только если есть исходное сообщение
             try:
                 user_error_msg = "Извините, не могу сейчас ответить на это. Попробуйте переформулировать."
                 if "заблокирован" in filtered.lower():
                     user_error_msg = "Мой ответ был заблокирован из-за ограничений безопасности. Попробуйте другой запрос."
                 elif "ошибка" in filtered.lower():
                      user_error_msg = "Возникла внутренняя ошибка при генерации ответа."
                 await update.message.reply_text(user_error_msg)
             except Exception as reply_err:
                  logger.error(f"Failed to send Gemini error message to chat {chat_id}: {reply_err}")
         # Запись об ошибке в историю не делаем, чтобы не засорять векторную базу
    else: # Пустой ответ после фильтрации
        logger.warning(f"Filtered response was empty for key {history_key}. Original raw: {response[:100]}...")
        if update.message:
            try:
                await update.message.reply_text("Простите, у меня возникли сложности с формированием ответа. Попробуйте еще раз или задайте другой вопрос.")
            except Exception as reply_err:
                 logger.error(f"Failed to send empty response message to chat {chat_id}: {reply_err}")


# --- Обработчик текстовых сообщений ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие текстовые сообщения."""
    if not update.message or not update.message.text or update.message.via_bot: return # Игнорируем сообщения от ботов
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat: return

    user_id = user.id
    chat_id = chat.id
    prompt_text = update.message.text.strip()
    chat_type = chat.type

    if not prompt_text: return # Игнорируем пустые сообщения

    # Обновляем информацию о пользователе
    await update_user_info(update)
    # Получаем предпочтительное имя или имя из ТГ
    user_name = user_preferred_name.get(user_id, user.first_name)
    history_key = chat_id if chat_type in ['group', 'supergroup'] else user_id

    # Получаем NER и Sentiment
    ner_model = get_ner_pipeline()
    entities = ner_model(prompt_text) if ner_model else None
    if entities: logger.info(f"RuBERT Entities: {entities}")

    sentiment_model = get_sentiment_pipeline()
    sentiment_result = sentiment_model(prompt_text) if sentiment_model else None
    sentiment = sentiment_result[0] if sentiment_result else None
    if sentiment: logger.info(f"RuBERT Sentiment: {sentiment}")

    # Добавляем ТЕКУЩЕЕ сообщение пользователя в историю ПЕРЕД поиском релевантной истории
    await add_to_history(history_key, USER_ROLE, prompt_text, user_name=user_name if chat_type != 'private' else None)

    # --- Сбор информации о пользователе из state.user_info_db ---
    profile_parts = []
    user_data = user_info_db.get(user_id, {})
    pref_name = user_preferred_name.get(user_id) # Имя для обращения уже в user_name
    tg_first_name = user_data.get('first_name', '')
    if user_name != tg_first_name: # Показываем только если отличается от ТГ
         profile_parts.append(f"Предпочитает имя: {user_name}")
    user_memory = user_data.get('memory')
    if user_memory:
         profile_parts.append(f"Запомненная информация: {user_memory}")
    # Добавьте другие поля, если нужно
    user_profile_info = ". ".join(profile_parts) if profile_parts else ""
    # --- Конец сбора информации ---

    # --- Логика ответа ---
    if chat_type == 'private':
        logger.info(f"Processing private message from {user_name} ({user_id}).")
        effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
        # Базовый системный промпт - основная роль и стиль
        system_message_base = f"{effective_style} Ты - {settings.BOT_NAME}."
        topic = user_topic.get(user_id)
        topic_context = f"Текущая тема разговора: {topic}." if topic else ""

        # Получаем релевантную историю из ChromaDB
        history_str = await query_relevant_history(
            history_key, prompt_text,
            n_results=settings.MAX_HISTORY_RESULTS,
            max_tokens=settings.MAX_HISTORY_TOKENS
        )

        # Формируем промпт
        prompt = prompt_builder.build_prompt(
            history_str=history_str,
            user_profile_info=user_profile_info,
            user_name=user_name,
            prompt_text=prompt_text,
            system_message_base=system_message_base,
            topic_context=topic_context,
            entities=entities,
            sentiment=sentiment
        )
        # Генерируем и отправляем ответ
        await _process_generation_and_reply(update, context, history_key, prompt, prompt_text)

    else: # Групповой чат
        if not should_process_message(get_bot_activity_percentage()):
            logger.debug(f"Message from {user_id} in group {chat_id} skipped: low activity.")
            return

        try: bot_username = (await context.bot.get_me()).username
        except Exception: bot_username = settings.BOT_NAME

        mentioned = f"@{bot_username}".lower() in prompt_text.lower() or settings.BOT_NAME.lower() in prompt_text.lower()
        is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id
        should_check_context = not (mentioned or is_reply_to_bot)
        is_related = await is_context_related(prompt_text, user_id, chat_id, chat_type) if should_check_context else False

        if mentioned or is_reply_to_bot or is_related:
            logger.info(f"Processing group message from {user_name} ({user_id}). Reason: M={mentioned}, R={is_reply_to_bot}, C={is_related}")
            effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
            system_message_base = f"{effective_style} Отвечай от первого лица как {settings.BOT_NAME}."
            topic = user_topic.get(user_id)
            topic_context = f"Текущая тема разговора с {user_name}: {topic}." if topic else ""

            history_str = await query_relevant_history(
                history_key, prompt_text,
                n_results=settings.MAX_HISTORY_RESULTS,
                max_tokens=settings.MAX_HISTORY_TOKENS
            )

            prompt = prompt_builder.build_prompt(
                history_str=history_str,
                user_profile_info=user_profile_info,
                user_name=user_name,
                prompt_text=prompt_text,
                system_message_base=system_message_base,
                topic_context=topic_context,
                entities=entities,
                sentiment=sentiment
            )
            await _process_generation_and_reply(update, context, history_key, prompt, prompt_text)
        else:
            logger.info(f"Group message from {user_id} ignored (no mention/reply/context): '{prompt_text[:50]}...'")


# --- Обработчик фотографий ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие фотографии."""
    if not update.message or not update.message.photo: return
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat: return

    user_id = user.id
    chat_id = chat.id
    chat_type = chat.type

    if chat_type != 'private' and not should_process_message(get_bot_activity_percentage()):
        logger.debug(f"Photo from {user_id} in group {chat_id} skipped: low activity.")
        return

    await update_user_info(update)
    user_name = user_preferred_name.get(user_id, user.first_name)
    history_key = chat_id if chat_type in ['group', 'supergroup'] else user_id

    logger.info(f"Processing photo from {user_name} ({user_id}) in chat {chat_id}")
    processing_msg = await update.message.reply_text("🖼️ Анализирую фото...")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.UPLOAD_PHOTO)
        photo_file = await update.message.photo[-1].get_file()
        file_bytes = await photo_file.download_as_bytearray()
        if not file_bytes:
             await processing_msg.edit_text("⚠️ Не удалось скачать фото.")
             return
        image = Image.open(BytesIO(file_bytes))
        caption = update.message.caption or ""

        # Добавляем запись о получении фото в историю
        history_entry = f"Получено фото" + (f" с подписью: '{caption}'" if caption else " без подписи")
        await add_to_history(history_key, USER_ROLE, history_entry, user_name=user_name if chat_type != 'private' else None)

        effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
        # Формируем промпт для Vision модели
        vision_prompt = f"{effective_style} "
        if chat_type != 'private': vision_prompt += f"Обращайся к {user_name}. "
        vision_prompt += f"Ты ({settings.BOT_NAME}) видишь фото"
        vision_prompt += f" с подписью: '{caption}'. " if caption else " без подписи. "
        vision_prompt += "Опиши кратко, что видишь, и эмоционально отреагируй на изображение и подпись (если есть)."

        contents = [vision_prompt, image] # Список для vision модели

        logger.debug(f"Sending image/prompt to Gemini Vision for key {history_key}")
        await processing_msg.edit_text("🤖 ИИ смотрит на фото...")

        response_text = await generate_vision_content_async(contents)
        filtered = filter_response(response_text)

        if filtered and not filtered.startswith("["):
            # Добавляем ответ бота на фото в историю
            await add_to_history(history_key, ASSISTANT_ROLE, filtered)
            await processing_msg.edit_text(filtered)
        elif filtered.startswith("["): # Ошибка Gemini
             logger.warning(f"Gemini Vision returned an error/block: {filtered}")
             user_error_msg = "Не удалось обработать изображение."
             if "заблокирован" in filtered.lower():
                 user_error_msg = "Не могу прокомментировать это изображение из-за ограничений безопасности."
             await processing_msg.edit_text(f"⚠️ {user_error_msg}")
        else: # Пустой ответ
            logger.warning(f"Gemini Vision returned empty response for photo from {user_id}")
            await processing_msg.edit_text("🤔 Не могу ничего сказать об этом изображении.")

    except Exception as e:
        logger.error(f"Error handling photo for user {user_id}: {e}", exc_info=True)
        try:
            await processing_msg.edit_text("❌ Произошла ошибка при обработке фото.")
        except Exception: pass


# --- Обработчик голосовых сообщений ---
async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие голосовые сообщения."""
    if not update.message or not update.message.voice: return
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat: return

    user_id = user.id
    chat_id = chat.id
    voice = update.message.voice
    chat_type = chat.type

    if chat_type != 'private' and not should_process_message(get_bot_activity_percentage()):
        logger.debug(f"Voice message from {user_id} skipped: low activity.")
        return

    await update_user_info(update)
    user_name = user_preferred_name.get(user_id, user.first_name)
    history_key = chat_id if chat_type in ['group', 'supergroup'] else user_id

    logger.info(f"Processing voice message from {user_name} ({user_id})")
    processing_msg = await update.message.reply_text("🎤 Обрабатываю голосовое...")

    original_file_path = None
    wav_path = None
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.RECORD_VOICE)
        voice_file = await voice.get_file()
        timestamp_id = int(time.time() * 1000)
        original_file_path = f"voice_{user_id}_{timestamp_id}.oga"
        wav_path = f"voice_{user_id}_{timestamp_id}.wav"

        await voice_file.download_to_drive(original_file_path)
        logger.debug(f"Downloaded voice file: {original_file_path}")

        # --- Конвертация OGG в WAV ---
        try:
            await processing_msg.edit_text("🎼 Конвертирую аудио...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: AudioSegment.from_file(original_file_path, format="ogg").export(wav_path, format="wav"))
            logger.debug(f"Converted {original_file_path} to {wav_path}")
            file_to_transcribe = wav_path
        except Exception as e:
            logger.error(f"Error converting voice {original_file_path} to WAV: {e}. Check ffmpeg.", exc_info=True)
            await processing_msg.edit_text("⚠️ Ошибка конвертации аудио. Убедитесь, что `ffmpeg` установлен.")
            return # Важно выйти здесь

        # --- Распознавание речи ---
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
        await processing_msg.edit_text("🗣️ Распознаю речь...")
        transcribed_text = await transcribe_voice(file_to_transcribe) # Эта функция удалит wav_path

        # --- Обработка распознанного текста ---
        if transcribed_text and not transcribed_text.startswith("["):
            logger.info(f"Transcription result for voice from {user_name}: '{transcribed_text}'")
            await processing_msg.edit_text("✍️ Формирую ответ...")

            # Добавляем распознанный текст в историю (ChromaDB)
            await add_to_history(history_key, USER_ROLE, transcribed_text + " (голос.)", user_name=user_name if chat_type != 'private' else None)

            # --- Сбор информации о пользователе ---
            profile_parts = []
            user_data = user_info_db.get(user_id, {})
            pref_name = user_preferred_name.get(user_id)
            tg_first_name = user_data.get('first_name', '')
            if user_name != tg_first_name: profile_parts.append(f"Предпочитает имя: {user_name}")
            user_memory = user_data.get('memory')
            if user_memory: profile_parts.append(f"Запомненная информация: {user_memory}")
            user_profile_info = ". ".join(profile_parts) if profile_parts else ""
            # --- Конец сбора информации ---

            # --- Логика генерации ответа ---
            ner_model = get_ner_pipeline()
            entities = ner_model(transcribed_text) if ner_model else None
            sentiment_model = get_sentiment_pipeline()
            sentiment_result = sentiment_model(transcribed_text) if sentiment_model else None
            sentiment = sentiment_result[0] if sentiment_result else None

            if chat_type == 'private':
                effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
                system_message_base = f"{effective_style} Ты - {settings.BOT_NAME}."
                topic = user_topic.get(user_id)
                topic_context = f"Тема: {topic}." if topic else ""

                history_str = await query_relevant_history(history_key, transcribed_text, n_results=settings.MAX_HISTORY_RESULTS, max_tokens=settings.MAX_HISTORY_TOKENS)
                prompt = prompt_builder.build_prompt(history_str=history_str, user_profile_info=user_profile_info, user_name=user_name, prompt_text=transcribed_text, system_message_base=system_message_base, topic_context=topic_context, entities=entities, sentiment=sentiment)
                await _process_generation_and_reply(update, context, history_key, prompt, transcribed_text)
            else: # Групповой чат
                try: bot_username = (await context.bot.get_me()).username
                except Exception: bot_username = settings.BOT_NAME
                mentioned = f"@{bot_username}".lower() in transcribed_text.lower() or settings.BOT_NAME.lower() in transcribed_text.lower()
                is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id
                should_check_context = not (mentioned or is_reply_to_bot)
                is_related = await is_context_related(transcribed_text, user_id, chat_id, chat_type) if should_check_context else False

                if mentioned or is_reply_to_bot or is_related:
                    effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
                    system_message_base = f"{effective_style} Отвечай от первого лица как {settings.BOT_NAME}."
                    topic = user_topic.get(user_id)
                    topic_context = f"Тема разговора с {user_name}: {topic}." if topic else ""

                    history_str = await query_relevant_history(history_key, transcribed_text, n_results=settings.MAX_HISTORY_RESULTS, max_tokens=settings.MAX_HISTORY_TOKENS)
                    prompt = prompt_builder.build_prompt(history_str=history_str, user_profile_info=user_profile_info, user_name=user_name, prompt_text=transcribed_text, system_message_base=system_message_base, topic_context=topic_context, entities=entities, sentiment=sentiment)
                    await _process_generation_and_reply(update, context, history_key, prompt, transcribed_text)
                else:
                     logger.info(f"Transcribed voice text from group ignored...")

            # Удаляем сообщение "Формирую ответ..."
            try: await processing_msg.delete()
            except Exception: pass

        elif transcribed_text and transcribed_text.startswith("["): # Ошибка распознавания
             logger.warning(f"Transcription failed for voice from {user_id}: {transcribed_text}")
             await processing_msg.edit_text(f"⚠️ {transcribed_text}")
        else: # Пустой результат распознавания
            logger.warning(f"Transcription returned empty for voice from {user_id}")
            await processing_msg.edit_text("⚠️ Не удалось распознать речь в сообщении.")

    except Exception as e:
        logger.error(f"Error handling voice message from {user_id}: {e}", exc_info=True)
        try:
            # Пытаемся отредактировать сообщение об ошибке
            if processing_msg:
                 await processing_msg.edit_text("❌ Произошла ошибка при обработке голосового сообщения.")
            else: # Если processing_msg не успело создаться
                 await update.message.reply_text("❌ Произошла ошибка при обработке голосового сообщения.")
        except Exception as e_reply:
             logger.error(f"Failed to send error reply for voice message: {e_reply}")
    finally:
         # Очистка временных файлов
        if wav_path and os.path.exists(wav_path):
             try: os.remove(wav_path)
             except OSError as e: logger.warning(f"Could not remove temp WAV {wav_path}: {e}")
        # Удаляем oga только если он был создан
        if original_file_path and os.path.exists(original_file_path):
             try: os.remove(original_file_path)
             except OSError as e: logger.warning(f"Could not remove temp OGA {original_file_path}: {e}")


# --- Обработчик видеосообщений ("кружочков") ---
async def handle_video_note_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает входящие видео-кружочки."""
    if not update.message or not update.message.video_note: return
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat: return

    user_id = user.id
    chat_id = chat.id
    video_note = update.message.video_note
    chat_type = chat.type

    if chat_type != 'private' and not should_process_message(get_bot_activity_percentage()):
        logger.debug(f"Video note from {user_id} skipped: low activity.")
        return

    await update_user_info(update)
    user_name = user_preferred_name.get(user_id, user.first_name)
    history_key = chat_id if chat_type in ['group', 'supergroup'] else user_id

    logger.info(f"Processing video note from {user_name} ({user_id})")
    processing_msg = await update.message.reply_text("📹 Обрабатываю видео-кружок...")

    original_file_path = None
    wav_path = None
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.RECORD_VIDEO_NOTE)
        video_note_file = await video_note.get_file()
        timestamp_id = int(time.time() * 1000)
        original_file_path = f"video_note_{user_id}_{timestamp_id}.mp4"
        wav_path = f"video_note_{user_id}_{timestamp_id}.wav"

        await video_note_file.download_to_drive(original_file_path)
        logger.debug(f"Downloaded video note file: {original_file_path}")

        # --- Извлечение аудио и конвертация в WAV ---
        try:
            await processing_msg.edit_text("🎼 Извлекаю аудио из видео...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: AudioSegment.from_file(original_file_path).export(wav_path, format="wav"))
            logger.debug(f"Extracted audio from {original_file_path} to {wav_path}")
            file_to_transcribe = wav_path
        except Exception as e:
            logger.error(f"Error extracting/converting audio from video note {original_file_path}: {e}. Check ffmpeg.", exc_info=True)
            await processing_msg.edit_text("⚠️ Ошибка извлечения аудио из видео. Убедитесь, что `ffmpeg` доступен.")
            return

        # --- Распознавание речи ---
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
        await processing_msg.edit_text("🗣️ Распознаю речь в видео...")
        transcribed_text = await transcribe_voice(file_to_transcribe) # Удалит wav_path

        # --- Обработка распознанного текста ---
        if transcribed_text and not transcribed_text.startswith("["):
            logger.info(f"Transcription result (video note) from {user_name}: '{transcribed_text}'")
            await processing_msg.edit_text("✍️ Формирую ответ...")

            # Добавляем распознанный текст в историю (ChromaDB)
            await add_to_history(history_key, USER_ROLE, transcribed_text + " (видео)", user_name=user_name if chat_type != 'private' else None)

            # --- Сбор информации о пользователе ---
            profile_parts = []
            user_data = user_info_db.get(user_id, {})
            pref_name = user_preferred_name.get(user_id)
            tg_first_name = user_data.get('first_name', '')
            if user_name != tg_first_name: profile_parts.append(f"Предпочитает имя: {user_name}")
            user_memory = user_data.get('memory')
            if user_memory: profile_parts.append(f"Запомненная информация: {user_memory}")
            user_profile_info = ". ".join(profile_parts) if profile_parts else ""
            # --- Конец сбора информации ---

            # --- Логика генерации ответа ---
            ner_model = get_ner_pipeline()
            entities = ner_model(transcribed_text) if ner_model else None
            sentiment_model = get_sentiment_pipeline()
            sentiment_result = sentiment_model(transcribed_text) if sentiment_model else None
            sentiment = sentiment_result[0] if sentiment_result else None

            if chat_type == 'private':
                 effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
                 system_message_base = f"{effective_style} Ты - {settings.BOT_NAME}."
                 topic = user_topic.get(user_id)
                 topic_context = f"Тема: {topic}." if topic else ""

                 history_str = await query_relevant_history(history_key, transcribed_text, n_results=settings.MAX_HISTORY_RESULTS, max_tokens=settings.MAX_HISTORY_TOKENS)
                 prompt = prompt_builder.build_prompt(history_str=history_str, user_profile_info=user_profile_info, user_name=user_name, prompt_text=transcribed_text, system_message_base=system_message_base, topic_context=topic_context, entities=entities, sentiment=sentiment)
                 await _process_generation_and_reply(update, context, history_key, prompt, transcribed_text)
            else: # Групповой чат
                 try: bot_username = (await context.bot.get_me()).username
                 except Exception: bot_username = settings.BOT_NAME
                 mentioned = f"@{bot_username}".lower() in transcribed_text.lower() or settings.BOT_NAME.lower() in transcribed_text.lower()
                 is_reply_to_bot = update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id
                 should_check_context = not (mentioned or is_reply_to_bot)
                 is_related = await is_context_related(transcribed_text, user_id, chat_id, chat_type) if should_check_context else False

                 if mentioned or is_reply_to_bot or is_related:
                     effective_style = await _get_effective_style(chat_id, user_id, user_name, chat_type)
                     system_message_base = f"{effective_style} Отвечай от первого лица как {settings.BOT_NAME}."
                     topic = user_topic.get(user_id)
                     topic_context = f"Тема разговора с {user_name}: {topic}." if topic else ""

                     history_str = await query_relevant_history(history_key, transcribed_text, n_results=settings.MAX_HISTORY_RESULTS, max_tokens=settings.MAX_HISTORY_TOKENS)
                     prompt = prompt_builder.build_prompt(history_str=history_str, user_profile_info=user_profile_info, user_name=user_name, prompt_text=transcribed_text, system_message_base=system_message_base, topic_context=topic_context, entities=entities, sentiment=sentiment)
                     await _process_generation_and_reply(update, context, history_key, prompt, transcribed_text)
                 else:
                      logger.info(f"Transcribed video note text from group ignored...")

            try: await processing_msg.delete()
            except Exception: pass

        elif transcribed_text and transcribed_text.startswith("["): # Ошибка распознавания
             logger.warning(f"Transcription failed (video note) for {user_id}: {transcribed_text}")
             await processing_msg.edit_text(f"⚠️ {transcribed_text}")
        else: # Пустой результат распознавания
             logger.warning(f"Transcription returned empty (video note) for {user_id}")
             await processing_msg.edit_text("⚠️ Не удалось распознать речь в видео-кружке.")

    except Exception as e:
        logger.error(f"Error handling video note from {user_id}: {e}", exc_info=True)
        try:
            if processing_msg:
                 await processing_msg.edit_text("❌ Произошла ошибка при обработке видео-кружка.")
            else:
                 await update.message.reply_text("❌ Произошла ошибка при обработке видео-кружка.")
        except Exception as e_reply:
             logger.error(f"Failed to send error reply for video note: {e_reply}")
    finally:
        # Очистка временных файлов
        if wav_path and os.path.exists(wav_path):
             try: os.remove(wav_path)
             except OSError as e: logger.warning(f"Could not remove temp WAV {wav_path}: {e}")
        if original_file_path and os.path.exists(original_file_path):
             try: os.remove(original_file_path)
             except OSError as e: logger.warning(f"Could not remove temp MP4 {original_file_path}: {e}")
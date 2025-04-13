import pinecone
from pinecone import Index as PineconeIndex # Переименуем, чтобы не путать с индексом в ChromaDB
from typing import List, Dict, Optional, Tuple # Оставьте нужные типы
import time
import uuid
import asyncio

# Импорты из проекта
from config import logger, settings, USER_ROLE, ASSISTANT_ROLE
from chromadb.utils import embedding_functions # Оставляем для sentence_transformer_ef

PINECONE_API_KEY = settings.PINECONE_API_KEY  # Предполагаем, что ключ добавлен в config.py
PINECONE_ENVIRONMENT = settings.PINECONE_ENVIRONMENT # Предполагаем, что окружение добавлено в config.py
PINECONE_INDEX_NAME_PREFIX = "chat-history-" # Измените по своему усмотрению

client: Optional[pinecone.Pinecone] = None
sentence_transformer_ef: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None

def initialize_vector_store():
    global client, sentence_transformer_ef
    if client and sentence_transformer_ef:
        logger.debug("Pinecone client and embedding function already initialized.")
        return True
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        logger.critical("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found in settings.")
        return False
    try:
        client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        logger.info(f"Pinecone client initialized successfully in environment: {PINECONE_ENVIRONMENT}")

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        logger.info(f"SentenceTransformer embedding function initialized with model: all-MiniLM-L6-v2")
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize Pinecone client or embedding model: {e}", exc_info=True)
        client = None
        sentence_transformer_ef = None
        return False

def get_collection_name(history_key: int) -> str:
    """Генерирует стандартизированное имя индекса Pinecone."""
    return f"{PINECONE_INDEX_NAME_PREFIX}{history_key}"

def get_or_create_collection(history_key: int) -> Optional[PineconeIndex]:
    if not client or not sentence_transformer_ef:
        if not initialize_vector_store():
             logger.error("Pinecone client or embedding function not initialized and failed to re-initialize.")
             return None
    index_name = get_collection_name(history_key)
    try:
        if not client: return None
        if index_name not in client.list_indexes().names():
            logger.info(f"Creating Pinecone index: {index_name}")
            client.create_index(
                name=index_name,
                dimension=384,  # Размерность эмбеддингов all-MiniLM-L6-v2
                metric="cosine" # Обычно используется для семантического поиска
            )
            logger.info(f"Pinecone index '{index_name}' created successfully.")
        index = client.Index(index_name)
        logger.debug(f"Accessed Pinecone index: {index_name}")
        return index
    except Exception as e:
        logger.error(f"Failed to get or create Pinecone index '{index_name}': {e}", exc_info=True)
        return None

def add_message_to_vector_store_sync(history_key: int, role: str, message: str, timestamp: float, user_name: Optional[str] = None):
    index = get_or_create_collection(history_key)
    if not index:
        logger.error(f"Cannot add message sync, Pinecone index unavailable for key {history_key}")
        return

    doc_id = f"{int(timestamp)}_{uuid.uuid4()}"
    metadata = {
        "role": role,
        "timestamp": timestamp,
        "message_preview": message[:100],
        **( {"user_name": user_name} if user_name and role == USER_ROLE else {} )
    }
    try:
        embeddings = sentence_transformer_ef.encode([message]).tolist()
        index.upsert(vectors=[(doc_id, embeddings[0], metadata)])
        logger.debug(f"SYNC Added message to Pinecone index {index.name} with ID {doc_id} (Role: {role})")
    except Exception as e:
        logger.error(f"SYNC Failed to upsert document to Pinecone index {index.name}: {e}", exc_info=True)

async def add_message(history_key: int, role: str, message: str, timestamp: float, user_name: Optional[str] = None):
    """Асинхронная обертка для добавления сообщения в векторную базу данных."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, add_message_to_vector_store_sync, history_key, role, message, timestamp, user_name
    )

async def query_relevant_history(history_key: int, query_text: str, n_results: int = 15, max_tokens: int = 2000) -> str:
    index = get_or_create_collection(history_key)
    if not index:
        logger.warning(f"Pinecone index not found for key {history_key} during query.")
        return ""
    if not query_text:
        logger.warning("Query text is empty, cannot perform search.")
        return ""

    try:
        embeddings = sentence_transformer_ef.encode([query_text]).tolist()
        query_results = index.query(
            vector=embeddings[0],
            top_k=n_results,
            include_values=False,
            include_metadata=True
        )

        logger.debug(f"Query results for key {history_key} (query: '{query_text[:50]}...'): Found {len(query_results.matches)} items.")

        if not query_results.matches:
            return ""

        messages_data = []
        for match in query_results.matches:
            metadata = match.metadata
            document = metadata.get('message_preview', '') # Pinecone не хранит сам документ, только метаданные
            score = match.score
            timestamp = metadata.get('timestamp', 0)
            messages_data.append({
                "id": match.id, "doc": document, "meta": metadata,
                "distance": 1 - score if score is not None else None, # Косинусная схожесть, Pinecone возвращает score
                "timestamp": timestamp
            })

        messages_data.sort(key=lambda x: x['timestamp'])

        history_lines = []
        current_tokens = 0
        token_factor = 4

        # Вам может потребоваться заново получить полные документы из Pinecone,
        # если 'message_preview' недостаточно. Либо хранить полный текст в метаданных (осторожно с размером).
        # В данном примере используется 'message_preview'.
        for item in messages_data:
            role = item['meta'].get('role', 'Unknown')
            user_name = item['meta'].get('user_name')
            message = item['meta'].get('message_preview', '') # Используем preview

            entry = f"{role} ({user_name}): {message}" if role == USER_ROLE and user_name else f"{role}: {message}"

            entry_tokens = len(entry) / token_factor
            if current_tokens + entry_tokens <= max_tokens:
                history_lines.append(entry)
                current_tokens += entry_tokens
            else:
                logger.debug(f"History token limit ({max_tokens}) reached for key {history_key}.")
                break

        history_str = "\n".join(history_lines)
        logger.debug(f"Formatted relevant history string for key {history_key} (approx tokens: {int(current_tokens)}). Preview: \n{history_str[:200]}...")
        return history_str

    except Exception as e:
        logger.error(f"Error querying Pinecone index for key {history_key}: {e}", exc_info=True)
        return "[Ошибка при поиске в истории]"

async def get_last_bot_message(history_key: int) -> Optional[str]:
    index = get_or_create_collection(history_key)
    if not index: return None

    try:
        embeddings = sentence_transformer_ef.encode(["последнее сообщение бота"]).tolist() # Запрос для поиска сообщений бота
        query_results = index.query(
            vector=embeddings[0],
            top_k=5, # Проверяем последние несколько сообщений
            include_values=False,
            include_metadata=True,
            filter={"role": {"$eq": ASSISTANT_ROLE}}
        )

        if not query_results.matches:
            return None

        latest_message: Optional[str] = None
        max_ts = 0.0

        for match in query_results.matches:
            metadata = match.metadata
            timestamp = metadata.get('timestamp', 0.0)
            message = metadata.get('message_preview')
            if timestamp > max_ts and message:
                max_ts = timestamp
                latest_message = message

        logger.debug(f"Last bot message found for key {history_key}: {'Yes' if latest_message else 'No'}")
        return latest_message

    except Exception as e:
        logger.error(f"Error getting last bot message from Pinecone index for key {history_key}: {e}", exc_info=True)
        return None

def delete_history_sync(history_key: int) -> bool:
    index_name = get_collection_name(history_key)
    if not client:
        logger.error("Pinecone client not initialized. Cannot delete history.")
        return False
    try:
        if index_name in client.list_indexes().names():
            client.delete_index(name=index_name)
            logger.info(f"Deleted Pinecone index: {index_name}")
            return True
        else:
            logger.warning(f"Pinecone index {index_name} not found for deletion.")
            return True
    except Exception as e:
        logger.error(f"Failed to delete Pinecone index {index_name}: {e}", exc_info=True)
        return False

async def delete_history(history_key: int) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, delete_history_sync, history_key)

def delete_old_messages_sync(history_key: int, ttl_seconds: int) -> int:
    index = get_or_create_collection(history_key)
    if not index:
        logger.warning(f"Cannot cleanup old messages, Pinecone index unavailable for key {history_key}")
        return 0

    cutoff_time = time.time() - ttl_seconds
    deleted_count = 0
    try:
        # Pinecone позволяет фильтровать по метаданным при удалении
        index.delete(filter={"timestamp": {"$lt": cutoff_time}})
        logger.info(f"Deleted old messages from Pinecone index {index.name} with timestamp less than {cutoff_time}")
        return -1 # Возвращаем -1, так как нет точного счетчика удаленных элементов
    except Exception as e:
        logger.warning(f"Failed to delete old messages from Pinecone index {index.name}: {e}", exc_info=True)
        return 0

async def cleanup_all_old_histories(ttl_seconds: int):
    if not client:
        logger.error("Pinecone client not initialized. Cannot run cleanup.")
        return
    logger.info(f"Starting cleanup of old messages across all Pinecone indexes (TTL: {ttl_seconds}s)...")
    total_deleted = 0
    loop = asyncio.get_running_loop()
    try:
        indexes = client.list_indexes().names()
        for index_name in indexes:
            if index_name.startswith(PINECONE_INDEX_NAME_PREFIX):
                history_key_str = index_name[len(PINECONE_INDEX_NAME_PREFIX):]
                if history_key_str.isdigit():
                    history_key = int(history_key_str)
                    deleted = await loop.run_in_executor(None, delete_old_messages_sync, history_key, ttl_seconds)
                    if deleted > 0: # В delete_old_messages_sync возвращается -1
                        total_deleted += 1 # Просто инкрементируем счетчик обработанных индексов
                else:
                    logger.warning(f"Skipping cleanup for Pinecone index with non-integer key: {index_name}")
    except Exception as e:
        logger.error(f"Error listing Pinecone indexes or running cleanup tasks: {e}", exc_info=True)

# Добавление функции для постоянной информации (резюме)
def add_persistent_info_sync(history_key: int, info_type: str, info_content: str):
    """Синхронная функция добавления постоянной информации (например, резюме) в Pinecone."""
    index = get_or_create_collection(history_key)
    if not index:
        logger.error(f"Cannot add persistent info sync, Pinecone index unavailable for key {history_key}")
        return

    timestamp = time.time()
    doc_id = f"persistent_{info_type}_{int(timestamp)}"
    metadata = {
        "type": "persistent_info",
        "info_type": info_type,
        "timestamp": timestamp,
        "content_preview": info_content[:100]
    }
    try:
        embeddings = sentence_transformer_ef.encode([info_content]).tolist()
        index.upsert(vectors=[(doc_id, embeddings[0], metadata)])
        logger.info(f"SYNC Added persistent info '{info_type}' to Pinecone index {index.name} with ID {doc_id}")
    except Exception as e:
        logger.error(f"SYNC Failed to add persistent info to Pinecone index {index.name}: {e}", exc_info=True)

async def add_persistent_info(history_key: int, info_type: str, info_content: str):
    """Асинхронная обертка для добавления постоянной информации в Pinecone."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, add_persistent_info_sync, history_key, info_type, info_content
    )

async def query_relevant_history_with_persistent_info(history_key: int, query_text: str, n_results: int = 15, max_tokens: int = 2000) -> str:
    index = get_or_create_collection(history_key)
    if not index:
        logger.warning(f"Pinecone index not found for key {history_key} during query.")
        return ""
    if not query_text:
        logger.warning("Query text is empty, cannot perform search.")
        return ""

    try:
        embeddings = sentence_transformer_ef.encode([query_text]).tolist()
        query_results = index.query(
            vector=embeddings[0],
            top_k=n_results,
            include_values=False,
            include_metadata=True
        )

        logger.debug(f"Query results for key {history_key} (query: '{query_text[:50]}...'): Found {len(query_results.matches)} items.")

        messages_data = []
        if query_results.matches:
            for match in query_results.matches:
                metadata = match.metadata
                document = metadata.get('message_preview', '')
                score = match.score
                timestamp = metadata.get('timestamp', 0)
                messages_data.append({
                    "id": match.id, "doc": document, "meta": metadata,
                    "distance": 1 - score if score is not None else None,
                    "timestamp": timestamp
                })

        # Получаем постоянную информацию
        query_persistent_results = index.query(
            vector=embeddings[0], # Можно использовать тот же вектор запроса или нулевой вектор
            top_k=100, # Получаем все постоянные записи (предполагаем, что их немного)
            include_values=False,
            include_metadata=True,
            filter={"type": {"$eq": "persistent_info"}}
        )
        if query_persistent_results.matches:
            for match in query_persistent_results.matches:
                metadata = match.metadata
                document = metadata.get('content_preview', '')
                timestamp = metadata.get('timestamp', 0)
                messages_data.append({
                    "id": match.id, "doc": document, "meta": metadata,
                    "distance": 0.0, # Придаем высокий приоритет
                    "timestamp": timestamp
                })


        messages_data.sort(key=lambda x: x['timestamp'])

        history_lines = []
        current_tokens = 0
        token_factor = 4

        for item in messages_data:
            role = item['meta'].get('role', item['meta'].get('info_type', 'Unknown'))
            user_name = item['meta'].get('user_name')
            message = item['doc']
            prefix = f"{role}: "
            if role == USER_ROLE and user_name:
                prefix = f"{role} ({user_name}): "
            elif item['meta'].get('type') == 'persistent_info':
                prefix = f"{item['meta']['info_type']}: "

            entry = prefix + message

            entry_tokens = len(entry) / token_factor
            if current_tokens + entry_tokens <= max_tokens:
                history_lines.append(entry)
                current_tokens += entry_tokens
            else:
                logger.debug(f"History token limit ({max_tokens}) reached for key {history_key}.")
                break

        history_str = "\n".join(history_lines)
        logger.debug(f"Formatted relevant history string for key {history_key} (approx tokens: {int(current_tokens)}). Preview: \n{history_str[:200]}...")
        return history_str

    except Exception as e:
        logger.error(f"Error querying Pinecone index for key {history_key}: {e}", exc_info=True)
        return "[Ошибка при поиске в истории]"

if __name__ == '__main__':
    async def main():
        if not initialize_vector_store():
            print("Failed to initialize vector store.")
            return

        test_history_key = 12345
        resume_content = """
        Имя: Иван Иванов
        Опыт работы: ...
        Навыки: ...
        """

        await add_persistent_info(test_history_key, "resume", resume_content)
        print(f"Резюме добавлено для ключа истории: {test_history_key}")

        question = "Расскажи об опыте работы этого кандидата."
        relevant_history = await query_relevant_history_with_persistent_info(test_history_key, question)
        print("\nРелевантная история, включая резюме:")
        print(relevant_history)

        # Очистка тестовой истории (опционально)
        await delete_history(test_history_key)
        print(f"\nИстория для ключа {test_history_key} удалена.")

    asyncio.run(main())

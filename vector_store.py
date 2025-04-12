import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.types import GetResult # Для type hints
import time
import uuid
from typing import List, Dict, Optional, Tuple # Tuple может быть не нужен, если убрать group_user_style_prompts из state
import asyncio

# Импорты из проекта
from config import logger, settings, USER_ROLE, ASSISTANT_ROLE # Используем настройки и роли из config

# ==============================================================================
# Начало: Содержимое vector_store.py
# ==============================================================================

# --- Константы ---
# CHROMA_DATA_PATH уже определен в config
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME_PREFIX = "chat_history_"

# --- Инициализация ---
client: Optional[chromadb.Client] = None
sentence_transformer_ef: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None

def initialize_vector_store():
    """Инициализирует клиент ChromaDB и модель эмбеддингов."""
    global client, sentence_transformer_ef
    if client and sentence_transformer_ef:
        logger.debug("Vector store already initialized.")
        return True
    try:
        # Используем путь из настроек
        client = chromadb.PersistentClient(path=settings.CHROMA_DATA_PATH)
        logger.info(f"ChromaDB PersistentClient initialized at path: {settings.CHROMA_DATA_PATH}")

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        logger.info(f"SentenceTransformer embedding function initialized with model: {EMBEDDING_MODEL_NAME}")
        return True
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB client or embedding model: {e}", exc_info=True)
        client = None
        sentence_transformer_ef = None
        return False

# --- Функции ---

def get_collection_name(history_key: int) -> str:
    """Генерирует стандартизированное имя коллекции."""
    return f"{COLLECTION_NAME_PREFIX}{history_key}"

def get_or_create_collection(history_key: int) -> Optional[chromadb.Collection]:
    """Получает или создает коллекцию для указанного ключа истории."""
    if not client or not sentence_transformer_ef:
        # Попытка инициализации, если не удалось при старте
        if not initialize_vector_store():
             logger.error("ChromaDB client or embedding function not initialized and failed to re-initialize.")
             return None
    collection_name = get_collection_name(history_key)
    try:
        # Проверка на None еще раз после попытки инициализации
        if not client or not sentence_transformer_ef: return None
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        logger.debug(f"Accessed or created ChromaDB collection: {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create collection '{collection_name}': {e}", exc_info=True)
        return None

def add_message_to_vector_store_sync(history_key: int, role: str, message: str, timestamp: float, user_name: Optional[str] = None):
    """Синхронная функция добавления сообщения (для вызова из to_thread)."""
    collection = get_or_create_collection(history_key)
    if not collection:
        logger.error(f"Cannot add message sync, collection unavailable for key {history_key}")
        return

    doc_id = f"{int(timestamp)}_{uuid.uuid4()}"
    metadata = {
        "role": role,
        "timestamp": timestamp,
        "message_preview": message[:100],
        **({"user_name": user_name} if user_name and role == USER_ROLE else {})
    }
    try:
        collection.add(
            documents=[message],
            metadatas=[metadata],
            ids=[doc_id]
        )
        logger.debug(f"SYNC Added message to collection {collection.name} with ID {doc_id} (Role: {role})")
    except Exception as e:
        logger.error(f"SYNC Failed to add document to collection {collection.name}: {e}", exc_info=True)


async def add_message(history_key: int, role: str, message: str, timestamp: float, user_name: Optional[str] = None):
    """Асинхронная обертка для добавления сообщения в векторную базу данных."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, add_message_to_vector_store_sync, history_key, role, message, timestamp, user_name
    )


async def query_relevant_history(history_key: int, query_text: str, n_results: int = 15, max_tokens: int = 2000) -> str:
    """Ищет релевантные сообщения, сортирует по времени и форматирует для промпта."""
    collection = get_or_create_collection(history_key)
    if not collection:
        logger.warning(f"Collection not found for key {history_key} during query.")
        return ""
    if not query_text:
        logger.warning("Query text is empty, cannot perform search.")
        return ""

    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
        )

        logger.debug(f"Query results for key {history_key} (query: '{query_text[:50]}...'): Found {len(results.get('ids', [[]])[0])} items.")

        if not results or not results.get('ids') or not results['ids'][0]:
            return ""

        messages_data = []
        if results['metadatas'] and results['documents']: # Проверяем наличие данных
            for i, doc_id in enumerate(results['ids'][0]):
                # Дополнительные проверки на существование индексов
                if i < len(results['metadatas'][0]) and i < len(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    document = results['documents'][0][i]
                    distance = results.get('distances', [[None]*len(results['ids'][0])])[0][i] # Безопасное получение дистанции
                    timestamp = metadata.get('timestamp', 0)
                    messages_data.append({
                        "id": doc_id, "doc": document, "meta": metadata,
                        "distance": distance, "timestamp": timestamp
                    })
                else:
                    logger.warning(f"Index out of range for doc_id {doc_id} in query results.")

        messages_data.sort(key=lambda x: x['timestamp'])

        history_lines = []
        current_tokens = 0
        token_factor = 4

        for item in messages_data:
            role = item['meta'].get('role', 'Unknown')
            user_name = item['meta'].get('user_name')
            message = item['doc']
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
        logger.error(f"Error querying collection for key {history_key}: {e}", exc_info=True)
        return "[Ошибка при поиске в истории]"

async def get_last_bot_message(history_key: int) -> Optional[str]:
    """Получает текст последнего сообщения бота из истории."""
    collection = get_or_create_collection(history_key)
    if not collection: return None

    try:
        loop = asyncio.get_running_loop()
        results: Optional[GetResult] = await loop.run_in_executor(
            None,
            lambda: collection.get(
                where={"role": ASSISTANT_ROLE},
                limit=5,
                include=['documents', 'metadatas']
            )
        )

        if not results or not results.get('ids'):
            return None

        latest_message: Optional[str] = None
        max_ts = 0.0
        # Итерируемся безопасно
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])
        for i, doc_id in enumerate(results['ids']):
             if i < len(metadatas) and i < len(documents):
                ts = metadatas[i].get('timestamp', 0.0)
                if ts > max_ts:
                    max_ts = ts
                    latest_message = documents[i]

        logger.debug(f"Last bot message found for key {history_key}: {'Yes' if latest_message else 'No'}")
        return latest_message

    except Exception as e:
        logger.error(f"Error getting last bot message for key {history_key}: {e}", exc_info=True)
        return None

def delete_history_sync(history_key: int) -> bool:
    """Синхронная функция удаления истории (для вызова из to_thread или напрямую)."""
    if not client:
        logger.error("ChromaDB client not initialized. Cannot delete history.")
        return False
    collection_name = get_collection_name(history_key)
    try:
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        else:
            logger.warning(f"Collection {collection_name} not found for deletion.")
            return True # Возвращаем True, так как её и не было
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_name}: {e}", exc_info=True)
        return False

async def delete_history(history_key: int) -> bool:
    """Асинхронная обертка для удаления истории."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, delete_history_sync, history_key)


def delete_old_messages_sync(history_key: int, ttl_seconds: int) -> int:
    """Синхронная функция удаления старых сообщений."""
    collection = get_or_create_collection(history_key)
    if not collection:
        logger.warning(f"Cannot cleanup old messages, collection unavailable for key {history_key}")
        return 0

    cutoff_time = time.time() - ttl_seconds
    deleted_count = 0
    try:
        # Используем where-фильтр, если поддерживается
        results = collection.get(
             where={"timestamp": {"$lt": cutoff_time}},
             include=[]
        )
        ids_to_delete = results.get('ids', [])

        if ids_to_delete:
             collection.delete(ids=ids_to_delete)
             deleted_count = len(ids_to_delete)
             logger.info(f"Deleted {deleted_count} old messages from {collection.name} using 'where'.")
        else:
             logger.debug(f"No old messages found to delete in {collection.name} using 'where'.")
    except Exception as e:
        logger.warning(f"Failed delete using 'where' for {collection.name} (error: {e}). Trying manual check.")
        # Fallback
        try:
            all_data = collection.get(include=['metadatas'])
            if all_data and all_data.get('ids') and all_data.get('metadatas'):
                ids_to_delete_manual = [
                    all_data['ids'][i] for i, meta in enumerate(all_data['metadatas'])
                    if meta.get('timestamp', float('inf')) < cutoff_time and i < len(all_data['ids'])
                ]
                if ids_to_delete_manual:
                    collection.delete(ids=ids_to_delete_manual)
                    deleted_count = len(ids_to_delete_manual)
                    logger.info(f"Deleted {deleted_count} old messages from {collection.name} using manual fallback.")
                else:
                    logger.debug(f"No old messages found in {collection.name} using manual fallback.")
        except Exception as fallback_e:
            logger.error(f"Error during manual fallback deletion for {collection.name}: {fallback_e}", exc_info=True)
    return deleted_count

async def cleanup_all_old_histories(ttl_seconds: int):
    """Асинхронная функция очистки старых сообщений во всех коллекциях."""
    if not client:
        logger.error("ChromaDB client not initialized. Cannot run cleanup.")
        return
    logger.info(f"Starting cleanup of old messages across all collections (TTL: {ttl_seconds}s)...")
    total_deleted = 0
    loop = asyncio.get_running_loop()
    try:
        collections = await loop.run_in_executor(None, client.list_collections)

        tasks = []
        for collection in collections:
            if collection.name.startswith(COLLECTION_NAME_PREFIX):
                try:
                    history_key_str = collection.name[len(COLLECTION_NAME_PREFIX):]
                    if history_key_str.isdigit():
                        history_key = int(history_key_str)
                        # Запускаем синхронную очистку в отдельном потоке для каждой коллекции
                        tasks.append(loop.run_in_executor(None, delete_old_messages_sync, history_key, ttl_seconds))
                    else:
                        logger.warning(f"Skipping cleanup for collection with non-integer key: {collection.name}")
                except Exception as inner_e:
                    logger.error(f"Error preparing cleanup task for {collection.name}: {inner_e}", exc_info=True)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during parallel history cleanup task: {result}", exc_info=result)
            elif isinstance(result, int):
                total_deleted += result
        logger.info(f"Cleanup finished. Total old messages deleted: {total_deleted}")
    except Exception as e:
        logger.error(f"Error listing collections or running cleanup tasks: {e}", exc_info=True)

# ==============================================================================
# Конец: Содержимое vector_store.py
# ==============================================================================
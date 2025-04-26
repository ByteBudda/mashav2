# -*- coding: utf-8 -*-
# vector_db.py
import os
import time
from typing import List, Dict, Optional, Tuple, Any
import threading
import asyncio # Для блокировки кэша

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    logger, settings, EMBEDDING_MODEL_NAME, CHROMA_DB_PATH,
    CHROMA_HISTORY_COLLECTION_PREFIX, CHROMA_FACTS_COLLECTION_NAME,
    VECTOR_SEARCH_K_HISTORY, VECTOR_SEARCH_K_FACTS
)

# --- Глобальные переменные ---
client: Optional[chromadb.Client] = None
model: Optional[SentenceTransformer] = None
dimension: Optional[int] = None
model_lock = threading.Lock() # Блокировка для потокобезопасного вызова model.encode()
collection_cache: Dict[str, chromadb.Collection] = {}
cache_lock = asyncio.Lock() # Асинхронная блокировка для доступа к кэшу коллекций

# --- Инициализация ---
def initialize_vector_db():
    """Инициализирует клиент ChromaDB и модель эмбеддингов."""
    global client, model, dimension
    if client and model: logger.info("Vector DB already initialized."); return

    logger.info(f"Initializing Vector DB (Chroma)... Path: {CHROMA_DB_PATH}")
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        with model_lock:
            if model is None:
                # Исправляем возможный race condition при многопоточной инициализации
                if model is None:
                    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                    dimension = model.get_sentence_embedding_dimension()
                    logger.info(f"Model '{EMBEDDING_MODEL_NAME}' loaded. Dimension: {dimension}")

        # Инициализация клиента ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info(f"ChromaDB client initialized. Path: {CHROMA_DB_PATH}")

        # Проверяем/создаем коллекцию фактов при инициализации (синхронно)
        get_or_create_facts_collection_sync()

    except Exception as e:
        logger.critical(f"Failed to initialize Vector DB: {e}", exc_info=True)
        client = None; model = None; dimension = None; raise

# --- Управление коллекциями ---
async def get_or_create_collection(name: str) -> Optional[chromadb.Collection]:
    """Асинхронно и потокобезопасно получает или создает коллекцию ChromaDB."""
    global client, collection_cache, cache_lock
    if not client: logger.error("ChromaDB client not initialized."); return None

    # Сначала проверяем кэш асинхронно
    async with cache_lock:
        if name in collection_cache:
            return collection_cache[name]

    # Если в кэше нет, пытаемся получить/создать
    try:
        # Используем get_or_create_collection для атомарности
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"} # Рекомендуется для эмбеддингов SBERT
        )
        logger.info(f"Accessed/Created ChromaDB collection '{name}'. Count: {collection.count()}")
        # Обновляем кэш после успешного получения/создания
        async with cache_lock:
            collection_cache[name] = collection
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create collection '{name}': {e}", exc_info=True)
        return None

# Синхронная версия для инициализации коллекции фактов
def get_or_create_facts_collection_sync() -> Optional[chromadb.Collection]:
    """Синхронно получает или создает единую коллекцию для фактов."""
    global client
    if not client: return None
    name = CHROMA_FACTS_COLLECTION_NAME
    try:
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Accessed/Created ChromaDB facts collection '{name}'. Count: {collection.count()}")
        # Добавляем в кэш (здесь можно без async lock, т.к. вызывается при старте)
        collection_cache[name] = collection
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create facts collection '{name}': {e}")
        return None

def get_history_collection_name(history_key: int) -> str:
    """Генерирует имя коллекции для истории чата."""
    # Используем префикс и ключ, проверяем на допустимые символы (Chroma требует特定формат)
    safe_key = str(history_key).replace('-', '_neg_') # Заменяем минус, если chat_id отрицательный
    return f"{CHROMA_HISTORY_COLLECTION_PREFIX}{safe_key}"

async def get_history_collection(history_key: int) -> Optional[chromadb.Collection]:
    """Получает или создает коллекцию для истории конкретного чата."""
    collection_name = get_history_collection_name(history_key)
    return await get_or_create_collection(collection_name)

# --- Добавление/Обновление Эмбеддингов ---

def add_message_embedding_sync(sqlite_id: int, history_key: int, role: str, text: str):
    """Синхронная функция добавления эмбеддинга сообщения (для to_thread)."""
    global model
    if model is None: return
    if not text or not text.strip(): return

    # Получаем коллекцию синхронно (блокирующий вызов)
    # Это может быть неоптимально, если много потоков будут создавать коллекции одновременно
    # Лучше передавать объект коллекции или использовать асинхронный вариант
    collection_name = get_history_collection_name(history_key)
    collection = asyncio.run(get_or_create_collection(collection_name)) # Запускаем async в sync контексте

    if collection is None:
        logger.error(f"Sync: Could not get history collection '{collection_name}'. Skipping embedding.")
        return

    chroma_id = str(sqlite_id)
    try:
        logger.debug(f"Sync: Generating history embedding for SQLite ID: {sqlite_id}")
        with model_lock: embedding = model.encode(text).tolist()
        metadata = {"history_key": history_key, "role": role, "sqlite_id": sqlite_id, "timestamp": time.time()}
        collection.upsert(ids=[chroma_id], embeddings=[embedding], metadatas=[metadata], documents=[text])
        logger.debug(f"Sync: Upserted history embedding for SQLite ID {sqlite_id} in '{collection.name}'.")
    except Exception as e:
        logger.error(f"Sync: Failed to add/upsert history embedding for SQLite ID {sqlite_id}: {e}", exc_info=True)


def add_fact_embedding_sync(fact_id: str, history_key: int, fact_type: str, fact_text: str):
    """Синхронная функция добавления эмбеддинга факта (для to_thread)."""
    global model
    if model is None: return
    if not fact_text or not fact_text.strip(): return

    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME)) # Запускаем async в sync
    if facts_collection is None:
        logger.error("Sync: Could not get facts collection. Skipping fact embedding.")
        return

    try:
        logger.debug(f"Sync: Generating fact embedding for Fact ID: {fact_id}")
        with model_lock: embedding = model.encode(fact_text).tolist()
        metadata = {"history_key": history_key, "type": fact_type, "timestamp": time.time()}
        facts_collection.upsert(ids=[fact_id], embeddings=[embedding], metadatas=[metadata], documents=[fact_text])
        logger.debug(f"Sync: Upserted fact embedding for Fact ID {fact_id}.")
    except Exception as e:
        logger.error(f"Sync: Failed to add/upsert fact embedding for Fact ID {fact_id}: {e}", exc_info=True)

# --- Поиск ---

def search_relevant_history_sync(history_key: int, query_text: str, k: int = VECTOR_SEARCH_K_HISTORY) -> List[Tuple[str, Dict[str, Any]]]:
    """Синхронная функция поиска релевантной истории (для to_thread)."""
    global model
    results = []
    if model is None: return results
    if not query_text or not query_text.strip(): return results

    collection = asyncio.run(get_history_collection(history_key)) # Запускаем async в sync
    if collection is None: return results

    try:
        count = collection.count(); effective_k = min(k, count)
        if effective_k <= 0: return results

        logger.debug(f"Sync: Searching history '{collection.name}' for key {history_key} (k={effective_k})")
        with model_lock: query_embedding = model.encode(query_text).tolist()

        query_results = collection.query(
            query_embeddings=[query_embedding], n_results=effective_k,
            include=['documents', 'metadatas', 'distances']
        )

        if query_results and query_results.get('ids') and query_results['ids'][0]:
            num_found = len(query_results['ids'][0]); logger.info(f"Sync: Found {num_found} relevant history messages for key {history_key}.")
            docs = query_results['documents'][0] if query_results.get('documents') else []; metadatas = query_results['metadatas'][0] if query_results.get('metadatas') else []
            for i in range(num_found):
                 doc_text = docs[i] if i < len(docs) else None; metadata = metadatas[i] if i < len(metadatas) else {}
                 if doc_text: results.append((doc_text, metadata))
        else: logger.debug(f"Sync: No relevant history messages found for key {history_key}.")

    except Exception as e: logger.error(f"Sync: Error during ChromaDB history search for key {history_key}: {e}", exc_info=True)
    return results


def search_relevant_facts_sync(history_key: int, query_text: str, k: int = VECTOR_SEARCH_K_FACTS) -> List[Tuple[str, Dict[str, Any]]]:
    """Синхронная функция поиска релевантных фактов (для to_thread)."""
    global model
    results = []
    if model is None: return results
    if not query_text or not query_text.strip(): return results

    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME)) # Запускаем async в sync
    if facts_collection is None: return results

    try:
        count = facts_collection.count(); effective_k = min(k, count)
        if effective_k <= 0: return results

        logger.debug(f"Sync: Searching facts collection for key {history_key} (k={effective_k})")
        with model_lock: query_embedding = model.encode(query_text).tolist()

        query_results = facts_collection.query(
            query_embeddings=[query_embedding], n_results=effective_k,
            where={"history_key": history_key},
            include=['documents', 'metadatas', 'distances'] # Запрашиваем distance
        )
        if query_results and query_results.get('ids') and query_results['ids'][0]:
            num_found = len(query_results['ids'][0]); logger.info(...)
            docs = query_results['documents'][0] if query_results.get('documents') else []
            metadatas = query_results['metadatas'][0] if query_results.get('metadatas') else []
            distances = query_results['distances'][0] if query_results.get('distances') else [] # Получаем расстояния

            for i in range(num_found):
                 doc_text = docs[i] if i < len(docs) else None
                 metadata = metadatas[i] if i < len(metadatas) else {}
                 distance = distances[i] if i < len(distances) else 1.0 # Ставим 1.0 (макс. расстояние для косинуса), если нет
                 if doc_text:
                     results.append((doc_text, metadata, distance))
        else: logger.debug(f"Sync: No relevant facts found for key {history_key}.")

    except Exception as e: logger.error(f"Sync: Error during ChromaDB facts search for key {history_key}: {e}", exc_info=True)
    return results

# --- Удаление эмбеддингов ---
def delete_embeddings_by_sqlite_ids_sync(history_key: int, sqlite_ids: List[int]):
    """Синхронная функция удаления эмбеддингов истории (для to_thread)."""
    if not sqlite_ids: return

    collection = asyncio.run(get_history_collection(history_key)) # Запускаем async в sync
    if collection is None: logger.warning(f"Sync: Cannot delete embeddings: collection for key {history_key} not found."); return

    chroma_ids_to_delete = [str(sid) for sid in sqlite_ids]
    logger.info(f"Sync: Attempting to delete {len(chroma_ids_to_delete)} history embeddings from '{collection.name}'...")
    try:
        if chroma_ids_to_delete: collection.delete(ids=chroma_ids_to_delete)
        logger.info(f"Sync: Sent delete request for history embeddings. New approx count for '{collection.name}': {collection.count()}")
    except Exception as e: logger.warning(f"Sync: Error during ChromaDB history deletion for key {history_key}: {e}")


def delete_fact_embeddings_by_ids_sync(fact_ids: List[str]):
    """Синхронная функция удаления эмбеддингов фактов по их ID (для to_thread)."""
    if not fact_ids: return

    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME)) # Запускаем async в sync
    if facts_collection is None: logger.warning("Sync: Cannot delete fact embeddings: facts collection not found."); return

    logger.info(f"Sync: Attempting to delete {len(fact_ids)} fact embeddings from '{facts_collection.name}'...")
    try:
        if fact_ids: facts_collection.delete(ids=fact_ids)
        logger.info(f"Sync: Sent delete request for fact embeddings. New approx count for facts: {facts_collection.count()}")
    except Exception as e: logger.warning(f"Sync: Error during ChromaDB facts deletion: {e}")


def delete_facts_by_history_key_sync(history_key: int):
    """Синхронная функция удаления ВСЕХ фактов для history_key (для to_thread)."""
    facts_collection = asyncio.run(get_or_create_collection(CHROMA_FACTS_COLLECTION_NAME)) # Запускаем async в sync
    if facts_collection is None: return
    logger.info(f"Sync: Attempting to delete ALL facts for history_key {history_key}...")
    try:
        facts_collection.delete(where={"history_key": history_key})
        logger.info(f"Sync: Sent delete request for facts of key {history_key}. New approx count for facts: {facts_collection.count()}")
    except Exception as e: logger.warning(f"Sync: Error during ChromaDB facts deletion for key {history_key}: {e}")
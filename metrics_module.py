# -*- coding: utf-8 -*-
# metrics_module.py

import asyncio
import time
from collections import deque, defaultdict
from typing import Dict, Any, Optional, Tuple, DefaultDict, Deque, Union, cast
import statistics # Для расчета медианы/перцентилей, если понадобится

# Импортируем логгер и настройки из config
from config import logger, settings

# --- Класс для сбора метрик ---

class Metrics:
    """
    Централизованный класс для сбора и предоставления различных метрик производительности и использования бота.
    Потокобезопасен для использования в асинхронной среде.
    """
    def __init__(self):
        # Структура для API вызовов: {provider: {call_type: {'total_time': float, 'count': int, 'errors': int}}}
        self._api_stats: DefaultDict[str, DefaultDict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {'total_time': 0.0, 'count': 0, 'errors': 0}))

        # Структура для времени обработки сообщений: {'total_proc_time': float, 'message_count': int, 'individual_times': deque}
        self._processing_stats: Dict[str, Union[float, int, Deque[float]]] = {'total_proc_time': 0.0, 'message_count': 0, 'individual_times': deque(maxlen=1000)} # Храним последние 1000 времен для перцентилей

        # Структура для использования команд: {command_name: count}
        self._command_usage: DefaultDict[str, int] = defaultdict(int)

        self._lock = asyncio.Lock() # Единый лок для всех структур
        logger.info("Metrics system initialized.")

    # --- Методы для записи метрик ---

    async def record_api_call(self, provider: str, call_type: str, duration: float, is_error: bool):
        """
        Записывает результат вызова внешнего API (LLM, etc.).

        Args:
            provider: Имя провайдера (e.g., 'gemini', 'mistral', 'telegram_api').
            call_type: Тип вызова (e.g., 'generation', 'vision', 'send_message').
            duration: Время выполнения вызова в секундах.
            is_error: True, если вызов завершился ошибкой.
        """
        async with self._lock:
            stats = self._api_stats[provider][call_type]
            stats['total_time'] += duration
            stats['count'] += 1
            if is_error:
                stats['errors'] += 1
        logger.debug(f"Metric API Call: {provider}/{call_type} - Duration={duration:.3f}s, Error={is_error}")

    async def record_message_processed(self, processing_time: float):
        """
        Записывает время, затраченное на полную обработку одного сообщения пользователя
        (от получения до отправки ответа).

        Args:
            processing_time: Время обработки в секундах.
        """
        async with self._lock:
            self._processing_stats['total_proc_time'] += processing_time
            self._processing_stats['message_count'] += 1
            # Добавляем в deque для расчета перцентилей/медианы
            cast(Deque[float], self._processing_stats['individual_times']).append(processing_time)
        logger.debug(f"Metric Message Processed: Duration={processing_time:.3f}s")

    async def record_command_usage(self, command_name: str):
        """
        Увеличивает счетчик использования для указанной команды.

        Args:
            command_name: Имя команды (e.g., '/start', '/help').
        """
        # Убираем '/' из начала команды для унификации
        clean_command = command_name.lstrip('/')
        async with self._lock:
            self._command_usage[clean_command] += 1
        logger.debug(f"Metric Command Usage: /{clean_command}")

    # --- Методы для получения статистики ---

    async def get_api_stats(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Возвращает статистику по вызовам API.

        Returns:
            Словарь вида: {provider: {call_type: {avg_time, count, errors, error_rate}}}
        """
        result: Dict[str, Dict[str, Dict[str, float]]] = {}
        async with self._lock:
            # Глубокое копирование для безопасных расчетов
            stats_copy = {p: {ct: d.copy() for ct, d in ct_data.items()} for p, ct_data in self._api_stats.items()}

        for provider, call_types_data in stats_copy.items():
             result[provider] = {}
             for call_type, data in call_types_data.items():
                 count = data.get('count', 0)
                 total_time = data.get('total_time', 0.0)
                 errors = data.get('errors', 0)
                 avg_time = (total_time / count) if count > 0 else 0.0
                 error_rate = (errors / count * 100) if count > 0 else 0.0
                 result[provider][call_type] = {
                     'avg_time': round(avg_time, 3),
                     'count': int(count),
                     'errors': int(errors),
                     'error_rate': round(error_rate, 2)
                 }
        return result

    async def get_processing_stats(self) -> Dict[str, Union[float, int]]:
        """
        Возвращает статистику по времени обработки сообщений.

        Returns:
            Словарь вида: {avg_time, count, median_time, p95_time}
        """
        async with self._lock:
            count = int(self._processing_stats['message_count'])
            total_time = float(self._processing_stats['total_proc_time'])
            # Копируем individual_times для безопасных расчетов
            individual_times = list(cast(Deque[float], self._processing_stats['individual_times']))

        avg_time = (total_time / count) if count > 0 else 0.0
        median_time = 0.0
        p95_time = 0.0

        if individual_times:
             try:
                 median_time = statistics.median(individual_times)
                 # Рассчитываем 95-й перцентиль
                 individual_times.sort()
                 p95_index = int(len(individual_times) * 0.95)
                 p95_time = individual_times[min(p95_index, len(individual_times) - 1)] # Берем индекс или последний элемент
             except statistics.StatisticsError:
                 logger.warning("Not enough data points for median/percentile calculation.")
             except IndexError:
                  logger.warning("Index error during percentile calculation.")


        return {
            'avg_time': round(avg_time, 3),
            'count': count,
            'median_time': round(median_time, 3),
            'p95_time': round(p95_time, 3)
        }

    async def get_command_stats(self) -> Dict[str, int]:
        """
        Возвращает статистику использования команд.

        Returns:
            Словарь вида: {command_name: count}
        """
        async with self._lock:
            # Возвращаем копию словаря
            return self._command_usage.copy()

    async def get_all_stats(self) -> Dict[str, Any]:
        """
        Возвращает всю собранную статистику в одном словаре.
        """
        api_stats, proc_stats, cmd_stats = await asyncio.gather(
            self.get_api_stats(),
            self.get_processing_stats(),
            self.get_command_stats()
        )
        return {
            'api_calls': api_stats,
            'message_processing': proc_stats,
            'command_usage': cmd_stats
        }

    async def reset(self):
        """Сбрасывает всю собранную статистику."""
        async with self._lock:
            self._api_stats.clear()
            self._processing_stats = {'total_proc_time': 0.0, 'message_count': 0, 'individual_times': deque(maxlen=1000)}
            self._command_usage.clear()
        logger.info("All metrics stats have been reset.")

# --- Класс для простого кэша ---

class SimpleCache:
    """
    Простой асинхронный кэш в памяти с ограничением размера и статистикой.
    Использует LRU-подобную стратегию вытеснения (при доступе элемент перемещается в конец).
    """
    def __init__(self, maxsize: int = 256):
        if maxsize <= 0:
            raise ValueError("Cache maxsize must be positive")
        self.maxsize = maxsize
        self._cache: Dict[Any, Any] = {}
        # deque хранит ключи в порядке от самого "старого" (давно не использованного) к новому
        self._keys: Deque[Any] = deque()
        self._hits: int = 0
        self._misses: int = 0
        self._lock = asyncio.Lock()
        logger.info(f"SimpleCache initialized with maxsize={maxsize}.")

    async def get(self, key: Any) -> Optional[Any]:
        """Получает значение из кэша. Перемещает ключ в конец при попадании (LRU)."""
        async with self._lock:
            value = self._cache.get(key, None)
            if value is not None:
                self._hits += 1
                # Перемещаем ключ в конец очереди (самый свежий)
                if key in self._keys: # Должен быть, но проверим
                    self._keys.remove(key)
                    self._keys.append(key)
                logger.debug(f"Cache HIT for key: {str(key)[:50]}...")
                return value
            else:
                self._misses += 1
                logger.debug(f"Cache MISS for key: {str(key)[:50]}...")
                return None

    async def set(self, key: Any, value: Any):
        """Добавляет или обновляет значение в кэше. Вытесняет самый старый при переполнении."""
        async with self._lock:
            if key in self._cache:
                # Обновляем значение
                self._cache[key] = value
                # Перемещаем ключ в конец очереди (самый свежий)
                self._keys.remove(key)
                self._keys.append(key)
            else:
                # Новый ключ
                # Проверяем размер *перед* добавлением
                if len(self._cache) >= self.maxsize:
                    # Вытесняем самый старый элемент (из начала deque)
                    oldest_key = self._keys.popleft()
                    if oldest_key in self._cache:
                        del self._cache[oldest_key]
                        logger.debug(f"Cache evicted (LRU): {str(oldest_key)[:50]}...")
                    else: # На всякий случай, если _keys и _cache рассинхронизированы
                         logger.warning(f"Cache inconsistency: oldest key {oldest_key} not found in cache during eviction.")

                # Добавляем новый элемент в кэш и в конец очереди
                self._cache[key] = value
                self._keys.append(key)
            logger.debug(f"Cache SET for key: {str(key)[:50]}...")

    async def clear(self):
        """Очищает весь кэш и статистику."""
        async with self._lock:
            self._cache.clear()
            self._keys.clear()
            self._hits = 0
            self._misses = 0
        logger.info("Cache cleared.")

    async def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику кэша."""
        async with self._lock:
            # Рассчитываем hit rate
            total_accesses = self._hits + self._misses
            hit_rate = (self._hits / total_accesses * 100) if total_accesses > 0 else 0.0
            stats = {
                'hits': self._hits,
                'misses': self._misses,
                'total_accesses': total_accesses,
                'hit_rate': round(hit_rate, 2),
                'size': len(self._cache),
                'maxsize': self.maxsize
            }
        logger.debug("Retrieved cache stats.")
        return stats

    @property
    async def size(self) -> int:
        """Текущий размер кэша."""
        async with self._lock: return len(self._cache)

# --- Глобальные экземпляры ---
# Инициализируем метрики и кэш при загрузке модуля
metrics = Metrics()
# Размер кэша берем из настроек, если есть, иначе значение по умолчанию
cache_maxsize = getattr(settings, 'SIMPLE_CACHE_MAXSIZE', 512) # Увеличил дефолт
cache = SimpleCache(maxsize=cache_maxsize)

# --- Функции-обертки для удобства записи метрик ---

async def record_api_call_metric(provider: str, call_type: str, start_time: float, is_error: bool):
    """Хелпер для записи метрики API вызова."""
    duration = time.monotonic() - start_time
    await metrics.record_api_call(provider, call_type, duration, is_error)

# Пример использования этой обертки:
# start = time.monotonic()
# error_flag = False
# try:
#     response = await some_api_call(...)
# except:
#     error_flag = True
# finally:
#     await record_api_call_metric('provider_name', 'call_type', start, error_flag)


async def record_message_processing_metric(start_time: float):
    """Хелпер для записи метрики времени обработки сообщения."""
    duration = time.monotonic() - start_time
    await metrics.record_message_processed(duration)

# Пример использования:
# async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     start_proc_time = time.monotonic()
#     # ... вся логика обработки ...
#     # В самом конце, после отправки ответа (или его планирования)
#     await record_message_processing_metric(start_proc_time)
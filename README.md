```markdown
# Маша v2 - Telegram Чат-Бот на Gemini и ChromaDB

Этот проект представляет собой многофункционального Telegram чат-бота по имени **Маша**, построенного с использованием Python, библиотеки `python-telegram-bot`, большой языковой модели Google Gemini и векторной базы данных ChromaDB для управления историей диалогов.

**Имя бота:** Маша (настраиваемое через `.env`)

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Если вы выберете MIT лицензию -->
<!-- [![Issues](https://img.shields.io/github/issues/ByteBudda/mashav2)](https://github.com/ByteBudda/mashav2/issues) -->
<!-- [![Forks](https://img.shields.io/github/forks/ByteBudda/mashav2)](https://github.com/ByteBudda/mashav2/network/members) -->
<!-- [![Stars](https://img.shields.io/github/stars/ByteBudda/mashav2)](https://github.com/ByteBudda/mashav2/stargazers) -->

## 🌟 Возможности

*   **Разговорный ИИ:** Использует Google Gemini (`gemini-2.0-flash-latest` по умолчанию) для генерации осмысленных и контекстно-зависимых ответов в заданном стиле.
*   **Семантическая Память:** Применяет векторную базу данных ChromaDB для хранения и поиска релевантной истории диалогов, позволяя боту поддерживать более длительные и связные разговоры.
*   **Обработка Разных Типов Сообщений:**
    *   Текстовые сообщения
    *   Фотографии (с использованием Gemini Vision для описания и реакции)
    *   Голосовые сообщения (автоматическое распознавание речи через Google Speech Recognition)
    *   Видео-сообщения ("кружочки") (извлечение аудио и распознавание речи)
*   **Анализ Текста:** Использует RuBERT (через `transformers`) для:
    *   Распознавания именованных сущностей (NER)
    *   Определения тональности сообщений
*   **Настраиваемость:**
    *   Изменение имени бота (`/setbotname`, админ.)
    *   Настройка глобального стиля общения (`/setdefaultstyle`, админ.)
    *   Установка индивидуального стиля для пользователя в группе (`/setgroupuserstyle`, админ.)
    *   Настройка "запоминания" фактов о пользователе (`/remember`)
    *   Управление активностью бота в группах (`/setactivity`, админ.)
    *   Установка предпочитаемого имени для пользователя (`/setmyname`)
*   **Управление Историей:**
    *   Пользователи могут очистить свою историю (`/clear_my_history`)
    *   Администраторы могут очистить историю любого пользователя (`/clear_history`)
    *   Автоматическая очистка старых записей в ChromaDB по истечении TTL (`HISTORY_TTL`).
    *   Сброс контекста текущего диалога (`/reset_context`).
*   **Администрирование:** Набор команд для управления ботом (список админов задается в `.env`).
*   **Логирование:** Подробное логирование действий и ошибок в файл `bot.log` с ротацией.
*   **Сохранение Состояния:** Основные настройки, информация о пользователях (кроме истории ChromaDB) сохраняются в JSON файлы (`learned_knowledge.json`, `user_data/`) для персистентности между перезапусками.

## 🛠️ Установка и Запуск

### Предварительные требования

*   Python 3.12 или выше
*   `pip` (менеджер пакетов Python)
*   `ffmpeg` (для обработки аудио/видео)
    *   **Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (Homebrew):** `brew install ffmpeg`
    *   **Windows:** Скачайте с [официального сайта](https://ffmpeg.org/download.html) и добавьте в системную переменную PATH.

### Установка зависимостей

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/ByteBudda/mashav2.git
    cd mashav2
    ```

2.  **(Рекомендуется) Создайте и активируйте виртуальное окружение:**
    ```bash
    python3 -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows (cmd):
    # venv\Scripts\activate.bat
    # Windows (PowerShell):
    # venv\Scripts\Activate.ps1
    ```

3.  **Установите зависимости:**
    ```bash
    pip install --upgrade pip
    # Установите зависимости из файла, если он есть:
    # pip install -r requirements.txt
    # Или установите вручную:
    pip install python-telegram-bot google-generativeai Pillow python-dotenv chromadb sentence-transformers pydub speechrecognition transformers torch torchvision torchaudio accelerate jsonschema
    ```
    *(Установка `torch` может потребовать специфичной команды для вашей системы/CUDA, см. [pytorch.org](https://pytorch.org/))*

### Конфигурация

1.  **Создайте файл `.env`** в корневой директории проекта.
2.  **Заполните файл `.env` вашими данными:**
    ```dotenv
    TELEGRAM_BOT_TOKEN=ВАШ_ТЕЛЕГРАМ_ТОКЕН
    GEMINI_API_KEY=ВАШ_GOOGLE_GEMINI_API_KEY
    ADMIN_IDS=ID_АДМИН_1,ID_АДМИН_2 # Через запятую, без пробелов

    # --- Опциональные настройки (можно оставить по умолчанию) ---
    DEFAULT_STYLE="Ты - Маша, 25-летняя девушка из Москвы. Ты очень веселая и раскрепощенная. Твой стиль общения непринужденный, эмоциональный, ты не стесняешься в выражениях и можешь использовать любую лексику, включая нецензурную если она уместна. Отвечай от первого лица, как будто ты - Маша. Подстраивайся под стиль общения собеседника"
    BOT_NAME=Маша
    HISTORY_TTL=86400 # Время жизни истории в секундах (1 день = 86400)
    CHROMA_DATA_PATH=./chroma_db_data # Путь для хранения базы ChromaDB
    MAX_HISTORY_RESULTS=15 # Сколько релевантных сообщений извлекать из ChromaDB
    MAX_HISTORY_TOKENS=2000 # Максимальное количество токенов истории для промпта
    GEMINI_MODEL_NAME=gemini-2.0-flash-latest # Модель Gemini
    ```

### Запуск бота

```bash
python main.py
```

Бот начнет работу. Логи будут выводиться в консоль и записываться в файл `bot.log`. Для остановки бота нажмите `Ctrl+C`.

## 🚀 Использование

Найдите вашего бота в Telegram по имени пользователя (username), которое вы установили при создании бота через @BotFather, и начните с ним диалог.

Используйте команду `/help`, чтобы увидеть список доступных команд.

## ⚙️ Структура Проекта

```
.
├── .env                   # Файл с секретами и конфигурацией 
├── bot.log                # Файл логов (создается автоматически)
├── learned_knowledge.json # Файл для сохранения общих данных бота
├── chroma_db_data/        # Директория для хранения данных ChromaDB
│   └── ...                # (создается и управляется ChromaDB)
├── user_data/             # Директория для хранения данных пользователей
│   └── user_12345.json    # Пример файла данных пользователя
├── config.py              # Загрузка настроек, константы, логгер
├── vector_store.py        # Модуль для взаимодействия с ChromaDB
├── state.py               # Управление состоянием бота (словари в памяти)
├── utils.py               # Вспомогательные функции (AI, аудио, PromptBuilder)
├── bot_commands.py        # Обработчики команд Telegram (/start, /help, и т.д.)
├── handlers.py            # Обработчики сообщений (текст, фото, голос, видео)
└── main.py                # Точка входа, инициализация, запуск бота
```

## 🤝 Вклад

Если вы хотите улучшить бота, исправить ошибки или добавить новые функции:

1.  Сделайте форк репозитория.
2.  Создайте новую ветку для ваших изменений (`git checkout -b feature/новая-фича`).
3.  Внесите изменения и сделайте коммиты (`git commit -am 'Добавлена новая фича'`).
4.  Отправьте изменения в ваш форк (`git push origin feature/новая-фича`).
5.  Создайте Pull Request в основной репозиторий.

Буду рад рассмотреть ваши предложения!

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. Вы можете свободно использовать, изменять и распространять код в соответствии с условиями лицензии. (Рекомендую добавить файл `LICENSE` с текстом [MIT License](https://opensource.org/licenses/MIT) в корень репозитория).

## 🙏 Благодарности

*   Команде [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
*   Google за [Gemini API](https://ai.google.dev/)
*   Разработчикам [ChromaDB](https://www.trychroma.com/)
*   Сообществу [Hugging Face](https://huggingface.co/) за модели (`sentence-transformers`, `RuBERT`)
*   Библиотекам `pydub`, `SpeechRecognition`

---

*Приятного использования и доработки Маши!* 😉

# RAG (YandexGPT + FAISS) + логи SQLite

**Сдача / отчёт:** репозиторий [PEcf09](https://github.com/samsa2846-sys/PEcf09) — текст отчёта для преподавателя: [`ОТЧЕТ.md`](ОТЧЕТ.md).

## Важно: версия Python

Используйте **Python 3.11–3.13**. Для **Python 3.14** пакет `faiss-cpu` пока без колёс под Windows.

Рекомендуется venv на **3.11** (как в `setup_venv.ps1`).

## Быстрый старт (Windows)

```powershell
cd c:\Cursor\DZ_2\rag-yandex-assistant
.\setup_venv.ps1
```

Активация (обратите внимание: папка **`Scripts`**, не `Script`):

```powershell
.\venv\Scripts\Activate.ps1
```

Если ругается на политику выполнения:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Скопируйте `.env.example` в `.env` и вставьте `YANDEX_API_KEY` и `YANDEX_FOLDER_ID`.

```powershell
python app.py
```

Команды в консоли: `stats`, `export` (CSV), `clear`, `exit`.

## Нестабильный интернет / pip

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt --default-timeout=300 --retries 10
```

## Telegram

В `.env`: `TELEGRAM_BOT_TOKEN=...`, затем `python telegram_bot.py`.

## Файлы

| Файл | Назначение |
|------|------------|
| `logs.db` | Логи взаимодействий |
| `yandexgpt_rag_cache.db` | Кеш ответов |
| `faiss_storage/` | Индекс FAISS |

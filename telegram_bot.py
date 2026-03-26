"""
Telegram-обёртка над RAG: те же логи в SQLite, user_id / username из Telegram.
Запуск: установите TELEGRAM_BOT_TOKEN в .env, затем python telegram_bot.py
"""
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from db_logger import DatabaseLogger
from rag_pipeline import RAGPipeline

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

_pipeline: Optional[RAGPipeline] = None
_logger: Optional[DatabaseLogger] = None


def _context_snippet(result: dict) -> str:
    docs = result.get("context_docs") or []
    parts = []
    for d in docs[:3]:
        if isinstance(d, dict):
            parts.append((d.get("text") or "")[:100])
        else:
            parts.append(str(d)[:100])
    return " | ".join(parts)


def _build_source(result: dict) -> str:
    if result.get("from_cache"):
        return "cache"
    return f"yandexgpt+faiss:{result.get('model', 'llm')}"


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            collection_name="yandexgpt_rag_collection",
            cache_db_path="yandexgpt_rag_cache.db",
            data_file="data/docs.txt",
            model="yandexgpt-lite/latest",
        )
    return _pipeline


def get_logger() -> DatabaseLogger:
    global _logger
    if _logger is None:
        _logger = DatabaseLogger(db_path=os.getenv("LOG_DB_PATH", "logs.db"))
    return _logger


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    m = update.effective_message
    if m:
        await m.reply_text("Задайте вопрос текстом. Команда /stats — статистика.")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    m = update.effective_message
    if not m:
        return
    p = get_pipeline()
    log = get_logger()
    st = p.get_stats()
    ls = log.get_stats()
    text = (
        f"Векторы: {st['vector_store']['count']} чанков\n"
        f"Кеш записей: {st['cache']['total_entries']}\n"
        f"Логов: {ls['total_interactions']}, ошибок: {ls['error_rows']}\n"
        f"Среднее мс: {ls['avg_response_time_ms']}"
    )
    await m.reply_text(text)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    q = update.message.text.strip()
    if not q:
        return
    uid = str(update.effective_user.id) if update.effective_user else "unknown"
    uname = (update.effective_user.username or update.effective_user.first_name or "user")[:200]
    p = get_pipeline()
    log = get_logger()
    t0 = time.perf_counter()
    try:
        result = p.query(q)
        ms = int((time.perf_counter() - t0) * 1000)
        log.log_interaction(
            query=q,
            response=result.get("answer") or "",
            source=_build_source(result),
            user_id=uid,
            username=uname,
            from_cache=bool(result.get("from_cache")),
            response_time_ms=ms,
            context_snippet=_context_snippet(result),
        )
        await update.message.reply_text(result.get("answer") or "")
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        log.log_error(q, str(e)[:2000], uid, uname, ms)
        await update.message.reply_text(f"Ошибка: {e}")


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Задайте TELEGRAM_BOT_TOKEN в .env")
    if not os.getenv("YANDEX_API_KEY") or not os.getenv("YANDEX_FOLDER_ID"):
        raise SystemExit("Нужны YANDEX_API_KEY и YANDEX_FOLDER_ID")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print("[OK] Бот запущен. Ctrl+C — остановка.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

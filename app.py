"""
Консольное приложение: RAG (YandexGPT) + логирование в SQLite.
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from db_logger import DatabaseLogger
from rag_pipeline import RAGPipeline

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

CONSOLE_USER_ID = "console"
CONSOLE_USERNAME = "local"


def _context_snippet(result: dict) -> str:
    docs = result.get("context_docs") or []
    if not docs:
        return ""
    parts = []
    for d in docs[:3]:
        if isinstance(d, dict):
            t = (d.get("text") or "")[:120]
        else:
            t = str(d)[:120]
        parts.append(t)
    return " | ".join(parts)


def _build_source(result: dict) -> str:
    if result.get("from_cache"):
        return "cache"
    return f"yandexgpt+faiss:{result.get('model', 'llm')}"


def print_banner():
    banner = """
============================================================
         RAG Ассистент (YandexGPT Mode)                 
  Retrieval-Augmented Generation через YandexGPT API    
============================================================
    """
    print(banner)
    print("Введите 'exit' или 'quit' для выхода")
    print("Введите 'stats' — статистика пайплайна и логов")
    print("Введите 'export' — выгрузить логи в CSV")
    print("Введите 'clear' для очистки кеша\n")


def print_response(result: dict):
    print(f"\n{'-'*60}")
    print(f"Вопрос: {result['query']}")
    print(f"{'-'*60}")
    if result["from_cache"]:
        print("[CACHE] Источник: КЕШ")
        if "cached_at" in result:
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"[API] Источник: YandexGPT API ({result.get('model', 'LLM')})")
        print(f"   Использовано документов: {len(result.get('context_docs', []))}")
    print(f"\nОтвет:\n{result['answer']}")
    if result.get("context_docs"):
        print("\nИспользованный контекст:")
        for i, doc in enumerate(result["context_docs"][:2], 1):
            text = doc["text"] if isinstance(doc, dict) else str(doc)
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"   {i}. {preview}")
    print(f"{'-'*60}\n")


def print_stats(pipeline: RAGPipeline, logger: DatabaseLogger):
    stats = pipeline.get_stats()
    print(f"\n{'='*60}")
    print("СТАТИСТИКА СИСТЕМЫ")
    print(f"{'='*60}")
    print("\nВекторное хранилище:")
    print(f"   Коллекция: {stats['vector_store']['name']}")
    print(f"   Документов: {stats['vector_store']['count']}")
    print(f"   Директория: {stats['vector_store']['persist_directory']}")
    print("\nКеш:")
    print(f"   Записей: {stats['cache']['total_entries']}")
    print(f"   Размер БД: {stats['cache']['db_size_mb']:.2f} MB")
    if stats["cache"]["oldest_entry"]:
        print(f"   Первая запись: {stats['cache']['oldest_entry']}")
    if stats["cache"]["newest_entry"]:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")
    print(f"\nМодель: {stats['model']}")
    print(f"Режим: {stats['mode']}")
    ls = logger.get_stats()
    print("\nЛоги (interactions):")
    print(f"   Всего записей: {ls['total_interactions']}")
    print(f"   Из кеша (по логам): {ls['cache_hits_logged']}")
    print(f"   Среднее время ответа (мс): {ls['avg_response_time_ms']}")
    print(f"   Строк с ошибкой: {ls['error_rows']}")
    print(f"{'='*60}\n")


def main():
    print_banner()
    if not os.getenv("YANDEX_API_KEY"):
        print("[ERROR] YANDEX_API_KEY не установлена")
        sys.exit(1)
    if not os.getenv("YANDEX_FOLDER_ID"):
        print("[ERROR] YANDEX_FOLDER_ID не установлена")
        sys.exit(1)

    log_db = os.getenv("LOG_DB_PATH", "logs.db")
    logger = DatabaseLogger(db_path=log_db)

    try:
        print("[INIT] Инициализация системы...\n")
        pipeline = RAGPipeline(
            collection_name="yandexgpt_rag_collection",
            cache_db_path="yandexgpt_rag_cache.db",
            data_file="data/docs.txt",
            model="yandexgpt-lite/latest",
        )
        print("\n[OK] Система готова к работе!\n")
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    while True:
        try:
            user_input = input("Ваш вопрос: ").strip()
            low = user_input.lower()
            if low in ("exit", "quit", "q"):
                print("\nДо свидания!")
                break
            if low == "stats":
                print_stats(pipeline, logger)
                continue
            if low == "export":
                out = logger.export_csv("logs_export.csv")
                print(f"[OK] Экспорт: {out}\n")
                continue
            if low == "clear":
                confirm = input("[WARNING] Очистить кеш? (yes/no): ")
                if confirm.lower() in ("yes", "y", "да"):
                    pipeline.cache.clear()
                    print("[OK] Кеш очищен")
                continue
            if not user_input:
                print("[WARNING] Введите вопрос\n")
                continue

            t0 = time.perf_counter()
            try:
                result = pipeline.query(user_input)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                logger.log_interaction(
                    query=user_input,
                    response=result.get("answer") or "",
                    source=_build_source(result),
                    user_id=CONSOLE_USER_ID,
                    username=CONSOLE_USERNAME,
                    from_cache=bool(result.get("from_cache")),
                    response_time_ms=elapsed_ms,
                    context_snippet=_context_snippet(result),
                )
                print_response(result)
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                logger.log_error(
                    query=user_input,
                    error_message=str(e)[:2000],
                    user_id=CONSOLE_USER_ID,
                    username=CONSOLE_USERNAME,
                    response_time_ms=elapsed_ms,
                )
                print(f"\n[ERROR] Ошибка: {e}\n")
                import traceback

                traceback.print_exc()
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До свидания!")
            break


if __name__ == "__main__":
    main()

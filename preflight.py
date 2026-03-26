"""Проверка venv, импортов и .env (без вывода секретов)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    print("Python:", sys.version.split()[0])
    root = Path(__file__).resolve().parent
    os.chdir(root)

    try:
        import faiss  # noqa: F401
        import numpy  # noqa: F401
        import requests  # noqa: F401
        from dotenv import load_dotenv  # noqa: F401
        import telegram  # noqa: F401
    except ImportError as e:
        print("[FAIL] Импорт:", e)
        return 1
    print("[OK] Зависимости импортируются")

    load_dotenv(root / ".env")
    key = os.getenv("YANDEX_API_KEY", "").strip()
    folder = os.getenv("YANDEX_FOLDER_ID", "").strip()
    placeholder = "your-" in key.lower() or "your-" in folder.lower()
    if not key or not folder or placeholder or len(key) < 12:
        print("[WARN] В .env нет реальных YANDEX_API_KEY / YANDEX_FOLDER_ID — app.py не запустится.")
        print("        Откройте .env и вставьте ключи из Yandex Cloud.")
        return 2

    print("[OK] Переменные Yandex заданы (значения не показываются)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

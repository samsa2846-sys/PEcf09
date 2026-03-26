"""SQLite-логирование взаимодействий RAG (урок 5–7)."""
import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatabaseLogger:
    """Логи без ПДн: user_id/username обезличены (console / telegram id)."""

    def __init__(self, db_path: str = "logs.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT,
                source TEXT NOT NULL,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                from_cache INTEGER NOT NULL DEFAULT 0,
                response_time_ms INTEGER,
                error TEXT,
                context_snippet TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _shorten(text: str, max_len: int = 8000) -> str:
        if text is None:
            return ""
        text = str(text)
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def log_interaction(
        self,
        query: str,
        response: str,
        source: str,
        user_id: str,
        username: str,
        from_cache: bool,
        response_time_ms: int,
        context_snippet: str = "",
        error: Optional[str] = None,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO interactions (
                created_at, query, response, source, user_id, username,
                from_cache, response_time_ms, error, context_snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat() + "Z",
                self._shorten(query, 4000),
                self._shorten(response or "", 8000) if response is not None else "",
                source[:200],
                user_id[:128],
                username[:256],
                1 if from_cache else 0,
                int(response_time_ms),
                self._shorten(error, 2000) if error else None,
                self._shorten(context_snippet, 1500),
            ),
        )
        conn.commit()
        conn.close()

    def log_error(
        self,
        query: str,
        error_message: str,
        user_id: str,
        username: str,
        response_time_ms: int = 0,
    ) -> None:
        self.log_interaction(
            query=query,
            response="",
            source="error",
            user_id=user_id,
            username=username,
            from_cache=False,
            response_time_ms=response_time_ms,
            error=error_message,
        )

    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM interactions")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM interactions WHERE from_cache = 1")
        cache_hits = cur.fetchone()[0]
        cur.execute("SELECT AVG(response_time_ms) FROM interactions WHERE error IS NULL")
        row = cur.fetchone()
        avg_ms = float(row[0]) if row[0] is not None else 0.0
        cur.execute("SELECT COUNT(*) FROM interactions WHERE error IS NOT NULL")
        errors = cur.fetchone()[0]
        conn.close()
        return {
            "total_interactions": total,
            "cache_hits_logged": cache_hits,
            "avg_response_time_ms": round(avg_ms, 2),
            "error_rows": errors,
        }

    def export_csv(self, path: str) -> str:
        out = Path(path)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, created_at, query, response, source, user_id, username,
                   from_cache, response_time_ms, error, context_snippet
            FROM interactions ORDER BY id
            """
        )
        rows: List[sqlite3.Row] = cur.fetchall()
        conn.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            cols = [
                "id",
                "created_at",
                "query",
                "response",
                "source",
                "user_id",
                "username",
                "from_cache",
                "response_time_ms",
                "error",
                "context_snippet",
            ]
            if not rows:
                csv.writer(f).writerow(cols)
            else:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(dict(r))
        return str(out.resolve())

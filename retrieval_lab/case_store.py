"""Durable case and reviewer feedback storage for evidence verification."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchCaseStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    claim TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    reviewer TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reviews_case_idx ON reviews(case_id, review_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def create(self, claim: str, result: Mapping[str, Any]) -> dict[str, Any]:
        case_id = uuid.uuid4().hex
        now = _now()
        status = "needs_review" if result.get("action") == "abstain" or result.get("label") == "NEI" or result.get("review_context", {}).get("errors") else "open"
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, claim, json.dumps(dict(result), ensure_ascii=False), status, now, now),
            )
        return self.get(case_id)

    def get(self, case_id: str) -> dict[str, Any]:
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            reviews = db.execute(
                "SELECT reviewer, verdict, note, created_at FROM reviews WHERE case_id = ? ORDER BY review_id",
                (case_id,),
            ).fetchall()
        fields = dict(row)
        fields.pop("result_json")
        return {
            **fields,
            "result": json.loads(row["result_json"]),
            "reviews": [dict(item) for item in reviews],
        }

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT case_id FROM cases ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [self.get(row["case_id"]) for row in rows]

    def review(self, case_id: str, reviewer: str, verdict: str, note: str) -> dict[str, Any]:
        now = _now()
        with closing(self._connect()) as db, db:
            if db.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,)).fetchone() is None:
                raise KeyError(case_id)
            db.execute(
                "INSERT INTO reviews(case_id, reviewer, verdict, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (case_id, reviewer, verdict, note, now),
            )
            db.execute("UPDATE cases SET status = 'reviewed', updated_at = ? WHERE case_id = ?", (now, case_id))
        return self.get(case_id)

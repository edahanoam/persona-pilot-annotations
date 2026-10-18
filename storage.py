from __future__ import annotations

import json
import os
import threading
import uuid
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
RESULTS_PATH = Path(os.environ.get("RESULTS_PATH", BASE_DIR / "data" / "responses.jsonl"))
_write_lock = threading.Lock()


class StorageError(RuntimeError):
    """Raised when persistent response storage is unavailable."""


def _supabase_settings() -> tuple[str, str] | None:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
    try:
        import streamlit as st

        section = st.secrets.get("supabase", {})
        url = str(section.get("url", url)).strip()
        key = str(section.get("secret_key", key)).strip()
    except Exception:
        pass

    if bool(url) != bool(key):
        raise StorageError(
            "Supabase storage is only partially configured. Both url and "
            "secret_key are required."
        )
    return (url, key) if url and key else None


@lru_cache(maxsize=1)
def _supabase_client(url: str, key: str) -> Any:
    try:
        from supabase import create_client
    except ModuleNotFoundError as exc:
        raise StorageError(
            "Supabase storage is configured, but the 'supabase' package is missing"
        ) from exc
    return create_client(url, key)


def storage_backend_name() -> str:
    return "supabase" if _supabase_settings() else "jsonl"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        raw_questions = json.load(handle)

    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("questions.json must contain a non-empty JSON array")

    questions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object")
        required = ("id", "question", "description")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise ValueError(
                f"Question {index} must have non-empty string fields: {', '.join(required)}"
            )
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate question id: {item['id']}")
        seen_ids.add(item["id"])
        questions.append({key: item[key].strip() for key in required})

    return questions


def save_response(
    *,
    participant_id: str,
    question: dict[str, str],
    answer: str,
    sequence: int,
    path: Path = RESULTS_PATH,
    experiment: str | None = None,
    persona_id: str | None = None,
    sampling_strategy: str | None = None,
    assignment_id: str | None = None,
    choices: list[str] | None = None,
) -> dict[str, Any]:
    response_id = (
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"persona-pilot:{participant_id}:{experiment}:{persona_id}:{question['id']}",
            )
        )
        if experiment and persona_id
        else str(uuid.uuid4())
    )
    record: dict[str, Any] = {
        "response_id": response_id,
        "participant_id": participant_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "sequence": sequence,
        "question_id": question["id"],
        "question": question["question"],
        "description": question["description"],
        "answer": answer,
    }
    if experiment is not None:
        record["experiment"] = experiment
    if persona_id is not None:
        record["persona_id"] = persona_id
    if sampling_strategy is not None:
        record["sampling_strategy"] = sampling_strategy
    if assignment_id is not None:
        record["assignment_id"] = assignment_id
    if choices is not None:
        record["choices"] = choices

    settings = _supabase_settings() if path == RESULTS_PATH else None
    if settings:
        try:
            (
                _supabase_client(*settings)
                .table("responses")
                .upsert(
                    record,
                    on_conflict="participant_id,experiment,persona_id,question_id",
                )
                .execute()
            )
        except Exception as exc:
            raise StorageError(f"Supabase could not save the response: {exc}") from exc
        return record

    line = json.dumps(record, ensure_ascii=False) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return record


def load_completed_question_ids(
    participant_id: str,
    *,
    experiment: str,
    persona_id: str,
    path: Path = RESULTS_PATH,
) -> set[str]:
    """Return saved question IDs so an interrupted survey can resume."""
    settings = _supabase_settings() if path == RESULTS_PATH else None
    if settings:
        try:
            result = (
                _supabase_client(*settings)
                .table("responses")
                .select("question_id")
                .eq("participant_id", participant_id)
                .eq("experiment", experiment)
                .eq("persona_id", persona_id)
                .execute()
            )
        except Exception as exc:
            raise StorageError(f"Supabase could not load survey progress: {exc}") from exc
        return {
            row["question_id"]
            for row in (result.data or [])
            if isinstance(row.get("question_id"), str)
        }

    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("participant_id") == participant_id
                and record.get("experiment") == experiment
                and record.get("persona_id") == persona_id
            ):
                question_id = record.get("question_id")
                if isinstance(question_id, str):
                    completed.add(question_id)
    return completed

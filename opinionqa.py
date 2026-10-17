"""OpinionQA sample loading and participant assignment."""

from __future__ import annotations

import ast
import csv
import hashlib
import os
from pathlib import Path
from typing import TypedDict

from sampling import Sampler, StableHashSampler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_PATH = BASE_DIR / "opinionqa_sample.csv"
REQUIRED_COLUMNS = {
    "persona_description",
    "id",
    "question",
    "question_identifier",
    "options",
    "answer",
}


class OpinionAssignment(TypedDict):
    assignment_id: str
    persona_id: str
    persona: str
    question_id: str
    question: str
    choices: list[str]


def _parse_choices(raw: str, row_number: int) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"OpinionQA row {row_number} has invalid options") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"OpinionQA row {row_number} options must be a list")
    choices = [str(choice).strip() for choice in parsed]
    if not choices or any(not choice for choice in choices):
        raise ValueError(f"OpinionQA row {row_number} has an empty option")
    if len(set(choices)) != len(choices):
        raise ValueError(f"OpinionQA row {row_number} has duplicate options")
    return choices


def load_opinion_assignments(path: Path | None = None) -> list[OpinionAssignment]:
    configured = os.environ.get("OPINIONQA_SAMPLE_PATH")
    source = path or (Path(configured) if configured else DEFAULT_SAMPLE_PATH)
    if not source.exists():
        raise FileNotFoundError(f"OpinionQA sample was not found at {source}")

    assignments: list[OpinionAssignment] = []
    seen_pairs: set[tuple[str, str]] = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "OpinionQA CSV is missing columns: " + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            persona_id = row["id"].strip()
            persona = row["persona_description"].strip()
            question_id = row["question_identifier"].strip()
            question = row["question"].strip()
            if not persona_id or not persona or not question_id or not question:
                raise ValueError(f"OpinionQA row {row_number} has a blank required value")
            pair = (persona_id, question_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"OpinionQA row {row_number} duplicates persona/question pair {pair}"
                )
            seen_pairs.add(pair)
            assignment_key = f"{persona_id}\x1f{question_id}"
            assignment_id = hashlib.sha256(assignment_key.encode("utf-8")).hexdigest()[:20]
            assignments.append(
                {
                    "assignment_id": assignment_id,
                    "persona_id": persona_id,
                    "persona": persona,
                    "question_id": question_id,
                    "question": question,
                    "choices": _parse_choices(row["options"], row_number),
                }
            )

    if not assignments:
        raise ValueError("The OpinionQA sample contains no assignments")
    assignments.sort(key=lambda item: item["assignment_id"])
    return assignments


def assign_opinion_question(
    participant_id: str,
    assignments: list[OpinionAssignment],
    sampler: Sampler | None = None,
) -> tuple[OpinionAssignment, str]:
    policy = sampler or StableHashSampler()
    return assignments[policy.choose_index(participant_id, len(assignments))], policy.name

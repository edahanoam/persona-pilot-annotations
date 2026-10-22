"""Run OpenAI and Gemini on the exact persona-condition pairs in the pilot.

The annotation response values are used only to identify completed assignments.
They are never included in a model prompt.

Example:
    python run_pilot_matched_inference.py responses.csv \
        --openai-model gpt-5.5 \
        --gemini-model gemini-3.5-flash \
        --output evaluations/pilot_models/model_responses.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from evaluate_pilot_responses import CUTOFF_PARTICIPANT, load_pilot_responses
from generate_persona_responses import make_answer_function
from run_bfi44_inference import ANSWER_LABELS
from twin2k import (
    PILOT_BFI44_NUMBERS,
    PersonaCondition,
    TwinPersona,
    bfi44_questions,
    load_twin_personas,
)


OUTPUT_COLUMNS = (
    "annotation_participant_id",
    "persona_id",
    "condition",
    "assignment_id",
    "question_id",
    "question_number",
    "display_number",
    "question",
    "answer_value",
    "answer_text",
    "provider",
    "model",
    "persona_text",
    "persona_text_sha256",
)


def persona_text_for_condition(
    persona: TwinPersona, condition: PersonaCondition
) -> str:
    """Return exactly the information shown for one pilot condition."""
    if condition == "demographics":
        return persona["demographics"].strip()
    if condition == "qualitative":
        return persona["qualitative"].strip()
    return f"{persona['demographics'].strip()}\n\n{persona['qualitative'].strip()}"


def build_batch_prompt(persona_text: str) -> str:
    questions = bfi44_questions(PILOT_BFI44_NUMBERS)
    statements = "\n".join(
        f"{question['display_number']}. I see myself as someone who "
        f"{str(question['question'])[0].lower() + str(question['question'])[1:]}"
        for question in questions
    )
    return f"""Answer all 20 questions as if you are the person described below.
Do not answer as an AI and do not use information beyond the persona description.

Persona description:
{persona_text}

For every statement, choose exactly one value:
1 = Disagree strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly

Statements:
{statements}

Return only a JSON array of 20 integers in question order, for example:
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]"""


def parse_batch_answers(raw_answer: str) -> list[int]:
    text = raw_answer.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw_answer!r}") from exc
    if (
        not isinstance(values, list)
        or len(values) != len(PILOT_BFI44_NUMBERS)
        or any(type(value) is not int or value not in ANSWER_LABELS for value in values)
    ):
        raise ValueError("Model response must be a JSON array of exactly 20 integers, 1-5")
    return values


def answer_batch_with_retries(
    answer_question: Callable[[str], str], prompt: str, attempts: int = 3
) -> list[int]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return parse_batch_answers(answer_question(prompt))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def load_completed_assignments(responses_csv: Path, cutoff: str) -> list[dict[str, str]]:
    responses = load_pilot_responses(responses_csv, cutoff)
    group_columns = ["participant_id", "persona_id", "condition"]
    assignments: list[dict[str, str]] = []
    for keys, group in responses.groupby(group_columns, sort=True):
        participant_id, persona_id, condition = keys
        if condition == "unknown":
            raise ValueError(
                f"Assignment condition is missing for participant {participant_id}, "
                f"persona {persona_id}"
            )
        question_ids = set(group["question_id"])
        expected_ids = {f"bfi44_{number:02d}" for number in PILOT_BFI44_NUMBERS}
        if question_ids != expected_ids:
            missing = sorted(expected_ids - question_ids)
            raise ValueError(
                f"Participant {participant_id}, persona {persona_id} is incomplete; "
                f"missing questions: {missing}"
            )
        assignment_values = {
            str(value).strip()
            for value in group.get("assignment_id", [])
            if str(value).strip()
        }
        assignments.append(
            {
                "participant_id": str(participant_id),
                "persona_id": str(persona_id),
                "condition": str(condition),
                "assignment_id": next(iter(assignment_values), ""),
            }
        )
    if not assignments:
        raise ValueError("No complete pilot assignments were found after the cutoff")
    return assignments


def completed_keys(output_csv: Path) -> set[tuple[str, str, str, str, str]]:
    if not output_csv.exists():
        return set()
    with output_csv.open("r", encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        missing = set(OUTPUT_COLUMNS).difference(rows.fieldnames or [])
        if missing:
            raise ValueError(
                f"Existing output CSV is missing columns: {sorted(missing)}"
            )
        answers_by_key: dict[tuple[str, str, str, str, str], set[str]] = {}
        for row in rows:
            key = (
                row["annotation_participant_id"],
                row["persona_id"],
                row["condition"],
                row["provider"],
                row["model"],
            )
            answers_by_key.setdefault(key, set()).add(row["question_id"])
        expected_ids = {f"bfi44_{number:02d}" for number in PILOT_BFI44_NUMBERS}
        incomplete = {
            key: sorted(expected_ids - question_ids)
            for key, question_ids in answers_by_key.items()
            if question_ids != expected_ids
        }
        if incomplete:
            example_key, missing_ids = next(iter(incomplete.items()))
            raise ValueError(
                "Existing output contains an incomplete assignment. Remove or repair "
                f"the rows for {example_key}; missing question IDs: {missing_ids}"
            )
        return set(answers_by_key)


def run_matched_inference(
    assignments: list[dict[str, str]],
    personas: list[TwinPersona],
    provider_models: list[tuple[str, str]],
    output_csv: Path,
    answer_factory: Callable[[str, str], Callable[[str], str]] = make_answer_function,
) -> int:
    persona_lookup = {persona["persona_id"]: persona for persona in personas}
    missing_personas = sorted(
        {assignment["persona_id"] for assignment in assignments}
        - set(persona_lookup)
    )
    if missing_personas:
        raise ValueError(f"Persona IDs were not found in Twin-2K: {missing_personas[:10]}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output_csv)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    questions = bfi44_questions(PILOT_BFI44_NUMBERS)
    generated = 0

    with output_csv.open("a", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()

        for provider, model in provider_models:
            answer_question = answer_factory(provider, model)
            for position, assignment in enumerate(assignments, start=1):
                key = (
                    assignment["participant_id"],
                    assignment["persona_id"],
                    assignment["condition"],
                    provider,
                    model,
                )
                if key in done:
                    print(
                        f"Skipping completed {provider}/{model} assignment "
                        f"{position}/{len(assignments)}"
                    )
                    continue
                persona = persona_lookup[assignment["persona_id"]]
                condition = assignment["condition"]
                persona_text = persona_text_for_condition(
                    persona, condition  # type: ignore[arg-type]
                )
                answers = answer_batch_with_retries(
                    answer_question, build_batch_prompt(persona_text)
                )
                persona_hash = hashlib.sha256(persona_text.encode("utf-8")).hexdigest()
                for question, answer in zip(questions, answers):
                    writer.writerow(
                        {
                            "annotation_participant_id": assignment["participant_id"],
                            "persona_id": assignment["persona_id"],
                            "condition": condition,
                            "assignment_id": assignment["assignment_id"],
                            "question_id": question["id"],
                            "question_number": question["number"],
                            "display_number": question["display_number"],
                            "question": question["question"],
                            "answer_value": answer,
                            "answer_text": ANSWER_LABELS[answer],
                            "provider": provider,
                            "model": model,
                            "persona_text": persona_text,
                            "persona_text_sha256": persona_hash,
                        }
                    )
                destination.flush()
                generated += len(answers)
                done.add(key)
                print(
                    f"Completed {provider}/{model} assignment "
                    f"{position}/{len(assignments)}"
                )
    return generated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run models on the exact persona and information-condition assignments "
            "used in the 20-question pilot."
        )
    )
    parser.add_argument("responses_csv", type=Path, help="Supabase response export")
    parser.add_argument("--openai-model", action="append", default=[])
    parser.add_argument("--gemini-model", action="append", default=[])
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional local Twin-2K dataset directory or JSONL file.",
    )
    parser.add_argument("--cutoff-participant", default=CUTOFF_PARTICIPANT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluations/pilot_models/model_responses.csv"),
    )
    args = parser.parse_args(argv)
    if not args.openai_model and not args.gemini_model:
        parser.error("Specify at least one --openai-model or --gemini-model")
    return args


def main() -> None:
    args = parse_args()
    assignments = load_completed_assignments(
        args.responses_csv, args.cutoff_participant
    )
    personas = list(load_twin_personas(args.dataset_path))
    provider_models = [
        *(("openai", model) for model in args.openai_model),
        *(("gemini", model) for model in args.gemini_model),
    ]
    generated = run_matched_inference(
        assignments, personas, provider_models, args.output
    )
    print(f"Generated {generated} new answers in {args.output.resolve()}")


if __name__ == "__main__":
    main()

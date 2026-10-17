"""Run the BFI-44 as personas stored by create_short_personas.py.

Example:
    python run_bfi44_inference.py openai gpt-5.5 10 d
    python run_bfi44_inference.py gemini gemini-3.5-flash 10 dm --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from create_short_personas import remove_bfi_aggregate_scores
from generate_persona_responses import make_answer_function


BFI44_ITEMS = (
    "Is talkative.",
    "Tends to find fault with others.",
    "Does a thorough job.",
    "Is depressed, blue.",
    "Is original, comes up with new ideas.",
    "Is reserved.",
    "Is helpful and unselfish with others.",
    "Can be somewhat careless.",
    "Is relaxed, handles stress well.",
    "Is curious about many different things.",
    "Is full of energy.",
    "Starts quarrels with others.",
    "Is a reliable worker.",
    "Can be tense.",
    "Is ingenious, a deep thinker.",
    "Generates a lot of enthusiasm.",
    "Has a forgiving nature.",
    "Tends to be disorganized.",
    "Worries a lot.",
    "Has an active imagination.",
    "Tends to be quiet.",
    "Is generally trusting.",
    "Tends to be lazy.",
    "Is emotionally stable, not easily upset.",
    "Is inventive.",
    "Has an assertive personality.",
    "Can be cold and aloof.",
    "Perseveres until the task is finished.",
    "Can be moody.",
    "Values artistic, aesthetic experiences.",
    "Is sometimes shy, inhibited.",
    "Is considerate and kind to almost everyone.",
    "Does things efficiently.",
    "Remains calm in tense situations.",
    "Prefers work that is routine.",
    "Is outgoing, sociable.",
    "Is sometimes rude to others.",
    "Makes plans and follows through with them.",
    "Gets nervous easily.",
    "Likes to reflect, play with ideas.",
    "Has few artistic interests.",
    "Likes to cooperate with others.",
    "Is easily distracted.",
    "Is sophisticated in art, music, or literature.",
)

ANSWER_LABELS = {
    1: "Disagree strongly",
    2: "Disagree a little",
    3: "Neither agree nor disagree",
    4: "Agree a little",
    5: "Agree strongly",
}

VERSION_COLUMNS = {
    "d": "persona_demographics",
    "dm": "persona_demographics_and_qualitative",
    "full": "persona_summary",
    "full_no_bfi": "persona_summary",
}

OUTPUT_COLUMNS = (
    "pid",
    "persona_version",
    "question_number",
    "question",
    "answer_value",
    "answer_text",
    "provider",
    "model",
)


def build_bfi_prompt(persona: str, item: str) -> str:
    return f"""Answer as if you are the person described below. Do not answer as an AI.

Persona:
{persona}

The following statement may or may not describe you:
I see myself as someone who {item[0].lower() + item[1:]}

Choose one response:
1 = Disagree strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly

Return only one digit: 1, 2, 3, 4, or 5."""


def parse_answer(raw_answer: str) -> int:
    """Accept a single 1-5 value, with optional surrounding whitespace."""
    match = re.fullmatch(r"\s*([1-5])(?:[.)])?\s*", raw_answer)
    if not match:
        raise ValueError(f"Model returned an invalid BFI answer: {raw_answer!r}")
    return int(match.group(1))


def answer_with_retries(
    answer_question: Callable[[str], str], prompt: str, attempts: int = 3
) -> int:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return parse_answer(answer_question(prompt))
        except Exception as exc:  # API errors and malformed answers are retryable.
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def sample_indices(dataset_size: int, n: int, seed: int) -> list[int]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > dataset_size:
        raise ValueError(f"n={n} exceeds the dataset size of {dataset_size}")
    return random.Random(seed).sample(range(dataset_size), n)


def run_inference(
    dataset: object,
    answer_question: Callable[[str], str],
    provider: str,
    model: str,
    n: int,
    version: str,
    seed: int,
    output_csv: Path,
) -> int:
    persona_column = VERSION_COLUMNS[version]
    column_names = set(dataset.column_names)  # type: ignore[attr-defined]
    if persona_column not in column_names:
        raise ValueError(
            f"Saved dataset has no {persona_column!r} column. Re-run "
            "create_short_personas.py to recreate it."
        )

    indices = sample_indices(len(dataset), n, seed)  # type: ignore[arg-type]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = 0

    with output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for persona_position, index in enumerate(indices, start=1):
            row = dataset[index]  # type: ignore[index]
            persona = row[persona_column]
            if version == "full_no_bfi":
                persona = remove_bfi_aggregate_scores(persona)
            for question_number, item in enumerate(BFI44_ITEMS, start=1):
                value = answer_with_retries(
                    answer_question, build_bfi_prompt(persona, item)
                )
                writer.writerow(
                    {
                        "pid": row["pid"],
                        "persona_version": version,
                        "question_number": question_number,
                        "question": item,
                        "answer_value": value,
                        "answer_text": ANSWER_LABELS[value],
                        "provider": provider,
                        "model": model,
                    }
                )
                destination.flush()
                completed += 1
                print(
                    f"Persona {persona_position}/{n}, question "
                    f"{question_number}/44 complete"
                )

    return completed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample saved personas and answer all BFI-44 questions."
    )
    parser.add_argument("provider", choices=("openai", "gemini"))
    parser.add_argument("model", help="Provider model name")
    parser.add_argument("n", type=int, help="Number of personas to sample")
    parser.add_argument(
        "version",
        choices=tuple(VERSION_COLUMNS),
        help=(
            "d=demographics, dm=demographics+qualitative, full=full summary, "
            "full_no_bfi=full summary without aggregate Big Five scores"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/twin_2k_500_short_personas"),
    )
    parser.add_argument("--output", type=Path, default=Path("bfi44_responses.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    try:
        from datasets import load_from_disk
    except ModuleNotFoundError as exc:
        if exc.name == "datasets":
            raise SystemExit(
                "Missing dependency 'datasets'. Install it with: pip install -r "
                "requirements-personas.txt"
            ) from exc
        raise RuntimeError(
            f"The datasets package is installed, but one of its dependencies is "
            f"missing: {exc.name!r}. Python executable: {__import__('sys').executable}"
        ) from exc

    args = parse_args()
    dataset = load_from_disk(str(args.dataset_dir))
    answer_question = make_answer_function(args.provider, args.model)
    completed = run_inference(
        dataset=dataset,
        answer_question=answer_question,
        provider=args.provider,
        model=args.model,
        n=args.n,
        version=args.version,
        seed=args.seed,
        output_csv=args.output,
    )
    print(f"Saved {completed} answers to {args.output.resolve()}")


if __name__ == "__main__":
    main()

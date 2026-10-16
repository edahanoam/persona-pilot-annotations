"""Generate survey answers from a persona without showing the model the real answer."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Callable, Sequence
from pathlib import Path


def build_prompt(persona: str, question: str, choices: str) -> str:
    """Build a prompt using only fields the model is allowed to see."""
    return f"""Answer the survey question as if you were the person described below.
Choose exactly one answer from the provided choices.
Return only the answer text, with no explanation or quotation marks.

Persona:
{persona}

Question:
{question}

Choices:
{choices}"""


def make_answer_function(provider: str, model: str) -> Callable[[str], str]:
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")

        from openai import OpenAI

        client = OpenAI()

        def answer(prompt: str) -> str:
            response = client.responses.create(model=model, input=prompt)
            return response.output_text.strip()

        return answer

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first.")

    from google import genai

    client = genai.Client()

    def answer(prompt: str) -> str:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()

    return answer


def process_csv(
    input_path: Path,
    output_path: Path,
    answer_question: Callable[[str], str],
    has_header: bool = False,
) -> int:
    """Copy each CSV row and append a model response."""
    count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination)

        if has_header:
            header = next(reader, None)
            if header is None:
                return 0
            writer.writerow([*header, "model_response"])

        for row_number, row in enumerate(reader, start=2 if has_header else 1):
            if not row:
                writer.writerow(row)
                continue
            if len(row) < 5:
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns; expected at least 5."
                )

            # Intentionally use only persona, question, and choices. In particular,
            # row[-1] (the human answer in the sample format) never enters the prompt.
            prompt = build_prompt(persona=row[0], question=row[2], choices=row[4])
            model_response = answer_question(prompt)
            writer.writerow([*row, model_response])
            count += 1
            print(f"Completed row {row_number}")

    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer survey questions as each row's persona."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("provider", choices=("openai", "gemini"))
    parser.add_argument("model", help="Model name, for example gpt-5.5 or gemini-3.5-flash")
    parser.add_argument(
        "--has-header", action="store_true", help="Treat the first row as column names"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    answer_question = make_answer_function(args.provider, args.model)
    count = process_csv(
        args.input_csv, args.output_csv, answer_question, args.has_header
    )
    print(f"Saved {count} model responses to {args.output_csv}")


if __name__ == "__main__":
    main()

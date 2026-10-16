"""Create two shortened persona summaries from Twin-2K-500.

Run:
    python create_short_personas.py

Load the saved result later:
    from datasets import load_from_disk
    personas = load_from_disk("data/twin_2k_500_short_personas")
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict


DATASET_NAME = "LLM-Digital-Twin/Twin-2K-500"
DATASET_CONFIG = "full_persona"
BIG_FIVE_MARKER = "The person's Big 5 scores are the following:"
QUALITATIVE_MARKER = (
    "The person also answered three purely qualitative questions about their "
    "concept of self"
)


class ShortPersona(TypedDict):
    persona_demographics: str
    persona_demographics_and_qualitative: str


def _marker_index(text: str, marker: str) -> int:
    """Return a case-insensitive marker position or raise a useful error."""
    index = text.casefold().find(marker.casefold())
    if index == -1:
        raise ValueError(f"Could not find expected marker: {marker!r}")
    return index


def create_short_versions(persona_summary: str) -> ShortPersona:
    """Extract demographics, and demographics plus open-ended answers."""
    if not isinstance(persona_summary, str) or not persona_summary.strip():
        raise ValueError("persona_summary must be a non-empty string")

    big_five_start = _marker_index(persona_summary, BIG_FIVE_MARKER)
    qualitative_start = _marker_index(persona_summary, QUALITATIVE_MARKER)
    if qualitative_start <= big_five_start:
        raise ValueError("The qualitative section occurs before the Big 5 section")

    demographics = persona_summary[:big_five_start].strip()
    qualitative = persona_summary[qualitative_start:].strip()

    return {
        "persona_demographics": demographics,
        "persona_demographics_and_qualitative": (
            f"{demographics}\n\n{qualitative}"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create short persona variants from Twin-2K-500."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/twin_2k_500_short_personas"),
        help="Directory used by datasets.save_to_disk().",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep persona_text, persona_summary, and persona_json in the output.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install it with: pip install -r "
            "requirements-personas.txt"
        ) from exc

    args = parse_args()
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="data")
    dataset = dataset.map(
        lambda row: create_short_versions(row["persona_summary"]),
        desc="Creating short persona summaries",
    )

    if not args.keep_original:
        columns_to_keep = {
            "pid",
            "persona_summary",
            "persona_demographics",
            "persona_demographics_and_qualitative",
        }
        dataset = dataset.remove_columns(
            [name for name in dataset.column_names if name not in columns_to_keep]
        )

    first = dataset[0]
    print("FIRST PERSONA: DEMOGRAPHICS ONLY")
    print(first["persona_demographics"])
    print("\nFIRST PERSONA: DEMOGRAPHICS + OPEN-ENDED ANSWERS")
    print(first["persona_demographics_and_qualitative"])

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(f"\nSaved {len(dataset):,} personas to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

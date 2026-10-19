"""Print answer accuracy for selected annotation participants.

Examples:
    python participant_accuracy.py responses.csv 1001 1002
    python participant_accuracy.py responses.csv "1001,1002,0003"

The responses file is a CSV export of the Supabase ``responses`` table. The
actual scoring is shared with ``evaluate_collected_responses.py`` so this
command uses the same OpinionQA and Twin-2K answer normalization rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_collected_responses import (
    bfi_source_question_id,
    download_twin_human_csv,
    load_opinionqa_reference,
    load_responses,
    load_twin_reference,
    participant_summary,
    score_responses,
)


def parse_participant_ids(values: list[str]) -> list[str]:
    """Split space/comma-separated IDs while preserving order and leading zeroes."""
    participant_ids: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in value.split(","):
            participant_id = item.strip()
            if participant_id and participant_id not in seen:
                participant_ids.append(participant_id)
                seen.add(participant_id)
    if not participant_ids:
        raise ValueError("Provide at least one participant ID")
    return participant_ids


def calculate_participant_accuracy(
    responses_csv: Path,
    participant_ids: list[str],
    *,
    opinionqa_csv: Path = Path("opinionqa_sample.csv"),
    twin_human_csv: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    responses = load_responses(responses_csv)
    selected = responses[responses["participant_id"].isin(participant_ids)].copy()
    found = set(selected["participant_id"])
    missing = [participant_id for participant_id in participant_ids if participant_id not in found]

    if selected.empty:
        return pd.DataFrame(), missing

    opinion_reference = None
    if (selected["experiment"] == "opinionqa").any():
        opinion_reference = load_opinionqa_reference(opinionqa_csv)

    twin_reference = None
    if (selected["experiment"] == "twin2k").any():
        twin_rows = selected[selected["experiment"] == "twin2k"]
        source_ids = {
            mapped
            for mapped in twin_rows["question_id"].map(bfi_source_question_id)
            if mapped is not None
        }
        human_path = twin_human_csv or download_twin_human_csv()
        twin_reference = load_twin_reference(human_path, source_ids)

    scored = score_responses(selected, opinion_reference, twin_reference)
    summary = participant_summary(scored)
    order = {participant_id: index for index, participant_id in enumerate(participant_ids)}
    summary["_requested_order"] = summary["participant_id"].map(order)
    summary = summary.sort_values(["_requested_order", "experiment"]).drop(
        columns="_requested_order"
    )
    return summary.reset_index(drop=True), missing


def format_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No responses found for the requested participant IDs."

    display = summary.copy()
    display["accuracy"] = display["accuracy"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.1%}"
    )
    display["complete"] = display["complete"].map({True: "yes", False: "no"})
    display = display.rename(
        columns={
            "participant_id": "participant",
            "scored_responses": "scored",
            "unscored_responses": "unscored",
            "expected_responses": "expected",
        }
    )
    columns = [
        "participant",
        "experiment",
        "matches",
        "scored",
        "accuracy",
        "unscored",
        "responses",
        "expected",
        "complete",
    ]
    return display.loc[:, columns].to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show answer accuracy for a list of participant IDs."
    )
    parser.add_argument("responses_csv", type=Path, help="Supabase responses CSV export")
    parser.add_argument(
        "participant_ids",
        nargs="+",
        help="Participant IDs, separated by spaces and/or commas",
    )
    parser.add_argument(
        "--opinionqa-csv", type=Path, default=Path("opinionqa_sample.csv")
    )
    parser.add_argument(
        "--twin-human-csv",
        type=Path,
        help="Local Twin-2K wave1_3_response.csv; downloaded if omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    participant_ids = parse_participant_ids(args.participant_ids)
    summary, missing = calculate_participant_accuracy(
        args.responses_csv,
        participant_ids,
        opinionqa_csv=args.opinionqa_csv,
        twin_human_csv=args.twin_human_csv,
    )
    print(format_summary(summary))
    if missing:
        print(f"\nNo responses found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()

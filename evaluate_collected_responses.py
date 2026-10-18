"""Score collected Twin-2K and OpinionQA annotations against source answers.

Example:
    python evaluate_collected_responses.py responses.csv

The responses CSV should be exported from the Supabase ``responses`` table.
OpinionQA references come from ``opinionqa_sample.csv``. Twin-2K BFI answers
come from the public Twin-2K-500 repository unless ``--twin-human-csv`` is set.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


HF_REPO = "LLM-Digital-Twin/Twin-2K-500"
HF_HUMAN_FILE = "question_catalog_and_human_response_csv/wave1_3_response.csv"
REQUIRED_RESPONSE_COLUMNS = {
    "participant_id",
    "experiment",
    "persona_id",
    "question_id",
    "answer",
}
EXPECTED_QUESTION_COUNTS = {"twin2k": 44, "opinionqa": 10}
BFI_QUESTION_PATTERN = re.compile(r"^bfi44_0*([1-9]|[1-3][0-9]|4[0-4])$")


def normalize_id(value: object) -> str:
    """Make identifiers such as 574, '574', and '574.0' comparable."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def bfi_source_question_id(question_id: object) -> str | None:
    """Map app IDs such as bfi44_01 to Twin-2K columns such as QID25_1."""
    match = BFI_QUESTION_PATTERN.fullmatch(str(question_id).strip())
    return f"QID25_{int(match.group(1))}" if match else None


def normalize_bfi_answer(value: object) -> str | None:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    integer = int(numeric)
    return str(integer) if numeric == integer and 1 <= integer <= 5 else None


def load_responses(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_RESPONSE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Responses CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Responses CSV contains no response rows")
    frame = frame.copy()
    frame["experiment"] = frame["experiment"].str.strip().str.casefold()
    unsupported = sorted(set(frame["experiment"]) - set(EXPECTED_QUESTION_COUNTS))
    if unsupported:
        raise ValueError(f"Unsupported experiment values: {unsupported}")
    for column in ("participant_id", "persona_id", "question_id", "answer"):
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Responses CSV contains a blank {column}")
    frame["_persona_key"] = frame["persona_id"].map(normalize_id)
    frame["_response_row"] = range(1, len(frame) + 1)
    return frame


def load_opinionqa_reference(path: Path) -> pd.DataFrame:
    reference = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"id", "question_identifier", "answer"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"OpinionQA CSV is missing columns: {sorted(missing)}")
    reference = reference.loc[:, ["id", "question_identifier", "answer"]].copy()
    reference["_persona_key"] = reference["id"].map(normalize_id)
    reference["_source_question_id"] = reference["question_identifier"].str.strip()
    reference["reference_answer"] = reference["answer"].str.strip()
    duplicate = reference.duplicated(["_persona_key", "_source_question_id"], keep=False)
    if duplicate.any():
        examples = reference.loc[duplicate, ["id", "question_identifier"]].head(5)
        raise ValueError(
            "OpinionQA contains duplicate persona/question pairs: "
            f"{examples.to_dict(orient='records')}"
        )
    return reference.loc[:, ["_persona_key", "_source_question_id", "reference_answer"]]


def download_twin_human_csv() -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install requirements.txt or supply --twin-human-csv"
        ) from exc
    return Path(
        hf_hub_download(repo_id=HF_REPO, filename=HF_HUMAN_FILE, repo_type="dataset")
    )


def load_twin_reference(path: Path, source_question_ids: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    required = {"pid", *source_question_ids}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"Twin-2K human CSV is missing columns: {sorted(missing)}")
    reference = pd.read_csv(path, usecols=sorted(required), dtype=str)
    reference["_persona_key"] = reference["pid"].map(normalize_id)
    if reference["_persona_key"].duplicated().any():
        raise ValueError("Twin-2K human CSV contains duplicate pid values")
    long = reference.melt(
        id_vars="_persona_key",
        value_vars=sorted(source_question_ids),
        var_name="_source_question_id",
        value_name="reference_answer",
    )
    long["reference_answer"] = long["reference_answer"].map(normalize_bfi_answer)
    return long


def score_responses(
    responses: pd.DataFrame,
    opinionqa_reference: pd.DataFrame | None,
    twin_reference: pd.DataFrame | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    opinion = responses[responses["experiment"] == "opinionqa"].copy()
    if not opinion.empty:
        if opinionqa_reference is None:
            raise ValueError("OpinionQA responses exist but no OpinionQA reference was loaded")
        opinion["_source_question_id"] = opinion["question_id"].str.strip()
        opinion = opinion.merge(
            opinionqa_reference,
            how="left",
            on=["_persona_key", "_source_question_id"],
            validate="many_to_one",
        )
        opinion["normalized_answer"] = opinion["answer"].str.strip()
        opinion["normalized_reference_answer"] = opinion["reference_answer"]
        parts.append(opinion)

    twin = responses[responses["experiment"] == "twin2k"].copy()
    if not twin.empty:
        if twin_reference is None:
            raise ValueError("Twin-2K responses exist but no Twin-2K reference was loaded")
        twin["_source_question_id"] = twin["question_id"].map(bfi_source_question_id)
        twin = twin.merge(
            twin_reference,
            how="left",
            on=["_persona_key", "_source_question_id"],
            validate="many_to_one",
        )
        twin["normalized_answer"] = twin["answer"].map(normalize_bfi_answer)
        twin["normalized_reference_answer"] = twin["reference_answer"]
        parts.append(twin)

    scored = pd.concat(parts, ignore_index=True).sort_values("_response_row")
    source_found = scored["reference_answer"].notna()
    response_valid = scored["normalized_answer"].notna()
    scored["comparison_status"] = "scored"
    scored.loc[~source_found, "comparison_status"] = "reference_not_found"
    scored.loc[source_found & ~response_valid, "comparison_status"] = "invalid_response"
    scored["is_match"] = pd.NA
    comparable = source_found & response_valid
    scored.loc[comparable, "is_match"] = (
        scored.loc[comparable, "normalized_answer"]
        == scored.loc[comparable, "normalized_reference_answer"]
    )
    return scored.drop(columns=["_persona_key", "_response_row"])


def summarize(scored: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in scored.groupby(grouper, dropna=False, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else keys
        comparable = group[group["comparison_status"] == "scored"]
        matches = comparable["is_match"].fillna(False).astype(bool)
        row = dict(zip(group_columns, key_values))
        row.update(
            {
                "responses": len(group),
                "scored_responses": len(comparable),
                "unscored_responses": len(group) - len(comparable),
                "matches": int(matches.sum()),
                "accuracy": float(matches.mean()) if len(matches) else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def participant_summary(scored: pd.DataFrame) -> pd.DataFrame:
    summary = summarize(scored, ["experiment", "participant_id"])
    summary["expected_responses"] = summary["experiment"].map(EXPECTED_QUESTION_COUNTS)
    summary["complete"] = summary["responses"] == summary["expected_responses"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exported annotations with OpinionQA and Twin-2K answers."
    )
    parser.add_argument("responses_csv", type=Path, help="Supabase responses CSV export")
    parser.add_argument(
        "--opinionqa-csv", type=Path, default=Path("opinionqa_sample.csv")
    )
    parser.add_argument(
        "--twin-human-csv",
        type=Path,
        help="Optional local wave1_3_response.csv; downloaded if omitted",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluations/collected_responses")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses = load_responses(args.responses_csv)

    opinion_reference = None
    if (responses["experiment"] == "opinionqa").any():
        opinion_reference = load_opinionqa_reference(args.opinionqa_csv)

    twin_reference = None
    if (responses["experiment"] == "twin2k").any():
        twin_rows = responses[responses["experiment"] == "twin2k"]
        source_ids = {
            mapped
            for mapped in twin_rows["question_id"].map(bfi_source_question_id)
            if mapped is not None
        }
        human_path = args.twin_human_csv or download_twin_human_csv()
        twin_reference = load_twin_reference(human_path, source_ids)

    scored = score_responses(responses, opinion_reference, twin_reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "scored_responses.csv", index=False)
    summarize(scored, ["experiment"]).to_csv(
        args.output_dir / "summary.csv", index=False
    )
    summarize(scored, ["experiment", "question_id"]).to_csv(
        args.output_dir / "by_question.csv", index=False
    )
    participant_summary(scored).to_csv(
        args.output_dir / "by_participant.csv", index=False
    )

    status_counts = scored["comparison_status"].value_counts().to_dict()
    print(f"Scored {len(scored):,} collected responses.")
    print(f"Comparison statuses: {status_counts}")
    print(f"Reports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

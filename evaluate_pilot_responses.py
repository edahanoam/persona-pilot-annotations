"""Score the 20-item, three-persona pilot and project profiles into human PCA.

Rows for the cutoff participant and every earlier annotation are excluded. When
``submitted_at`` is available, "after" is based on the cutoff participant's
latest timestamp. Otherwise, CSV row order after that participant's last row is
used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_collected_responses import (
    download_twin_human_csv,
    normalize_bfi_answer,
    normalize_id,
)
from twin2k import PILOT_BFI44_NUMBERS


CUTOFF_PARTICIPANT = "0noamlasttest"
QUESTION_IDS = tuple(f"bfi44_{number:02d}" for number in PILOT_BFI44_NUMBERS)
SOURCE_COLUMNS = tuple(f"QID25_{number}" for number in PILOT_BFI44_NUMBERS)
CONDITION_PATTERN = re.compile(
    r"^persona-\d+-(demographics|combined|qualitative)$"
)
CONDITION_COLORS = {
    "demographics": "#dc2626",
    "combined": "#16a34a",
    "qualitative": "#2563eb",
    "unknown": "#9333ea",
}
CONDITION_LABELS = {
    "demographics": "Demographics",
    "combined": "Demographics + self-description",
    "qualitative": "Self-description",
    "unknown": "Condition unavailable",
}


def load_pilot_responses(path: Path, cutoff_participant: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"participant_id", "persona_id", "question_id", "answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Responses CSV is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["_row_order"] = range(len(frame))
    frame["participant_id"] = frame["participant_id"].map(normalize_id)
    marker = frame["participant_id"].eq(normalize_id(cutoff_participant))
    if not marker.any():
        raise ValueError(
            f"Cutoff participant {cutoff_participant!r} was not found in the CSV"
        )

    if "submitted_at" in frame.columns:
        timestamps = pd.to_datetime(frame["submitted_at"], errors="coerce", utc=True)
        marker_times = timestamps[marker].dropna()
    else:
        timestamps = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        marker_times = timestamps[marker].dropna()

    if not marker_times.empty:
        cutoff_time = marker_times.max()
        frame = frame.loc[timestamps > cutoff_time].copy()
        frame["_submitted_at"] = timestamps.loc[frame.index]
        cutoff_method = f"submitted_at > {cutoff_time.isoformat()}"
    else:
        cutoff_row = int(frame.loc[marker, "_row_order"].max())
        frame = frame.loc[frame["_row_order"] > cutoff_row].copy()
        frame["_submitted_at"] = pd.NaT
        cutoff_method = f"CSV row after {cutoff_row + 1}"

    if "experiment" in frame.columns:
        frame = frame.loc[frame["experiment"].str.strip().str.casefold() == "twin2k"]
    frame = frame.loc[frame["question_id"].isin(QUESTION_IDS)].copy()
    if frame.empty:
        raise ValueError("No pilot responses remain after applying the cutoff")

    frame["persona_id"] = frame["persona_id"].map(normalize_id)
    frame["answer_value"] = frame["answer"].map(normalize_bfi_answer)
    frame["condition"] = (
        frame.get("assignment_id", pd.Series("", index=frame.index))
        .str.extract(CONDITION_PATTERN, expand=False)
        .fillna("unknown")
    )
    frame["question_number"] = frame["question_id"].str.extract(
        r"(\d+)$", expand=False
    ).astype(int)
    frame = frame.sort_values(["_submitted_at", "_row_order"], na_position="first")
    frame = frame.drop_duplicates(
        ["participant_id", "persona_id", "question_id"], keep="last"
    )
    frame.attrs["cutoff_method"] = cutoff_method
    return frame


def load_human_subset(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    required = {"pid", *SOURCE_COLUMNS}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"Human CSV is missing columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=["pid", *SOURCE_COLUMNS], dtype={"pid": str})
    frame["persona_id"] = frame["pid"].map(normalize_id)
    if frame["persona_id"].duplicated().any():
        raise ValueError("Human CSV contains duplicate pid values")
    values = frame.loc[:, list(SOURCE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values.columns = PILOT_BFI44_NUMBERS
    values.index = frame["persona_id"]
    values.index.name = "persona_id"
    return values


def score_responses(responses: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    reference = human.stack().rename("human_answer").reset_index()
    reference.columns = ["persona_id", "question_number", "human_answer"]
    scored = responses.merge(
        reference,
        how="left",
        on=["persona_id", "question_number"],
        validate="many_to_one",
    )
    scored["comparison_status"] = "scored"
    scored.loc[scored["human_answer"].isna(), "comparison_status"] = (
        "reference_not_found"
    )
    scored.loc[
        scored["human_answer"].notna() & scored["answer_value"].isna(),
        "comparison_status",
    ] = "invalid_response"
    comparable = scored["comparison_status"].eq("scored")
    scored["is_match"] = pd.NA
    scored.loc[comparable, "is_match"] = (
        scored.loc[comparable, "answer_value"].astype(float)
        == scored.loc[comparable, "human_answer"].astype(float)
    )
    scored["absolute_error"] = np.nan
    scored.loc[comparable, "absolute_error"] = (
        scored.loc[comparable, "answer_value"].astype(float)
        - scored.loc[comparable, "human_answer"].astype(float)
    ).abs()
    return scored


def summarize(scored: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = [((), scored)] if not group_columns else scored.groupby(
        group_columns, dropna=False, sort=True
    )
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        if group_columns and not isinstance(keys, tuple):
            keys = (keys,)
        comparable = group.loc[group["comparison_status"] == "scored"]
        matches = comparable["is_match"].fillna(False).astype(bool)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "responses": len(group),
                "scored_responses": len(comparable),
                "matches": int(matches.sum()),
                "exact_accuracy": float(matches.mean()) if len(matches) else np.nan,
                "mean_absolute_error": float(comparable["absolute_error"].mean())
                if len(comparable)
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def complete_annotation_profiles(responses: pd.DataFrame) -> pd.DataFrame:
    matrix = responses.pivot_table(
        index=["participant_id", "persona_id", "condition"],
        columns="question_number",
        values="answer_value",
        aggfunc="last",
    )
    matrix = matrix.reindex(columns=PILOT_BFI44_NUMBERS)
    return matrix.dropna()


def run_pca(
    human: pd.DataFrame, responses: pd.DataFrame, output_dir: Path
) -> tuple[int, int]:
    from sklearn.decomposition import PCA

    human_complete = human.dropna()
    annotations = complete_annotation_profiles(responses)
    if len(human_complete) < 2:
        raise ValueError("At least two complete human profiles are required for PCA")
    if annotations.empty:
        raise ValueError("No complete 20-answer annotation profiles are available for PCA")

    pca = PCA(n_components=min(human_complete.shape))
    human_xy = pca.fit_transform(human_complete.to_numpy(dtype=float))
    annotation_xy = pca.transform(annotations.to_numpy(dtype=float))

    human_coordinates = pd.DataFrame(
        {
            "source": "human",
            "participant_id": "",
            "persona_id": human_complete.index,
            "condition": "human",
            "PC1": human_xy[:, 0],
            "PC2": human_xy[:, 1],
        }
    )
    annotation_coordinates = annotations.index.to_frame(index=False)
    annotation_coordinates.insert(0, "source", "annotation")
    annotation_coordinates["PC1"] = annotation_xy[:, 0]
    annotation_coordinates["PC2"] = annotation_xy[:, 1]
    coordinates = pd.concat(
        [human_coordinates, annotation_coordinates], ignore_index=True
    )
    coordinates.to_csv(output_dir / "pca_coordinates.csv", index=False)
    pd.DataFrame(
        {
            "component": range(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    ).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        human_xy[:, 0], human_xy[:, 1], s=16, alpha=0.22,
        color="#6b7280", label="Human reference profiles",
    )
    human_lookup = {
        persona_id: human_xy[index, :2]
        for index, persona_id in enumerate(human_complete.index)
    }
    for condition in CONDITION_COLORS:
        mask = annotation_coordinates["condition"].eq(condition).to_numpy()
        if not mask.any():
            continue
        points = annotation_xy[mask]
        rows = annotation_coordinates.loc[mask]
        for point, persona_id in zip(points, rows["persona_id"]):
            target = human_lookup.get(persona_id)
            if target is not None:
                axis.plot(
                    [target[0], point[0]], [target[1], point[1]],
                    color=CONDITION_COLORS[condition], alpha=0.20, linewidth=0.8,
                )
        axis.scatter(
            points[:, 0], points[:, 1], s=48, alpha=0.90,
            color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition],
        )
    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100
    axis.set_xlabel(f"Human PCA component 1 ({pc1:.1f}% variance)")
    axis.set_ylabel(f"Human PCA component 2 ({pc2:.1f}% variance)")
    axis.set_title("Pilot annotations projected into 20-item human PCA space")
    axis.axhline(0, color="#9ca3af", linewidth=0.6)
    axis.axvline(0, color="#9ca3af", linewidth=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "pca_projection.png", dpi=200)
    plt.close(figure)
    return len(human_complete), len(annotations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the three-persona, 20-question annotation pilot."
    )
    parser.add_argument("responses_csv", type=Path)
    parser.add_argument(
        "--human-csv",
        type=Path,
        help="Local wave1_3_response.csv; downloaded from Hugging Face if omitted.",
    )
    parser.add_argument(
        "--cutoff-participant", default=CUTOFF_PARTICIPANT,
        help="Exclude this participant and all annotations before it.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluations/pilot_responses")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses = load_pilot_responses(args.responses_csv, args.cutoff_participant)
    cutoff_method = responses.attrs.get("cutoff_method", "unknown")
    human_path = args.human_csv or download_twin_human_csv()
    human = load_human_subset(human_path)
    scored = score_responses(responses, human)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "scored_responses.csv", index=False)
    summarize(scored, []).to_csv(args.output_dir / "overall.csv", index=False)
    by_participant = summarize(scored, ["participant_id"])
    by_participant["expected_responses"] = len(QUESTION_IDS) * 3
    by_participant["complete"] = (
        by_participant["responses"] == by_participant["expected_responses"]
    )
    by_participant.to_csv(
        args.output_dir / "by_participant.csv", index=False
    )
    by_participant_condition = summarize(scored, ["participant_id", "condition"])
    by_participant_condition["expected_responses"] = len(QUESTION_IDS)
    by_participant_condition["complete"] = (
        by_participant_condition["responses"]
        == by_participant_condition["expected_responses"]
    )
    by_participant_condition.to_csv(
        args.output_dir / "by_participant_condition.csv", index=False
    )
    summarize(scored, ["participant_id", "persona_id", "condition"]).to_csv(
        args.output_dir / "by_participant_persona.csv", index=False
    )
    summarize(scored, ["condition"]).to_csv(
        args.output_dir / "by_condition.csv", index=False
    )
    human_count, annotation_count = run_pca(human, responses, args.output_dir)

    print(f"Cutoff applied using {cutoff_method}.")
    print(f"Retained {len(responses):,} response rows.")
    print(
        f"PCA fitted on {human_count:,} complete human profiles and projected "
        f"{annotation_count:,} complete annotation profiles."
    )
    print(f"Reports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

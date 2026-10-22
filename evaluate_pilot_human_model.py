"""Compare pilot annotators and models and project both into human PCA space."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_collected_responses import download_twin_human_csv, normalize_id
from evaluate_pilot_responses import (
    CONDITION_LABELS,
    CUTOFF_PARTICIPANT,
    load_human_subset,
    load_pilot_responses,
)
from twin2k import PILOT_BFI44_NUMBERS


ASSIGNMENT_COLUMNS = ["participant_id", "persona_id", "condition"]
EXPECTED_QUESTION_IDS = {
    f"bfi44_{number:02d}" for number in PILOT_BFI44_NUMBERS
}


def annotation_rows(path: Path, cutoff_participant: str) -> pd.DataFrame:
    frame = load_pilot_responses(path, cutoff_participant).copy()
    frame["answer_value"] = pd.to_numeric(frame["answer_value"], errors="coerce")
    frame["agent_type"] = "human_annotator"
    frame["source"] = "Human annotators"
    frame["provider"] = ""
    frame["model"] = ""
    return frame.loc[
        :,
        [
            *ASSIGNMENT_COLUMNS,
            "question_id",
            "question_number",
            "answer_value",
            "agent_type",
            "source",
            "provider",
            "model",
        ],
    ]


def model_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "annotation_participant_id",
        "persona_id",
        "condition",
        "question_id",
        "question_number",
        "answer_value",
        "provider",
        "model",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Model CSV is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["participant_id"] = frame["annotation_participant_id"].map(normalize_id)
    frame["persona_id"] = frame["persona_id"].map(normalize_id)
    frame["condition"] = frame["condition"].str.strip()
    frame["question_id"] = frame["question_id"].str.strip()
    frame["question_number"] = pd.to_numeric(
        frame["question_number"], errors="raise"
    ).astype(int)
    frame["answer_value"] = pd.to_numeric(
        frame["answer_value"], errors="coerce"
    )
    if not frame["question_id"].isin(EXPECTED_QUESTION_IDS).all():
        unexpected = sorted(set(frame["question_id"]) - EXPECTED_QUESTION_IDS)
        raise ValueError(f"Model CSV contains unexpected question IDs: {unexpected}")
    if not frame["answer_value"].between(1, 5).all():
        raise ValueError("Model CSV contains a missing answer or a value outside 1-5")
    frame["agent_type"] = "model"
    frame["source"] = frame["provider"].str.strip() + ":" + frame["model"].str.strip()
    duplicate_columns = ["source", *ASSIGNMENT_COLUMNS, "question_id"]
    if frame.duplicated(duplicate_columns).any():
        raise ValueError("Model CSV contains duplicate source/assignment/question rows")
    return frame.loc[
        :,
        [
            *ASSIGNMENT_COLUMNS,
            "question_id",
            "question_number",
            "answer_value",
            "agent_type",
            "source",
            "provider",
            "model",
        ],
    ]


def _complete_assignment_keys(group: pd.DataFrame, source: str) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for assignment_key, assignment in group.groupby(ASSIGNMENT_COLUMNS, sort=False):
        question_ids = set(assignment["question_id"])
        if question_ids == EXPECTED_QUESTION_IDS:
            keys.add(tuple(str(value) for value in assignment_key))
    if not keys:
        raise ValueError(f"Source {source!r} has no complete 20-question assignments")
    return keys


def align_common_assignments(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key_sets = [
        _complete_assignment_keys(group, str(source))
        for source, group in rows.groupby("source", sort=True)
    ]
    common = set.intersection(*key_sets)
    if not common:
        raise ValueError("No complete assignments are shared by annotators and all models")
    row_keys = pd.Series(
        list(zip(*(rows[column].astype(str) for column in ASSIGNMENT_COLUMNS))),
        index=rows.index,
    )
    aligned = rows.loc[row_keys.isin(common)].copy()
    expected_rows = len(common) * len(EXPECTED_QUESTION_IDS) * rows["source"].nunique()
    if len(aligned) != expected_rows:
        raise ValueError(
            "Aligned rows are not unique and complete across every response source"
        )
    return aligned, len(common)


def score_rows(rows: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    reference = human.stack().rename("reference_answer").reset_index()
    reference.columns = ["persona_id", "question_number", "reference_answer"]
    scored = rows.merge(
        reference,
        how="left",
        on=["persona_id", "question_number"],
        validate="many_to_one",
    )
    if scored["reference_answer"].isna().any():
        missing = sorted(scored.loc[scored["reference_answer"].isna(), "persona_id"].unique())
        raise ValueError(f"Human reference answers were not found for personas: {missing}")
    scored["error"] = scored["answer_value"] - scored["reference_answer"]
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"].pow(2)
    scored["is_match"] = scored["error"].eq(0)
    return scored


def profile_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["source", "agent_type", "provider", "model", *ASSIGNMENT_COLUMNS]
    rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(group_columns, dropna=False, sort=True):
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "profile_exact_accuracy": float(group["is_match"].mean()),
                "profile_mae": float(group["absolute_error"].mean()),
                "profile_rmse": float(np.sqrt(group["squared_error"].mean())),
                "profile_euclidean_distance": float(
                    np.linalg.norm(group["error"].to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(scored: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    profiles = profile_metrics(scored)
    rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys))
        profile_filter = pd.Series(True, index=profiles.index)
        for column, value in key_values.items():
            profile_filter &= profiles[column].eq(value)
        matching_profiles = profiles.loc[profile_filter]
        predicted = group["answer_value"].to_numpy(dtype=float)
        reference = group["reference_answer"].to_numpy(dtype=float)
        correlation = (
            float(np.corrcoef(predicted, reference)[0, 1])
            if np.std(predicted) > 0 and np.std(reference) > 0
            else math.nan
        )
        rows.append(
            {
                **key_values,
                "profiles": len(matching_profiles),
                "responses": len(group),
                "matches": int(group["is_match"].sum()),
                "exact_accuracy": float(group["is_match"].mean()),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "pearson_r": correlation,
                "mean_profile_euclidean_distance": float(
                    matching_profiles["profile_euclidean_distance"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def make_winner_report(agent_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, group in agent_summary.groupby("condition", sort=True):
        human = group.loc[group["agent_type"] == "human_annotator"]
        models = group.loc[group["agent_type"] == "model"]
        if len(human) != 1 or len(models) != 1:
            raise ValueError(f"Could not compare human and model rows for {condition}")
        human_row = human.iloc[0]
        model_row = models.iloc[0]
        rows.append(
            {
                "condition": condition,
                "description_type": CONDITION_LABELS.get(condition, condition),
                "human_exact_accuracy": human_row["exact_accuracy"],
                "model_exact_accuracy": model_row["exact_accuracy"],
                "exact_accuracy_winner": (
                    "human_annotators"
                    if human_row["exact_accuracy"] > model_row["exact_accuracy"]
                    else "models"
                    if model_row["exact_accuracy"] > human_row["exact_accuracy"]
                    else "tie"
                ),
                "human_mae": human_row["mae"],
                "model_mae": model_row["mae"],
                "mae_winner": (
                    "human_annotators"
                    if human_row["mae"] < model_row["mae"]
                    else "models"
                    if model_row["mae"] < human_row["mae"]
                    else "tie"
                ),
                "human_rmse": human_row["rmse"],
                "model_rmse": model_row["rmse"],
                "rmse_winner": (
                    "human_annotators"
                    if human_row["rmse"] < model_row["rmse"]
                    else "models"
                    if model_row["rmse"] < human_row["rmse"]
                    else "tie"
                ),
                "human_pearson_r": human_row["pearson_r"],
                "model_pearson_r": model_row["pearson_r"],
                "correlation_winner": (
                    "human_annotators"
                    if human_row["pearson_r"] > model_row["pearson_r"]
                    else "models"
                    if model_row["pearson_r"] > human_row["pearson_r"]
                    else "tie"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_pca(
    human: pd.DataFrame, aligned: pd.DataFrame, output_dir: Path
) -> tuple[int, int]:
    human_complete = human.dropna()
    if len(human_complete) < 2:
        raise ValueError("At least two complete human reference profiles are required")
    profile_index = [
        "source",
        "agent_type",
        "provider",
        "model",
        *ASSIGNMENT_COLUMNS,
    ]
    profiles = aligned.pivot(
        index=profile_index, columns="question_number", values="answer_value"
    ).reindex(columns=PILOT_BFI44_NUMBERS)
    if profiles.isna().any().any():
        raise ValueError("An aligned annotator or model profile is incomplete")

    human_values = human_complete.to_numpy(dtype=float)
    human_mean = human_values.mean(axis=0)
    centered_human = human_values - human_mean
    _, singular_values, components = np.linalg.svd(
        centered_human, full_matrices=False
    )
    explained_variance = singular_values**2 / (len(human_complete) - 1)
    total_variance = explained_variance.sum()
    if total_variance <= 0:
        raise ValueError("Human reference profiles have no variance for PCA")
    explained_variance_ratio = explained_variance / total_variance
    reference_xy = centered_human @ components.T
    projected_xy = (profiles.to_numpy(dtype=float) - human_mean) @ components.T

    reference_coordinates = pd.DataFrame(
        {
            "source": "Human reference",
            "agent_type": "human_reference",
            "provider": "",
            "model": "",
            "participant_id": "",
            "persona_id": human_complete.index,
            "condition": "reference",
            "PC1": reference_xy[:, 0],
            "PC2": reference_xy[:, 1],
        }
    )
    projected_coordinates = profiles.index.to_frame(index=False)
    projected_coordinates["PC1"] = projected_xy[:, 0]
    projected_coordinates["PC2"] = projected_xy[:, 1]
    coordinates = pd.concat(
        [reference_coordinates, projected_coordinates], ignore_index=True
    )
    coordinates.to_csv(output_dir / "pca_coordinates.csv", index=False)
    pd.DataFrame(
        {
            "component": range(1, len(explained_variance_ratio) + 1),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance_ratio": np.cumsum(
                explained_variance_ratio
            ),
        }
    ).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ["demographics", "combined", "qualitative"]
    sources = list(projected_coordinates["source"].drop_duplicates())
    palette = ["#1d4ed8", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    source_colors = {source: palette[index % len(palette)] for index, source in enumerate(sources)}
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    reference_lookup = {
        persona_id: reference_xy[index, :2]
        for index, persona_id in enumerate(human_complete.index)
    }
    for axis, condition in zip(axes, conditions):
        axis.scatter(
            reference_xy[:, 0], reference_xy[:, 1], s=13, alpha=0.16,
            color="#6b7280", label="Human reference",
        )
        condition_rows = projected_coordinates.loc[
            projected_coordinates["condition"] == condition
        ]
        for source in sources:
            source_rows = condition_rows.loc[condition_rows["source"] == source]
            if source_rows.empty:
                continue
            marker = "o" if source == "Human annotators" else "^"
            color = source_colors[source]
            for row in source_rows.itertuples(index=False):
                target = reference_lookup.get(str(row.persona_id))
                if target is not None:
                    axis.plot(
                        [target[0], row.PC1], [target[1], row.PC2],
                        color=color, alpha=0.12, linewidth=0.7,
                    )
            axis.scatter(
                source_rows["PC1"], source_rows["PC2"], s=48,
                alpha=0.82, color=color, marker=marker, label=source,
            )
        axis.set_title(CONDITION_LABELS.get(condition, condition))
        axis.axhline(0, color="#9ca3af", linewidth=0.6)
        axis.axvline(0, color="#9ca3af", linewidth=0.6)
    pc1 = explained_variance_ratio[0] * 100
    pc2 = explained_variance_ratio[1] * 100
    figure.supxlabel(f"Human-reference PCA component 1 ({pc1:.1f}% variance)")
    figure.supylabel(f"Human-reference PCA component 2 ({pc2:.1f}% variance)")
    figure.suptitle("Annotator and model profiles in 20-item human PCA space")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(2, len(labels)), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_dir / "pca_human_model_by_condition.png", dpi=200)
    plt.close(figure)
    return len(human_complete), len(profiles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human annotators and models on matched pilot assignments."
    )
    parser.add_argument("annotations_csv", type=Path)
    parser.add_argument(
        "model_responses_csv",
        type=Path,
        nargs="+",
        help="One or more model-response CSV files to combine for evaluation.",
    )
    parser.add_argument("--human-csv", type=Path)
    parser.add_argument("--cutoff-participant", default=CUTOFF_PARTICIPANT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("evaluations/pilot_human_model")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = annotation_rows(args.annotations_csv, args.cutoff_participant)
    models = pd.concat(
        [model_rows(path) for path in args.model_responses_csv],
        ignore_index=True,
    )
    combined = pd.concat([annotations, models], ignore_index=True)
    aligned, assignment_count = align_common_assignments(combined)
    human_path = args.human_csv or download_twin_human_csv()
    human = load_human_subset(human_path)
    scored = score_rows(aligned, human)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "scored_responses.csv", index=False)
    profile_metrics(scored).to_csv(args.output_dir / "profile_metrics.csv", index=False)
    source_summary = metric_summary(
        scored, ["condition", "source", "agent_type", "provider", "model"]
    )
    source_summary.to_csv(args.output_dir / "metrics_by_condition_source.csv", index=False)
    agent_summary = metric_summary(scored, ["condition", "agent_type"])
    agent_summary.to_csv(args.output_dir / "metrics_by_condition_agent_type.csv", index=False)
    make_winner_report(agent_summary).to_csv(
        args.output_dir / "winner_by_condition.csv", index=False
    )
    human_count, projected_count = run_pca(human, aligned, args.output_dir)

    print(
        f"Compared {assignment_count} assignments shared by annotators and all models."
    )
    print(
        f"PCA fitted on {human_count} human-reference profiles and projected "
        f"{projected_count} annotator/model profiles."
    )
    print(f"Reports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

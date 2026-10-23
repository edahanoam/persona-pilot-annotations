"""Add Big Five trait labels and summarize prediction accuracy by trait."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BFI_TRAIT_BY_QUESTION = {
    **{number: "Extraversion" for number in (1, 6, 11, 16, 21, 26, 31, 36)},
    **{number: "Agreeableness" for number in (2, 7, 12, 17, 22, 27, 32, 37, 42)},
    **{
        number: "Conscientiousness"
        for number in (3, 8, 13, 18, 23, 28, 33, 38, 43)
    },
    **{number: "Neuroticism" for number in (4, 9, 14, 19, 24, 29, 34, 39)},
    **{number: "Openness" for number in (5, 10, 15, 20, 25, 30, 35, 40, 41, 44)},
}


def metrics(group: pd.DataFrame) -> dict[str, object]:
    return {
        "responses": len(group),
        "exact_matches": int(group["is_match"].sum()),
        "exact_accuracy": float(group["is_match"].mean()),
        "within_one_matches": int(group["is_within_one"].sum()),
        "within_one_accuracy": float(group["is_within_one"].mean()),
        "mean_absolute_error": float(group["absolute_error"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add BFI trait labels and compare accuracy by trait."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluations/pilot_human_model/scored_responses.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/pilot_human_model"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {
        "question_number",
        "answer_value",
        "reference_answer",
        "condition",
        "source",
        "agent_type",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    data["question_number"] = pd.to_numeric(
        data["question_number"], errors="raise"
    ).astype(int)
    data["answer_value"] = pd.to_numeric(data["answer_value"], errors="raise")
    data["reference_answer"] = pd.to_numeric(
        data["reference_answer"], errors="raise"
    )
    data["bfi_trait"] = data["question_number"].map(BFI_TRAIT_BY_QUESTION)
    if data["bfi_trait"].isna().any():
        missing_numbers = sorted(
            data.loc[data["bfi_trait"].isna(), "question_number"].unique()
        )
        raise ValueError(f"No BFI trait mapping for questions: {missing_numbers}")

    data["absolute_error"] = (
        data["answer_value"] - data["reference_answer"]
    ).abs()
    data["is_match"] = data["absolute_error"].eq(0)
    data["is_within_one"] = data["absolute_error"].le(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        args.output_dir / "scored_responses_with_bfi_traits.csv", index=False
    )

    detailed_rows: list[dict[str, object]] = []
    for keys, group in data.groupby(
        ["bfi_trait", "condition", "source", "agent_type"], sort=True
    ):
        trait, condition, source, agent_type = keys
        detailed_rows.append(
            {
                "bfi_trait": trait,
                "condition": condition,
                "source": source,
                "agent_type": agent_type,
                **metrics(group),
            }
        )
    pd.DataFrame(detailed_rows).to_csv(
        args.output_dir / "accuracy_by_bfi_trait_and_condition.csv", index=False
    )

    humans = data.loc[data["agent_type"] == "human_annotator"]
    models = data.loc[data["agent_type"] == "model"]
    comparison_rows: list[dict[str, object]] = []
    for source, model_rows in models.groupby("source", sort=True):
        for trait in sorted(data["bfi_trait"].unique()):
            human_metrics = metrics(humans.loc[humans["bfi_trait"] == trait])
            model_metrics = metrics(model_rows.loc[model_rows["bfi_trait"] == trait])
            comparison_rows.append(
                {
                    "bfi_trait": trait,
                    "model": source,
                    "human_exact_accuracy": human_metrics["exact_accuracy"],
                    "model_exact_accuracy": model_metrics["exact_accuracy"],
                    "human_within_one_accuracy": human_metrics[
                        "within_one_accuracy"
                    ],
                    "model_within_one_accuracy": model_metrics[
                        "within_one_accuracy"
                    ],
                    "human_mean_absolute_error": human_metrics[
                        "mean_absolute_error"
                    ],
                    "model_mean_absolute_error": model_metrics[
                        "mean_absolute_error"
                    ],
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.output_dir / "accuracy_comparison_by_bfi_trait.csv", index=False
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

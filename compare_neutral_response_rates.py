"""Compare how often annotators and models choose the neutral response (3)."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_COLUMNS = ["participant_id", "persona_id", "condition", "question_id"]


def paired_permutation_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided sign-flip test over participant-level differences."""
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return float("nan")

    observed = abs(float(differences.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(differences)):
        permuted = differences * np.asarray(signs)
        if abs(float(permuted.mean())) >= observed - 1e-12:
            extreme += 1
        total += 1
    return extreme / total


def compare_group(
    paired: pd.DataFrame, *, source: str, condition: str
) -> dict[str, object]:
    human_rate = float(paired["human_is_three"].mean())
    model_rate = float(paired["model_is_three"].mean())
    reference_rate = float(paired["reference_is_three"].mean())
    participant_differences = np.asarray(
        [
            float(group["model_is_three"].mean() - group["human_is_three"].mean())
            for _, group in paired.groupby("participant_id", sort=True)
        ]
    )
    return {
        "source": source,
        "condition": condition,
        "participants": paired["participant_id"].nunique(),
        "paired_responses": len(paired),
        "reference_three_rate": reference_rate,
        "human_three_rate": human_rate,
        "model_three_rate": model_rate,
        "human_minus_reference_percentage_points": (human_rate - reference_rate) * 100,
        "model_minus_reference_percentage_points": (model_rate - reference_rate) * 100,
        "model_minus_human_percentage_points": (model_rate - human_rate) * 100,
        "model_to_human_rate_ratio": (
            model_rate / human_rate if human_rate > 0 else float("nan")
        ),
        "participant_level_permutation_p": paired_permutation_pvalue(
            participant_differences
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare neutral-response rates for annotators and models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluations/pilot_human_model/scored_responses.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluations/pilot_human_model/neutral_response_comparison.csv"
        ),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {
        *PAIR_COLUMNS,
        "source",
        "agent_type",
        "answer_value",
        "reference_answer",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    data["answer_value"] = pd.to_numeric(data["answer_value"], errors="raise")
    data["reference_answer"] = pd.to_numeric(
        data["reference_answer"], errors="raise"
    )
    humans = data.loc[
        data["agent_type"] == "human_annotator",
        [*PAIR_COLUMNS, "answer_value", "reference_answer"],
    ].rename(columns={"answer_value": "human_answer"})

    rows: list[dict[str, object]] = []
    for source, model in data.loc[data["agent_type"] == "model"].groupby(
        "source", sort=True
    ):
        model = model[[*PAIR_COLUMNS, "answer_value"]].rename(
            columns={"answer_value": "model_answer"}
        )
        paired = humans.merge(
            model, on=PAIR_COLUMNS, how="inner", validate="one_to_one"
        )
        paired["reference_is_three"] = paired["reference_answer"].eq(3).astype(int)
        paired["human_is_three"] = paired["human_answer"].eq(3).astype(int)
        paired["model_is_three"] = paired["model_answer"].eq(3).astype(int)

        rows.append(compare_group(paired, source=str(source), condition="all_conditions"))
        for condition, condition_rows in paired.groupby("condition", sort=True):
            rows.append(
                compare_group(
                    condition_rows,
                    source=str(source),
                    condition=str(condition),
                )
            )

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

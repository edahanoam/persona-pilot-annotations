"""Compare inferred BFI-44 answers with humans and project into human PCA space.

The human answers are downloaded from LLM-Digital-Twin/Twin-2K-500 unless a
local --human-csv is supplied. PCA is fitted only on the aligned human vectors;
the model vectors are transformed with that fitted PCA.
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


HF_REPO = "LLM-Digital-Twin/Twin-2K-500"
HF_HUMAN_FILE = (
    "question_catalog_and_human_response_csv/wave1_3_response.csv"
)
VERSIONS = ("full", "d", "dm")
BFI_PREFIX = "QID25_"
N_ITEMS = 44


def normalize_pid(value: object) -> str:
    """Normalize CSV identifiers such as 7, '7', and 7.0 to the same key."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_model_answers(path: Path, expected_version: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"pid": str})
    required = {"pid", "persona_version", "question_number", "answer_value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    versions = set(frame["persona_version"].dropna().astype(str))
    if versions != {expected_version}:
        raise ValueError(
            f"{path} should contain version {expected_version!r}; found {versions}"
        )

    result = frame.loc[:, ["pid", "question_number", "answer_value"]].copy()
    result["pid"] = result["pid"].map(normalize_pid)
    result["question_number"] = pd.to_numeric(
        result["question_number"], errors="raise"
    ).astype(int)
    result["answer_value"] = pd.to_numeric(
        result["answer_value"], errors="raise"
    ).astype(float)

    if not result["question_number"].between(1, N_ITEMS).all():
        raise ValueError(f"{path} contains a question number outside 1-{N_ITEMS}")
    if not result["answer_value"].between(1, 5).all():
        raise ValueError(f"{path} contains an answer outside the 1-5 scale")
    if result.duplicated(["pid", "question_number"]).any():
        raise ValueError(f"{path} contains duplicate participant/question rows")

    matrix = result.pivot(
        index="pid", columns="question_number", values="answer_value"
    ).sort_index(axis=1)
    expected_items = list(range(1, N_ITEMS + 1))
    if list(matrix.columns) != expected_items:
        missing_items = sorted(set(expected_items) - set(matrix.columns))
        raise ValueError(f"{path} is missing BFI items: {missing_items}")
    incomplete = matrix.index[matrix.isna().any(axis=1)].tolist()
    if incomplete:
        raise ValueError(f"{path} has incomplete participants: {incomplete[:10]}")
    return matrix


def find_bfi_columns(columns: list[str]) -> list[str]:
    candidates = [name for name in columns if name.startswith(BFI_PREFIX)]

    def suffix_number(name: str) -> int:
        suffix = name.removeprefix(BFI_PREFIX)
        if not suffix.isdigit():
            raise ValueError(f"Unexpected BFI column name: {name}")
        return int(suffix)

    ordered = sorted(candidates, key=suffix_number)
    suffixes = [suffix_number(name) for name in ordered]
    if len(ordered) != N_ITEMS or suffixes != list(range(1, N_ITEMS + 1)):
        raise ValueError(
            f"Expected QID25_1 through QID25_44; found {len(ordered)} columns "
            f"with suffixes {suffixes}"
        )
    return ordered


def load_human_answers(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if "pid" not in header.columns:
        raise ValueError(f"{path} has no 'pid' column")
    bfi_columns = find_bfi_columns(list(header.columns))
    frame = pd.read_csv(path, usecols=["pid", *bfi_columns], dtype={"pid": str})
    frame["pid"] = frame["pid"].map(normalize_pid)
    if frame["pid"].duplicated().any():
        raise ValueError("Human response CSV contains duplicate pid values")

    numeric = frame[bfi_columns].apply(pd.to_numeric, errors="coerce")
    numeric.columns = range(1, N_ITEMS + 1)
    numeric.index = frame["pid"]
    numeric.index.name = "pid"
    return numeric


def comparison_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left - right
    left_flat = left.ravel()
    right_flat = right.ravel()
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return {
        "n_persons": float(left.shape[0]),
        "n_answers": float(left.size),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "exact_agreement": float(np.mean(left == right)),
        "pearson_r": correlation,
        "mean_person_euclidean": float(
            np.mean(np.linalg.norm(difference, axis=1))
        ),
    }


def make_comparison_reports(
    matrices: dict[str, pd.DataFrame], output_dir: Path
) -> pd.DataFrame:
    pairs = [(version, "human") for version in VERSIONS]
    pairs.extend(combinations(VERSIONS, 2))
    summary_rows: list[dict[str, object]] = []
    person_rows: list[dict[str, object]] = []
    question_rows: list[dict[str, object]] = []

    for left_name, right_name in pairs:
        left = matrices[left_name].to_numpy(dtype=float)
        right = matrices[right_name].to_numpy(dtype=float)
        label = f"{left_name}_vs_{right_name}"
        summary_rows.append(
            {
                "comparison": label,
                "left": left_name,
                "right": right_name,
                **comparison_metrics(left, right),
            }
        )

        for row_index, pid in enumerate(matrices[left_name].index):
            metrics = comparison_metrics(
                left[row_index : row_index + 1], right[row_index : row_index + 1]
            )
            person_rows.append({"comparison": label, "pid": pid, **metrics})

        for item_index in range(N_ITEMS):
            item_left = left[:, item_index]
            item_right = right[:, item_index]
            difference = item_left - item_right
            correlation = (
                float(np.corrcoef(item_left, item_right)[0, 1])
                if np.std(item_left) > 0 and np.std(item_right) > 0
                else math.nan
            )
            question_rows.append(
                {
                    "comparison": label,
                    "question_number": item_index + 1,
                    "mae": float(np.mean(np.abs(difference))),
                    "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                    "exact_agreement": float(np.mean(item_left == item_right)),
                    "pearson_r": correlation,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    pd.DataFrame(person_rows).to_csv(
        output_dir / "comparison_by_person.csv", index=False
    )
    pd.DataFrame(question_rows).to_csv(
        output_dir / "comparison_by_question.csv", index=False
    )
    return summary


def save_aligned_answers(
    matrices: dict[str, pd.DataFrame], output_dir: Path
) -> None:
    rows: list[dict[str, object]] = []
    for pid in matrices["human"].index:
        for question_number in range(1, N_ITEMS + 1):
            rows.append(
                {
                    "pid": pid,
                    "question_number": question_number,
                    **{
                        source: matrices[source].loc[pid, question_number]
                        for source in ("human", *VERSIONS)
                    },
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "aligned_answers.csv", index=False)


def run_pca(matrices: dict[str, pd.DataFrame], output_dir: Path) -> None:
    from sklearn.decomposition import PCA

    human = matrices["human"].to_numpy(dtype=float)
    pca = PCA(n_components=min(human.shape[0], human.shape[1]))
    pca.fit(human)

    coordinate_rows: list[dict[str, object]] = []
    transformed: dict[str, np.ndarray] = {}
    for source in ("human", *VERSIONS):
        transformed[source] = pca.transform(matrices[source].to_numpy(dtype=float))
        for row_index, pid in enumerate(matrices[source].index):
            coordinate_rows.append(
                {
                    "pid": pid,
                    "source": source,
                    "PC1": transformed[source][row_index, 0],
                    "PC2": transformed[source][row_index, 1],
                }
            )

    pd.DataFrame(coordinate_rows).to_csv(
        output_dir / "pca_coordinates.csv", index=False
    )
    pd.DataFrame(
        {
            "component": range(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    ).to_csv(output_dir / "pca_explained_variance.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"human": "#111827", "full": "#2563eb", "d": "#dc2626", "dm": "#16a34a"}
    labels = {
        "human": "Human",
        "full": "Full profile",
        "d": "Demographics",
        "dm": "Demographics + qualitative",
    }
    figure, axis = plt.subplots(figsize=(10, 8))
    human_xy = transformed["human"][:, :2]
    for source in VERSIONS:
        model_xy = transformed[source][:, :2]
        for human_point, model_point in zip(human_xy, model_xy):
            axis.plot(
                [human_point[0], model_point[0]],
                [human_point[1], model_point[1]],
                color=colors[source],
                alpha=0.10,
                linewidth=0.7,
            )
    for source in ("human", *VERSIONS):
        points = transformed[source]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=34 if source == "human" else 25,
            alpha=0.82,
            color=colors[source],
            label=labels[source],
        )
    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100
    axis.set_xlabel(f"Human PCA component 1 ({pc1:.1f}% variance)")
    axis.set_ylabel(f"Human PCA component 2 ({pc2:.1f}% variance)")
    axis.set_title("BFI-44 model answers projected into human PCA space")
    axis.axhline(0, color="#9ca3af", linewidth=0.6)
    axis.axvline(0, color="#9ca3af", linewidth=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "pca_projection.png", dpi=200)
    plt.close(figure)


def download_human_csv() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=HF_REPO, filename=HF_HUMAN_FILE, repo_type="dataset")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare human and model BFI-44 answers and run human-fitted PCA."
    )
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--d", type=Path, required=True)
    parser.add_argument("--dm", type=Path, required=True)
    parser.add_argument(
        "--human-csv",
        type=Path,
        help="Local wave1_3_response.csv; downloaded from Hugging Face if omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/bfi44"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_matrices = {
        version: load_model_answers(getattr(args, version), version)
        for version in VERSIONS
    }
    human_path = args.human_csv or download_human_csv()
    human = load_human_answers(human_path)

    common_pids = set(human.index)
    for matrix in model_matrices.values():
        common_pids &= set(matrix.index)
    ordered_pids = sorted(common_pids, key=lambda pid: (not pid.isdigit(), int(pid) if pid.isdigit() else pid))
    if len(ordered_pids) < 2:
        raise ValueError(
            f"Only {len(ordered_pids)} participants are shared across human/full/d/dm"
        )

    matrices = {"human": human.loc[ordered_pids]}
    matrices.update(
        {version: matrix.loc[ordered_pids] for version, matrix in model_matrices.items()}
    )
    for source, matrix in matrices.items():
        if matrix.isna().any().any():
            missing = int(matrix.isna().sum().sum())
            raise ValueError(f"{source} has {missing} missing answers after alignment")

    save_aligned_answers(matrices, args.output_dir)
    summary = make_comparison_reports(matrices, args.output_dir)
    run_pca(matrices, args.output_dir)

    coverage_rows = [
        {"source": "human", "input_personas": len(human), "aligned_personas": len(ordered_pids)}
    ]
    coverage_rows.extend(
        {
            "source": version,
            "input_personas": len(model_matrices[version]),
            "aligned_personas": len(ordered_pids),
        }
        for version in VERSIONS
    )
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "coverage.csv", index=False)

    print(f"Aligned {len(ordered_pids)} participants across all four sources.")
    print(summary.to_string(index=False))
    print(f"\nReports saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

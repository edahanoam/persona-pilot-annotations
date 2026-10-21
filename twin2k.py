"""Twin-2K data loading and replaceable persona assignment policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

from create_short_personas import (
    DATASET_CONFIG,
    DATASET_NAME,
    QUALITATIVE_MARKER,
    create_short_versions,
)
from run_bfi44_inference import ANSWER_LABELS, BFI44_ITEMS
from sampling import Sampler, StableHashSampler


class TwinPersona(TypedDict):
    persona_id: str
    demographics: str
    qualitative: str
    combined: str


PersonaCondition = Literal["demographics", "combined", "qualitative"]


class PersonaRound(TypedDict):
    persona: TwinPersona
    condition: PersonaCondition


def _qualitative_from_combined(demographics: str, combined: str) -> str:
    """Support saved datasets that predate persona_qualitative."""
    marker_position = combined.casefold().find(QUALITATIVE_MARKER.casefold())
    if marker_position != -1:
        return combined[marker_position:].strip()
    prefix = demographics.strip()
    if combined.startswith(prefix):
        return combined[len(prefix) :].strip()
    raise ValueError("Could not separate the qualitative persona section")


def _normalize_persona(row: dict[str, object]) -> TwinPersona:
    persona_id = str(row.get("pid", "")).strip()
    demographics = str(row.get("persona_demographics", "")).strip()
    combined = str(row.get("persona_demographics_and_qualitative", "")).strip()
    qualitative = str(row.get("persona_qualitative", "")).strip()
    if not qualitative and demographics and combined:
        qualitative = _qualitative_from_combined(demographics, combined)
    if not persona_id or not demographics or not qualitative or not combined:
        raise ValueError(
            "Every Twin-2K persona needs pid, demographics, qualitative, and combined text"
        )
    return {
        "persona_id": persona_id,
        "demographics": demographics,
        "qualitative": qualitative,
        "combined": combined,
    }


def _load_huggingface_personas() -> list[TwinPersona]:
    """Stream only the two columns needed from the public Hugging Face dataset."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading Twin-2K requires the 'datasets' package"
        ) from exc

    try:
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split="data",
            streaming=True,
        ).select_columns(["pid", "persona_summary"])
        personas: list[TwinPersona] = []
        for row in dataset:
            shortened = create_short_versions(str(row["persona_summary"]))
            personas.append(
                _normalize_persona(
                    {
                        "pid": row["pid"],
                        **shortened,
                    }
                )
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {DATASET_NAME} from Hugging Face: {exc}"
        ) from exc
    return personas


def load_twin_personas(path: Path | None = None) -> list[TwinPersona]:
    """Load Twin-2K online, or use an explicitly configured local cache."""
    configured = os.environ.get("TWIN2K_DATASET_PATH")
    if path is None and not configured:
        personas = _load_huggingface_personas()
        if not personas:
            raise ValueError("The Twin-2K Hugging Face dataset is empty")
        return personas

    source = path or Path(configured)  # type: ignore[arg-type]
    if not source.exists():
        raise FileNotFoundError(
            f"Configured Twin-2K data was not found at {source}"
        )

    if source.is_file():
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        try:
            from datasets import load_from_disk
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Loading the generated Twin-2K dataset requires the 'datasets' package"
            ) from exc
        rows = load_from_disk(str(source))

    personas = [_normalize_persona(dict(row)) for row in rows]
    if not personas:
        raise ValueError("The Twin-2K persona collection is empty")
    return personas


def assign_persona(
    participant_id: str,
    personas: list[TwinPersona],
    sampler: Sampler | None = None,
) -> tuple[TwinPersona, str]:
    policy = sampler or StableHashSampler()
    return personas[policy.choose_index(participant_id, len(personas))], policy.name


def assign_persona_rounds(
    participant_id: str,
    personas: list[TwinPersona],
    sampler: Sampler | None = None,
) -> tuple[list[PersonaRound], str]:
    """Assign three distinct personas and a stable randomized condition order."""
    policy = sampler or StableHashSampler()
    persona_indices = policy.choose_indices(participant_id, len(personas), 3)
    conditions: list[PersonaCondition] = [
        "demographics",
        "combined",
        "qualitative",
    ]
    condition_indices = policy.choose_indices(f"{participant_id}:conditions", 3, 3)
    rounds = [
        {
            "persona": personas[persona_index],
            "condition": conditions[condition_index],
        }
        for persona_index, condition_index in zip(persona_indices, condition_indices)
    ]
    return rounds, policy.name


def bfi44_questions() -> list[dict[str, object]]:
    return [
        {
            "id": f"bfi44_{number:02d}",
            "number": number,
            "question": item,
            "description": "How accurately does this statement describe the persona?",
            "choices": dict(ANSWER_LABELS),
        }
        for number, item in enumerate(BFI44_ITEMS, start=1)
    ]

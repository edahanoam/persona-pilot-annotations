"""Replaceable assignment policies shared by all experiments."""

from __future__ import annotations

import hashlib
import random
from typing import Protocol


class Sampler(Protocol):
    name: str

    def choose_index(self, participant_id: str, population_size: int) -> int: ...

    def choose_indices(
        self, participant_id: str, population_size: int, count: int
    ) -> list[int]: ...


class StableHashSampler:
    """Random-looking assignment that is stable for a participant ID."""

    name = "stable_hash_v1"

    def choose_index(self, participant_id: str, population_size: int) -> int:
        if population_size < 1:
            raise ValueError("Cannot sample from an empty collection")
        digest = hashlib.sha256(participant_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % population_size

    def choose_indices(
        self, participant_id: str, population_size: int, count: int
    ) -> list[int]:
        if population_size < 1:
            raise ValueError("Cannot sample from an empty collection")
        if count < 1 or count > population_size:
            raise ValueError("Sample count must be between 1 and the collection size")
        digest = hashlib.sha256(f"{participant_id}:batch".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big")).sample(
            range(population_size), count
        )

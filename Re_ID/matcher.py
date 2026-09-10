from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchCandidate:
    track_key: tuple[str, int, int]
    similarity: float


class CosineMatcher:
    def __init__(self):
        self._embeddings: dict[tuple[str, int, int], np.ndarray] = {}

    def register(self, track_key: tuple[str, int, int], embedding: np.ndarray) -> None:
        self._embeddings[track_key] = embedding

    def query(self, embedding: np.ndarray, exclude_camera_id: str | None = None,
              top_k: int = 5, min_similarity: float = 0.0) -> list[MatchCandidate]:
        candidates: list[MatchCandidate] = []
        for track_key, stored_emb in self._embeddings.items():
            if exclude_camera_id is not None and track_key[0] == exclude_camera_id:
                continue
            similarity = float(np.dot(embedding, stored_emb))
            if similarity >= min_similarity:
                candidates.append(MatchCandidate(track_key=track_key, similarity=similarity))

        candidates.sort(key=lambda c: c.similarity, reverse=True)
        return candidates[:top_k]

    def __len__(self) -> int:
        return len(self._embeddings)
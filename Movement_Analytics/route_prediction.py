from __future__ import annotations

from collections import defaultdict


class MarkovRoutePredictor:
    def __init__(self):
        self._transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_from: dict[str, int] = defaultdict(int)

    def observe_sequence(self, camera_sequence: list[str]) -> None:
        for i in range(len(camera_sequence) - 1):
            frm, to = camera_sequence[i], camera_sequence[i + 1]
            self._transitions[frm][to] += 1
            self._total_from[frm] += 1

    def fit(self, sequences: dict[str, list[str]]) -> None:
        self._transitions = defaultdict(lambda: defaultdict(int))
        self._total_from = defaultdict(int)
        for camera_sequence in sequences.values():
            self.observe_sequence(camera_sequence)

    def predict_next(self, current_camera: str, top_k: int = 3) -> list[tuple[str, float]]:
        total = self._total_from.get(current_camera, 0)
        if total == 0:
            return []

        counts = self._transitions[current_camera]
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [(camera, count / total) for camera, count in ranked[:top_k]]

    def transition_count(self, camera_from: str, camera_to: str) -> int:
        return self._transitions.get(camera_from, {}).get(camera_to, 0)

    def observed_cameras(self) -> set[str]:
        return set(self._total_from.keys())


def fit_from_db(conn, min_sightings: int = 2) -> MarkovRoutePredictor:
    from .movement_history import get_all_plate_sequences

    predictor = MarkovRoutePredictor()
    sequences = get_all_plate_sequences(conn, min_sightings=min_sightings)
    predictor.fit(sequences)
    return predictor
"""
Step 4 (body-type half) — coarse shape bucket, rule-based.
"""

from __future__ import annotations


def estimate_body_type(vehicle_class_name: str, bbox_xyxy: tuple[float, float, float, float]) -> tuple[str, float]:
    if vehicle_class_name != "car":
        return vehicle_class_name, 1.0

    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    if height <= 0:
        return "unknown", 0.0

    ratio = width / height

    bins = [
        ("hatchback", 0.0, 1.6),
        ("sedan", 1.6, 2.1),
        ("suv", 2.1, 3.0),
    ]

    for name, lo, hi in bins:
        if lo <= ratio < hi:
            mid = (lo + hi) / 2
            half_width = (hi - lo) / 2
            confidence = 1.0 - min(abs(ratio - mid) / half_width, 1.0) * 0.5
            return name, round(confidence, 2)

    if ratio < bins[0][2]:
        return bins[0][0], 0.3
    return bins[-1][0], 0.3
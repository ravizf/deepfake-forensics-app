"""High-level inference labeling helpers for SnapTrace."""

from __future__ import annotations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def final_label(fake_score: float):
    """Map a normalized fake score to a user-facing label and risk tier."""
    fake_score = _clamp01(fake_score)
    if fake_score >= 0.70:
        return "Fake", "High"
    if fake_score <= 0.30:
        return "Real", "Low"
    return "Uncertain", "Medium"

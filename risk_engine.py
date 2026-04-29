"""Risk score helpers for reviewer-facing result summaries."""

from __future__ import annotations


def calculate_risk_score(fake_score, metadata_found, face_detected=True):
    score = float(fake_score or 0.0)
    if score <= 1.0:
        score *= 100.0

    if str(metadata_found).strip().lower() == "no":
        score += 8.0

    if not face_detected:
        score += 5.0

    score = max(0.0, min(score, 100.0))

    if score >= 70:
        risk_level = "High"
    elif score >= 40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return round(score, 2), risk_level

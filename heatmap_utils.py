"""Prototype heatmap helpers and labels."""

from __future__ import annotations


def heatmap_label(analysis_mode):
    if analysis_mode == "trained_model":
        return "Heatmap-style visual evidence"
    return "Prototype visual evidence"

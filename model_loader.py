"""Wrapper utilities for detector loading and status exposure."""

from __future__ import annotations

from forensics import detector_descriptor as _detector_descriptor


def detector_descriptor():
    return _detector_descriptor()


def get_model_status():
    descriptor = detector_descriptor()
    return {
        "status": descriptor.get("status"),
        "mode": descriptor.get("mode"),
        "engine": descriptor.get("engine"),
        "detector_version": descriptor.get("detector_version"),
        "model_version": descriptor.get("model_version"),
        "dataset_version": descriptor.get("dataset_version"),
        "architecture": descriptor.get("architecture"),
    }

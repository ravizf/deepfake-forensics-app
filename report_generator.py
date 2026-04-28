"""Downloadable report generation wrapper."""

from __future__ import annotations

from forensics import generate_report_document


def generate_downloadable_report(analysis, audit_trail, report_dir):
    return generate_report_document(
        analysis=analysis,
        audit_trail=audit_trail,
        report_dir=report_dir,
    )

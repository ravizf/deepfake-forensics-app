"""Downloadable report generation wrapper."""

from __future__ import annotations

import os
from datetime import datetime

from forensics import generate_report_document


def generate_downloadable_report(analysis, audit_trail, report_dir):
    try:
        from fpdf import FPDF
    except Exception:
        return generate_report_document(
            analysis=analysis,
            audit_trail=audit_trail,
            report_dir=report_dir,
        )

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"prototype_report_{analysis['analysis_id']}.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "SnapTrace Prototype Report", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.cell(0, 8, f"Prediction: {analysis.get('display_prediction') or analysis.get('prediction')}", ln=True)
    pdf.cell(0, 8, f"Confidence: {analysis.get('confidence', 'N/A')}%", ln=True)
    pdf.cell(
        0,
        8,
        f"Risk Score: {analysis.get('risk_score', analysis.get('fraud_score', 'N/A'))}/100",
        ln=True,
    )
    pdf.cell(0, 8, f"Risk Level: {analysis.get('display_risk_level') or analysis.get('risk_level')}", ln=True)
    pdf.cell(0, 8, f"Model Status: {analysis.get('model_status_label', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Metadata Found: {analysis.get('metadata_found', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Face Detected: {analysis.get('face_detected', 'N/A')}", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Why This Result?", ln=True)
    pdf.set_font("Arial", "", 11)
    for reason in analysis.get("analysis_reasons") or []:
        pdf.multi_cell(0, 7, f"- {reason}")

    metadata_summary = analysis.get("metadata_summary") or {}
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Metadata", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Camera: {metadata_summary.get('camera_make') or 'Not found'} {metadata_summary.get('camera_model') or ''}".strip(), ln=True)
    pdf.cell(0, 7, f"Software: {metadata_summary.get('software') or 'Not found'}", ln=True)
    pdf.multi_cell(0, 7, f"Metadata Check: {analysis.get('metadata_check') or 'N/A'}")

    image_path = analysis.get("stored_path")
    if image_path and os.path.exists(image_path):
        try:
            pdf.ln(3)
            pdf.image(image_path, x=10, w=80)
        except Exception:
            pass

    heatmap_path = analysis.get("heatmap_path")
    if heatmap_path and os.path.exists(heatmap_path):
        try:
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Heatmap Evidence", ln=True)
            pdf.ln(2)
            pdf.image(heatmap_path, x=10, w=170)
        except Exception:
            pass

    pdf.ln(60)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0,
        6,
        analysis.get("prototype_notice")
        or "This is AI-assisted analysis, not legal proof.",
    )
    pdf.output(report_path)
    return report_path

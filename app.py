import os
import hashlib
import mimetypes
import json
import secrets
import smtplib
from functools import wraps
from datetime import datetime, timedelta
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

from database import (
    create_analysis,
    create_media_upload,
    create_report,
    create_user,
    ensure_public_demo_user,
    get_admin_summary,
    get_analysis_detail,
    get_case_audit_logs,
    get_dashboard_summary,
    get_user_by_email,
    get_user_by_id,
    get_user_by_verification_token_hash,
    init_db,
    list_user_history,
    log_audit_event,
    mark_user_email_verified,
    set_user_email_verification_token,
    set_upload_status,
    verify_user_credentials,
)
from evaluation import (
    EVALUATION_ROOT,
    evaluation_dataset_exists,
    load_latest_evaluation,
    run_evaluation,
)
from metadata_utils import extract_media_metadata as read_media_metadata
from model_loader import detector_descriptor
from risk_engine import calculate_risk_score
from forensics import (
    allowed_file,
    analyze_media_file,
    infer_media_type,
)
from report_generator import generate_downloadable_report


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
CORS(app)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "snaptrace-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
app.config["HEATMAP_FOLDER"] = os.path.join(app.root_path, "artifacts", "heatmaps")
app.config["REPORT_FOLDER"] = os.path.join(app.root_path, "artifacts", "reports")
app.config["PUBLIC_API_ENABLED"] = os.getenv("PUBLIC_API_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
app.config["EMAIL_VERIFICATION_TOKEN_TTL_HOURS"] = int(
    os.getenv("EMAIL_VERIFICATION_TOKEN_TTL_HOURS", "48")
)
app.config["SHOW_EMAIL_VERIFICATION_LINK"] = os.getenv(
    "SHOW_EMAIL_VERIFICATION_LINK", "true"
).lower() in {"1", "true", "yes", "on"}
app.config["SMTP_HOST"] = os.getenv("SMTP_HOST")
app.config["SMTP_PORT"] = int(os.getenv("SMTP_PORT", "587"))
app.config["SMTP_USERNAME"] = os.getenv("SMTP_USERNAME")
app.config["SMTP_PASSWORD"] = os.getenv("SMTP_PASSWORD")
app.config["SMTP_SENDER"] = os.getenv("SMTP_SENDER") or os.getenv("SMTP_USERNAME")
app.config["SMTP_USE_TLS"] = os.getenv("SMTP_USE_TLS", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
app.config["SMTP_USE_SSL"] = os.getenv("SMTP_USE_SSL", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

for folder in (
    app.config["UPLOAD_FOLDER"],
    app.config["HEATMAP_FOLDER"],
    app.config["REPORT_FOLDER"],
):
    os.makedirs(folder, exist_ok=True)

init_db()

DEMO_SAMPLE_ROOTS = [
    Path(app.root_path) / "static" / "demo",
    Path(app.root_path) / "static" / "demo-samples",
    Path(app.root_path) / "dataset" / "test",
]
DEMO_SAMPLE_LIMIT_PER_CLASS = 2
DEMO_SAMPLE_CATALOG = [
    {
        "id": "real-selfie",
        "filename": "real-selfie.jpg",
        "label": "real",
        "title": "Real selfie",
    },
    {
        "id": "ai-face",
        "filename": "ai-face.jpg",
        "label": "fake",
        "title": "AI face",
    },
    {
        "id": "compressed-whatsapp",
        "filename": "compressed-whatsapp.jpg",
        "label": "real",
        "title": "Compressed WhatsApp-style image",
    },
    {
        "id": "cropped-image",
        "filename": "cropped-image.jpg",
        "label": "review",
        "title": "Cropped image",
    },
    {
        "id": "low-quality",
        "filename": "low-quality.jpg",
        "label": "review",
        "title": "Low-quality image",
    },
    {
        "id": "low-light",
        "filename": "low-light.jpg",
        "label": "real",
        "title": "Low-light image",
    },
]
MODEL_METRICS_PATH = Path(app.root_path) / "model_metrics.json"
MODEL_MANIFEST_PATH = Path(app.root_path) / "model_manifest.json"
SYNC_REPORT_GENERATION = os.getenv("SYNC_REPORT_GENERATION", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HISTORY_PAGE_LIMIT = int(os.getenv("HISTORY_PAGE_LIMIT", "25"))
MAX_BATCH_UPLOADS = int(os.getenv("MAX_BATCH_UPLOADS", "10"))


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def verification_token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def smtp_is_configured():
    return bool(app.config["SMTP_HOST"] and app.config["SMTP_SENDER"])


def send_plain_email(to_email, subject, body):
    if not smtp_is_configured():
        return False, "SMTP email is not configured for this deployment."

    message = EmailMessage()
    message["From"] = app.config["SMTP_SENDER"]
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    try:
        if app.config["SMTP_USE_SSL"]:
            server = smtplib.SMTP_SSL(
                app.config["SMTP_HOST"],
                app.config["SMTP_PORT"],
                timeout=15,
            )
        else:
            server = smtplib.SMTP(
                app.config["SMTP_HOST"],
                app.config["SMTP_PORT"],
                timeout=15,
            )
        with server:
            if app.config["SMTP_USE_TLS"] and not app.config["SMTP_USE_SSL"]:
                server.starttls()
            if app.config["SMTP_USERNAME"] and app.config["SMTP_PASSWORD"]:
                server.login(app.config["SMTP_USERNAME"], app.config["SMTP_PASSWORD"])
            server.send_message(message)
        return True, None
    except Exception as exc:
        app.logger.exception("Verification email failed")
        return False, str(exc)


def send_email_verification(user):
    token = secrets.token_urlsafe(32)
    set_user_email_verification_token(user["id"], verification_token_hash(token))
    verification_url = url_for("verify_email", token=token, _external=True)
    sent, error = send_plain_email(
        user["email"],
        "Verify your SnapTrace account",
        "\n".join(
            [
                f"Hi {user['full_name']},",
                "",
                "Verify your SnapTrace Forensics account using this link:",
                verification_url,
                "",
                (
                    "This verification link expires in "
                    f"{app.config['EMAIL_VERIFICATION_TOKEN_TTL_HOURS']} hours."
                ),
                "",
                "If you did not create this account, you can ignore this email.",
            ]
        ),
    )
    return {
        "sent": sent,
        "error": error,
        "verification_url": verification_url
        if app.config["SHOW_EMAIL_VERIFICATION_LINK"] or not sent
        else None,
    }


def verification_token_is_expired(user):
    sent_at = user.get("email_verification_sent_at")
    if not sent_at:
        return True
    try:
        sent_time = datetime.fromisoformat(sent_at)
    except ValueError:
        return True
    expires_at = sent_time + timedelta(
        hours=app.config["EMAIL_VERIFICATION_TOKEN_TTL_HOURS"]
    )
    return datetime.utcnow() > expires_at


def render_verification_sent(user, delivery):
    return render_template(
        "verification_sent.html",
        title="Verify Email",
        email=user["email"],
        sent=delivery["sent"],
        verification_url=delivery.get("verification_url"),
        error_message=delivery.get("error"),
        token_hours=app.config["EMAIL_VERIFICATION_TOKEN_TTL_HOURS"],
        smtp_configured=smtp_is_configured(),
    )


def detector_status_label(detector, benchmark_report=None):
    if benchmark_report:
        return "Benchmark Available"
    return "Benchmark Pending"


def detector_status_note(detector, benchmark_report=None):
    label = detector_status_label(detector, benchmark_report)
    if label == "Benchmark Available":
        return "Benchmark metrics are available for the current detector setup."
    if detector.get("status") == "error":
        return "Checkpoint is unavailable or incompatible, so benchmark reporting is still pending."
    return "Checkpoint is loaded, but benchmark metrics have not been published yet."


def analysis_engine_label(_detector=None):
    return "Active"


def detection_mode_label(_detector=None):
    return "AI-Assisted Prototype"


def evaluation_status_label(_detector=None, benchmark_report=None, metrics_report=None):
    if benchmark_report or metrics_report:
        return "Benchmark Available"
    return "Benchmark Pending"


def manifest_descriptor():
    if not MODEL_MANIFEST_PATH.exists():
        return {
            "engine": "SnapTrace Analysis Engine",
            "architecture": "efficientnet_b0_binary",
            "model_version": "N/A",
            "dataset_version": "Evaluation Pending",
            "status": "metadata_only",
        }
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "engine": "SnapTrace Analysis Engine",
            "architecture": "efficientnet_b0_binary",
            "model_version": "N/A",
            "dataset_version": "Evaluation Pending",
            "status": "metadata_unavailable",
        }
    return {
        "engine": manifest.get("display_name") or "SnapTrace Analysis Engine",
        "architecture": manifest.get("architecture", "efficientnet_b0_binary"),
        "detector_version": manifest.get("detector_version"),
        "model_version": manifest.get("model_version"),
        "dataset_version": manifest.get("dataset_version"),
        "training_date": manifest.get("training_date"),
        "temperature": manifest.get("temperature"),
        "calibration_method": manifest.get("calibration_method"),
        "status": "metadata_only",
    }


def list_demo_samples():
    samples = []
    catalog_root = Path(app.root_path) / "static" / "demo"
    for item in DEMO_SAMPLE_CATALOG:
        path = catalog_root / item["filename"]
        if path.is_file():
            samples.append(
                {
                    **item,
                    "path": path,
                    "preview_url": url_for("static", filename=f"demo/{item['filename']}"),
                }
            )
    if samples:
        return samples

    label_map = {
        "real": "Real sample",
        "fake": "AI-generated sample",
    }
    sample_roots = [root for root in DEMO_SAMPLE_ROOTS if root.is_dir()]
    for label, title in label_map.items():
        files = []
        for root in sample_roots:
            canonical_files = [
                root / f"{label}1.jpg",
                root / f"{label}2.jpg",
            ]
            existing_canonical = [path for path in canonical_files if path.is_file()]
            if existing_canonical:
                files = existing_canonical
                break
            sample_dir = root / label
            if sample_dir.is_dir():
                files = sorted(p for p in sample_dir.iterdir() if p.is_file())
                if files:
                    break
        for path in files[:DEMO_SAMPLE_LIMIT_PER_CLASS]:
            preview_url = None
            if "static" in path.parts:
                static_index = path.parts.index("static") + 1
                static_path = "/".join(path.parts[static_index:])
                preview_url = url_for("static", filename=static_path)
            samples.append(
                {
                    "id": f"{label}-{path.stem}",
                    "label": label,
                    "title": title,
                    "filename": path.name,
                    "path": path,
                    "preview_url": preview_url,
                }
            )
    return samples


def get_demo_sample(sample_id):
    for sample in list_demo_samples():
        if sample["id"] == sample_id:
            return sample
    return None


def benchmark_summary(report, detector=None):
    if not report:
        return {
            "accuracy": "Evaluation Pending",
            "precision": "Evaluation Pending",
            "recall": "Evaluation Pending",
            "f1_score": "Evaluation Pending",
            "test_dataset": "Evaluation Pending",
        }

    precision = float(report.get("ai_metrics", {}).get("precision") or 0.0)
    recall = float(report.get("ai_metrics", {}).get("recall") or 0.0)
    f1_score = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "accuracy": f"{report.get('accuracy')}%",
        "precision": f"{precision}%",
        "recall": f"{recall}%",
        "f1_score": f"{round(f1_score, 2)}%",
        "test_dataset": report.get("dataset_version") or detector.get("dataset_version") or "Not declared",
    }


def load_model_metrics():
    if not MODEL_METRICS_PATH.exists():
        return None
    try:
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def training_metrics_summary(metrics_report, benchmark_report=None, detector=None):
    architecture = (detector or {}).get("architecture")
    model_name = {
        "efficientnet_b0_binary": "EfficientNet-B0",
        "resnet50_binary": "ResNet-50",
        "resnet18_binary": "ResNet-18",
    }.get(architecture, architecture or "EfficientNet-B0")

    if metrics_report:
        def _metric_value(key):
            value = metrics_report.get(key, "Pending")
            if isinstance(value, (int, float)):
                return f"{value}%"
            return str(value)

        return {
            "accuracy": _metric_value("accuracy"),
            "precision": _metric_value("precision"),
            "recall": _metric_value("recall"),
            "f1_score": _metric_value("f1_score"),
            "test_dataset": metrics_report.get("dataset_version") or "Benchmark Available After Training",
            "model_name": metrics_report.get("model") or "EfficientNet-B0",
            "evaluation_split": metrics_report.get("evaluation_split") or "val",
            "sample_count": metrics_report.get("sample_count") or "N/A",
            "notes": metrics_report.get("notes") or "",
            "external_benchmarks": metrics_report.get("external_benchmarks") or [],
            "per_class_accuracy": metrics_report.get("per_class_accuracy") or {},
        }

    fallback = benchmark_summary(benchmark_report, detector)
    fallback["model_name"] = model_name
    fallback["evaluation_split"] = "evaluation"
    fallback["sample_count"] = "N/A"
    fallback["notes"] = ""
    fallback["external_benchmarks"] = []
    fallback["per_class_accuracy"] = {}
    return fallback


def percentage(part, total):
    if not total:
        return 0
    return round((float(part) / float(total)) * 100, 1)


def dashboard_visual_summary(summary, training_metrics):
    total = summary.get("total_cases") or 0
    return {
        "fake_percent": percentage(summary.get("fake_cases") or 0, total),
        "real_percent": percentage(summary.get("real_cases") or 0, total),
        "review_percent": percentage(summary.get("review_cases") or 0, total),
        "accuracy": training_metrics.get("accuracy", "Pending"),
        "precision": training_metrics.get("precision", "Pending"),
        "recall": training_metrics.get("recall", "Pending"),
        "f1_score": training_metrics.get("f1_score", "Pending"),
    }


def image_quality_warning(analysis):
    if analysis.get("media_type") != "image":
        return "Video analysis uses the fallback scoring path in this prototype."

    face_count = int(analysis.get("face_count") or 0)
    if face_count == 0:
        return "Image-level visual patterns analyzed."
    if analysis.get("confidence_band") in {"Low", "Review Required"}:
        return "Low-confidence image. Compression, blur, or weak facial detail may reduce reliability."
    return "No major image-quality warning was triggered during this run."


def extract_media_metadata(analysis):
    return read_media_metadata(
        analysis.get("stored_path"),
        media_type=analysis.get("media_type"),
        file_sha256=analysis.get("file_sha256"),
    )


def metadata_check_text(analysis):
    metadata = extract_media_metadata(analysis)
    if metadata["media_type"] != "image":
        return metadata["warning"] or "Metadata inspection is limited for non-image files."
    if metadata["exif_present"]:
        parts = ["EXIF present"]
        if metadata["camera_make"] or metadata["camera_model"]:
            parts.append(
                f"camera={metadata['camera_make'] or 'unknown'} {metadata['camera_model'] or ''}".strip()
            )
        if metadata["software"]:
            parts.append(f"software={metadata['software']}")
        return " | ".join(parts)
    if metadata["warning"]:
        return metadata["warning"]
    return (
        "No EXIF metadata found. This can happen in edited, compressed, camera-exported, "
        "or AI-generated images. Missing metadata alone does not confirm manipulation."
    )


def analysis_reasons(analysis):
    reasons = []
    fake_score = float(analysis.get("fake_prob") or 0.0)
    real_score = float(analysis.get("real_prob") or 0.0)
    face_count = int(analysis.get("face_count") or 0)
    detector_breakdown = analysis.get("detector_breakdown") or {}
    artifact_score = float(detector_breakdown.get("artifact_detector") or 0.0)
    frequency_score = float(detector_breakdown.get("frequency_detector") or 0.0)
    diffusion_score = float(detector_breakdown.get("diffusion_gan_detector") or 0.0)

    if face_detection_label(analysis) == "Estimated region":
        reasons.append("Estimated face-region crop used after image-level scan.")
    elif face_count > 0:
        if artifact_score >= 0.55:
            reasons.append("Face texture and blending artifacts were elevated.")
        else:
            reasons.append("Face texture and blending artifacts stayed low.")
    else:
        reasons.append("Image-level visual patterns analyzed.")

    if frequency_score >= 0.55:
        reasons.append("Compression and frequency artifacts were elevated.")
    else:
        reasons.append("Compression and frequency artifacts did not dominate the score.")

    if diffusion_score >= 0.55 or fake_score >= 70:
        reasons.append("Synthetic-generation pattern score was elevated.")
    elif real_score >= 70:
        reasons.append("Visual score favored natural image patterns over synthetic ones.")
    else:
        reasons.append("Visual score stayed close enough to require context review.")

    if artifact_score >= 0.55:
        reasons.append("Lighting, texture, or edge consistency signals showed a possible mismatch.")
    else:
        reasons.append("Lighting, texture, and edge consistency did not show a strong mismatch.")

    metadata_message = metadata_check_text(analysis)
    if analysis.get("metadata_summary", {}).get("exif_present"):
        reasons.append("Camera metadata is present in the uploaded file.")
    elif metadata_message:
        reasons.append("Metadata is unavailable, so confidence should be read as visual-only.")

    if not reasons:
        if fake_score > real_score:
            reasons.append("The model score leaned toward AI-generated characteristics.")
        else:
            reasons.append("The model score leaned toward natural-photo characteristics.")

    return reasons[:5]


def decision_summary(analysis):
    prediction = str(analysis.get("display_prediction") or analysis.get("prediction") or "")
    risk_level = str(analysis.get("risk_level") or "").lower()
    confidence = float(analysis.get("confidence") or 0.0)

    if prediction == "Real" and risk_level == "low":
        return "Low risk because visual analysis strongly favors a real image."
    if prediction == "AI Generated" and risk_level == "high":
        return "High risk due to strong indicators of synthetic generation."
    if prediction == "Uncertain":
        return "Review needed because the visual score is not strongly separated."
    if confidence >= 75:
        return f"{analysis.get('risk_level', 'Medium')} risk because the visual score shows a clear lead."
    return "Review recommended because the visual score and supporting signals are mixed."


def confidence_warning(analysis):
    if analysis.get("metadata_found") == "No":
        return (
            "Confidence is based mainly on visual analysis because EXIF metadata is unavailable. "
            "Missing metadata alone does not confirm manipulation."
        )
    if analysis.get("confidence_band") in {"Low", "Review Required"}:
        return "Confidence is limited because the detector score is close to the review range."
    return "Confidence combines visual scoring, artifact checks, and available metadata context."


def face_detection_label(analysis):
    evidence_text = " ".join(str(item) for item in analysis.get("evidence_points") or [])
    if analysis.get("face_strategy") == "estimated-face-region" or "estimated-face-region" in evidence_text:
        return "Estimated region"
    if int(analysis.get("face_count") or 0) > 0:
        return "Yes"
    return "No"


def fraud_score_and_risk(analysis):
    metadata_found = analysis.get("metadata_found") or "No"
    face_detected = int(analysis.get("face_count") or 0) > 0
    return calculate_risk_score(
        analysis.get("fake_prob") or 0.0,
        metadata_found,
        face_detected=face_detected,
    )


def ensure_public_demo_case(analysis):
    demo_user = ensure_public_demo_user()
    if not analysis or analysis.get("user_id") != demo_user["id"]:
        abort(403)


def enrich_public_demo_analysis(analysis):
    analysis = enrich_analysis(analysis)
    if not analysis:
        return None
    analysis["result_url"] = url_for(
        "public_result_page", analysis_id=analysis["analysis_id"]
    )
    if analysis.get("heatmap_path"):
        analysis["heatmap_url"] = url_for(
            "public_artifact_file",
            kind="heatmaps",
            filename=os.path.basename(analysis["heatmap_path"]),
        )
    analysis["evidence_url"] = url_for("public_evidence_page", analysis_id=analysis["analysis_id"])
    analysis["report_url"] = None
    analysis["report_download_url"] = url_for(
        "public_download_report", analysis_id=analysis["analysis_id"]
    )
    analysis["report_artifact_url"] = None
    return analysis


@app.before_request
def load_user():
    g.current_user = current_user()


@app.context_processor
def inject_globals():
    active_detector = manifest_descriptor()
    metrics_report = load_model_metrics()
    return {
        "current_user": g.get("current_user"),
        "app_name": "SnapTrace Forensics",
        "active_detector": active_detector,
        "active_detector_label": detector_status_label(active_detector, metrics_report),
        "active_detector_note": detector_status_note(active_detector, metrics_report),
        "analysis_engine_label": analysis_engine_label(active_detector),
        "detection_mode_label": detection_mode_label(active_detector),
        "evaluation_status_label": evaluation_status_label(
            active_detector,
            None,
            metrics_report,
        ),
    }


def audit(action, target_type, target_id=None, details=""):
    user = g.get("current_user")
    log_audit_event(
        user_id=user["id"] if user else None,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details,
        ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not g.get("current_user"):
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        user = g.get("current_user")
        if not user:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login"))
        if user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view


def enrich_analysis(analysis):
    if not analysis:
        return None

    analysis = dict(analysis)
    heatmap_path = analysis.get("heatmap_path")
    report_path = analysis.get("report_path")

    analysis["heatmap_url"] = (
        url_for("artifact_file", kind="heatmaps", filename=os.path.basename(heatmap_path))
        if heatmap_path
        else None
    )
    analysis["report_download_url"] = (
        url_for("download_report", analysis_id=analysis["analysis_id"])
    )
    analysis["report_artifact_url"] = (
        url_for("artifact_file", kind="reports", filename=os.path.basename(report_path))
        if report_path
        else None
    )
    analysis["result_url"] = url_for("result_page", analysis_id=analysis["analysis_id"])
    analysis["evidence_url"] = url_for("evidence_page", analysis_id=analysis["analysis_id"])
    analysis["report_url"] = url_for("report_page", analysis_id=analysis["analysis_id"])
    analysis["source_api_url"] = url_for(
        "api_source_attribution", analysis_id=analysis["analysis_id"]
    )
    latest_report = load_latest_evaluation()
    analysis["analysis_engine_label"] = analysis_engine_label()
    analysis["detection_mode_label"] = detection_mode_label()
    metrics_report = load_model_metrics()
    analysis["evaluation_status_label"] = evaluation_status_label(
        manifest_descriptor(),
        latest_report,
        metrics_report,
    )
    analysis["detector_badge"] = analysis["evaluation_status_label"]
    analysis["model_status_label"] = analysis["evaluation_status_label"]
    analysis["face_detected"] = face_detection_label(analysis)
    analysis["image_quality_warning"] = image_quality_warning(analysis)
    analysis["metadata_check"] = metadata_check_text(analysis)
    analysis["metadata_summary"] = extract_media_metadata(analysis)
    analysis["metadata_found"] = "Yes" if analysis["metadata_summary"].get("exif_present") else "No"
    analysis["analysis_reasons"] = analysis_reasons(analysis)
    analysis["risk_score"], analysis["risk_level"] = fraud_score_and_risk(analysis)
    analysis["fraud_score"] = analysis["risk_score"]
    analysis["display_prediction"] = (
        str(analysis.get("prediction") or "").replace("AI-Generated", "AI Generated")
    )
    analysis["display_risk_level"] = analysis["risk_level"]
    analysis["decision_summary"] = decision_summary(analysis)
    analysis["confidence_warning"] = confidence_warning(analysis)
    analysis["prototype_notice"] = "This is AI-assisted analysis and not legal proof."
    return analysis


def enrich_history_case(analysis, public=False):
    if not analysis:
        return None

    analysis = dict(analysis)
    if public:
        analysis["result_url"] = url_for(
            "public_result_page", analysis_id=analysis["analysis_id"]
        )
        analysis["evidence_url"] = url_for(
            "public_evidence_page", analysis_id=analysis["analysis_id"]
        )
        analysis["report_url"] = analysis["result_url"]
        analysis["report_download_url"] = url_for(
            "public_download_report", analysis_id=analysis["analysis_id"]
        )
    else:
        analysis["result_url"] = url_for("result_page", analysis_id=analysis["analysis_id"])
        analysis["evidence_url"] = url_for("evidence_page", analysis_id=analysis["analysis_id"])
        analysis["report_url"] = url_for("report_page", analysis_id=analysis["analysis_id"])
        analysis["report_download_url"] = url_for(
            "download_report", analysis_id=analysis["analysis_id"]
        )
    analysis["display_prediction"] = (
        str(analysis.get("prediction") or "").replace("AI-Generated", "AI Generated")
    )
    analysis["face_detected"] = face_detection_label(analysis)
    analysis["risk_score"], analysis["risk_level"] = calculate_risk_score(
        analysis.get("fake_prob") or 0.0,
        "No",
        face_detected=int(analysis.get("face_count") or 0) > 0,
    )
    analysis["detector_badge"] = analysis.get("analysis_mode") or "analysis"
    return analysis


def build_analysis_response(analysis):
    return {
        "analysis_id": analysis["analysis_id"],
        "prediction": analysis.get("display_prediction") or analysis["prediction"],
        "raw_prediction": analysis["prediction"],
        "display_prediction": analysis.get("display_prediction"),
        "binary_prediction": analysis.get("binary_prediction"),
        "leaning_prediction": analysis.get("leaning_prediction"),
        "confidence": analysis["confidence"],
        "confidence_band": analysis.get("confidence_band"),
        "review_status": analysis.get("review_status"),
        "model_status_label": analysis.get("model_status_label"),
        "analysis_engine_label": analysis.get("analysis_engine_label"),
        "detection_mode_label": analysis.get("detection_mode_label"),
        "evaluation_status_label": analysis.get("evaluation_status_label"),
        "face_detected": analysis.get("face_detected"),
        "image_quality_warning": analysis.get("image_quality_warning"),
        "metadata_check": analysis.get("metadata_check"),
        "metadata_summary": analysis.get("metadata_summary"),
        "metadata_found": analysis.get("metadata_found"),
        "analysis_reasons": analysis.get("analysis_reasons"),
        "risk_score": analysis.get("risk_score", analysis.get("fraud_score")),
        "fraud_score": analysis.get("fraud_score"),
        "risk_level": analysis.get("display_risk_level") or analysis.get("risk_level"),
        "decision_summary": analysis.get("decision_summary"),
        "confidence_warning": analysis.get("confidence_warning"),
        "prototype_notice": analysis.get("prototype_notice"),
        "fake_prob": analysis.get("fake_prob"),
        "real_prob": analysis.get("real_prob"),
        "source_model": analysis["source_model"],
        "heatmap_url": analysis["heatmap_url"],
        "result_url": analysis["result_url"],
        "report_url": analysis["report_url"],
        "inference_engine": analysis.get("inference_engine"),
        "detector_version": analysis.get("detector_version"),
        "model_version": analysis.get("model_version"),
        "dataset_version": analysis.get("dataset_version"),
        "training_date": analysis.get("training_date"),
        "temperature": analysis.get("temperature"),
        "calibration_method": analysis.get("calibration_method"),
        "analysis_mode": analysis.get("analysis_mode"),
        "ensemble_fake_score": analysis.get("ensemble_fake_score"),
        "uncertainty_score": analysis.get("uncertainty_score"),
        "detector_breakdown": analysis.get("detector_breakdown"),
        "file_sha256": analysis.get("file_sha256"),
        "uploaded_at": analysis.get("uploaded_at"),
        "analyzed_at": analysis.get("created_at"),
    }


def ensure_case_access(analysis):
    user = g.get("current_user")
    if not analysis or not user:
        abort(404)
    if user["role"] == "admin":
        return
    if analysis["user_id"] != user["id"]:
        abort(403)


def persist_upload(file_storage):
    original_name = secure_filename(file_storage.filename or "")
    if not original_name:
        raise ValueError("Please choose a valid file.")

    extension = os.path.splitext(original_name)[1].lower()
    stored_name = f"{os.urandom(8).hex()}{extension}"
    destination = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file_storage.save(destination)
    size = os.path.getsize(destination)
    file_sha256 = sha256_file(destination)
    return destination, original_name, stored_name, size, file_sha256


def preprocess_image(file_path):
    try:
        with Image.open(file_path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            if max(normalized.size) > 960:
                normalized.thumbnail((960, 960), Image.Resampling.LANCZOS)
            save_kwargs = {}
            if os.path.splitext(file_path.lower())[1] in {".jpg", ".jpeg"}:
                save_kwargs = {"quality": 88, "optimize": True}
            normalized.save(file_path, **save_kwargs)
            return normalized.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Unsupported or corrupted image file.") from exc


def max_upload_size_mb():
    return round(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024), 1)


def ensure_report_record(analysis):
    if analysis.get("report_path") and os.path.exists(analysis["report_path"]):
        return analysis

    analysis = enrich_analysis(analysis)
    audit_trail = get_case_audit_logs(
        upload_id=analysis["upload_id"], analysis_id=analysis["analysis_id"]
    )
    report_path = generate_downloadable_report(
        analysis=analysis,
        audit_trail=audit_trail,
        report_dir=app.config["REPORT_FOLDER"],
    )
    create_report(analysis["analysis_id"], report_path, "ready")
    audit(
        "REPORT_GENERATED",
        "report",
        analysis["analysis_id"],
        f"Regenerated forensic report for analysis #{analysis['analysis_id']}",
    )
    return get_analysis_detail(analysis["analysis_id"])


def run_analysis_workflow(file_storage, acting_user=None, audit_prefix=None):
    user = acting_user or g.get("current_user")
    if not user:
        raise PermissionError("Authentication required.")

    if not allowed_file(file_storage.filename):
        raise ValueError("Unsupported file format. Upload an image or video file.")

    file_path, original_name, stored_name, size, file_sha256 = persist_upload(file_storage)
    media_type = infer_media_type(original_name)
    if media_type == "image":
        preprocess_image(file_path)

    upload_id = create_media_upload(
        user_id=user["id"],
        original_filename=original_name,
        stored_filename=stored_name,
        stored_path=file_path,
        media_type=media_type,
        file_size=size,
        file_sha256=file_sha256,
        status="processing",
    )
    audit(
        "UPLOAD_SUBMITTED",
        "upload",
        upload_id,
        f"{audit_prefix or 'Submitted'} {media_type} evidence file {original_name}",
    )

    try:
        active_detector = detector_descriptor()
        analysis_result = analyze_media_file(file_path, media_type, app.config["HEATMAP_FOLDER"])
        analysis_id = create_analysis(upload_id, analysis_result)
        set_upload_status(upload_id, "complete")
        audit(
            "ANALYSIS_COMPLETED",
            "analysis",
            analysis_id,
            " | ".join(
                [
                    f"Prediction={analysis_result['prediction']}",
                    f"confidence={analysis_result['confidence']}",
                    f"engine={analysis_result.get('inference_engine') or active_detector['engine']}",
                    f"detector_version={analysis_result.get('detector_version') or active_detector['detector_version']}",
                    f"model_version={analysis_result.get('model_version') or active_detector.get('model_version')}",
                    f"dataset_version={analysis_result.get('dataset_version') or active_detector.get('dataset_version')}",
                    f"confidence_band={analysis_result.get('confidence_band')}",
                ]
            ),
        )
        analysis = enrich_analysis(get_analysis_detail(analysis_id))
        if SYNC_REPORT_GENERATION:
            try:
                audit_trail = get_case_audit_logs(upload_id=upload_id, analysis_id=analysis_id)
                report_path = generate_downloadable_report(
                    analysis=analysis,
                    audit_trail=audit_trail,
                    report_dir=app.config["REPORT_FOLDER"],
                )
                create_report(analysis_id, report_path, "ready")
                audit(
                    "REPORT_GENERATED",
                    "report",
                    analysis_id,
                    f"Generated forensic report for analysis #{analysis_id}",
                )
            except Exception as report_exc:
                app.logger.exception("Report generation failed")
                audit(
                    "REPORT_GENERATION_FAILED",
                    "report",
                    analysis_id,
                    f"Report generation failed: {report_exc}",
                )
        else:
            audit(
                "REPORT_DEFERRED",
                "report",
                analysis_id,
                f"Report generation deferred until requested for analysis #{analysis_id}",
            )
        return enrich_analysis(get_analysis_detail(analysis_id))
    except Exception as exc:
        app.logger.exception("Analysis workflow failed")
        set_upload_status(upload_id, "failed")
        audit(
            "ANALYSIS_FAILED",
            "upload",
            upload_id,
            f"Processing failed for {original_name}",
        )
        raise


@app.route("/")
def home():
    latest_report = None
    active_detector = manifest_descriptor()
    metrics_report = load_model_metrics()
    return render_template(
        "home.html",
        title="Home",
        demo_samples=list_demo_samples(),
        latest_report=latest_report,
        benchmark_summary=benchmark_summary(latest_report, active_detector),
        training_metrics=training_metrics_summary(metrics_report, latest_report, active_detector),
        github_url="https://github.com/ravizf/deepfake-forensics-app",
    )


@app.route("/model")
def model_page():
    latest_report = None
    detector_status = manifest_descriptor()
    metrics_report = load_model_metrics()
    return render_template(
        "model.html",
        title="Model Details",
        detector_status=detector_status,
        latest_report=latest_report,
        benchmark_summary=benchmark_summary(latest_report, detector_status),
        training_metrics=training_metrics_summary(metrics_report, latest_report, detector_status),
        model_metrics_report=metrics_report,
    )


@app.route("/model-info")
def model_info_page():
    return model_page()


@app.route("/demo")
def demo_page():
    latest_report = None
    active_detector = manifest_descriptor()
    return render_template(
        "demo.html",
        title="Try Demo",
        demo_samples=list_demo_samples(),
        benchmark_summary=benchmark_summary(latest_report, active_detector),
    )


@app.route("/api-docs")
def api_docs_page():
    return render_template(
        "api_docs.html",
        title="API Docs",
        public_api_enabled=app.config["PUBLIC_API_ENABLED"],
    )


@app.route("/privacy")
def privacy_page():
    return render_template("privacy.html", title="Privacy")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not full_name or not email or not password:
            flash("Name, email, and password are required.", "danger")
        elif get_user_by_email(email):
            flash("That email is already registered.", "danger")
        else:
            user_id = create_user(
                full_name=full_name,
                email=email,
                password_hash=generate_password_hash(password),
            )
            user = get_user_by_id(user_id)
            delivery = send_email_verification(user)
            log_audit_event(
                user_id=user_id,
                action="USER_REGISTERED",
                target_type="user",
                target_id=user_id,
                details="Created account; email verification required",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
            log_audit_event(
                user_id=user_id,
                action="EMAIL_VERIFICATION_SENT",
                target_type="user",
                target_id=user_id,
                details="Sent email verification link"
                if delivery["sent"]
                else f"Email verification link generated locally: {delivery.get('error')}",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
            flash("Account created. Verify your email before signing in.", "success")
            return render_verification_sent(user, delivery)

    return render_template(
        "auth.html",
        title="Register",
        form_mode="register",
        page_heading="Create your forensic workspace",
        page_blurb="Register an analyst account to upload evidence, review reports, and maintain chain-of-custody records.",
        submit_label="Create Account",
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = verify_user_credentials(email, password)

        if not user:
            flash("Invalid email or password.", "danger")
        elif user.get("requires_email_verification"):
            delivery = send_email_verification(user)
            log_audit_event(
                user_id=user["id"],
                action="EMAIL_VERIFICATION_REQUIRED",
                target_type="user",
                target_id=user["id"],
                details="Login blocked until email verification is completed",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
            flash("Please verify your email before signing in.", "warning")
            return render_verification_sent(user, delivery)
        else:
            session["user_id"] = user["id"]
            log_audit_event(
                user_id=user["id"],
                action="USER_LOGIN",
                target_type="user",
                target_id=user["id"],
                details="User authenticated successfully",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
            flash("Welcome back.", "success")
            return redirect(url_for("dashboard"))

    return render_template(
        "auth.html",
        title="Login",
        form_mode="login",
        page_heading="Sign in to SnapTrace",
        page_blurb="Continue your forensic workflow, review flagged cases, and download evidence reports.",
        submit_label="Sign In",
    )


@app.route("/verify-email/<token>")
def verify_email(token):
    user = get_user_by_verification_token_hash(verification_token_hash(token))
    if not user:
        flash("Verification link is invalid or has already been used.", "danger")
        return redirect(url_for("resend_verification"))

    if user.get("email_verified_at"):
        flash("Your email is already verified. Please sign in.", "success")
        return redirect(url_for("login"))

    if verification_token_is_expired(user):
        flash("Verification link expired. Request a fresh link below.", "warning")
        return redirect(url_for("resend_verification", email=user["email"]))

    verified_user = mark_user_email_verified(user["id"])
    log_audit_event(
        user_id=user["id"],
        action="EMAIL_VERIFIED",
        target_type="user",
        target_id=user["id"],
        details="User verified email address",
        ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
    )
    flash(f"Email verified for {verified_user['email']}. You can sign in now.", "success")
    return redirect(url_for("login"))


@app.route("/resend-verification", methods=["GET", "POST"])
def resend_verification():
    prefill_email = request.args.get("email", "").strip().lower()
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = get_user_by_email(email)
        if user and user.get("email_verified_at"):
            flash("That email is already verified. Please sign in.", "success")
            return redirect(url_for("login"))
        if user:
            delivery = send_email_verification(user)
            log_audit_event(
                user_id=user["id"],
                action="EMAIL_VERIFICATION_RESENT",
                target_type="user",
                target_id=user["id"],
                details="Resent email verification link"
                if delivery["sent"]
                else f"Email verification link regenerated locally: {delivery.get('error')}",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
            flash("Verification instructions are ready.", "success")
            return render_verification_sent(user, delivery)

        flash(
            "If that email belongs to an unverified account, a verification link will be sent.",
            "success",
        )
        return redirect(url_for("login"))

    return render_template(
        "resend_verification.html",
        title="Resend Verification",
        email=prefill_email,
    )


@app.route("/logout")
@login_required
def logout():
    audit("USER_LOGOUT", "user", g.current_user["id"], "User ended the session")
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    summary = get_dashboard_summary(g.current_user["id"])
    summary["recent_cases"] = [enrich_analysis(case) for case in summary["recent_cases"]]
    detector = manifest_descriptor()
    metrics_report = load_model_metrics()
    training_metrics = training_metrics_summary(metrics_report, None, detector)
    return render_template(
        "dashboard.html",
        title="Dashboard",
        summary=summary,
        training_metrics=training_metrics,
        dashboard_visual=dashboard_visual_summary(summary, training_metrics),
    )


@app.route("/evaluation", methods=["GET", "POST"])
@login_required
def evaluation_page():
    if request.method == "POST":
        if g.current_user["role"] != "admin":
            abort(403)
        try:
            report = run_evaluation()
            flash(
                f"Evaluation completed on {report['sample_count']} samples. Accuracy: {report['accuracy']}%",
                "success",
            )
        except Exception as exc:
            flash(f"Evaluation failed: {exc}", "danger")
        return redirect(url_for("evaluation_page"))

    report = load_latest_evaluation()
    return render_template(
        "evaluation.html",
        title="Evaluation",
        report=report,
        detector_status=detector_descriptor(),
        dataset_root=EVALUATION_ROOT,
        dataset_exists=evaluation_dataset_exists(),
    )


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if request.method == "POST":
        try:
            files = [file for file in request.files.getlist("file") if file and file.filename]
            if not files:
                flash("Choose an image or video file to analyze.", "danger")
            else:
                if len(files) > MAX_BATCH_UPLOADS:
                    raise ValueError(
                        f"Upload up to {MAX_BATCH_UPLOADS} files at once for stable Render processing."
                    )

                is_public = not bool(g.get("current_user"))
                acting_user = g.get("current_user") or ensure_public_demo_user()
                batch_results = []
                for file_storage in files:
                    try:
                        analysis = run_analysis_workflow(
                            file_storage,
                            acting_user=acting_user,
                            audit_prefix="Submitted public upload" if is_public else None,
                        )
                        view_analysis = (
                            enrich_public_demo_analysis(analysis)
                            if is_public
                            else enrich_analysis(analysis)
                        )
                        batch_results.append(
                            {
                                "ok": True,
                                "filename": file_storage.filename,
                                "analysis": view_analysis,
                            }
                        )
                    except Exception as item_exc:
                        app.logger.exception("Batch upload item failed")
                        batch_results.append(
                            {
                                "ok": False,
                                "filename": file_storage.filename,
                                "error": str(item_exc),
                            }
                        )

                successful = [item for item in batch_results if item["ok"]]
                failed = [item for item in batch_results if not item["ok"]]
                if len(files) == 1 and successful:
                    flash("Analysis completed successfully.", "success")
                    analysis = successful[0]["analysis"]
                    return redirect(analysis["result_url"])

                if successful:
                    flash(
                        f"Analyzed {len(successful)} file(s)"
                        + (f"; {len(failed)} failed." if failed else "."),
                        "success" if not failed else "warning",
                    )
                    return render_template(
                        "upload_batch.html",
                        title="Batch Results",
                        batch_results=batch_results,
                        public_demo=is_public,
                    )

                raise ValueError(
                    failed[0]["error"] if failed else "No files could be analyzed."
                )
        except RequestEntityTooLarge:
            message = f"Upload is too large. Please use a file under {max_upload_size_mb()} MB."
            app.logger.warning("Upload rejected because it exceeded MAX_CONTENT_LENGTH")
            flash(message, "danger")
            return render_template(
                "upload.html",
                title="Upload Evidence",
                error_message=message,
                max_upload_mb=max_upload_size_mb(),
                max_batch_uploads=MAX_BATCH_UPLOADS,
            ), 413
        except ValueError as exc:
            app.logger.exception("Validation error during upload")
            flash(str(exc), "danger")
            return render_template(
                "upload.html",
                title="Upload Evidence",
                error_message=str(exc),
                max_upload_mb=max_upload_size_mb(),
                max_batch_uploads=MAX_BATCH_UPLOADS,
            ), 400
        except Exception as exc:
            app.logger.exception("Unhandled error during analysis")
            message = (
                "Analysis failed on the server. Please try a smaller JPG/PNG image, "
                "or check the Render logs for the full traceback."
            )
            flash(message, "danger")
            return render_template(
                "upload.html",
                title="Upload Evidence",
                error_message=message,
                technical_error=str(exc),
                max_upload_mb=max_upload_size_mb(),
                max_batch_uploads=MAX_BATCH_UPLOADS,
            ), 500

    return render_template(
        "upload.html",
        title="Upload Evidence",
        error_message=None,
        max_upload_mb=max_upload_size_mb(),
        max_batch_uploads=MAX_BATCH_UPLOADS,
    )


@app.route("/analysis/<int:analysis_id>")
@login_required
def result_page(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    return render_template("result.html", title="Analysis Result", analysis=analysis)


@app.route("/analysis/<int:analysis_id>/evidence")
@login_required
def evidence_page(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    return render_template("evidence.html", title="Visual Evidence", analysis=analysis)


@app.route("/analysis/<int:analysis_id>/report")
@login_required
def report_page(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    refreshed = ensure_report_record(analysis)
    analysis = enrich_analysis(refreshed)
    audit_trail = get_case_audit_logs(
        upload_id=analysis["upload_id"], analysis_id=analysis["analysis_id"]
    )
    return render_template(
        "report.html",
        title="Forensic Report",
        analysis=analysis,
        audit_trail=audit_trail,
    )


@app.route("/history")
def history_page():
    is_public = not bool(g.get("current_user"))
    user = g.get("current_user") or ensure_public_demo_user()
    include_all = (not is_public) and user["role"] == "admin"
    limit = 50 if include_all else HISTORY_PAGE_LIMIT
    try:
        history_rows = list_user_history(user["id"], include_all, limit=limit)
    except Exception as exc:
        app.logger.exception("History query failed")
        flash(f"History is temporarily unavailable: {exc}", "warning")
        history_rows = []
    cases = [enrich_history_case(case, public=is_public) for case in history_rows]
    return render_template(
        "history.html",
        title="Report History",
        cases=cases,
        limit=limit,
        public_history=is_public,
    )


@app.route("/admin")
@admin_required
def admin_page():
    summary = get_admin_summary()
    summary["recent_cases"] = [enrich_analysis(case) for case in summary["recent_cases"]]
    return render_template("admin.html", title="Admin Panel", summary=summary)


@app.route("/artifacts/<kind>/<path:filename>")
@login_required
def artifact_file(kind, filename):
    folders = {
        "heatmaps": app.config["HEATMAP_FOLDER"],
        "reports": app.config["REPORT_FOLDER"],
    }
    if kind not in folders:
        abort(404)
    return send_from_directory(folders[kind], filename)


@app.route("/public/artifacts/<kind>/<path:filename>")
def public_artifact_file(kind, filename):
    folders = {
        "heatmaps": app.config["HEATMAP_FOLDER"],
    }
    if kind not in folders:
        abort(404)
    return send_from_directory(folders[kind], filename)


@app.route("/download/report/<int:analysis_id>")
@login_required
def download_report(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    refreshed = ensure_report_record(analysis)
    analysis = enrich_analysis(refreshed)
    return send_file(
        analysis["report_path"],
        as_attachment=True,
        download_name=os.path.basename(analysis["report_path"]),
    )


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        analysis = run_analysis_workflow(file)
        return jsonify(build_analysis_response(analysis))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("API analyze failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/public-analyze", methods=["POST"])
def api_public_analyze():
    if not app.config["PUBLIC_API_ENABLED"]:
        return jsonify({"error": "Public API is disabled"}), 403

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        demo_user = ensure_public_demo_user()
        analysis = run_analysis_workflow(
            file,
            acting_user=demo_user,
            audit_prefix="Submitted public demo",
        )
        return jsonify(build_analysis_response(analysis))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Public API analyze failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def api_predict():
    if not app.config["PUBLIC_API_ENABLED"]:
        return jsonify({"error": "Public prediction API is disabled"}), 403

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        demo_user = ensure_public_demo_user()
        analysis = run_analysis_workflow(
            file,
            acting_user=demo_user,
            audit_prefix="Submitted public predict API request",
        )
        payload = build_analysis_response(analysis)
        return jsonify(
            {
                "prediction": payload["prediction"],
                "raw_prediction": payload.get("raw_prediction"),
                "binary_prediction": payload.get("binary_prediction"),
                "confidence": payload["confidence"],
                "risk_score": payload.get("risk_score"),
                "fraud_score": payload.get("fraud_score"),
                "risk_level": payload.get("risk_level"),
                "analysis_engine": payload.get("analysis_engine_label"),
                "detection_mode": payload.get("detection_mode_label"),
                "evaluation_status": payload.get("evaluation_status_label"),
                "face_detected": payload.get("face_detected"),
                "metadata_found": payload.get("metadata_found"),
                "image_quality_warning": payload.get("image_quality_warning"),
                "metadata_check": payload.get("metadata_check"),
                "analysis_reasons": payload.get("analysis_reasons"),
                "heatmap_url": payload.get("heatmap_url"),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Predict API failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/demo/sample/<sample_id>")
def run_demo_sample(sample_id):
    if not app.config["PUBLIC_API_ENABLED"]:
        abort(403)

    sample = get_demo_sample(sample_id)
    if not sample:
        abort(404)

    demo_user = ensure_public_demo_user()
    with sample["path"].open("rb") as stream:
        payload = BytesIO(stream.read())
        file_storage = FileStorage(
            stream=payload,
            filename=sample["filename"],
            content_type=mimetypes.guess_type(sample["filename"])[0] or "application/octet-stream",
        )
        analysis = run_analysis_workflow(
            file_storage,
            acting_user=demo_user,
            audit_prefix=f"Submitted demo sample ({sample['label']})",
        )

    return redirect(url_for("public_result_page", analysis_id=analysis["analysis_id"]))


@app.route("/demo/analysis/<int:analysis_id>")
def public_result_page(analysis_id):
    analysis = get_analysis_detail(analysis_id)
    ensure_public_demo_case(analysis)
    analysis = enrich_public_demo_analysis(analysis)
    return render_template("result.html", title="Demo Result", analysis=analysis, public_demo=True)


@app.route("/demo/analysis/<int:analysis_id>/evidence")
def public_evidence_page(analysis_id):
    analysis = get_analysis_detail(analysis_id)
    ensure_public_demo_case(analysis)
    analysis = enrich_public_demo_analysis(analysis)
    return render_template(
        "evidence.html",
        title="Demo Visual Evidence",
        analysis=analysis,
        public_demo=True,
    )


@app.route("/demo/download/report/<int:analysis_id>")
def public_download_report(analysis_id):
    analysis = get_analysis_detail(analysis_id)
    ensure_public_demo_case(analysis)
    refreshed = ensure_report_record(analysis)
    analysis = enrich_analysis(refreshed)
    return send_file(
        analysis["report_path"],
        as_attachment=True,
        download_name=os.path.basename(analysis["report_path"]),
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/health")
def api_health():
    detector = manifest_descriptor()
    return jsonify(
        {
            "status": "ok",
            "public_api_enabled": app.config["PUBLIC_API_ENABLED"],
            "detector": detector,
        }
    )


@app.route("/api/analysis/<int:analysis_id>")
@login_required
def api_analysis(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    return jsonify(analysis)


@app.route("/api/source-attribution/<int:analysis_id>")
@login_required
def api_source_attribution(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    return jsonify(
        {
            "analysis_id": analysis["analysis_id"],
            "source_model": analysis["source_model"],
            "attribution": analysis["attribution"],
            "summary": analysis["summary"],
        }
    )


@app.route("/api/report/<int:analysis_id>")
@login_required
def api_report(analysis_id):
    analysis = enrich_analysis(get_analysis_detail(analysis_id))
    ensure_case_access(analysis)
    refreshed = ensure_report_record(analysis)
    analysis = enrich_analysis(refreshed)
    return jsonify(
        {
            "analysis_id": analysis["analysis_id"],
            "report_path": analysis["report_path"],
            "download_url": analysis["report_download_url"],
            "viewer_url": analysis["report_artifact_url"],
            "generated_at": analysis["generated_at"],
        }
    )


@app.route("/api/evaluation/latest")
@login_required
def api_latest_evaluation():
    report = load_latest_evaluation()
    if report is None:
        return jsonify(
            {
                "error": "No evaluation report found",
                "dataset_root": EVALUATION_ROOT,
                "dataset_exists": evaluation_dataset_exists(),
            }
        ), 404
    return jsonify(report)


@app.route("/api/model-status")
@login_required
def api_model_status():
    return jsonify(detector_descriptor())


@app.errorhandler(403)
def forbidden(_error):
    return render_template("error.html", title="Access Denied", code=403), 403


@app.errorhandler(404)
def not_found(_error):
    return render_template("error.html", title="Not Found", code=404), 404


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    message = f"Upload is too large. Please use a file under {max_upload_size_mb()} MB."
    return render_template(
        "upload.html",
        title="Upload Evidence",
        error_message=message,
        max_upload_mb=max_upload_size_mb(),
        max_batch_uploads=MAX_BATCH_UPLOADS,
    ), 413


@app.errorhandler(500)
def internal_error(error):
    app.logger.error("Unhandled application error: %s", error, exc_info=True)
    return render_template(
        "error.html",
        title="Server Error",
        code=500,
        message=(
            "The server hit an unexpected error. If this happened during upload, "
            "try a smaller JPG/PNG image while the Render logs are checked."
        ),
    ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False,
        use_reloader=False,
    )

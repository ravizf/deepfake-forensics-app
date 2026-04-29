import os
import hashlib
import mimetypes
from functools import wraps
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
    init_db,
    list_user_history,
    log_audit_event,
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

for folder in (
    app.config["UPLOAD_FOLDER"],
    app.config["HEATMAP_FOLDER"],
    app.config["REPORT_FOLDER"],
):
    os.makedirs(folder, exist_ok=True)

init_db()

DEMO_SAMPLE_ROOT = Path(app.root_path) / "static" / "demo-samples"
DEMO_SAMPLE_LIMIT_PER_CLASS = 2


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def detector_status_label(detector, benchmark_report=None):
    if detector.get("mode") == "trained_model" and detector.get("status") == "loaded":
        if not benchmark_report:
            return "Prototype Model / Evaluation Pending"
        return "Trained Model"
    if detector.get("status") == "error":
        return "Fallback Mode"
    return "Demo Mode"


def detector_status_note(detector, benchmark_report=None):
    label = detector_status_label(detector, benchmark_report)
    if label == "Trained Model":
        return "Checkpoint loaded with calibrated scoring."
    if label == "Prototype Model / Evaluation Pending":
        return "Checkpoint is loaded, but benchmark metrics have not been published yet."
    if label == "Fallback Mode":
        return "Checkpoint unavailable or incompatible, so the demo fallback is active."
    return "Prototype heuristic mode is active until a trained checkpoint is available."


def list_demo_samples():
    samples = []
    label_map = {
        "real": "Real sample",
        "fake": "AI-generated sample",
    }
    for label, title in label_map.items():
        sample_dir = DEMO_SAMPLE_ROOT / label
        if not sample_dir.is_dir():
            continue
        for path in sorted(p for p in sample_dir.iterdir() if p.is_file())[:DEMO_SAMPLE_LIMIT_PER_CLASS]:
            samples.append(
                {
                    "id": f"{label}-{path.stem}",
                    "label": label,
                    "title": title,
                    "filename": path.name,
                    "path": path,
                    "preview_url": url_for(
                        "static",
                        filename=f"demo-samples/{label}/{path.name}",
                    ),
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
        dataset_name = None
        if detector:
            dataset_name = detector.get("dataset_version")
        return {
            "accuracy": "Evaluation pending",
            "precision": "Evaluation pending",
            "recall": "Evaluation pending",
            "f1_score": "Evaluation pending",
            "test_dataset": dataset_name or "Evaluation pending",
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


def image_quality_warning(analysis):
    if analysis.get("media_type") != "image":
        return "Video analysis uses the fallback scoring path in this prototype."

    face_count = int(analysis.get("face_count") or 0)
    if face_count == 0:
        return "No clear face was detected, so this result relied on full-image analysis only."
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
    return "No EXIF metadata found. This can happen in edited, compressed, or AI-generated images."


def analysis_reasons(analysis):
    reasons = []
    fake_score = float(analysis.get("fake_prob") or 0.0)
    real_score = float(analysis.get("real_prob") or 0.0)
    face_count = int(analysis.get("face_count") or 0)
    detector_breakdown = analysis.get("detector_breakdown") or {}

    if face_count > 0:
        reasons.append("Face detected and analyzed for blending or texture anomalies.")
    else:
        reasons.append("No clear face detected, so the decision relied on whole-image patterns.")

    if detector_breakdown.get("frequency_detector", 0) >= 0.55:
        reasons.append("Frequency-domain irregularities were stronger than typical natural-image patterns.")
    if detector_breakdown.get("artifact_detector", 0) >= 0.55:
        reasons.append("Texture inconsistency and sensor-noise mismatch raised the artifact score.")
    if detector_breakdown.get("diffusion_gan_detector", 0) >= 0.55:
        reasons.append("Synthetic texture signatures resembled common AI image artifacts.")

    metadata_message = metadata_check_text(analysis)
    if metadata_message:
        reasons.append(metadata_message)

    if not reasons:
        if fake_score > real_score:
            reasons.append("The model score leaned toward AI-generated characteristics.")
        else:
            reasons.append("The model score leaned toward natural-photo characteristics.")

    return reasons[:4]


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
    active_detector = detector_descriptor()
    latest_report = load_latest_evaluation()
    return {
        "current_user": g.get("current_user"),
        "app_name": "SnapTrace Forensics",
        "active_detector": active_detector,
        "active_detector_label": detector_status_label(active_detector, latest_report),
        "active_detector_note": detector_status_note(active_detector, latest_report),
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
        if report_path
        else None
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
    analysis_mode = analysis.get("analysis_mode")
    latest_report = load_latest_evaluation()
    if analysis_mode == "trained_model":
        model_status = (
            "Trained Model"
            if latest_report
            else "Prototype Model / Evaluation Pending"
        )
    elif "fallback" in str(analysis.get("inference_engine") or "").lower():
        model_status = "Fallback Mode"
    else:
        model_status = "Demo Mode"
    analysis["detector_badge"] = model_status
    analysis["model_status_label"] = model_status
    analysis["face_detected"] = "Yes" if int(analysis.get("face_count") or 0) > 0 else "No"
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
    analysis["prototype_notice"] = "This is not legal proof, only AI-assisted analysis."
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
        "face_detected": analysis.get("face_detected"),
        "image_quality_warning": analysis.get("image_quality_warning"),
        "metadata_check": analysis.get("metadata_check"),
        "metadata_summary": analysis.get("metadata_summary"),
        "metadata_found": analysis.get("metadata_found"),
        "analysis_reasons": analysis.get("analysis_reasons"),
        "risk_score": analysis.get("risk_score", analysis.get("fraud_score")),
        "fraud_score": analysis.get("fraud_score"),
        "risk_level": analysis.get("display_risk_level") or analysis.get("risk_level"),
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
            normalized.save(file_path)
            return normalized.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Unsupported or corrupted image file.") from exc


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
    latest_report = load_latest_evaluation()
    active_detector = detector_descriptor()
    return render_template(
        "home.html",
        title="Home",
        demo_samples=list_demo_samples(),
        latest_report=latest_report,
        benchmark_summary=benchmark_summary(latest_report, active_detector),
    )


@app.route("/model")
def model_page():
    latest_report = load_latest_evaluation()
    detector_status = detector_descriptor()
    return render_template(
        "model.html",
        title="Model Details",
        detector_status=detector_status,
        latest_report=latest_report,
        benchmark_summary=benchmark_summary(latest_report, detector_status),
    )


@app.route("/model-info")
def model_info_page():
    return model_page()


@app.route("/demo")
def demo_page():
    latest_report = load_latest_evaluation()
    active_detector = detector_descriptor()
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
            log_audit_event(
                user_id=user_id,
                action="USER_REGISTERED",
                target_type="user",
                target_id=user_id,
                details="Created account",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
            flash("Account created. Please sign in.", "success")
            return redirect(url_for("login"))

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
    return render_template("dashboard.html", title="Dashboard", summary=summary)


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
@login_required
def upload_page():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Choose an image or video file to analyze.", "danger")
        else:
            try:
                analysis = run_analysis_workflow(file)
                flash("Analysis completed successfully.", "success")
                return redirect(url_for("result_page", analysis_id=analysis["analysis_id"]))
            except ValueError as exc:
                app.logger.exception("Validation error during upload")
                flash(str(exc), "danger")
            except Exception as exc:
                app.logger.exception("Unhandled error during analysis")
                flash(f"Analysis failed: {exc}", "danger")
                return render_template(
                    "upload.html",
                    title="Upload Evidence",
                    error_message=str(exc),
                ), 500

    return render_template("upload.html", title="Upload Evidence", error_message=None)


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
@login_required
def history_page():
    include_all = g.current_user["role"] == "admin"
    cases = [enrich_analysis(case) for case in list_user_history(g.current_user["id"], include_all)]
    return render_template("history.html", title="Report History", cases=cases)


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
                "face_detected": payload.get("face_detected"),
                "metadata_found": payload.get("metadata_found"),
                "image_quality_warning": payload.get("image_quality_warning"),
                "metadata_check": payload.get("metadata_check"),
                "analysis_reasons": payload.get("analysis_reasons"),
                "model_status": payload.get("model_status_label"),
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
    detector = detector_descriptor()
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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        debug=False,
        use_reloader=False,
    )

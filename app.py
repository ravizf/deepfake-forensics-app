import os
import hashlib
from functools import wraps

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
from forensics import (
    allowed_file,
    analyze_media_file,
    detector_descriptor,
    generate_report_document,
    infer_media_type,
)


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


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


@app.before_request
def load_user():
    g.current_user = current_user()


@app.context_processor
def inject_globals():
    return {
        "current_user": g.get("current_user"),
        "app_name": "SnapTrace Forensics",
        "active_detector": detector_descriptor(),
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
    analysis["detector_badge"] = (
        "Trained Model" if analysis.get("analysis_mode") == "trained_model" else "Fallback Demo"
    )
    return analysis


def build_analysis_response(analysis):
    return {
        "analysis_id": analysis["analysis_id"],
        "prediction": analysis["prediction"],
        "binary_prediction": analysis.get("binary_prediction"),
        "leaning_prediction": analysis.get("leaning_prediction"),
        "confidence": analysis["confidence"],
        "confidence_band": analysis.get("confidence_band"),
        "review_status": analysis.get("review_status"),
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

    audit_trail = get_case_audit_logs(
        upload_id=analysis["upload_id"], analysis_id=analysis["analysis_id"]
    )
    report_path = generate_report_document(
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
        analysis = get_analysis_detail(analysis_id)
        try:
            audit_trail = get_case_audit_logs(upload_id=upload_id, analysis_id=analysis_id)
            report_path = generate_report_document(
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
    return render_template("home.html", title="Home")


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
                return render_template("upload.html", title="Upload Evidence"), 500

    return render_template("upload.html", title="Upload Evidence")


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

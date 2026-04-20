import json
import os
import sqlite3
from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash


DB_PATH = os.path.join(os.path.dirname(__file__), "snaptrace.db")
LEGACY_DETECTOR_VERSION = "snaptrace-heuristic-v1"
LEGACY_INFERENCE_ENGINE = "Legacy heuristic fallback"
LEGACY_ANALYSIS_MODE = "fallback_demo"
PUBLIC_DEMO_EMAIL = "public-demo@snaptrace.local"
PUBLIC_DEMO_NAME = "Public Demo"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'analyst',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS media_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_sha256 TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            binary_prediction TEXT,
            leaning_prediction TEXT,
            confidence REAL NOT NULL,
            confidence_band TEXT,
            review_status TEXT,
            fake_prob REAL,
            real_prob REAL,
            source_model TEXT NOT NULL,
            frequency_score REAL NOT NULL,
            face_count INTEGER NOT NULL,
            heatmap_path TEXT,
            ensemble_fake_score REAL,
            uncertainty_score REAL,
            detector_breakdown_json TEXT,
            attribution_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            inference_engine TEXT,
            detector_version TEXT,
            model_version TEXT,
            dataset_version TEXT,
            training_date TEXT,
            temperature REAL,
            calibration_method TEXT,
            analysis_mode TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (upload_id) REFERENCES media_uploads (id)
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL UNIQUE,
            report_path TEXT NOT NULL,
            report_status TEXT NOT NULL DEFAULT 'ready',
            generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id INTEGER,
            details TEXT NOT NULL,
            ip_address TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
    )
    _ensure_analysis_columns(connection)
    connection.commit()
    connection.close()


def _ensure_analysis_columns(connection):
    upload_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(media_uploads)").fetchall()
    }
    if "file_sha256" not in upload_columns:
        connection.execute("ALTER TABLE media_uploads ADD COLUMN file_sha256 TEXT")

    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(analyses)").fetchall()
    }
    if "inference_engine" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN inference_engine TEXT")
    if "detector_version" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN detector_version TEXT")
    if "model_version" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN model_version TEXT")
    if "dataset_version" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN dataset_version TEXT")
    if "training_date" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN training_date TEXT")
    if "temperature" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN temperature REAL")
    if "calibration_method" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN calibration_method TEXT")
    if "analysis_mode" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN analysis_mode TEXT")
    if "binary_prediction" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN binary_prediction TEXT")
    if "leaning_prediction" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN leaning_prediction TEXT")
    if "ensemble_fake_score" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN ensemble_fake_score REAL")
    if "uncertainty_score" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN uncertainty_score REAL")
    if "detector_breakdown_json" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN detector_breakdown_json TEXT")
    if "fake_prob" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN fake_prob REAL")
    if "real_prob" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN real_prob REAL")
    if "confidence_band" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN confidence_band TEXT")
    if "review_status" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN review_status TEXT")

    connection.execute(
        """
        UPDATE analyses
        SET
            inference_engine = COALESCE(inference_engine, ?),
            detector_version = COALESCE(detector_version, ?),
            analysis_mode = COALESCE(analysis_mode, ?)
        """,
        (LEGACY_INFERENCE_ENGINE, LEGACY_DETECTOR_VERSION, LEGACY_ANALYSIS_MODE),
    )


def _parse_analysis_payload(row):
    if not row:
        return None
    payload = dict(row)
    payload["attribution"] = json.loads(payload.get("attribution_json") or "{}")
    payload["evidence_points"] = json.loads(payload.get("evidence_json") or "[]")
    payload["detector_breakdown"] = json.loads(payload.get("detector_breakdown_json") or "{}")
    payload.pop("attribution_json", None)
    payload.pop("evidence_json", None)
    payload.pop("detector_breakdown_json", None)
    return payload


def create_user(full_name, email, password_hash):
    connection = get_connection()
    user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    role = "admin" if user_count == 0 else "analyst"
    cursor = connection.execute(
        """
        INSERT INTO users (full_name, email, password_hash, role)
        VALUES (?, ?, ?, ?)
        """,
        (full_name, email, password_hash, role),
    )
    connection.commit()
    user_id = cursor.lastrowid
    connection.close()
    return user_id


def get_user_by_email(email):
    connection = get_connection()
    row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    connection.close()
    return dict(row) if row else None


def ensure_public_demo_user():
    connection = get_connection()
    existing = connection.execute(
        "SELECT * FROM users WHERE email = ?", (PUBLIC_DEMO_EMAIL,)
    ).fetchone()
    if existing:
        connection.close()
        return dict(existing)

    cursor = connection.execute(
        """
        INSERT INTO users (full_name, email, password_hash, role)
        VALUES (?, ?, ?, ?)
        """,
        (
            PUBLIC_DEMO_NAME,
            PUBLIC_DEMO_EMAIL,
            generate_password_hash(os.urandom(16).hex()),
            "analyst",
        ),
    )
    connection.commit()
    user_id = cursor.lastrowid
    row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    connection.close()
    return dict(row)


def get_user_by_id(user_id):
    connection = get_connection()
    row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    connection.close()
    return dict(row) if row else None


def verify_user_credentials(email, password):
    connection = get_connection()
    row = connection.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        connection.close()
        return None

    last_login = datetime.utcnow().isoformat(timespec="seconds")
    connection.execute(
        "UPDATE users SET last_login = ? WHERE id = ?", (last_login, row["id"])
    )
    connection.commit()
    updated = connection.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
    connection.close()
    return dict(updated)


def log_audit_event(user_id, action, target_type, target_id, details, ip_address=None):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO audit_logs (user_id, action, target_type, target_id, details, ip_address)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, action, target_type, target_id, details, ip_address),
    )
    connection.commit()
    connection.close()


def create_media_upload(
    user_id,
    original_filename,
    stored_filename,
    stored_path,
    media_type,
    file_size,
    file_sha256=None,
    status="queued",
):
    connection = get_connection()
    cursor = connection.execute(
        """
        INSERT INTO media_uploads (
            user_id, original_filename, stored_filename, stored_path, media_type, file_size, file_sha256, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            original_filename,
            stored_filename,
            stored_path,
            media_type,
            file_size,
            file_sha256,
            status,
        ),
    )
    connection.commit()
    upload_id = cursor.lastrowid
    connection.close()
    return upload_id


def set_upload_status(upload_id, status):
    connection = get_connection()
    connection.execute(
        "UPDATE media_uploads SET status = ? WHERE id = ?", (status, upload_id)
    )
    connection.commit()
    connection.close()


def create_analysis(upload_id, result):
    connection = get_connection()
    cursor = connection.execute(
        """
        INSERT INTO analyses (
            upload_id,
            prediction,
            binary_prediction,
            leaning_prediction,
            confidence,
            confidence_band,
            review_status,
            fake_prob,
            real_prob,
            source_model,
            frequency_score,
            face_count,
            heatmap_path,
            ensemble_fake_score,
            uncertainty_score,
            detector_breakdown_json,
            attribution_json,
            evidence_json,
            summary,
            inference_engine,
            detector_version,
            model_version,
            dataset_version,
            training_date,
            temperature,
            calibration_method,
            analysis_mode
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            upload_id,
            result["prediction"],
            result.get("binary_prediction"),
            result.get("leaning_prediction"),
            result["confidence"],
            result.get("confidence_band"),
            result.get("review_status"),
            result.get("fake_prob"),
            result.get("real_prob"),
            result["source_model"],
            result["frequency_score"],
            result["face_count"],
            result.get("heatmap_path"),
            result.get("ensemble_fake_score"),
            result.get("uncertainty_score"),
            json.dumps(result.get("detector_breakdown", {})),
            json.dumps(result.get("attribution", {})),
            json.dumps(result.get("evidence_points", [])),
            result["summary"],
            result.get("inference_engine", LEGACY_INFERENCE_ENGINE),
            result.get("detector_version", LEGACY_DETECTOR_VERSION),
            result.get("model_version"),
            result.get("dataset_version"),
            result.get("training_date"),
            result.get("temperature"),
            result.get("calibration_method"),
            result.get("analysis_mode", LEGACY_ANALYSIS_MODE),
        ),
    )
    connection.commit()
    analysis_id = cursor.lastrowid
    connection.close()
    return analysis_id


def create_report(analysis_id, report_path, report_status="ready"):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO reports (analysis_id, report_path, report_status, generated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(analysis_id)
        DO UPDATE SET
            report_path = excluded.report_path,
            report_status = excluded.report_status,
            generated_at = CURRENT_TIMESTAMP
        """,
        (analysis_id, report_path, report_status),
    )
    connection.commit()
    connection.close()


def get_analysis_detail(analysis_id):
    connection = get_connection()
    row = connection.execute(
        """
        SELECT
            a.id AS analysis_id,
            a.prediction,
            a.binary_prediction,
            a.leaning_prediction,
            a.confidence,
            a.confidence_band,
            a.review_status,
            a.fake_prob,
            a.real_prob,
            a.source_model,
            a.frequency_score,
            a.face_count,
            a.heatmap_path,
            a.ensemble_fake_score,
            a.uncertainty_score,
            a.detector_breakdown_json,
            a.attribution_json,
            a.evidence_json,
            a.summary,
            a.inference_engine,
            a.detector_version,
            a.model_version,
            a.dataset_version,
            a.training_date,
            a.temperature,
            a.calibration_method,
            a.analysis_mode,
            a.created_at,
            m.id AS upload_id,
            m.user_id,
            m.original_filename,
            m.stored_filename,
            m.stored_path,
            m.media_type,
            m.file_size,
            m.file_sha256,
            m.status,
            m.uploaded_at,
            u.full_name,
            u.email,
            u.role,
            r.id AS report_id,
            r.report_path,
            r.report_status,
            r.generated_at
        FROM analyses a
        JOIN media_uploads m ON m.id = a.upload_id
        JOIN users u ON u.id = m.user_id
        LEFT JOIN reports r ON r.analysis_id = a.id
        WHERE a.id = ?
        """,
        (analysis_id,),
    ).fetchone()
    connection.close()
    return _parse_analysis_payload(row)


def list_user_history(user_id, include_all=False, limit=100):
    connection = get_connection()
    if include_all:
        rows = connection.execute(
            """
            SELECT
                a.id AS analysis_id,
                a.prediction,
                a.binary_prediction,
                a.leaning_prediction,
                a.confidence,
                a.confidence_band,
                a.review_status,
                a.fake_prob,
                a.real_prob,
                a.source_model,
                a.frequency_score,
                a.face_count,
                a.heatmap_path,
                a.ensemble_fake_score,
                a.uncertainty_score,
                a.detector_breakdown_json,
                a.attribution_json,
                a.evidence_json,
                a.summary,
                a.inference_engine,
                a.detector_version,
                a.model_version,
                a.dataset_version,
                a.training_date,
                a.temperature,
                a.calibration_method,
                a.analysis_mode,
                a.created_at,
                m.id AS upload_id,
                m.user_id,
                m.original_filename,
                m.stored_filename,
                m.stored_path,
                m.media_type,
                m.file_size,
                m.file_sha256,
                m.status,
                m.uploaded_at,
                u.full_name,
                u.email,
                u.role,
                r.id AS report_id,
                r.report_path,
                r.report_status,
                r.generated_at
            FROM analyses a
            JOIN media_uploads m ON m.id = a.upload_id
            JOIN users u ON u.id = m.user_id
            LEFT JOIN reports r ON r.analysis_id = a.id
            ORDER BY a.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT
                a.id AS analysis_id,
                a.prediction,
                a.binary_prediction,
                a.leaning_prediction,
                a.confidence,
                a.confidence_band,
                a.review_status,
                a.fake_prob,
                a.real_prob,
                a.source_model,
                a.frequency_score,
                a.face_count,
                a.heatmap_path,
                a.ensemble_fake_score,
                a.uncertainty_score,
                a.detector_breakdown_json,
                a.attribution_json,
                a.evidence_json,
                a.summary,
                a.inference_engine,
                a.detector_version,
                a.model_version,
                a.dataset_version,
                a.training_date,
                a.temperature,
                a.calibration_method,
                a.analysis_mode,
                a.created_at,
                m.id AS upload_id,
                m.user_id,
                m.original_filename,
                m.stored_filename,
                m.stored_path,
                m.media_type,
                m.file_size,
                m.file_sha256,
                m.status,
                m.uploaded_at,
                u.full_name,
                u.email,
                u.role,
                r.id AS report_id,
                r.report_path,
                r.report_status,
                r.generated_at
            FROM analyses a
            JOIN media_uploads m ON m.id = a.upload_id
            JOIN users u ON u.id = m.user_id
            LEFT JOIN reports r ON r.analysis_id = a.id
            WHERE m.user_id = ?
            ORDER BY a.created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    connection.close()
    return [_parse_analysis_payload(row) for row in rows]


def get_dashboard_summary(user_id):
    connection = get_connection()
    totals = connection.execute(
        """
        SELECT
            COUNT(*) AS total_cases,
            SUM(CASE WHEN a.prediction IN ('AI-Generated', 'Fake', 'Likely AI Generated') THEN 1 ELSE 0 END) AS fake_cases,
            SUM(CASE WHEN r.id IS NOT NULL THEN 1 ELSE 0 END) AS reports_ready
        FROM analyses a
        JOIN media_uploads m ON m.id = a.upload_id
        LEFT JOIN reports r ON r.analysis_id = a.id
        WHERE m.user_id = ?
        """,
        (user_id,),
    ).fetchone()
    audit_count = connection.execute(
        "SELECT COUNT(*) FROM audit_logs WHERE user_id = ?", (user_id,)
    ).fetchone()[0]
    recent_rows = connection.execute(
        """
        SELECT
            a.id AS analysis_id,
            a.prediction,
            a.binary_prediction,
            a.leaning_prediction,
            a.confidence,
            a.confidence_band,
            a.review_status,
            a.fake_prob,
            a.real_prob,
            a.source_model,
            a.frequency_score,
            a.face_count,
            a.heatmap_path,
            a.ensemble_fake_score,
            a.uncertainty_score,
            a.detector_breakdown_json,
            a.attribution_json,
            a.evidence_json,
            a.summary,
            a.inference_engine,
            a.detector_version,
            a.model_version,
            a.dataset_version,
            a.training_date,
            a.temperature,
            a.calibration_method,
            a.analysis_mode,
            a.created_at,
            m.id AS upload_id,
            m.user_id,
            m.original_filename,
            m.stored_filename,
            m.stored_path,
            m.media_type,
            m.file_size,
            m.file_sha256,
            m.status,
            m.uploaded_at,
            u.full_name,
            u.email,
            u.role,
            r.id AS report_id,
            r.report_path,
            r.report_status,
            r.generated_at
        FROM analyses a
        JOIN media_uploads m ON m.id = a.upload_id
        JOIN users u ON u.id = m.user_id
        LEFT JOIN reports r ON r.analysis_id = a.id
        WHERE m.user_id = ?
        ORDER BY a.created_at DESC
        LIMIT 5
        """,
        (user_id,),
    ).fetchall()
    connection.close()
    return {
        "total_cases": totals["total_cases"] or 0,
        "fake_cases": totals["fake_cases"] or 0,
        "reports_ready": totals["reports_ready"] or 0,
        "audit_events": audit_count or 0,
        "recent_cases": [_parse_analysis_payload(row) for row in recent_rows],
    }


def get_admin_summary():
    connection = get_connection()
    totals = connection.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM users) AS total_users,
            (SELECT COUNT(*) FROM media_uploads) AS total_uploads,
            (SELECT COUNT(*) FROM analyses) AS total_analyses,
            (SELECT COUNT(*) FROM analyses WHERE prediction IN ('AI-Generated', 'Fake', 'Likely AI Generated')) AS flagged_cases,
            (SELECT COUNT(*) FROM audit_logs) AS audit_events
        """
    ).fetchone()
    recent_logs = connection.execute(
        """
        SELECT
            l.id,
            l.action,
            l.target_type,
            l.target_id,
            l.details,
            l.ip_address,
            l.created_at,
            u.full_name
        FROM audit_logs l
        LEFT JOIN users u ON u.id = l.user_id
        ORDER BY l.created_at DESC
        LIMIT 12
        """
    ).fetchall()
    recent_cases = connection.execute(
        """
        SELECT
            a.id AS analysis_id,
            a.prediction,
            a.binary_prediction,
            a.leaning_prediction,
            a.confidence,
            a.confidence_band,
            a.review_status,
            a.fake_prob,
            a.real_prob,
            a.source_model,
            a.frequency_score,
            a.face_count,
            a.heatmap_path,
            a.ensemble_fake_score,
            a.uncertainty_score,
            a.detector_breakdown_json,
            a.attribution_json,
            a.evidence_json,
            a.summary,
            a.inference_engine,
            a.detector_version,
            a.model_version,
            a.dataset_version,
            a.training_date,
            a.temperature,
            a.calibration_method,
            a.analysis_mode,
            a.created_at,
            m.id AS upload_id,
            m.user_id,
            m.original_filename,
            m.stored_filename,
            m.stored_path,
            m.media_type,
            m.file_size,
            m.file_sha256,
            m.status,
            m.uploaded_at,
            u.full_name,
            u.email,
            u.role,
            r.id AS report_id,
            r.report_path,
            r.report_status,
            r.generated_at
        FROM analyses a
        JOIN media_uploads m ON m.id = a.upload_id
        JOIN users u ON u.id = m.user_id
        LEFT JOIN reports r ON r.analysis_id = a.id
        ORDER BY a.created_at DESC
        LIMIT 8
        """
    ).fetchall()
    connection.close()
    return {
        "total_users": totals["total_users"] or 0,
        "total_uploads": totals["total_uploads"] or 0,
        "total_analyses": totals["total_analyses"] or 0,
        "flagged_cases": totals["flagged_cases"] or 0,
        "audit_events": totals["audit_events"] or 0,
        "recent_logs": [dict(row) for row in recent_logs],
        "recent_cases": [_parse_analysis_payload(row) for row in recent_cases],
    }


def get_case_audit_logs(upload_id, analysis_id):
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT
            l.id,
            l.action,
            l.target_type,
            l.target_id,
            l.details,
            l.ip_address,
            l.created_at,
            u.full_name
        FROM audit_logs l
        LEFT JOIN users u ON u.id = l.user_id
        WHERE
            (l.target_type = 'upload' AND l.target_id = ?)
            OR (l.target_type IN ('analysis', 'report') AND l.target_id = ?)
        ORDER BY l.created_at ASC
        """,
        (upload_id, analysis_id),
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]

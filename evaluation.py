import json
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime

from forensics import (
    AI_LABEL,
    REAL_LABEL,
    UNCERTAIN_LABEL,
    analyze_media_file,
    detector_descriptor,
    infer_media_type,
)


EVALUATION_ROOT = os.path.join("dataset", "evaluation")
EVALUATION_ARTIFACT_DIR = os.path.join("artifacts", "evaluations")
HARD_EXAMPLE_ROOT = os.path.join("artifacts", "hard_examples")
LATEST_EVALUATION_PATH = os.path.join(EVALUATION_ARTIFACT_DIR, "latest.json")


def evaluation_dataset_exists(dataset_root=EVALUATION_ROOT):
    return os.path.isdir(dataset_root)


def _expected_label_from_path(path):
    parts = [part.lower() for part in path.replace("\\", "/").split("/") if part]
    joined = " ".join(parts)
    if any(token in joined for token in ("real", "natural", "authentic", "camera")):
        return REAL_LABEL
    if any(
        token in joined
        for token in (
            "ai",
            "fake",
            "synthetic",
            "generated",
            "diffusion",
            "gan",
            "faceswap",
            "deepfake",
        )
    ):
        return AI_LABEL
    return None


def _safe_rate(numerator, denominator):
    return round((numerator / denominator) * 100, 2) if denominator else 0.0


def _precision_recall(samples, positive_label):
    tp = sum(1 for sample in samples if sample["expected"] == positive_label and sample["predicted"] == positive_label)
    fp = sum(1 for sample in samples if sample["expected"] != positive_label and sample["predicted"] == positive_label)
    fn = sum(1 for sample in samples if sample["expected"] == positive_label and sample["predicted"] != positive_label)
    return {
        "precision": _safe_rate(tp, tp + fp),
        "recall": _safe_rate(tp, tp + fn),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _per_class_accuracy(samples, label):
    relevant = [sample for sample in samples if sample["expected"] == label]
    correct = sum(1 for sample in relevant if sample["predicted"] == label)
    return _safe_rate(correct, len(relevant))


def _roc_auc(samples, positive_label):
    positives = [
        sample["fake_probability"]
        for sample in samples
        if sample["expected"] == positive_label and sample["fake_probability"] is not None
    ]
    negatives = [
        sample["fake_probability"]
        for sample in samples
        if sample["expected"] != positive_label and sample["fake_probability"] is not None
    ]
    if not positives or not negatives:
        return None

    wins = 0.0
    for positive_score in positives:
        for negative_score in negatives:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return round((wins / (len(positives) * len(negatives))) * 100, 2)


def _run_id():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _safe_filename(path):
    candidate = os.path.basename(path).replace(" ", "_")
    return candidate or "sample"


def _export_hard_examples(samples, run_id, cap_per_type=100):
    run_dir = os.path.join(HARD_EXAMPLE_ROOT, run_id)
    false_positive_dir = os.path.join(run_dir, "false_positives")
    false_negative_dir = os.path.join(run_dir, "false_negatives")
    os.makedirs(false_positive_dir, exist_ok=True)
    os.makedirs(false_negative_dir, exist_ok=True)

    false_positives = [
        sample for sample in samples if sample["expected"] == REAL_LABEL and sample["predicted"] == AI_LABEL
    ]
    false_negatives = [
        sample for sample in samples if sample["expected"] == AI_LABEL and sample["predicted"] == REAL_LABEL
    ]
    false_positives.sort(key=lambda sample: sample.get("confidence") or 0.0, reverse=True)
    false_negatives.sort(key=lambda sample: sample.get("confidence") or 0.0, reverse=True)
    false_positives = false_positives[:cap_per_type]
    false_negatives = false_negatives[:cap_per_type]

    manifest = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "samples": [],
    }

    for index, sample in enumerate(false_positives, start=1):
        destination_name = f"{index:03d}_{_safe_filename(sample['source_path'])}"
        destination = os.path.join(false_positive_dir, destination_name)
        shutil.copy2(sample["source_path"], destination)
        manifest["samples"].append(
            {
                "error_type": "false_positive",
                "source_path": sample["source_path"],
                "exported_path": destination,
                "expected_label": sample["expected"],
                "predicted_label": sample["predicted"],
                "confidence": sample["confidence"],
                "fake_probability": sample.get("fake_probability"),
                "real_probability": sample.get("real_probability"),
                "detector_version": sample.get("detector_version"),
                "dataset_category": sample.get("category"),
                "review_status": "pending",
            }
        )

    for index, sample in enumerate(false_negatives, start=1):
        destination_name = f"{index:03d}_{_safe_filename(sample['source_path'])}"
        destination = os.path.join(false_negative_dir, destination_name)
        shutil.copy2(sample["source_path"], destination)
        manifest["samples"].append(
            {
                "error_type": "false_negative",
                "source_path": sample["source_path"],
                "exported_path": destination,
                "expected_label": sample["expected"],
                "predicted_label": sample["predicted"],
                "confidence": sample["confidence"],
                "fake_probability": sample.get("fake_probability"),
                "real_probability": sample.get("real_probability"),
                "detector_version": sample.get("detector_version"),
                "dataset_category": sample.get("category"),
                "review_status": "pending",
            }
        )

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)

    return run_dir


def run_evaluation(dataset_root=EVALUATION_ROOT, output_path=LATEST_EVALUATION_PATH, heatmap_dir=os.path.join("artifacts", "heatmaps")):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Evaluation dataset not found at {dataset_root}. Create folders like dataset/evaluation/real and dataset/evaluation/ai_generated."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(HARD_EXAMPLE_ROOT, exist_ok=True)
    detector_meta = detector_descriptor()
    run_id = _run_id()
    samples = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".mp4", ".mov", ".avi", ".mkv", ".webm"}

    for root, _dirs, files in os.walk(dataset_root):
        for name in files:
            extension = os.path.splitext(name)[1].lower()
            if extension not in supported_extensions:
                continue
            full_path = os.path.join(root, name)
            expected = _expected_label_from_path(os.path.relpath(full_path, dataset_root))
            if expected is None:
                continue
            media_type = infer_media_type(name)
            result = analyze_media_file(full_path, media_type, heatmap_dir)
            category = os.path.relpath(root, dataset_root)
            samples.append(
                {
                    "file": os.path.relpath(full_path, dataset_root),
                    "source_path": os.path.abspath(full_path),
                    "category": category,
                    "expected": expected,
                    "predicted": result["prediction"],
                    "binary_prediction": result.get("binary_prediction"),
                    "confidence": result["confidence"],
                    "confidence_band": result.get("confidence_band"),
                    "review_status": result.get("review_status"),
                    "fake_probability": result.get("fake_prob"),
                    "real_probability": result.get("real_prob"),
                    "analysis_mode": result.get("analysis_mode"),
                    "detector_version": result.get("detector_version"),
                    "inference_engine": result.get("inference_engine"),
                }
            )

    if not samples:
        raise ValueError(
            "No labeled evaluation samples were found. Use folder names containing 'real' or 'ai/fake/generated'."
        )

    total = len(samples)
    correct = sum(1 for sample in samples if sample["expected"] == sample["predicted"])
    resolved = [sample for sample in samples if sample["predicted"] != UNCERTAIN_LABEL]
    resolved_correct = sum(1 for sample in resolved if sample["expected"] == sample["predicted"])
    confusion = Counter((sample["expected"], sample["predicted"]) for sample in samples)
    per_category = defaultdict(lambda: {"total": 0, "correct": 0, "uncertain": 0})
    for sample in samples:
        category = per_category[sample["category"]]
        category["total"] += 1
        category["correct"] += int(sample["expected"] == sample["predicted"])
        category["uncertain"] += int(sample["predicted"] == UNCERTAIN_LABEL)

    ai_metrics = _precision_recall(samples, AI_LABEL)
    real_metrics = _precision_recall(samples, REAL_LABEL)
    false_positive_rate = _safe_rate(
        ai_metrics["false_positives"],
        sum(1 for sample in samples if sample["expected"] == REAL_LABEL),
    )
    false_negative_rate = _safe_rate(
        ai_metrics["false_negatives"],
        sum(1 for sample in samples if sample["expected"] == AI_LABEL),
    )
    failure_samples = [
        sample
        for sample in samples
        if sample["expected"] != sample["predicted"]
    ]
    failure_samples.sort(
        key=lambda sample: (
            sample.get("confidence") or 0.0,
            sample.get("fake_probability") or 0.0,
        ),
        reverse=True,
    )
    hard_example_export_path = _export_hard_examples(samples, run_id)

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_id": run_id,
        "dataset_root": dataset_root,
        "model_version": detector_meta.get("model_version"),
        "dataset_version": detector_meta.get("dataset_version"),
        "training_date": detector_meta.get("training_date"),
        "temperature": detector_meta.get("temperature"),
        "hard_example_export_path": hard_example_export_path,
        "sample_count": total,
        "accuracy": _safe_rate(correct, total),
        "resolved_accuracy": _safe_rate(resolved_correct, len(resolved)),
        "coverage": _safe_rate(len(resolved), total),
        "uncertain_count": total - len(resolved),
        "false_positives": ai_metrics["false_positives"],
        "false_negatives": ai_metrics["false_negatives"],
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "roc_auc": _roc_auc(samples, AI_LABEL),
        "ai_metrics": ai_metrics,
        "real_metrics": real_metrics,
        "per_class_accuracy": {
            "ai_generated": _per_class_accuracy(samples, AI_LABEL),
            "real": _per_class_accuracy(samples, REAL_LABEL),
        },
        "confusion_matrix": {
            "expected_real_predicted_real": confusion[(REAL_LABEL, REAL_LABEL)],
            "expected_real_predicted_ai": confusion[(REAL_LABEL, AI_LABEL)],
            "expected_real_predicted_uncertain": confusion[(REAL_LABEL, UNCERTAIN_LABEL)],
            "expected_ai_predicted_real": confusion[(AI_LABEL, REAL_LABEL)],
            "expected_ai_predicted_ai": confusion[(AI_LABEL, AI_LABEL)],
            "expected_ai_predicted_uncertain": confusion[(AI_LABEL, UNCERTAIN_LABEL)],
        },
        "per_category": [
            {
                "category": category,
                "total": values["total"],
                "accuracy": _safe_rate(values["correct"], values["total"]),
                "uncertain_rate": _safe_rate(values["uncertain"], values["total"]),
            }
            for category, values in sorted(per_category.items())
        ],
        "top_failure_samples": failure_samples[:10],
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    return report


def load_latest_evaluation(output_path=LATEST_EVALUATION_PATH):
    if not os.path.exists(output_path):
        return None
    with open(output_path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)

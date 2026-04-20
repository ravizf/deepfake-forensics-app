import html
import json
import math
import os
import uuid
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

REAL_LABEL = "Real"
AI_LABEL = "AI-Generated"
UNCERTAIN_LABEL = "Uncertain"
LIKELY_REAL_LABEL = REAL_LABEL
LIKELY_AI_LABEL = AI_LABEL

MODEL_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "model_manifest.json")
FALLBACK_ENGINE_NAME = "Heuristic fallback"
FALLBACK_VIDEO_ENGINE_NAME = "Heuristic video fallback"
FALLBACK_DETECTOR_VERSION = "snaptrace-heuristic-v3"
LEGACY_DETECTOR_VERSION = "snaptrace-heuristic-v1"

_MTCNN_DETECTOR = None
_MTCNN_ATTEMPTED = False
_MODEL = None
_MODEL_META = None
_MODEL_ATTEMPTED = False


def allowed_file(filename):
    return os.path.splitext((filename or "").lower())[1] in ALLOWED_EXTENSIONS


def infer_media_type(filename):
    extension = os.path.splitext((filename or "").lower())[1]
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def _log(message):
    print(f"[SnapTrace] {message}", flush=True)


def _clamp01(value):
    return float(np.clip(value, 0.0, 1.0))


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _normalize(array):
    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(maximum, minimum):
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def _dct_matrix(size):
    coordinates = np.arange(size)
    basis = np.zeros((size, size), dtype=np.float64)
    factor = math.pi / (2.0 * size)
    for k in range(size):
        scale = math.sqrt(1.0 / size) if k == 0 else math.sqrt(2.0 / size)
        basis[k, :] = scale * np.cos((2 * coordinates + 1) * k * factor)
    return basis


def _dct2(image_array):
    size = image_array.shape[0]
    basis = _dct_matrix(size)
    return basis @ image_array @ basis.T


def _load_model_manifest():
    if not os.path.exists(MODEL_MANIFEST_PATH):
        return None

    with open(MODEL_MANIFEST_PATH, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    checkpoint_path = manifest.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("model_manifest.json is missing checkpoint_path")

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(__file__), checkpoint_path)
    checkpoint_path = os.path.normpath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Configured checkpoint not found: {checkpoint_path}")

    class_names = manifest.get("class_names")
    if class_names is not None and len(class_names) != 2:
        raise ValueError("model_manifest.json class_names must contain exactly 2 entries when set")

    manifest["checkpoint_path"] = checkpoint_path
    manifest["framework"] = manifest.get("framework", "pytorch_state_dict")
    manifest["input_size"] = manifest.get("input_size", [224, 224])
    manifest["class_names"] = class_names or ["fake", "real"]
    manifest["normalization_mean"] = manifest.get(
        "normalization_mean", [0.485, 0.456, 0.406]
    )
    manifest["normalization_std"] = manifest.get(
        "normalization_std", [0.229, 0.224, 0.225]
    )
    manifest["real_threshold"] = float(manifest.get("real_threshold", 0.60))
    manifest["fake_threshold"] = float(manifest.get("fake_threshold", 0.92))
    manifest["prefer_face_crop"] = bool(manifest.get("prefer_face_crop", False))
    manifest["positive_label"] = manifest.get("positive_label", "ai_generated")
    manifest["channel_order"] = manifest.get("channel_order", "nchw")
    manifest["model_version"] = manifest.get(
        "model_version",
        manifest.get("detector_version", "snaptrace-trained-v1"),
    )
    manifest["dataset_version"] = manifest.get("dataset_version", "unknown")
    manifest["training_date"] = manifest.get("training_date")
    manifest["temperature"] = max(float(manifest.get("temperature", 1.0)), 1e-6)
    manifest["calibration_method"] = manifest.get("calibration_method", "uncalibrated")
    return manifest


def _build_pytorch_model_from_manifest(manifest):
    architecture = manifest.get("architecture", "efficientnet_b0_binary")
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "torch and torchvision are required for PyTorch checkpoints."
        ) from exc

    if architecture == "efficientnet_b0_binary":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        return model
    if architecture == "resnet18_binary":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model
    if architecture == "resnet50_binary":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model
    if architecture == "mobilenet_v3_small_binary":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
        return model
    if architecture == "convnext_tiny_binary":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
        return model

    raise ValueError(
        "Unsupported architecture. Use efficientnet_b0_binary, resnet18_binary, resnet50_binary, "
        "mobilenet_v3_small_binary, or convnext_tiny_binary."
    )


def _load_external_model():
    manifest = _load_model_manifest()
    if manifest is None:
        return None, None

    framework = manifest["framework"]
    checkpoint_path = manifest["checkpoint_path"]

    if framework == "pytorch_state_dict":
        import torch

        model = _build_pytorch_model_from_manifest(manifest)
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
    elif framework == "pytorch_torchscript":
        import torch

        model = torch.jit.load(checkpoint_path, map_location="cpu")
        model.eval()
    elif framework == "tensorflow_keras":
        import tensorflow as tf

        model = tf.keras.models.load_model(checkpoint_path)
    elif framework == "onnx":
        import onnxruntime as ort

        model = ort.InferenceSession(checkpoint_path)
    else:
        raise ValueError(f"Unsupported framework in model manifest: {framework}")

    metadata = {
        "name": manifest.get("display_name", "Trained checkpoint"),
        "path": checkpoint_path,
        "type": framework,
        "architecture": manifest.get("architecture"),
        "detector_version": manifest.get("detector_version", "snaptrace-trained-v1"),
        "model_version": manifest["model_version"],
        "dataset_version": manifest["dataset_version"],
        "training_date": manifest["training_date"],
        "class_names": manifest["class_names"],
        "input_size": tuple(manifest["input_size"]),
        "normalization_mean": manifest["normalization_mean"],
        "normalization_std": manifest["normalization_std"],
        "real_threshold": manifest["real_threshold"],
        "fake_threshold": manifest["fake_threshold"],
        "prefer_face_crop": manifest["prefer_face_crop"],
        "positive_label": manifest["positive_label"],
        "channel_order": manifest["channel_order"],
        "temperature": manifest["temperature"],
        "calibration_method": manifest["calibration_method"],
        "confidence_notes": manifest.get(
            "confidence_notes",
            "Confidence reflects the loaded trained checkpoint output.",
        ),
    }
    return model, metadata


def _get_classifier():
    global _MODEL, _MODEL_META, _MODEL_ATTEMPTED
    if _MODEL_ATTEMPTED:
        return _MODEL, _MODEL_META

    _MODEL_ATTEMPTED = True
    try:
        _MODEL, _MODEL_META = _load_external_model()
        if _MODEL is not None and _MODEL_META is not None:
            _log(
                "Model loaded successfully "
                f"({_MODEL_META['name']} @ {_MODEL_META['path']})"
            )
            _log(f"Loaded detector metadata: {_MODEL_META}")
        else:
            _log("No trained checkpoint found. Using heuristic fallback.")
    except Exception as exc:
        _MODEL = None
        _MODEL_META = {
            "load_error": str(exc),
            "name": "Unavailable external model",
            "detector_version": FALLBACK_DETECTOR_VERSION,
        }
        _log(f"Model load failed, using heuristic fallback: {exc}")

    print("MODEL LOADED:", _MODEL is not None, flush=True)
    print("MODEL META:", _MODEL_META, flush=True)
    return _MODEL, _MODEL_META


def detector_descriptor():
    model, meta = _get_classifier()
    if model is not None and meta is not None and not meta.get("load_error"):
        return {
            "engine": meta["name"],
            "detector_version": meta["detector_version"],
            "model_version": meta.get("model_version"),
            "dataset_version": meta.get("dataset_version"),
            "training_date": meta.get("training_date"),
            "temperature": meta.get("temperature"),
            "calibration_method": meta.get("calibration_method"),
            "mode": "trained_model",
            "checkpoint_path": meta.get("path"),
            "framework": meta.get("type"),
            "status": "loaded",
            "class_names": meta.get("class_names"),
        }
    if meta is not None and meta.get("load_error"):
        return {
            "engine": meta["name"],
            "detector_version": FALLBACK_DETECTOR_VERSION,
            "mode": "fallback_demo",
            "status": "error",
            "error": meta["load_error"],
        }
    return {
        "engine": FALLBACK_ENGINE_NAME,
        "detector_version": FALLBACK_DETECTOR_VERSION,
        "mode": "fallback_demo",
        "status": "fallback_only",
    }


def _get_mtcnn_detector():
    global _MTCNN_DETECTOR, _MTCNN_ATTEMPTED
    if _MTCNN_ATTEMPTED:
        return _MTCNN_DETECTOR

    _MTCNN_ATTEMPTED = True
    try:
        from mtcnn import MTCNN

        _MTCNN_DETECTOR = MTCNN()
    except Exception:
        _MTCNN_DETECTOR = None
    return _MTCNN_DETECTOR


def _extract_faces(image):
    detector = _get_mtcnn_detector()
    image_array = np.array(image)
    faces = []

    if detector is not None:
        detections = detector.detect_faces(image_array)
        for detection in detections:
            x, y, width, height = detection.get("box", [0, 0, 0, 0])
            x = max(0, x)
            y = max(0, y)
            width = max(1, width)
            height = max(1, height)
            crop = image.crop((x, y, min(image.width, x + width), min(image.height, y + height)))
            if crop.size[0] > 10 and crop.size[1] > 10:
                faces.append(crop)

    if faces:
        return faces, len(faces), "mtcnn"

    return [], 0, "no-face-detected"


def _prepare_image_array(image, metadata):
    resized = image.resize(tuple(metadata["input_size"]))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    std = np.asarray(metadata["normalization_std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    image_array = (image_array - mean) / std
    return image_array


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _confidence_band(confidence):
    if confidence >= 90:
        return "High"
    if confidence >= 75:
        return "Medium"
    if confidence >= 60:
        return "Low"
    return "Review Required"


def _review_status(prediction, confidence):
    if prediction == UNCERTAIN_LABEL or confidence < 60:
        return "Manual review recommended"
    if confidence >= 90:
        return "Passed automatic review"
    if confidence >= 75:
        return "Analyst spot-check recommended"
    return "Secondary review recommended"


def _binary_prediction(label):
    if label == LIKELY_AI_LABEL:
        return "AI Generated"
    if label == LIKELY_REAL_LABEL:
        return "Real"
    return "Review Required"


def _coerce_prediction_scores(output, positive_label, class_names=None):
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return None

    if values.size == 1:
        prediction = float(values[0])
        if prediction < 0.0 or prediction > 1.0:
            prediction = _sigmoid(prediction)
        fake_prob = prediction if positive_label != "real" else 1.0 - prediction
        real_prob = 1.0 - fake_prob
        return _clamp01(fake_prob), _clamp01(real_prob)

    probabilities = _softmax(values)
    if values.size == 2:
        labels = [label.lower() for label in (class_names or ["fake", "real"])]
        if "fake" in labels and "real" in labels:
            fake_prob = float(probabilities[labels.index("fake")])
            real_prob = float(probabilities[labels.index("real")])
        elif positive_label == "real":
            real_prob = float(probabilities[1])
            fake_prob = float(probabilities[0])
        else:
            fake_prob = float(probabilities[0])
            real_prob = float(probabilities[1])
        return _clamp01(fake_prob), _clamp01(real_prob)

    fake_prob = float(probabilities[0])
    real_prob = float(probabilities[-1])
    return _clamp01(fake_prob), _clamp01(real_prob)


def _apply_temperature_scaling(output, temperature):
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size == 0 or math.isclose(float(temperature), 1.0, rel_tol=1e-6):
        return values

    temperature = max(float(temperature), 1e-6)
    if values.size == 1:
        score = float(values[0])
        if 0.0 <= score <= 1.0:
            score = float(np.clip(score, 1e-6, 1.0 - 1e-6))
            score = math.log(score / (1.0 - score))
        return np.asarray([score / temperature], dtype=np.float32)

    if np.all((0.0 <= values) & (values <= 1.0)) and math.isclose(
        float(values.sum()), 1.0, rel_tol=1e-3, abs_tol=1e-3
    ):
        values = np.log(np.clip(values, 1e-6, 1.0))

    return values / temperature


def decide_label(
    ai_score,
    *,
    face_detected=False,
    fake_threshold=0.92,
    real_threshold=0.60,
):
    ai_score = float(ai_score)
    if face_detected:
        ai_score *= 0.65

    ai_score = _clamp01(ai_score)
    real_score = 1.0 - ai_score
    fake_threshold = _clamp01(fake_threshold)
    real_threshold = _clamp01(real_threshold)

    if face_detected and ai_score < 0.80:
        prediction = LIKELY_REAL_LABEL
        confidence = real_score * 100.0
    elif ai_score >= fake_threshold:
        prediction = LIKELY_AI_LABEL
        confidence = ai_score * 100.0
    elif real_score >= real_threshold:
        prediction = LIKELY_REAL_LABEL
        confidence = real_score * 100.0
    else:
        prediction = UNCERTAIN_LABEL
        confidence = max(ai_score, real_score) * 100.0

    return {
        "prediction": prediction,
        "binary_prediction": _binary_prediction(prediction),
        "confidence": round(confidence, 2),
        "confidence_band": _confidence_band(confidence),
        "review_status": _review_status(prediction, confidence),
        "raw_probability": round(ai_score, 4),
        "real_probability": round(real_score, 4),
        "probability_gap": round(abs(ai_score - real_score), 4),
    }


def _trained_prediction_from_probs(
    fake_prob,
    real_prob,
    *,
    face_detected=False,
    fake_threshold=0.92,
    real_threshold=0.60,
):
    fake_prob = _clamp01(fake_prob)
    real_prob = _clamp01(real_prob)
    total = fake_prob + real_prob
    if total > 0:
        fake_prob /= total
        real_prob /= total

    return decide_label(
        fake_prob,
        face_detected=face_detected,
        fake_threshold=fake_threshold,
        real_threshold=real_threshold,
    )


def _classify_with_resnet(face_image):
    model, meta = _get_classifier()
    if model is None or meta is None:
        return None

    resized = face_image.resize(meta["input_size"])
    image_array = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.asarray(meta.get("normalization_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(meta.get("normalization_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    std = np.where(std == 0, 1.0, std)
    image_array = (image_array - mean) / std

    if meta["type"] == "tensorflow_keras":
        batch = np.expand_dims(image_array, axis=0)
        output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)

    elif meta["type"] == "onnx":
        batch = np.expand_dims(image_array, axis=0)
        input_name = model.get_inputs()[0].name
        output = np.asarray(model.run(None, {input_name: batch})[0]).reshape(-1)

    elif meta["type"] in {"pytorch_torchscript", "pytorch_state_dict"}:
        import torch

        chw = np.transpose(image_array, (2, 0, 1))
        batch = torch.from_numpy(np.expand_dims(chw, axis=0)).float()
        with torch.no_grad():
            output = model(batch)
        output = np.asarray(output).reshape(-1)

    else:
        return None

    positive_label = meta.get("positive_label", "ai_generated")
    calibrated_output = _apply_temperature_scaling(output, meta.get("temperature", 1.0))
    scores = _coerce_prediction_scores(
        calibrated_output, positive_label, meta.get("class_names")
    )
    if scores is None:
        return None

    fake_prob, real_prob = scores
    prediction = _trained_prediction_from_probs(
        fake_prob,
        real_prob,
        fake_threshold=meta.get("fake_threshold", 0.92),
        real_threshold=meta.get("real_threshold", 0.60),
    )
    print("Fake prob:", fake_prob, flush=True)
    print("Real prob:", real_prob, flush=True)
    print("Prediction:", prediction["prediction"], flush=True)
    return {
        **prediction,
        "engine": meta["name"],
        "detector_version": meta["detector_version"],
        "model_version": meta.get("model_version"),
        "dataset_version": meta.get("dataset_version"),
        "training_date": meta.get("training_date"),
        "temperature": meta.get("temperature"),
        "calibration_method": meta.get("calibration_method"),
        "fake_threshold": meta.get("fake_threshold", 0.92),
        "real_threshold": meta.get("real_threshold", 0.60),
        "mode": "trained_model",
    }


def _colorize_heatmap(normalized_map):
    red = (255 * np.clip(normalized_map * 1.2, 0, 1)).astype(np.uint8)
    green = (255 * np.clip(normalized_map * 0.75, 0, 1)).astype(np.uint8)
    blue = (255 * np.clip(0.15 + normalized_map * 0.2, 0, 1)).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def _save_heatmap(image, intensity_map, heatmap_dir, prefix):
    normalized = _normalize(intensity_map)
    colorized = Image.fromarray(_colorize_heatmap(normalized)).resize(image.size)
    overlay = Image.blend(image.convert("RGB"), colorized, alpha=0.46)
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(heatmap_dir, filename)
    overlay.save(output_path)
    return output_path


def _source_attribution_for_image(freq_score, edge_density, contrast, prediction):
    if prediction == REAL_LABEL:
        return {
            "Authentic / Unknown": 0.78,
            "StyleGAN": 0.08,
            "Stable Diffusion": 0.08,
            "FaceSwap": 0.06,
        }
    if prediction == UNCERTAIN_LABEL:
        return {
            "Authentic / Unknown": 0.34,
            "StyleGAN": 0.24,
            "Stable Diffusion": 0.24,
            "FaceSwap": 0.18,
        }

    stylegan = 0.28 + freq_score * 0.32 + contrast * 0.1
    stable_diffusion = 0.24 + edge_density * 0.28 + (1 - contrast) * 0.14
    faceswap = 0.22 + max(0.0, 0.5 - edge_density) * 0.16 + freq_score * 0.18
    total = stylegan + stable_diffusion + faceswap
    return {
        "StyleGAN": round(stylegan / total, 3),
        "Stable Diffusion": round(stable_diffusion / total, 3),
        "FaceSwap": round(faceswap / total, 3),
    }


def _source_attribution_for_video(entropy_score, transition_score, prediction):
    if prediction == REAL_LABEL:
        return {
            "Authentic / Unknown": 0.8,
            "FaceSwap Video": 0.1,
            "Wav2Lip": 0.1,
        }
    if prediction == UNCERTAIN_LABEL:
        return {
            "Authentic / Unknown": 0.34,
            "FaceSwap Video": 0.34,
            "Wav2Lip": 0.32,
        }

    faceswap_video = 0.34 + transition_score * 0.36
    wav2lip = 0.26 + entropy_score * 0.34
    reenactment = 0.2 + (1 - entropy_score) * 0.22
    total = faceswap_video + wav2lip + reenactment
    return {
        "FaceSwap Video": round(faceswap_video / total, 3),
        "Wav2Lip": round(wav2lip / total, 3),
        "Neural Reenactment": round(reenactment / total, 3),
    }


def _best_source_model(scores):
    return max(scores.items(), key=lambda item: item[1])[0]


def _heuristic_label(fake_score, uncertainty_score):
    if uncertainty_score >= 0.55 or 0.40 <= fake_score <= 0.60:
        return UNCERTAIN_LABEL
    if fake_score > 0.60:
        return AI_LABEL
    return REAL_LABEL


def _heuristic_binary_prediction(label):
    return _binary_prediction(label)


def _heuristic_confidence(fake_score, uncertainty_score, label):
    margin = abs(fake_score - 0.5) * 2.0
    confidence = 50.0 + margin * 34.0 + (1.0 - uncertainty_score) * 11.0
    if label == UNCERTAIN_LABEL:
        confidence = min(confidence, 68.0)
    return round(float(np.clip(confidence, 50.0, 96.0)), 2)


def _label_from_score(fake_score, uncertainty_score):
    return _heuristic_label(fake_score, uncertainty_score)


def _confidence_from_scores(fake_score, uncertainty_score, prediction):
    return _heuristic_confidence(fake_score, uncertainty_score, prediction)


def _leaning_prediction(fake_score):
    return LIKELY_AI_LABEL if fake_score >= 0.5 else LIKELY_REAL_LABEL


def _analyze_image(file_path, heatmap_dir):
    image = Image.open(file_path).convert("RGB")
    max_side = 1280
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    faces, face_count, face_strategy = _extract_faces(image)
    primary_face = faces[0] if faces else image

    target = primary_face.resize((32, 32))
    gray = ImageOps.grayscale(target)
    gray_array = np.asarray(gray, dtype=np.float64) / 255.0
    dct_coefficients = np.abs(_dct2(gray_array))
    mask = np.add.outer(np.arange(32), np.arange(32)) > 18
    high_frequency = dct_coefficients[mask].sum()
    total_energy = dct_coefficients.sum() + 1e-8
    freq_score = float(np.clip(high_frequency / total_energy, 0.0, 1.0))

    full_gray = ImageOps.grayscale(image)
    edge_map = np.asarray(full_gray.filter(ImageFilter.FIND_EDGES), dtype=np.float64) / 255.0
    edge_density = _clamp01(edge_map.mean() * 2.2)
    contrast = _clamp01(np.asarray(full_gray, dtype=np.float64).std() / 64.0)

    face_rgb = primary_face.resize((224, 224))
    face_gray = ImageOps.grayscale(face_rgb)
    face_gray_array = np.asarray(face_gray, dtype=np.float64)
    residual = face_gray_array - np.asarray(
        face_gray.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float64
    )
    sensor_noise = _clamp01(residual.std() / 32.0)
    face_rgb_array = np.asarray(face_rgb, dtype=np.float64)
    chroma_variation = _clamp01(face_rgb_array.std(axis=2).mean() / 64.0)
    smoothness_penalty = 1.0 - chroma_variation

    artifact_score = _clamp01(
        0.34 * smoothness_penalty
        + 0.24 * sensor_noise
        + 0.20 * contrast
        + 0.12 * edge_density
        + 0.10 * freq_score
    )
    frequency_detector = _clamp01(0.74 * freq_score + 0.26 * sensor_noise)
    diffusion_detector = _clamp01(
        0.42 * smoothness_penalty + 0.26 * contrast + 0.20 * freq_score + 0.12 * edge_density
    )

    full_image_prediction = _classify_with_resnet(image)
    face_prediction = _classify_with_resnet(primary_face) if face_count else None

    component_scores = {
        "artifact_detector": round(artifact_score, 4),
        "frequency_detector": round(frequency_detector, 4),
        "diffusion_gan_detector": round(diffusion_detector, 4),
    }

    if full_image_prediction is not None:
        full_fake_score = _clamp01(full_image_prediction["raw_probability"])
        component_scores["full_image_classifier"] = round(full_fake_score, 4)
        if face_prediction is not None:
            face_fake_score = _clamp01(face_prediction["raw_probability"])
            component_scores["face_classifier"] = round(face_fake_score, 4)
            fake_score = _clamp01(0.60 * full_fake_score + 0.40 * face_fake_score)
            face_strategy = "mtcnn"
        else:
            face_fake_score = None
            fake_score = full_fake_score
            face_strategy = "full-image-only"

        real_score = 1.0 - fake_score
        trained_prediction = _trained_prediction_from_probs(
            fake_score,
            real_score,
            face_detected=bool(face_count),
            fake_threshold=full_image_prediction.get("fake_threshold", 0.92),
            real_threshold=full_image_prediction.get("real_threshold", 0.60),
        )
        uncertainty_score = _clamp01(1.0 - trained_prediction["probability_gap"])
        prediction = trained_prediction["prediction"]
        confidence = trained_prediction["confidence"]
        confidence_band = trained_prediction["confidence_band"]
        review_status = trained_prediction["review_status"]
        binary_prediction = trained_prediction["binary_prediction"]
        inference_engine = full_image_prediction["engine"]
        detector_version = full_image_prediction["detector_version"]
        model_version = full_image_prediction.get("model_version")
        dataset_version = full_image_prediction.get("dataset_version")
        training_date = full_image_prediction.get("training_date")
        temperature = full_image_prediction.get("temperature")
        calibration_method = full_image_prediction.get("calibration_method")
        analysis_mode = "trained_model"
    else:
        _log("Using heuristic fallback")
        fake_score = _clamp01(
            0.40 * artifact_score + 0.30 * frequency_detector + 0.30 * diffusion_detector
        )
        detector_spread = _clamp01(np.std(list(component_scores.values())) * 2.0)
        uncertainty_score = _clamp01(
            0.68 * (1.0 - abs(fake_score - 0.5) * 2.0) + 0.32 * detector_spread
        )
        prediction = _label_from_score(fake_score, uncertainty_score)
        confidence = _confidence_from_scores(fake_score, uncertainty_score, prediction)
        confidence_band = _confidence_band(confidence)
        review_status = _review_status(prediction, confidence)
        binary_prediction = _binary_prediction(prediction)
        inference_engine = FALLBACK_ENGINE_NAME
        detector_version = FALLBACK_DETECTOR_VERSION
        model_version = FALLBACK_DETECTOR_VERSION
        dataset_version = "heuristic-demo"
        training_date = None
        temperature = 1.0
        calibration_method = "uncalibrated"
        analysis_mode = "fallback_demo"
        full_fake_score = None
        face_fake_score = None

    fake_prob = round(fake_score * 100, 2)
    real_prob = round((1.0 - fake_score) * 100, 2)

    blurred = np.asarray(full_gray.filter(ImageFilter.GaussianBlur(radius=10)), dtype=np.float64)
    full_gray_array = np.asarray(full_gray, dtype=np.float64)
    intensity_map = np.abs(full_gray_array - blurred)
    heatmap_path = _save_heatmap(image, intensity_map, heatmap_dir, "heatmap")

    attribution = _source_attribution_for_image(freq_score, edge_density, contrast, prediction)
    summary = (
        f"{prediction} assessment generated by {inference_engine.lower()} using "
        f"{'combined full-image and face analysis' if face_count else 'full-image analysis only'}, "
        f"with a {confidence_band.lower()} confidence band and localized artifact concentration highlighted "
        "in the evidence map."
    )
    evidence_points = [
        f"Inference engine: {inference_engine}",
        f"Detector version: {detector_version}",
        f"Model version: {model_version}",
        f"Dataset version: {dataset_version}",
        f"Calibration: {calibration_method} (temperature={round(float(temperature), 4)})",
        f"Detector used: {'Two-stage image detector' if analysis_mode == 'trained_model' else 'Heuristic fallback detector'}",
        f"Face region analyzed: {face_strategy} ({face_count} region{'s' if face_count != 1 else ''})",
        f"Final fake score: {round(fake_score * 100, 2)}",
        f"Final real score: {round((1.0 - fake_score) * 100, 2)}",
        f"Confidence band: {confidence_band}",
        f"Review status: {review_status}",
        f"Uncertainty score: {round(uncertainty_score * 100, 2)}",
        f"Artifact detector score: {round(artifact_score * 100, 2)}",
        f"Frequency detector score: {round(frequency_detector * 100, 2)}",
        f"Diffusion / GAN detector score: {round(diffusion_detector * 100, 2)}",
        f"Frequency-domain anomaly score: {round(freq_score * 100, 2)}",
        f"Contrast score: {round(contrast * 100, 2)}",
        f"Sensor-noise residual score: {round(sensor_noise * 100, 2)}",
        f"Smoothness penalty: {round(smoothness_penalty * 100, 2)}",
        f"Edge inconsistency score: {round(edge_density * 100, 2)}",
    ]
    if full_image_prediction is not None:
        evidence_points.append(
            f"Full-image model fake probability: {round(full_fake_score * 100, 2)}"
        )
        evidence_points.append(
            f"Full-image model real probability: {round((1.0 - full_fake_score) * 100, 2)}"
        )
        if face_fake_score is not None:
            evidence_points.append(
                f"Face model fake probability: {round(face_fake_score * 100, 2)}"
            )
            evidence_points.append(
                f"Face model real probability: {round((1.0 - face_fake_score) * 100, 2)}"
            )
            evidence_points.append(
                "Final decision used 60% full-image score plus 40% face score."
            )
        else:
            evidence_points.append(
                "No face was detected, so the final decision used only the full-image detector."
            )
        evidence_points.append(
            "Confidence explanation: lower confidence appears when fake and real scores are close together."
        )
        evidence_points.append(
            "The detector looks for synthetic texture patterns, frequency anomalies, and sensor-noise mismatch."
        )
    else:
        evidence_points.append(
            "No trained checkpoint was loaded, so this result used the heuristic fallback engine."
        )

    return {
        "prediction": prediction,
        "binary_prediction": binary_prediction,
        "leaning_prediction": _leaning_prediction(fake_score),
        "confidence": confidence,
        "confidence_band": confidence_band,
        "review_status": review_status,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "source_model": _best_source_model(attribution),
        "frequency_score": round(freq_score * 100, 2),
        "face_count": face_count,
        "face_strategy": face_strategy,
        "heatmap_path": heatmap_path,
        "attribution": attribution,
        "evidence_points": evidence_points,
        "summary": summary,
        "inference_engine": inference_engine,
        "detector_version": detector_version,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "training_date": training_date,
        "temperature": round(float(temperature), 6) if temperature is not None else None,
        "calibration_method": calibration_method,
        "analysis_mode": analysis_mode,
        "ensemble_fake_score": round(fake_score, 4),
        "uncertainty_score": uncertainty_score,
        "detector_breakdown": component_scores,
    }


def _build_video_heatmap(byte_values, heatmap_dir):
    canvas = np.zeros((64, 64), dtype=np.float64)
    usable = min(len(byte_values), canvas.size)
    if usable:
        canvas.flat[:usable] = byte_values[:usable]
    canvas = _normalize(canvas)
    colored = Image.fromarray(_colorize_heatmap(canvas))
    draw = ImageDraw.Draw(colored)
    draw.rectangle((3, 3, 61, 61), outline=(255, 245, 220), width=1)
    draw.text((8, 24), "VIDEO", fill=(255, 245, 220))
    filename = f"video_heatmap_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(heatmap_dir, filename)
    colored.resize((512, 512), resample=Image.Resampling.NEAREST).save(output_path)
    return output_path


def _analyze_video(file_path, heatmap_dir):
    with open(file_path, "rb") as stream:
        payload = np.frombuffer(stream.read(32768), dtype=np.uint8)

    if payload.size == 0:
        raise ValueError("The uploaded video file appears to be empty.")

    normalized = payload.astype(np.float64) / 255.0
    histogram = np.bincount(payload, minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    entropy = float(
        -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    )
    entropy_score = float(np.clip(entropy / 8.0, 0.0, 1.0))
    transitions = np.abs(np.diff(normalized)).mean() if normalized.size > 1 else 0.0
    transition_score = float(np.clip(transitions * 2.2, 0.0, 1.0))
    spectrum = np.abs(np.fft.rfft(normalized - normalized.mean()))
    frequency_score = float(np.clip(spectrum.mean() / (spectrum.max() + 1e-8), 0.0, 1.0))

    artifact_score = _clamp01(
        0.45 * transition_score + 0.35 * frequency_score + 0.20 * entropy_score
    )
    compression_score = _clamp01(0.55 * entropy_score + 0.45 * transition_score)
    temporal_score = _clamp01(0.60 * transition_score + 0.40 * frequency_score)
    fake_score = _clamp01(
        0.40 * artifact_score + 0.30 * compression_score + 0.30 * temporal_score
    )
    uncertainty_value = _clamp01(1.0 - abs(fake_score - 0.5) * 2.0)
    prediction = _heuristic_label(fake_score, uncertainty_value)
    confidence = _heuristic_confidence(fake_score, uncertainty_value, prediction)
    confidence_band = _confidence_band(confidence)
    review_status = _review_status(prediction, confidence)
    attribution = _source_attribution_for_video(entropy_score, transition_score, prediction)
    heatmap_path = _build_video_heatmap(payload, heatmap_dir)

    return {
        "prediction": prediction,
        "binary_prediction": _heuristic_binary_prediction(prediction),
        "leaning_prediction": AI_LABEL if fake_score >= 0.5 else REAL_LABEL,
        "confidence": confidence,
        "confidence_band": confidence_band,
        "review_status": review_status,
        "fake_prob": round(fake_score * 100, 2),
        "real_prob": round((1.0 - fake_score) * 100, 2),
        "source_model": _best_source_model(attribution),
        "frequency_score": round(frequency_score * 100, 2),
        "face_count": 0,
        "face_strategy": "video-stream",
        "heatmap_path": heatmap_path,
        "attribution": attribution,
        "evidence_points": [
            "Using heuristic fallback",
            f"Inference engine: {FALLBACK_VIDEO_ENGINE_NAME}",
            f"Detector version: {FALLBACK_DETECTOR_VERSION}",
            f"Model version: {FALLBACK_DETECTOR_VERSION}",
            f"Dataset version: heuristic-demo",
            "Calibration: uncalibrated (temperature=1.0)",
            f"Prediction: {prediction}",
            f"Leaning result: {AI_LABEL if fake_score >= 0.5 else REAL_LABEL}",
            f"Confidence band: {confidence_band}",
            f"Review status: {review_status}",
            f"AI score: {round(fake_score * 100, 2)}",
            f"Real score: {round((1.0 - fake_score) * 100, 2)}",
            f"Byte entropy score: {round(entropy_score * 100, 2)}",
            f"Transition irregularity: {round(transition_score * 100, 2)}",
            f"Frequency residue: {round(frequency_score * 100, 2)}",
            "No trained video model is configured, so uploaded videos use the fallback scorer.",
        ],
        "summary": (
            f"{prediction} classification generated from heuristic video scoring because no "
            "trained video checkpoint is configured."
        ),
        "inference_engine": FALLBACK_VIDEO_ENGINE_NAME,
        "detector_version": FALLBACK_DETECTOR_VERSION,
        "model_version": FALLBACK_DETECTOR_VERSION,
        "dataset_version": "heuristic-demo",
        "training_date": None,
        "temperature": 1.0,
        "calibration_method": "uncalibrated",
        "analysis_mode": "fallback_demo",
        "ensemble_fake_score": round(fake_score, 4),
        "uncertainty_score": round(uncertainty_value, 4),
        "detector_breakdown": {
            "video_artifact_detector": round(artifact_score, 4),
            "compression_detector": round(compression_score, 4),
            "temporal_detector": round(temporal_score, 4),
        },
    }


def analyze_media_file(file_path, media_type, heatmap_dir):
    if media_type == "image":
        return _analyze_image(file_path, heatmap_dir)
    if media_type == "video":
        return _analyze_video(file_path, heatmap_dir)
    raise ValueError("Unsupported media type.")


def generate_report_document(analysis, audit_trail, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    report_name = f"forensic_report_{analysis['analysis_id']}.html"
    report_path = os.path.join(report_dir, report_name)

    evidence_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in analysis.get("evidence_points", [])
    )
    attribution_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{score:.3f}</td></tr>"
        for label, score in analysis.get("attribution", {}).items()
    )
    audit_rows = "".join(
        "<tr>"
        f"<td>{html.escape(entry['created_at'])}</td>"
        f"<td>{html.escape(entry.get('full_name') or 'System')}</td>"
        f"<td>{html.escape(entry['action'])}</td>"
        f"<td>{html.escape(entry['details'])}</td>"
        "</tr>"
        for entry in audit_trail
    )

    markup = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Forensic Report #{analysis['analysis_id']}</title>
  <style>
    body {{
      font-family: 'Trebuchet MS', 'Aptos', sans-serif;
      background: #f4efe5;
      color: #1f2937;
      margin: 0;
      padding: 32px;
    }}
    .shell {{
      max-width: 920px;
      margin: 0 auto;
      background: white;
      border-radius: 24px;
      padding: 32px;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.14);
    }}
    h1, h2 {{
      margin-bottom: 12px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .meta div, .panel {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 16px;
      padding: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid #e5e7eb;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>SnapTrace Forensic Report</h1>
    <p>Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC for case #{analysis['analysis_id']}.</p>

    <div class="meta">
      <div><strong>Report ID</strong><br>{html.escape(str(analysis.get('report_id') or 'Pending'))}</div>
      <div><strong>Analyst</strong><br>{html.escape(analysis['full_name'])}</div>
      <div><strong>File</strong><br>{html.escape(analysis['original_filename'])}</div>
      <div><strong>Prediction</strong><br>{html.escape(analysis['prediction'])}</div>
      <div><strong>Binary Result</strong><br>{html.escape(analysis.get('binary_prediction') or analysis['prediction'])}</div>
      <div><strong>Confidence</strong><br>{analysis['confidence']}%</div>
      <div><strong>Confidence Band</strong><br>{html.escape(analysis.get('confidence_band') or 'N/A')}</div>
      <div><strong>Review Status</strong><br>{html.escape(analysis.get('review_status') or 'N/A')}</div>
      <div><strong>Real Score</strong><br>{analysis.get('real_prob', 0)}%</div>
      <div><strong>AI Score</strong><br>{analysis.get('fake_prob', 0)}%</div>
      <div><strong>Model</strong><br>{html.escape(analysis.get('inference_engine') or FALLBACK_ENGINE_NAME)}</div>
      <div><strong>Detector Version</strong><br>{html.escape(analysis.get('detector_version') or LEGACY_DETECTOR_VERSION)}</div>
      <div><strong>Model Version</strong><br>{html.escape(analysis.get('model_version') or 'N/A')}</div>
      <div><strong>Dataset Version</strong><br>{html.escape(analysis.get('dataset_version') or 'N/A')}</div>
      <div><strong>Training Date</strong><br>{html.escape(analysis.get('training_date') or 'N/A')}</div>
      <div><strong>Calibration</strong><br>{html.escape(analysis.get('calibration_method') or 'N/A')} ({analysis.get('temperature', 'N/A')})</div>
      <div><strong>Analysis Mode</strong><br>{html.escape(analysis.get('analysis_mode') or 'unknown')}</div>
      <div><strong>Face Count</strong><br>{html.escape(str(analysis.get('face_count', 0)))}</div>
      <div><strong>Upload Timestamp</strong><br>{html.escape(analysis.get('uploaded_at') or 'N/A')}</div>
      <div><strong>Analysis Timestamp</strong><br>{html.escape(analysis.get('created_at') or 'N/A')}</div>
      <div><strong>Report Timestamp</strong><br>{html.escape(analysis.get('generated_at') or 'Pending')}</div>
      <div><strong>SHA-256</strong><br><code>{html.escape(analysis.get('file_sha256') or 'Unavailable')}</code></div>
    </div>

    <div class="panel">
      <h2>Executive Summary</h2>
      <p>{html.escape(analysis['summary'])}</p>
      <ul>{evidence_items}</ul>
    </div>

    <div class="panel">
      <h2>Source Attribution Scores</h2>
      <table>
        <thead><tr><th>Candidate Model</th><th>Score</th></tr></thead>
        <tbody>{attribution_rows}</tbody>
      </table>
    </div>

    <div class="panel">
      <h2>Chain of Custody / Audit Trail</h2>
      <table>
        <thead><tr><th>Timestamp</th><th>Actor</th><th>Action</th><th>Details</th></tr></thead>
        <tbody>{audit_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(markup)

    return report_path


def load_report_markup(report_path):
    if not report_path or not os.path.exists(report_path):
        return "<p>Report file is not available yet.</p>"
    with open(report_path, "r", encoding="utf-8") as report_file:
        return report_file.read()

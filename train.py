"""
Training script for a real-vs-fake image classifier.

Expected dataset layout:

dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/

This script trains a PyTorch image classifier and saves:
- deepfake_model.pth
- model_manifest.json
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone


DATASET_ROOT = Path("dataset")
CHECKPOINT_PATH = Path("deepfake_model.pth")
MODEL_MANIFEST_PATH = Path("model_manifest.json")
DATASET_VERSION_FILE = DATASET_ROOT / "version.txt"
IMAGE_SIZE = 224
CLASS_NAMES = ["fake", "real"]
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
REAL_CLASS_WEIGHT_BOOST = 1.2
DEFAULT_REAL_THRESHOLD = 0.65
DEFAULT_FAKE_THRESHOLD = 0.85


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise SystemExit(
            "PyTorch and torchvision are required. Install them before training."
        ) from exc
    return torch, nn, F, DataLoader, datasets, models, transforms


def _utc_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_dataset_version():
    return f"dataset-{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def _resolve_dataset_version():
    env_value = os.getenv("SNAPTRACE_DATASET_VERSION", "").strip()
    if env_value:
        return env_value
    if DATASET_VERSION_FILE.exists():
        file_value = DATASET_VERSION_FILE.read_text(encoding="utf-8").strip()
        if file_value:
            return file_value
    return _default_dataset_version()


def _resolve_model_version(architecture):
    env_value = os.getenv("SNAPTRACE_MODEL_VERSION", "").strip()
    if env_value:
        return env_value
    if architecture == "efficientnet_b0_binary":
        return "snaptrace-efficientnet-b0-v1"
    if architecture == "resnet50_binary":
        return "snaptrace-resnet50-v1"
    if architecture == "resnet18_binary":
        return "snaptrace-resnet18-v1"
    return f"snaptrace-{architecture}-v1"


def build_dataloaders():
    torch, _nn, _F, DataLoader, datasets, _models, transforms = _require_torch()

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, shear=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(DATASET_ROOT / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_ROOT / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(DATASET_ROOT / "test", transform=eval_transform)

    print("class_names:", train_dataset.classes)
    if train_dataset.classes != CLASS_NAMES:
        raise SystemExit(
            f"Unexpected class order {train_dataset.classes}. Expected {CLASS_NAMES}."
        )

    class_counts = [0 for _ in CLASS_NAMES]
    for _path, label in train_dataset.samples:
        class_counts[label] += 1

    return (
        DataLoader(train_dataset, batch_size=16, shuffle=True),
        DataLoader(val_dataset, batch_size=16, shuffle=False),
        DataLoader(test_dataset, batch_size=16, shuffle=False),
        {
            "train_class_counts": class_counts,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        },
    )


def build_model(architecture="efficientnet_b0_binary"):
    torch, nn, _F, _DataLoader, _datasets, models, _transforms = _require_torch()

    if architecture == "efficientnet_b0_binary":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model
    if architecture == "resnet18_binary":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if architecture == "resnet50_binary":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    raise SystemExit(
        "Unsupported architecture. Use efficientnet_b0_binary, resnet18_binary, or resnet50_binary."
    )


def evaluate_model(model, dataloader, device):
    torch, _nn, F, _DataLoader, _datasets, _models, _transforms = _require_torch()

    model.eval()
    total = 0
    correct = 0
    fake_tp = fake_fp = fake_fn = 0
    real_tp = real_fp = real_fn = 0
    class_totals = {name: 0 for name in CLASS_NAMES}
    class_correct = {name: 0 for name in CLASS_NAMES}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            fake_index = CLASS_NAMES.index("fake")
            real_index = CLASS_NAMES.index("real")
            fake_tp += ((preds == fake_index) & (labels == fake_index)).sum().item()
            fake_fp += ((preds == fake_index) & (labels != fake_index)).sum().item()
            fake_fn += ((preds != fake_index) & (labels == fake_index)).sum().item()
            real_tp += ((preds == real_index) & (labels == real_index)).sum().item()
            real_fp += ((preds == real_index) & (labels != real_index)).sum().item()
            real_fn += ((preds != real_index) & (labels == real_index)).sum().item()

            for class_name, class_index in zip(CLASS_NAMES, (fake_index, real_index)):
                matches = labels == class_index
                class_totals[class_name] += matches.sum().item()
                class_correct[class_name] += ((preds == class_index) & matches).sum().item()

    precision = (fake_tp / (fake_tp + fake_fp)) * 100 if (fake_tp + fake_fp) else 0.0
    recall = (fake_tp / (fake_tp + fake_fn)) * 100 if (fake_tp + fake_fn) else 0.0
    real_precision = (real_tp / (real_tp + real_fp)) * 100 if (real_tp + real_fp) else 0.0
    real_recall = (real_tp / (real_tp + real_fn)) * 100 if (real_tp + real_fn) else 0.0
    accuracy = (correct / total) * 100 if total else 0.0
    return {
        "accuracy": round(accuracy, 2),
        "precision_fake": round(precision, 2),
        "recall_fake": round(recall, 2),
        "precision_real": round(real_precision, 2),
        "recall_real": round(real_recall, 2),
        "per_class_accuracy": {
            class_name: round(
                (class_correct[class_name] / class_totals[class_name]) * 100, 2
            )
            if class_totals[class_name]
            else 0.0
            for class_name in CLASS_NAMES
        },
        "total": total,
    }


def _collect_logits_and_labels(model, dataloader, device):
    torch, _nn, _F, _DataLoader, _datasets, _models, _transforms = _require_torch()
    logits_list = []
    labels_list = []
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            logits_list.append(logits.detach())
            labels_list.append(labels.detach())

    if not logits_list:
        return None, None

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def calibrate_temperature(model, dataloader, device):
    torch, _nn, _F, _DataLoader, _datasets, _models, _transforms = _require_torch()
    logits, labels = _collect_logits_and_labels(model, dataloader, device)
    if logits is None or labels is None or logits.shape[0] < 2:
        return 1.0, "uncalibrated"

    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def _closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    calibrated_temperature = float(
        torch.exp(log_temperature).clamp(min=1e-3, max=100.0).detach().cpu().item()
    )
    return round(calibrated_temperature, 6), "temperature_scaling"


def train_model(epochs=12, architecture="efficientnet_b0_binary"):
    torch, _nn, F, _DataLoader, _datasets, _models, _transforms = _require_torch()
    if not (DATASET_ROOT / "train").is_dir() or not (DATASET_ROOT / "val").is_dir():
        raise SystemExit("Expected dataset/train and dataset/val directories before training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, dataset_stats = build_dataloaders()
    model = build_model(architecture=architecture).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    class_counts = dataset_stats["train_class_counts"]
    if not all(class_counts):
        raise SystemExit(f"Training set must contain both classes. Found counts: {class_counts}")

    total_train = float(sum(class_counts))
    class_weights = [total_train / (len(CLASS_NAMES) * count) for count in class_counts]
    real_index = CLASS_NAMES.index("real")
    class_weights[real_index] *= REAL_CLASS_WEIGHT_BOOST
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    best_val_acc = -1.0
    dataset_version = _resolve_dataset_version()
    model_version = _resolve_model_version(architecture)
    training_date = _utc_timestamp()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate_model(model, val_loader, device)
        print(
            f"epoch={epoch + 1} loss={running_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_metrics['accuracy']} val_precision_fake={val_metrics['precision_fake']} "
            f"val_recall_fake={val_metrics['recall_fake']}"
        )

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best checkpoint at val_acc={best_val_acc} to {CHECKPOINT_PATH}")

    print(f"Best validation accuracy: {best_val_acc}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    temperature, calibration_method = calibrate_temperature(model, val_loader, device)
    print(
        f"Calibrated validation temperature={temperature} method={calibration_method}"
    )

    manifest = {
        "display_name": "SnapTrace Deepfake Detector",
        "framework": "pytorch_state_dict",
        "architecture": architecture,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "training_date": training_date,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "class_names": CLASS_NAMES,
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "normalization_mean": NORMALIZE_MEAN,
        "normalization_std": NORMALIZE_STD,
        "positive_label": "ai_generated",
        "channel_order": "nchw",
        "real_threshold": DEFAULT_REAL_THRESHOLD,
        "fake_threshold": DEFAULT_FAKE_THRESHOLD,
        "prefer_face_crop": False,
        "detector_version": model_version,
        "temperature": temperature,
        "calibration_method": calibration_method,
        "confidence_notes": (
            "Confidence reflects the loaded PyTorch checkpoint output with a stricter "
            "fake threshold to reduce false positives on real photos."
        ),
    }
    MODEL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {MODEL_MANIFEST_PATH}")

    if (DATASET_ROOT / "test").is_dir():
        test_metrics = evaluate_model(model, test_loader, device)
        print(
            "test_acc={accuracy} test_precision_fake={precision_fake} "
            "test_recall_fake={recall_fake} test_precision_real={precision_real} "
            "test_recall_real={recall_real}".format(**test_metrics)
        )
        print(f"test_per_class_accuracy={test_metrics['per_class_accuracy']}")


if __name__ == "__main__":
    train_model(epochs=12, architecture="efficientnet_b0_binary")

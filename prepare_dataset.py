"""
Prepare a train/val/test image dataset for SnapTrace.

Examples:

1) Download a Hugging Face dataset repo and split images by folder names:
   python prepare_dataset.py --hf-dataset PrithivMLmods/Deepfake-vs-Real-60K

2) Prepare from a Hugging Face dataset exposed through the datasets library:
   python prepare_dataset.py --hf-dataset dragonintelligence/CIFAKE-image-dataset --hf-loader datasets

3) Prepare from an existing local folder:
   python prepare_dataset.py --source-dir path\\to\\images

The script infers labels from path names using keywords such as:
- fake, ai, generated, synthetic, diffusion, gan, deepfake
- real, natural, authentic, camera
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
HF_CACHE_ROOT = PROJECT_ROOT / "dataset_source"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = ("fake", "real")
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, help="Local image source root")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset repo id")
    parser.add_argument(
        "--hf-loader",
        choices=("snapshot", "datasets"),
        default="snapshot",
        help="Use snapshot for folder-style repos, or datasets for parquet/image datasets.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class after label inference",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="Use hardlinks by default to avoid duplicating large datasets.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing files inside dataset/train,val,test before writing.",
    )
    return parser.parse_args()


def _infer_label(path: Path):
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in ("fake", "ai", "generated", "synthetic", "diffusion", "gan", "deepfake")):
        return "fake"
    if any(token in text for token in ("real", "natural", "authentic", "camera")):
        return "real"
    return None


def _download_hf_dataset(repo_id: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for --hf-dataset. Install it first."
        ) from exc

    local_dir = HF_CACHE_ROOT / repo_id.replace("/", "__")
    print(f"Downloading Hugging Face dataset {repo_id} into {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return local_dir


def _load_hf_dataset_with_datasets(repo_id: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required for --hf-loader datasets. Install it first."
        ) from exc

    print(f"Loading Hugging Face dataset {repo_id} with datasets.load_dataset")
    return load_dataset(repo_id)


def _iter_images(source_root: Path):
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _gather_labeled_images(source_root: Path, max_per_class=None):
    buckets = defaultdict(list)
    for path in _iter_images(source_root):
        label = _infer_label(path.relative_to(source_root))
        if label in CLASS_NAMES:
            buckets[label].append(path)

    for label in CLASS_NAMES:
        buckets[label].sort()
        if max_per_class is not None:
            buckets[label] = buckets[label][:max_per_class]

    return buckets


def _clear_existing_targets():
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            target = DATASET_ROOT / split / label
            if not target.exists():
                continue
            for item in target.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()


def _ensure_targets():
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            (DATASET_ROOT / split / label).mkdir(parents=True, exist_ok=True)


def _split_items(items, train_ratio, val_ratio):
    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count :]
    return {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }


def _place_file(src: Path, dst: Path, mode: str):
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _label_from_dataset_value(value):
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("fake", "deepfake", "ai", "ai_generated", "generated"):
            return "fake"
        if lowered in ("real", "authentic", "natural"):
            return "real"
    if isinstance(value, (int, float)):
        # Common binary mapping used by CIFAKE-style datasets.
        if int(value) == 0:
            return "fake"
        if int(value) == 1:
            return "real"
    return None


def _prepare_from_hf_dataset_object(dataset_dict, seed, max_per_class, copy_mode, clear_existing):
    _ensure_targets()
    if clear_existing:
        _clear_existing_targets()

    rng = random.Random(seed)
    gathered = defaultdict(list)
    temp_root = HF_CACHE_ROOT / "prepared_images"
    temp_root.mkdir(parents=True, exist_ok=True)

    for split_name, split in dataset_dict.items():
        for index, sample in enumerate(split):
            if "image" not in sample or "label" not in sample:
                continue
            label = _label_from_dataset_value(sample["label"])
            if label not in CLASS_NAMES:
                continue
            gathered[label].append((split_name, index, sample["image"]))

    counts = {label: len(gathered[label]) for label in CLASS_NAMES}
    print("Detected HF dataset samples:", counts)
    if not all(counts.values()):
        raise SystemExit(
            "Could not detect both fake and real labels from the Hugging Face dataset."
        )

    for label in CLASS_NAMES:
        rng.shuffle(gathered[label])
        if max_per_class is not None:
            gathered[label] = gathered[label][:max_per_class]
        split_map = _split_items(gathered[label], 0.8, 0.1)
        for target_split, items in split_map.items():
            for out_index, (source_split, sample_index, image_obj) in enumerate(items):
                target_dir = DATASET_ROOT / target_split / label
                filename = target_dir / f"{source_split}_{sample_index:06d}.png"
                if filename.exists():
                    filename.unlink()
                image_obj.save(filename)

    for split in ("train", "val", "test"):
        fake_count = len(list((DATASET_ROOT / split / "fake").glob("*")))
        real_count = len(list((DATASET_ROOT / split / "real").glob("*")))
        print(f"  {split}: fake={fake_count} real={real_count}")


def prepare_dataset(source_root: Path, train_ratio, val_ratio, seed, max_per_class, copy_mode, clear_existing):
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise SystemExit("Use ratios where train_ratio > 0, val_ratio >= 0, and train+val < 1.")

    labeled = _gather_labeled_images(source_root, max_per_class=max_per_class)
    counts = {label: len(labeled[label]) for label in CLASS_NAMES}
    print("Detected images:", counts)
    if not all(counts.values()):
        raise SystemExit(
            "Could not find labeled images for both fake and real classes. "
            "Check the dataset structure or path keywords."
        )

    _ensure_targets()
    if clear_existing:
        _clear_existing_targets()

    rng = random.Random(seed)
    split_counts = defaultdict(dict)

    for label in CLASS_NAMES:
        items = labeled[label][:]
        rng.shuffle(items)
        split_map = _split_items(items, train_ratio, val_ratio)
        for split, split_items in split_map.items():
            split_counts[split][label] = len(split_items)
            target_dir = DATASET_ROOT / split / label
            for index, src in enumerate(split_items):
                dst = target_dir / f"{index:06d}{src.suffix.lower()}"
                _place_file(src, dst, copy_mode)

    print("Prepared dataset splits:")
    for split in ("train", "val", "test"):
        print(f"  {split}: fake={split_counts[split].get('fake', 0)} real={split_counts[split].get('real', 0)}")


def main():
    args = parse_args()
    if bool(args.source_dir) == bool(args.hf_dataset):
        raise SystemExit("Pass exactly one of --source-dir or --hf-dataset.")

    if args.hf_dataset:
        if args.hf_loader == "datasets":
            dataset_dict = _load_hf_dataset_with_datasets(args.hf_dataset)
            _prepare_from_hf_dataset_object(
                dataset_dict=dataset_dict,
                seed=args.seed,
                max_per_class=args.max_per_class,
                copy_mode=args.copy_mode,
                clear_existing=args.clear_existing,
            )
            return
        source_root = _download_hf_dataset(args.hf_dataset)
    else:
        source_root = args.source_dir.resolve()
        if not source_root.is_dir():
            raise SystemExit(f"Source directory not found: {source_root}")

    prepare_dataset(
        source_root=source_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
        copy_mode=args.copy_mode,
        clear_existing=args.clear_existing,
    )


if __name__ == "__main__":
    main()

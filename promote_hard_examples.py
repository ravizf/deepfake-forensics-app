import argparse
import json
import shutil
from pathlib import Path


HARD_EXAMPLE_ROOT = Path("artifacts") / "hard_examples"
DATASET_TRAIN_ROOT = Path("dataset") / "train"
EXPECTED_TO_CLASS = {
    "Real": "real",
    "AI-Generated": "fake",
}


def _latest_manifest_path():
    if not HARD_EXAMPLE_ROOT.exists():
        raise SystemExit("No hard-example exports found yet.")

    run_dirs = sorted(
        [path for path in HARD_EXAMPLE_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    if not run_dirs:
        raise SystemExit("No hard-example run directories were found.")
    manifest_path = run_dirs[0] / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found at {manifest_path}")
    return manifest_path


def _load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _destination_for(sample, copied_index):
    expected_label = sample.get("expected_label")
    class_name = EXPECTED_TO_CLASS.get(expected_label)
    if class_name is None:
        raise ValueError(f"Unsupported expected label for promotion: {expected_label}")

    source_path = Path(sample["source_path"])
    destination_dir = DATASET_TRAIN_ROOT / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_name = f"hard_{copied_index:03d}_{source_path.name}"
    return destination_dir / destination_name


def promote_approved_samples(manifest_path):
    manifest = _load_manifest(manifest_path)
    approved_samples = [
        sample for sample in manifest.get("samples", []) if sample.get("review_status") == "approved"
    ]
    if not approved_samples:
        print("No approved hard examples found in the manifest.")
        return 0

    copied = 0
    for copied, sample in enumerate(approved_samples, start=1):
        source_path = Path(sample["source_path"])
        if not source_path.exists():
            print(f"Skipping missing source file: {source_path}")
            continue
        destination = _destination_for(sample, copied)
        shutil.copy2(source_path, destination)
        print(f"Promoted {source_path} -> {destination}")

    print(f"Promoted {copied} approved hard examples.")
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Promote approved hard examples into dataset/train/{fake,real}."
    )
    parser.add_argument(
        "--manifest",
        help="Path to a hard-example manifest.json. Defaults to the latest run.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else _latest_manifest_path()
    promote_approved_samples(manifest_path)


if __name__ == "__main__":
    main()

"""Convenience launcher for the real trained SnapTrace detector.

Expected dataset layout:

dataset/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/   # optional but recommended
    real/
    fake/

This wrapper uses the project's main training pipeline in train.py and
produces:
- deepfake_model.pth
- model_manifest.json
- model_metrics.json
"""

from train import train_model


if __name__ == "__main__":
    train_model(epochs=8, architecture="efficientnet_b0_binary")

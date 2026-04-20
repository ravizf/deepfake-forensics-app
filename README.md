# SnapTrace Deepfake Forensics

## Deploy Online

### Recommended: Netlify Frontend + Render Backend

This project is a Flask app, so the Python backend should run on Render while the static demo frontend can be hosted on Netlify.

#### Render backend

The repo now includes:

- `Procfile`
- `render.yaml`
- `requirements.txt` with backend runtime packages
- public health route: `/api/health`
- public analysis route: `/api/public-analyze`

Deploy the repo to Render as a Python web service. After deploy, your backend should expose:

- `https://your-render-service.onrender.com/api/health`
- `https://your-render-service.onrender.com/api/public-analyze`

#### Netlify frontend

The repo now includes:

- `index.html` as a static upload/analyze client
- `netlify.toml`
- `config.js`
- `config.example.js`

For Netlify:

1. Deploy this repo as a static site.
2. Edit `config.js` and set:

```js
window.SNAPTRACE_API_BASE = "https://your-render-service.onrender.com";
```

3. Redeploy Netlify.

The static page will then call the Render backend for public analysis.

### Local + deployed behavior

- Local frontend uses same-origin or `localhost` by default.
- Netlify frontend should point to Render through `config.js`.
- Public demo uploads go through the seeded `public-demo@snaptrace.local` backend account.

## Trained Model Setup

### Option 1 (Best)

Put the model file here exactly:

`C:\me\code\deepfake-forensics-app\deepfake_model.pth`

This must be in the project root, at the same level as `app.py`, `forensics.py`, and `model_manifest.json`.

### Option 2

If the model is somewhere else, update `checkpoint_path` in `model_manifest.json` to the correct relative path. Example:

```json
{
  "checkpoint_path": "models/deepfake_model.pth"
}
```

### Manifest Check

Confirm `model_manifest.json` matches the trained PyTorch checkpoint:

```json
{
  "display_name": "SnapTrace Deepfake Detector",
  "framework": "pytorch_state_dict",
  "checkpoint_path": "deepfake_model.pth",
  "architecture": "resnet50_binary",
  "input_size": [224, 224],
  "positive_label": "ai_generated",
  "channel_order": "nchw",
  "normalization_mean": [0.485, 0.456, 0.406],
  "normalization_std": [0.229, 0.224, 0.225],
  "detector_version": "snaptrace-deepfake-v1"
}
```

### Step 3

Restart the server after adding the file:

```powershell
python app.py
```

### Step 4

Check terminal output. A successful load should show:

```text
MODEL LOADED: True
```

If you see `MODEL LOADED: False`, the path is still wrong or the checkpoint is incompatible.

### Step 5

Test `/api/analyze` again. A trained run should stop showing `Heuristic fallback` and should return trained output such as:

```json
{
  "prediction": "Likely Real",
  "binary_prediction": "Real",
  "confidence": 91.2,
  "inference_engine": "SnapTrace Deepfake Detector",
  "analysis_mode": "trained_model"
}
```

## Common Problems

- Missing file: `deepfake_model.pth` is not in the configured path.
- Wrong architecture: the app expects a binary `resnet50_binary` checkpoint.
- Wrong output layer: the checkpoint must match the model head expected by the app.
- Wrong classes: the app expects binary real vs AI-generated prediction.

Without a valid `.pth` file, SnapTrace will stay in heuristic fallback mode.

## Build a Training Dataset

If you want to train your own model, the app now includes a dataset preparation script.

Prepare from Hugging Face:

```powershell
python prepare_dataset.py --hf-dataset PrithivMLmods/Deepfake-vs-Real-60K --clear-existing
```

Prepare from a local folder:

```powershell
python prepare_dataset.py --source-dir path\to\images --clear-existing
```

This creates the layout expected by `train.py`:

```text
dataset/
  train/
    fake/
    real/
  val/
    fake/
    real/
  test/
    fake/
    real/
```

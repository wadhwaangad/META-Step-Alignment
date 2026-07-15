# Step Aligner

Local-friendly replication of the seven-stage step alignment pipeline for an 8 GB VRAM GPU.

This implementation keeps the expensive language/vision-language stages on Gemini and runs only visual feature extraction locally. It is designed to be resumable: every stage writes cached artifacts under the chosen output directory.

## What It Builds

```text
video
  -> frame sampling
  -> local visual embeddings
  -> temporal clustering
  -> Gemini segment captions
  -> Gemini step grouping
  -> Gemini step alignment
  -> Gemini coherence scoring
  -> transcript/alignment/qa artifacts
```

## Install

Create an environment with Python 3.10+:

```powershell
cd C:\Users\wadhw\Documents\Codex\2026-07-14\i\outputs\step-aligner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

For the VJEPA Hugging Face backend, install PyTorch CUDA separately for your GPU, then:

```powershell
pip install transformers accelerate pillow opencv-python scikit-learn scipy numpy pydantic google-genai
```

For a quick dry run without a GPU/model, use `--feature-backend colorhist`.

## Gemini Key

Set your API key:

```powershell
$env:GEMINI_API_KEY="your_key_here"
```

## Video Metadata

Create a JSON file like:

```json
{
  "activity": "Make a peanut butter sandwich",
  "steps": [
    "Place bread slices on the plate",
    "Spread peanut butter on one slice",
    "Put the slices together"
  ]
}
```

## Run

```powershell
step-align run `
  --video C:\path\to\video.mp4 `
  --metadata C:\path\to\metadata.json `
  --out C:\path\to\run-output `
  --feature-backend vjepa_hf `
  --vjepa-repo facebook/vjepa2-vitl-fpc64-256
```

For an 8 GB card, start with:

```powershell
step-align run `
  --video C:\path\to\video.mp4 `
  --metadata C:\path\to\metadata.json `
  --out C:\path\to\run-output `
  --feature-backend vjepa_hf `
  --vjepa-repo facebook/vjepa2-vitl-fpc64-256 `
  --window-size 32 `
  --window-stride 16 `
  --image-size 256 `
  --caption-frames 8
```

When Gemini returns multiple action lines for one segment, the runner recursively splits that segment by time and captions each half. The default is one split level; increase with `--max-caption-splits 2` if your videos are long or visually dense.

If VRAM is still tight, use `--feature-backend colorhist` first to validate the end-to-end Gemini stages, then swap in VJEPA.

## Run facebook/wearable-ai

The Wearable AI Dataset is hosted at `facebook/wearable-ai` on Hugging Face. It is gated, so first open the dataset page in a browser, accept the terms, and log in with `huggingface-cli login`.

The dataset has three configs:

- `egoproactive`: 700 streaming proactive-assistant videos, about 23 GB. Start here.
- `egoconv`: 700 conversational QA videos, about 91 GB.
- `egolongqa`: 700 long-form MCQ videos, about 203 GB.

Download only the smallest/most relevant task first:

```powershell
huggingface-cli download facebook/wearable-ai `
  --repo-type dataset `
  --include "egoproactive/**" "starter_kit/**" "*.md" "LICENSE" `
  --local-dir C:\datasets\wearable-ai
```

Then run a small pilot:

```powershell
step-align run-wearable `
  --dataset-root C:\datasets\wearable-ai `
  --config egoproactive `
  --out C:\wearable-ai-runs `
  --limit 5 `
  --metadata-source gemini_plan `
  --feature-backend colorhist `
  --skip-missing
```

After the pilot works, switch to VJEPA:

```powershell
step-align run-wearable `
  --dataset-root C:\datasets\wearable-ai `
  --config egoproactive `
  --out C:\wearable-ai-runs `
  --limit 25 `
  --metadata-source gemini_plan `
  --feature-backend vjepa_hf `
  --vjepa-repo facebook/vjepa2-vitl-fpc64-256 `
  --window-size 32 `
  --window-stride 16 `
  --caption-frames 8 `
  --skip-missing
```

`--metadata-source gemini_plan` asks Gemini to infer a procedural reference outline from the dataset row. This is useful because Wearable AI is a video QA/proactive-assistant benchmark, not a native step-alignment dataset with explicit recipe-style substeps.

## Outputs

Each run writes:

- `frames/`: sampled frames
- `features.npz`: local visual embeddings
- `segments.json`: temporal clusters
- `captions.json`: Gemini captions
- `grouped_steps.json`: grouped step transcript
- `alignment.json`: reference-step indices
- `qa.json`: coherence score and issues
- `transcript.jsonl`: final transcript rows

## GitHub Pages Results Site

Build the static dark dashboard:

```powershell
step-align build-site `
  --runs C:\wearable-ai-runs `
  --site-dir C:\path\to\your-github-io-repo\docs
```

The site writes:

- `index.html`
- `styles.css`
- `app.js`
- `data.json`
- `assets/*.jpg` thumbnails
- `.nojekyll`

For GitHub Pages, commit the `docs/` folder to your `username.github.io` repo or to a project repo configured to publish from `/docs`.

This repository also includes an empty generated site scaffold at `docs/` so the page structure is ready before results exist.

## Notes

The original paper-style pipeline used VJEPA2 ViT-Giant and much larger hosted VLM/LLM models. This project replicates the method, not the exact model stack. On 8 GB VRAM, the practical route is a smaller VJEPA/VJEPA2.1 encoder plus Gemini for the semantic stages.
## Built-In Local Video Folder

This repo can process your own local videos without the Facebook dataset.

Folder layout:

```text
step-aligner/
  videos/
    raw/              # put .mp4, .mov, .mkv, .avi, .webm, .m4v files here
    metadata/         # optional per-video JSON files
  runs/               # generated pipeline outputs
  docs/               # generated GitHub Pages dashboard
```

Optional metadata files go in `videos/metadata` and should match the video stem:

```text
videos/raw/make_sandwich.mp4
videos/metadata/make_sandwich.json
```

Example metadata:

```json
{
  "activity": "Make a peanut butter sandwich",
  "steps": [
    "Place two bread slices on the plate",
    "Spread peanut butter on one bread slice",
    "Put the bread slices together"
  ]
}
```

Run a small dry run with the lightweight feature backend:

```powershell
step-align run-folder `
  --feature-backend colorhist `
  --limit 3
```

Run with VJEPA on an 8 GB GPU:

```powershell
step-align run-folder `
  --feature-backend vjepa_hf `
  --vjepa-repo facebook/vjepa2-vitl-fpc64-256 `
  --window-size 32 `
  --window-stride 16 `
  --caption-frames 8
```

By default, missing metadata is inferred with Gemini from the filename using `--metadata-source gemini_plan`. If you want the simplest fallback, use:

```powershell
step-align run-folder --metadata-source default
```

After each folder run, the dark GitHub Pages site is rebuilt in `docs/`.
## Built-In Local Video Folder

This repo can process your own local videos without the Facebook dataset.

Folder layout:

```text
step-aligner/
  videos/
    raw/              # put .mp4, .mov, .mkv, .avi, .webm, .m4v files here
    metadata/         # optional per-video JSON files
  runs/               # generated pipeline outputs
  docs/               # generated GitHub Pages dashboard
```

Optional metadata files go in `videos/metadata` and should match the video stem:

```text
videos/raw/make_sandwich.mp4
videos/metadata/make_sandwich.json
```

Example metadata:

```json
{
  "activity": "Make a peanut butter sandwich",
  "steps": [
    "Place two bread slices on the plate",
    "Spread peanut butter on one bread slice",
    "Put the bread slices together"
  ]
}
```

Run a small dry run with the lightweight feature backend:

```powershell
step-align run-folder `
  --feature-backend colorhist `
  --limit 3
```

Run with VJEPA on an 8 GB GPU:

```powershell
step-align run-folder `
  --feature-backend vjepa_hf `
  --vjepa-repo facebook/vjepa2-vitl-fpc64-256 `
  --window-size 32 `
  --window-stride 16 `
  --caption-frames 8
```

By default, missing metadata is inferred with Gemini from the filename using `--metadata-source gemini_plan`. If you want the simplest fallback, use:

```powershell
step-align run-folder --metadata-source default
```

After each folder run, the dark GitHub Pages site is rebuilt in `docs/`.

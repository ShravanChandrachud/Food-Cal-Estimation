# Food Macro Estimation from Cooking Videos

Identifies ingredients in cooking videos using CLIP ViT-L/14 embeddings with a multi-label MLP classifier, then estimates macronutrients via USDA data and a local LLM.

**Author:** Shravan Chandrachud | Northeastern University | Spring 2026 | AI Foundations  
**GitHub:** https://github.com/ShravanChandrachud/Food-Cal-Estimation.git

---

## Requirements

- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 3060)
- [Ollama](https://ollama.com/download/windows) (local LLM runtime)
- [FFmpeg 8.1](https://www.gyan.dev/ffmpeg/builds/) (video processing)
- [Deno v2.7+](https://github.com/denoland/deno/releases) (required by yt-dlp for YouTube)

FFmpeg and Deno must be on your system PATH.

## Setup

### 1. Install FFmpeg

1. Download `ffmpeg-release-essentials.zip` from https://www.gyan.dev/ffmpeg/builds/
2. Extract to a folder, e.g. `C:\ffmpeg`
3. Add the `bin` folder to your system PATH:
   - Press `Win + S`, search **"Environment Variables"**
   - Click **"Edit the system environment variables"** → **"Environment Variables"**
   - Under **System variables**, find `Path`, click **Edit** → **New**
   - Add the path to the bin folder, e.g. `C:\ffmpeg\bin`
   - Click OK on all dialogs
4. Open a **new** terminal and verify:
   ```bash
   ffmpeg -version
   ```

### 2. Install Deno

1. Download `deno-x86_64-pc-windows-msvc.zip` from https://github.com/denoland/deno/releases
2. Extract to a folder, e.g. `C:\deno`
3. Add that folder to your system PATH (same steps as FFmpeg above), e.g. 
`C:\deno`
4. Open a **new** terminal and verify:
   ```bash
   deno --version
   ```

> **Why Deno?** Since November 2025, YouTube presents JavaScript challenges that yt-dlp must solve to download videos above 360p. Deno is a JavaScript runtime that yt-dlp uses automatically when it's on PATH.

### 3. Create virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 4. Install and start Ollama

1. Download and install from https://ollama.com/download/windows (installs to C: by default, ~150MB — this is fine)
2. Set the model storage location to avoid filling your C: drive. Add a **system environment variable** (same Environment Variables panel as above):
   - Variable: `OLLAMA_MODELS`
   - Value: your preferred path, e.g. `C:\Users\<username>\.ollama\models`
3. Optionally add these for VRAM optimization:
   - `OLLAMA_FLASH_ATTENTION` = `1`
   - `OLLAMA_KV_CACHE_TYPE` = `q8_0`
4. Open a **new** terminal and pull the model:
   ```bash
   ollama pull qwen2.5:3b
   ollama serve                  # keep this terminal open
   ```

### 5. Set up environment variables

Create a `.env` file in the project root:

```
USDA_API_KEY=your_key_here
```

Get a free API key at https://fdc.nal.usda.gov/api-key-signup/ (optional — the offline database handles most lookups).

## How to Run (Step by Step)

### Phase 1: Data Preparation

> **Note:** The `datasets/` folder is not included in the repository due to size (~16,990 images). Download the raw datasets (citations to the datasets given in final progress report submission) and place them in `datasets/raw/`, then run:

```bash
python scripts/organize_data.py       # raw/ → raw_organized/
python scripts/split_data.py          # raw_organized/ → train/ val/ test/
python scripts/generate_config.py     # → config/dataset_config.json
python scripts/visualize_data.py      # prints dataset statistics table
python scripts/visualize_samples.py   # → outputs/*.png (sample grids)
```

### Phase 2: Embedding Extraction and Model Training

```bash
# Extract CLIP ViT-L/14 embeddings for all splits
python scripts/extract_embeddings.py

# Train MLP classifier + tune thresholds + generate evaluation charts
python notebooks/02_train_and_evaluate.py
```

This produces:
- `data/embeddings_train.npz`, `data/embeddings_val.npz`, `data/embeddings_test.npz`
- `data/mlp_classifier.pth` (trained model weights)
- `data/thresholds.json` (per-class decision thresholds)
- `data/class_names.json` (ordered class list)
- `outputs/02_train_eval/` (PCA, t-SNE, confusion matrix, F1 chart, threshold chart)

### Phase 3: Build USDA Database

```bash
python scripts/build_usda_db.py       # → data/usda_nutrition.db
```

### Phase 4: Run on Cooking Videos

#### Download a YouTube cooking video (this is a dummy link in practice find and replace the youtube link before executing command)

```bash
python scripts/download_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

The video saves to `videos/VIDEO_ID.mp4`.

#### Or use your own videos

Manually place any `.mp4`, `.mkv`, `.avi`, `.mov`, or `.webm` file into the `videos/` folder.

#### Run the pipeline

Make sure `ollama serve` is running in a separate terminal, then:

```bash
# Interactive mode — processes all videos, asks you to confirm ingredients
python scripts/run_pipeline.py

# Process a specific video
python scripts/run_pipeline.py videos/your_video.mp4

# Notebook version — generates matplotlib charts at each step
python notebooks/03_video_demo.py
```

The interactive pipeline will:
1. Extract keyframes from the video
2. Classify each frame using the trained MLP
3. Show detected ingredients and ask you to confirm (Y/n) and provide quantities
4. Look up USDA nutritional data
5. Generate a macro summary via the LLM
6. Save everything to `outputs/v2_<video_name>/`

---

## Outputs

| File | Location | Description |
|------|----------|-------------|
| Evaluation charts | `outputs/02_train_eval/` | PCA, t-SNE, confusion matrix, F1, thresholds |
| Video demo results | `outputs/v2_demo_<name>/` | Keyframes, heatmap, frequency chart, macro breakdown |
| Pipeline results | `outputs/v2_<name>/` | frame_detections.json, summary.json, report.txt |
| Final report | `submission-docs/final-progress/` | LaTeX source + compiled PDF |

---

## Quick Test (5 minutes)

If you just want to verify the pipeline works end-to-end:

```bash
# 1. Make sure ollama is running
ollama serve

# 2. Download a short cooking video
python scripts/download_video.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# 3. Run the pipeline
python scripts/run_pipeline.py
```

---

## Technical Stack

| Component | Tool |
|-----------|------|
| Vision encoder | OpenCLIP ViT-L/14 (LAION-2B, 768D, frozen) |
| Classifier | MLP 768→256→20, sigmoid + BCEWithLogitsLoss |
| Keyframes | PySceneDetect AdaptiveDetector + OpenCV |
| Video download | yt-dlp + Deno + FFmpeg |
| Nutrition data | USDA FoodData Central SR Legacy (offline SQLite) |
| LLM | Qwen 2.5 3B via Ollama (local, structured JSON) |
| Fuzzy matching | RapidFuzz |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Test Macro F1 | 0.9923 |
| Test Micro F1 | 0.9955 |
| Classes | 20 subclasses across 6 food categories |
| Training images | ~13,000 |
| Training time (MLP) | < 30 seconds on precomputed embeddings |

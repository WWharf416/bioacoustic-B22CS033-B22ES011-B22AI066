# BioAcoustic Inference Setup

This repository contains the inference scripts for DCASE 2024 Task 5 BioAcoustic event detection pipeline. It uses a frozen BEATs encoder along with a few-shot prototypical network methodology.

## 1. Setup

First, install the Python requirements:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

These scripts do not implicitly download the DCASE 2024 Task 5 dataset. You must download it manually and ensure it is available locally. 

By default, the scripts look for the data in `dcase2024_task5/Development_Set` within the project root. If your dataset is located elsewhere, you must set the `BIOACOUSTIC_DATA_ROOT` environment variable before running the scripts:

```bash
export BIOACOUSTIC_DATA_ROOT=/path/to/your/dcase2024_task5
```

You'll need the BEATs pre-trained checkpoint or rely on the HuggingFace `microsoft/BEATs-Base` loaded automatically if local checkpoints are missing.

## 2. Using Single File Inference
`inference.py` allows testing the model on a quick query audio sample for a given support class.

```bash
python inference.py --audio /path/to/query.wav --class-name WMW --n-shot 1
```

**Options**:
- `--audio`: Valid path to a query `.wav` file.
- `--class-name`: The corresponding DCASE class for few-shot learning (e.g. `WMW`).
- `--n-shot`: How many samples to use to build the prototype representation.

The script decodes and outputs predicted events to `outputs/demo_single/demo_pred_events.csv`, and draws a prediction plot of raw/smoothed prototype similarity scores.

## 3. Using Batch Inference
`batch_inference.py` enables processing multiple files (up to 50 audio files by default) and prints the overall F1 score across the query samples.

```bash
python batch_inference.py --class-name WMW --n-shot 5
```

This will run through the query files within the support class set, extract embeddings, build a prototype, calculate prototypical scores, and dump prediction CSVs along with a final summary metric (`outputs/demo_specific/final_f1_scores.csv`).

## 4. Interactive Demo UI

An interactive web-based demo UI is available under `demo_ui/`. It connects to the real inference pipeline and lets you configure parameters, select audio files, and visualize results — all from your browser.

### Running the Demo Server

The demo requires Flask (`pip install flask`) in addition to the base requirements. Start the server with:

```bash
python demo_ui/server.py --port 8765
```

Then open `http://localhost:8765` in your browser.

**Options**:
- `--port`: Port to serve on (default: `8765`).
- `--host`: Host to bind (default: `0.0.0.0`).
- `--preload`: Pre-load the BEATs encoder and DCASE annotations on startup instead of lazy-loading on first request.

### What the UI Provides

- **Class & file browser** — dynamically lists all DCASE classes and audio files from the dataset.
- **Parameter controls** — adjust N-shot, threshold, median kernel, max seconds, and merge gap via sliders.
- **Real inference** — runs the actual BEATs encoder + prototypical scoring pipeline on the server.
- **Score visualization** — interactive canvas chart showing raw scores, smoothed scores, threshold, predicted events, and ground truth.
- **Results table** — lists each detected event with onset/offset, duration, and IoU match status.
- **F1 metric** — computes and displays the F1 score against ground truth annotations.
- **File upload** — upload your own `.wav` files for inference against any support class.

## 5. Key Notes
- Audio files are typically clamped to the `[0, 12.0s]` interval by default via `--max-seconds 12.0`. Set `--max-seconds 0.0` for full audio inference.
- An adaptive threshold is calculated from `mean + 1*std` of prototypical similarity scores on the fly for deciding event detection.

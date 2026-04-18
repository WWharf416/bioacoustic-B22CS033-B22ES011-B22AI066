# BioAcoustic Inference Setup

This repository contains the inference scripts for DCASE 2024 Task 5 BioAcoustic event detection pipeline. It uses a frozen BEATs encoder along with a few-shot prototypical network methodology.

## 1. Setup

First, install the Python requirements:
```bash
pip install -r requirements.txt
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

## 4. Key Notes
- Audio files are typically clamped to the `[0, 12.0s]` interval by default via `--max-seconds 12.0`. Set `--max-seconds 0.0` for full audio inference.
- An adaptive threshold is calculated from `mean + 1*std` of prototypical similarity scores on the fly for deciding event detection.

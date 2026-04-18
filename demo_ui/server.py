"""
Flask backend for the BioAcoustic Inference Demo UI.

Wraps the real inference.py pipeline — loads BEATs, encodes audio,
builds prototypes, computes scores, decodes events — and exposes
the results as JSON API endpoints.
"""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# ── Bootstrap the real inference module ──────────────────────
# Add the parent directory (containing inference.py) to the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import inference as inf  # noqa: E402  — the real inference module

# ── Flask App ────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")

# ── Global encoder (lazy-loaded once) ───────────────────────
_ENCODER = None
_ANN_DF = None


def get_encoder():
    """Lazy-load and cache the BEATs encoder."""
    global _ENCODER
    if _ENCODER is None:
        print("[server] Loading BEATs encoder (this may take ~10s the first time)...")
        _ENCODER = inf.BEATsEncoder().to(inf.DEVICE)
        print(f"[server] Encoder ready — embed_dim={_ENCODER.embed_dim}, device={inf.DEVICE}")
    return _ENCODER


def get_annotations():
    """Lazy-load and cache the DCASE annotation dataframe."""
    global _ANN_DF
    if _ANN_DF is None:
        print(f"[server] Loading DCASE annotations from {inf.DEV_SET_DIR} ...")
        _ANN_DF = inf.load_dcase_annotations(inf.DEV_SET_DIR)
        print(f"[server] Loaded {len(_ANN_DF)} positive event rows across {_ANN_DF['class'].nunique()} classes")
    return _ANN_DF


# ── Static file serving ─────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── API: List available classes + files ──────────────────────
@app.route("/api/classes")
def api_classes():
    """Return available DCASE classes and their audio files."""
    ann_df = get_annotations()

    result = {}
    for cls_name in sorted(ann_df["class"].unique()):
        cls_df = ann_df[ann_df["class"] == cls_name]
        files = sorted(cls_df["Audiofilename"].dropna().unique().tolist())
        result[cls_name] = {
            "files": [
                {"path": f, "name": Path(f).name}
                for f in files
            ],
            "event_count": len(cls_df),
        }

    return jsonify(result)


# ── API: Run inference ───────────────────────────────────────
@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Run single-file prototypical inference.

    Expects JSON body:
    {
      "class_name": "WMW",
      "n_shot": 5,
      "threshold": 0.0,
      "median_kernel": 13,
      "merge_gap_ms": 500,
      "max_seconds": 12.0,
      "query_file": "/path/to/audio.wav"  (optional – auto-picked if missing)
    }
    """
    try:
        data = request.get_json(force=True)
        class_name = data.get("class_name", "WMW")
        n_shot = max(1, int(data.get("n_shot", 5)))
        threshold = float(data.get("threshold", 0.0))
        median_kernel = int(data.get("median_kernel", 13))
        merge_gap_ms = float(data.get("merge_gap_ms", 500.0))
        max_seconds = float(data.get("max_seconds", 12.0))
        chunk_seconds = float(data.get("chunk_seconds", 4.0))
        min_chunk_seconds = float(data.get("min_chunk_seconds", 0.5))
        query_file = data.get("query_file", None)

        ann_df = get_annotations()
        encoder = get_encoder()

        # Cache directory
        cache_dir = Path(inf.PROJECT_ROOT) / "embed_cache" / "demo_ui"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Pick support and query
        chosen_class, support_files, query_file_resolved = inf.pick_support_and_query(
            ann_df,
            class_name=class_name,
            n_shot=n_shot,
            query_audio=query_file,
            max_seconds=max_seconds,
        )

        # Prepare annotations
        support_ann = ann_df[
            (ann_df["class"] == chosen_class)
            & (ann_df["Audiofilename"].isin(support_files))
        ]
        support_ann = inf.clip_event_rows_to_window(support_ann, max_seconds=max_seconds)

        query_gt_df = ann_df[
            (ann_df["class"] == chosen_class)
            & (ann_df["Audiofilename"].apply(
                lambda x: Path(str(x)).resolve() == Path(query_file_resolved).resolve()
            ))
        ]
        query_gt_df = inf.clip_event_rows_to_window(query_gt_df, max_seconds=max_seconds)

        # Build prototypes
        pos_proto, neg_proto = inf.build_prototypes(
            support_files=support_files,
            support_ann=support_ann,
            enc=encoder,
            cache_dir=cache_dir,
            max_seconds=max_seconds,
            chunk_seconds=chunk_seconds,
            min_chunk_seconds=min_chunk_seconds,
            use_cache=True,
        )

        # Score query
        scores = inf.prototypical_scores(
            query_file_resolved,
            pos_proto,
            neg_proto,
            encoder,
            cache_dir=cache_dir,
            max_seconds=max_seconds,
            chunk_seconds=chunk_seconds,
            min_chunk_seconds=min_chunk_seconds,
            use_cache=True,
        )

        # Decode events
        pred_events, smooth_scores, effective_threshold = inf.decode_events(
            scores,
            threshold=threshold,
            median_kernel=median_kernel,
            merge_gap_ms=merge_gap_ms,
        )

        # Ground truth
        gt_events = []
        if not query_gt_df.empty:
            gt_events = [
                {"onset": float(r.Starttime), "offset": float(r.Endtime)}
                for _, r in query_gt_df.iterrows()
            ]

        # F1
        f1 = inf.quick_f1(pred_events, gt_events) if gt_events else 0.0

        # Build time axis
        hop_sec = inf.HOP_MS / 1000.0
        times = [i * hop_sec for i in range(len(scores))]

        # Compute stats
        import numpy as np
        raw_scores = scores.tolist()
        smooth_list = smooth_scores.tolist()

        raw_arr = np.array(raw_scores)
        smooth_arr = np.array(smooth_list)

        response = {
            "status": "ok",
            "class_name": chosen_class,
            "query_file": Path(query_file_resolved).name,
            "query_file_path": str(query_file_resolved),
            "support_files": [Path(f).name for f in support_files],
            "n_shot": n_shot,

            # Scores (downsample for network efficiency if very long)
            "times": _downsample(times, 2000),
            "raw_scores": _downsample(raw_scores, 2000),
            "smoothed_scores": _downsample(smooth_list, 2000),
            "threshold": float(effective_threshold),

            # Events
            "pred_events": pred_events,
            "gt_events": gt_events,
            "f1": float(f1),

            # Stats
            "stats": {
                "raw_min": float(raw_arr.min()),
                "raw_max": float(raw_arr.max()),
                "raw_mean": float(raw_arr.mean()),
                "raw_std": float(raw_arr.std()),
                "smooth_min": float(smooth_arr.min()),
                "smooth_max": float(smooth_arr.max()),
                "smooth_mean": float(smooth_arr.mean()),
                "frames_above": int((smooth_arr >= effective_threshold).sum()),
                "total_frames": len(scores),
                "hop_ms": inf.HOP_MS,
                "duration": times[-1] if times else 0,
                "kernel_size": median_kernel,
            },
        }
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ── API: Upload audio for inference ──────────────────────────
@app.route("/api/upload_and_infer", methods=["POST"])
def api_upload_and_infer():
    """Upload a .wav file and run inference on it."""
    try:
        if "audio" not in request.files:
            return jsonify({"status": "error", "message": "No audio file in request"}), 400

        audio_file = request.files["audio"]
        class_name = request.form.get("class_name", "WMW")
        n_shot = max(1, int(request.form.get("n_shot", 5)))
        threshold = float(request.form.get("threshold", 0.0))
        median_kernel = int(request.form.get("median_kernel", 13))
        merge_gap_ms = float(request.form.get("merge_gap_ms", 500.0))
        max_seconds = float(request.form.get("max_seconds", 12.0))

        # Save uploaded file temporarily
        upload_dir = Path(inf.PROJECT_ROOT) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = upload_dir / audio_file.filename
        audio_file.save(str(tmp_path))

        # Now run inference using the regular endpoint logic
        data = {
            "class_name": class_name,
            "n_shot": n_shot,
            "threshold": threshold,
            "median_kernel": median_kernel,
            "merge_gap_ms": merge_gap_ms,
            "max_seconds": max_seconds,
            "query_file": str(tmp_path),
        }

        # Reuse the infer logic
        with app.test_request_context(
            "/api/infer", method="POST",
            content_type="application/json",
            data=json.dumps(data),
        ):
            return api_infer()

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def _downsample(arr, max_points):
    """Downsample an array for efficient JSON transfer."""
    if len(arr) <= max_points:
        return arr
    step = len(arr) / max_points
    return [arr[int(i * step)] for i in range(max_points)]


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioAcoustic Demo UI Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--preload", action="store_true", help="Pre-load encoder on startup")
    args = parser.parse_args()

    if args.preload:
        print("[server] Pre-loading model and annotations...")
        get_encoder()
        get_annotations()

    print(f"[server] Starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

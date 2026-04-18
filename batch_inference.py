import argparse
import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.signal import medfilt
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path("/scratch/kundalwal.1/bioAcoustic").resolve()
DATA_ROOT = Path(os.environ.get("BIOACOUSTIC_DATA_ROOT", PROJECT_ROOT / "dcase2024_task5"))
DEV_SET_DIR = DATA_ROOT / "Development_Set"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
HOP_MS = 20.83  # BEATs: 48 frames per 1s of audio at 16kHz → 1000/48 ≈ 20.83ms


class BEATsEncoder(nn.Module):
    """Frozen BEATs encoder wrapper for local checkpoint or HF model."""

    MODEL_ID = "microsoft/BEATs-Base"
    DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "BEATs_iter3_plus_AS2M.pt"
    DEFAULT_CODE_DIR = PROJECT_ROOT / "models" / "beats"

    def __init__(self, model_id: str | None = None):
        super().__init__()
        mid = Path(str(model_id or os.environ.get("BIOACOUSTIC_BEATS_MODEL") or self.DEFAULT_CHECKPOINT))
        self.backend = "hf"

        if mid.exists() and mid.is_file() and mid.suffix == ".pt":
            code_dir = Path(os.environ.get("BIOACOUSTIC_BEATS_CODE_DIR", str(self.DEFAULT_CODE_DIR)))
            if not code_dir.exists():
                raise FileNotFoundError(
                    "BEATs checkpoint found but code directory is missing. "
                    f"Expected BEATs code at {code_dir}."
                )
            sys.path.insert(0, str(code_dir))
            from BEATs import BEATs, BEATsConfig  # pylint: disable=import-error

            print(f"Loading BEATs checkpoint from {mid} ...")
            checkpoint = torch.load(str(mid), map_location="cpu")
            cfg = BEATsConfig(checkpoint["cfg"])
            self.model = BEATs(cfg)
            self.model.load_state_dict(checkpoint["model"])
            if hasattr(self.model, "predictor"):
                self.model.predictor = None
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.embed_dim = getattr(cfg, "encoder_embed_dim", 768)
            self.backend = "beats_checkpoint"
            return

        from transformers import AutoFeatureExtractor, AutoModel

        mid_str = str(model_id or os.environ.get("BIOACOUSTIC_BEATS_MODEL") or self.MODEL_ID)
        load_kwargs = {"local_files_only": True} if Path(mid_str).exists() else {}
        print(f"Loading BEATs HF model from {mid_str} ...")
        self.processor = AutoFeatureExtractor.from_pretrained(mid_str, **load_kwargs)
        self.model = AutoModel.from_pretrained(mid_str, **load_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = 768

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if self.backend == "beats_checkpoint":
            padding_mask = torch.zeros(wav.shape, dtype=torch.bool, device=wav.device)
            return self.model.extract_features(wav, padding_mask=padding_mask)[0]

        inputs = self.processor(wav.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.last_hidden_state


def load_and_resample(path: str, target_sr: int = TARGET_SR, max_seconds: float = 12.0):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if max_seconds > 0:
        max_samples = int(target_sr * max_seconds)
        wav = wav[..., :max_samples]
    return wav.squeeze(0), target_sr


def encode_waveform_in_chunks(
    wav: torch.Tensor,
    enc: nn.Module,
    chunk_seconds: float,
    min_chunk_seconds: float,
    sr: int = TARGET_SR,
) -> torch.Tensor:
    if wav.dim() != 1:
        wav = wav.squeeze(0)

    chunk_samples = max(1, int(chunk_seconds * sr))
    min_chunk_samples = max(1, int(min_chunk_seconds * sr))

    def prepare(piece: torch.Tensor) -> torch.Tensor:
        if piece.numel() < min_chunk_samples:
            piece = F.pad(piece, (0, min_chunk_samples - piece.numel()))
        return piece

    if wav.numel() <= chunk_samples:
        return enc(prepare(wav).to(DEVICE)).squeeze(0).detach().cpu()

    chunks = []
    for start in range(0, wav.numel(), chunk_samples):
        piece = wav[start : start + chunk_samples]
        if piece.numel() == 0:
            continue
        piece = prepare(piece)
        piece_emb = enc(piece.to(DEVICE)).squeeze(0).detach().cpu()
        chunks.append(piece_emb)
    if not chunks:
        return torch.empty((0, enc.embed_dim))
    return torch.cat(chunks, dim=0)


def load_dcase_annotations(ann_dir: Path) -> pd.DataFrame:
    rows = []
    base_cols = {"Audiofilename", "Starttime", "Endtime"}

    def resolve_audio_path(csv_file: Path, raw_name: str) -> str:
        name = Path(str(raw_name).strip()).name
        wav_name = str(Path(name).with_suffix(".wav"))
        wav_path = csv_file.with_name(wav_name)
        if wav_path.exists():
            return str(wav_path)
        return str(csv_file.with_name(name))

    for csv_file in ann_dir.rglob("*.csv"):
        df = pd.read_csv(csv_file)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if not base_cols.issubset(df.columns):
            continue

        task_name = csv_file.parent.name
        label_cols = [c for c in df.columns if c not in base_cols]
        for _, row in df.iterrows():
            labels = {col: str(row[col]).strip().upper() for col in label_cols if pd.notna(row[col])}
            is_positive = labels.get("Q") == "POS" if "Q" in labels else any(v == "POS" for v in labels.values())
            if not is_positive:
                continue

            rows.append(
                {
                    "Audiofilename": resolve_audio_path(csv_file, row["Audiofilename"]),
                    "Starttime": float(row["Starttime"]),
                    "Endtime": float(row["Endtime"]),
                    "class": task_name,
                }
            )
    return pd.DataFrame(rows)


def get_frame_embeddings(
    fpath: str,
    enc: nn.Module,
    cache_dir: Path,
    max_seconds: float,
    chunk_seconds: float,
    min_chunk_seconds: float,
    use_cache: bool,
) -> np.ndarray:
    cache_key = f"{Path(fpath).stem}_sec{int(max_seconds)}_chunks{str(chunk_seconds).replace('.', 'p')}"
    cache_file = cache_dir / f"{cache_key}_frames.npy"

    if use_cache and cache_file.exists():
        return np.load(str(cache_file))

    wav, sr = load_and_resample(fpath, max_seconds=max_seconds)
    emb = encode_waveform_in_chunks(
        wav,
        enc,
        chunk_seconds=chunk_seconds,
        min_chunk_seconds=min_chunk_seconds,
        sr=sr,
    ).numpy()

    if use_cache:
        np.save(str(cache_file), emb)
    return emb


def build_prototypes(
    support_files: list[str],
    support_ann: pd.DataFrame,
    enc: nn.Module,
    cache_dir: Path,
    max_seconds: float,
    chunk_seconds: float,
    min_chunk_seconds: float,
    use_cache: bool,
) -> tuple[np.ndarray, np.ndarray]:
    hop = HOP_MS / 1000.0
    positives = []
    negatives = []

    for support_file in support_files:
        embs = get_frame_embeddings(
            support_file,
            enc,
            cache_dir,
            max_seconds,
            chunk_seconds,
            min_chunk_seconds,
            use_cache,
        )
        if embs.size == 0:
            continue

        frame_times = np.arange(embs.shape[0]) * hop
        file_ann = support_ann[support_ann["Audiofilename"] == support_file]

        pos_mask = np.zeros(embs.shape[0], dtype=bool)
        for _, row in file_ann.iterrows():
            pos_mask |= (frame_times >= row["Starttime"]) & (frame_times <= row["Endtime"])

        if pos_mask.any():
            positives.append(embs[pos_mask])
        neg_mask = ~pos_mask
        if neg_mask.any():
            negatives.append(embs[neg_mask])

    def _make_proto(frame_list: list[np.ndarray], embed_dim: int) -> np.ndarray:
        if frame_list:
            proto = np.concatenate(frame_list, axis=0).mean(axis=0)
        else:
            proto = np.zeros(embed_dim, dtype=np.float32)
        return proto / (np.linalg.norm(proto) + 1e-8)

    pos_proto = _make_proto(positives, enc.embed_dim)
    neg_proto = _make_proto(negatives, enc.embed_dim)

    n_pos = sum(f.shape[0] for f in positives) if positives else 0
    n_neg = sum(f.shape[0] for f in negatives) if negatives else 0
    print(f"  Prototype stats: {n_pos} positive frames, {n_neg} negative frames")
    print(f"  Pos-neg proto cosine similarity: {np.dot(pos_proto, neg_proto):.3f}")

    return pos_proto, neg_proto


def prototypical_scores(
    query_file: str,
    pos_prototype: np.ndarray,
    neg_prototype: np.ndarray,
    enc: nn.Module,
    cache_dir: Path,
    max_seconds: float,
    chunk_seconds: float,
    min_chunk_seconds: float,
    use_cache: bool,
) -> np.ndarray:
    embs = get_frame_embeddings(
        query_file,
        enc,
        cache_dir,
        max_seconds,
        chunk_seconds,
        min_chunk_seconds,
        use_cache,
    )
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    pos_scores = embs @ pos_prototype
    neg_scores = embs @ neg_prototype
    return pos_scores - neg_scores


def decode_events(
    scores: np.ndarray,
    threshold: float,
    hop_ms: float = HOP_MS,
    min_duration_ms: float = 60.0,
    median_kernel: int = 13,
    merge_gap_ms: float = 500.0,
) -> tuple[list[dict], np.ndarray, float]:
    if len(scores) == 0:
        return [], scores, threshold

    median_kernel = max(1, int(median_kernel))
    if median_kernel % 2 == 0:
        median_kernel += 1
    median_kernel = min(median_kernel, len(scores) if len(scores) % 2 == 1 else max(1, len(scores) - 1))
    smooth = medfilt(scores.astype(float), kernel_size=median_kernel)

    if threshold <= 0:
        threshold = float(smooth.mean() + 1.0 * smooth.std())

    binary = (smooth >= threshold).astype(int)

    events = []
    in_event = False
    onset = None

    for i, val in enumerate(binary):
        if val == 1 and not in_event:
            onset = i
            in_event = True
        elif val == 0 and in_event:
            offset = i - 1
            on_s = onset * hop_ms / 1000.0
            off_s = offset * hop_ms / 1000.0
            if (off_s - on_s) * 1000 >= min_duration_ms:
                events.append({"onset": on_s, "offset": off_s})
            in_event = False

    if in_event:
        on_s = onset * hop_ms / 1000.0
        off_s = len(scores) * hop_ms / 1000.0
        events.append({"onset": on_s, "offset": off_s})

    merged_events = []
    for ev in events:
        if not merged_events:
            merged_events.append(ev)
        else:
            prev_ev = merged_events[-1]
            if (ev["onset"] - prev_ev["offset"]) * 1000.0 <= merge_gap_ms:
                prev_ev["offset"] = max(prev_ev["offset"], ev["offset"])
            else:
                merged_events.append(ev)

    return merged_events, smooth, threshold

def quick_f1(pred_events: list[dict], gt_events: list[dict], iou_threshold: float = 0.5) -> float:
    if not gt_events:
        return 1.0 if not pred_events else 0.0
    if not pred_events:
        return 0.0

    iou_matrix = np.zeros((len(gt_events), len(pred_events)))
    for i, g in enumerate(gt_events):
        for j, p in enumerate(pred_events):
            inter = max(0.0, min(g["offset"], p["offset"]) - max(g["onset"], p["onset"]))
            union = (g["offset"] - g["onset"]) + (p["offset"] - p["onset"]) - inter
            if union > 0:
                iou_matrix[i, j] = inter / union

    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    tp = 0
    for r, c in zip(gt_indices, pred_indices):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1

    prec = tp / len(pred_events)
    rec = tp / len(gt_events)
    
    return (2 * prec * rec) / (prec + rec + 1e-8)

def clip_event_rows_to_window(df: pd.DataFrame, max_seconds: float) -> pd.DataFrame:
    if df.empty or max_seconds <= 0:
        return df.copy()

    clipped = df.copy()
    clipped["Starttime"] = clipped["Starttime"].clip(lower=0.0, upper=max_seconds)
    clipped["Endtime"] = clipped["Endtime"].clip(lower=0.0, upper=max_seconds)
    clipped = clipped[clipped["Endtime"] > clipped["Starttime"]].copy()
    return clipped


def parse_args():
    parser = argparse.ArgumentParser(description="Quick batch inference.")
    parser.add_argument("--class-name", type=str, default="WMW", help="Support class to use from DCASE annotations.")
    parser.add_argument("--n-shot", type=int, default=5, help="Number of support files for prototype.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Event decoding threshold. 0 = adaptive (mean+1*std of smoothed scores).")
    parser.add_argument("--median-kernel", type=int, default=13, help="Median filter kernel size for score smoothing.")
    parser.add_argument("--merge-gap-ms", type=float, default=500.0, help="Maximum gap in ms to merge close events.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Truncate each file for speed (0.0 = full audio).")
    parser.add_argument("--chunk-seconds", type=float, default=4.0, help="Encoding chunk length in seconds.")
    parser.add_argument("--min-chunk-seconds", type=float, default=0.5, help="Minimum chunk padding length.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "demo_specific"),
        help="Where demo artifacts are saved.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "embed_cache" / "demo_specific"),
        help="Where frame embedding cache is stored.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable frame cache for this run.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Device: {DEVICE}")
    print(f"Project root: {PROJECT_ROOT}")

    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not DEV_SET_DIR.exists():
        raise FileNotFoundError(f"DCASE Development_Set not found at {DEV_SET_DIR}")

    ann_df = load_dcase_annotations(DEV_SET_DIR)
    if ann_df.empty:
        raise RuntimeError("No positive events found in annotations.")

    chosen_class = args.class_name
    class_df = ann_df[ann_df["class"] == chosen_class].copy()
    class_df_window = clip_event_rows_to_window(class_df, max_seconds=args.max_seconds)
    window_files = class_df_window["Audiofilename"].dropna().drop_duplicates().sort_values().tolist()
    
    if len(window_files) < max(2, args.n_shot + 1):
        raise RuntimeError(f"Not enough files for {chosen_class} in the given window.")

    support_files = window_files[:args.n_shot]
    query_candidates = [f for f in window_files if f not in support_files]
    query_files = query_candidates[:50]

    print(f"Chosen class: {chosen_class}")
    print(f"Support files ({len(support_files)}):")
    for sf in support_files:
        print(f"  - {sf}")
    print(f"Query files count: {len(query_files)}")

    support_ann = ann_df[(ann_df["class"] == chosen_class) & (ann_df["Audiofilename"].isin(support_files))]
    support_ann = clip_event_rows_to_window(support_ann, max_seconds=args.max_seconds)

    if support_ann.empty:
        print("WARNING: Support events do not overlap the selected max-seconds window.")

    encoder = BEATsEncoder().to(DEVICE)
    print(f"Encoder ready. Embed dim: {encoder.embed_dim}")

    pos_prototype, neg_prototype = build_prototypes(
        support_files=support_files,
        support_ann=support_ann,
        enc=encoder,
        cache_dir=cache_dir,
        max_seconds=args.max_seconds,
        chunk_seconds=args.chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        use_cache=not args.no_cache,
    )

    results = []

    for idx, query_file in enumerate(query_files):
        print(f"\n[{idx+1}/{len(query_files)}] Processing {Path(query_file).name}...")

        scores = prototypical_scores(
            query_file,
            pos_prototype,
            neg_prototype,
            encoder,
            cache_dir=cache_dir,
            max_seconds=args.max_seconds,
            chunk_seconds=args.chunk_seconds,
            min_chunk_seconds=args.min_chunk_seconds,
            use_cache=not args.no_cache,
        )
        
        pred_events, smooth_scores, effective_threshold = decode_events(
            scores,
            threshold=args.threshold,
            median_kernel=args.median_kernel,
            merge_gap_ms=args.merge_gap_ms,
        )

        base_name = Path(query_file).stem
        pred_path = out_dir / f"demo_pred_events_{base_name}.csv"
        pd.DataFrame(pred_events).to_csv(pred_path, index=False)

        query_gt_df = ann_df[
            (ann_df["class"] == chosen_class)
            & (ann_df["Audiofilename"].apply(lambda x: Path(str(x)).resolve() == Path(query_file).resolve()))
        ]
        query_gt_df = clip_event_rows_to_window(query_gt_df, max_seconds=args.max_seconds)

        f1 = 0.0
        if not query_gt_df.empty:
            gt_events = [
                {"onset": float(r.Starttime), "offset": float(r.Endtime)}
                for _, r in query_gt_df.iterrows()
            ]
            f1 = quick_f1(pred_events, gt_events)
            print(f"  F1 score: {f1:.3f} (Predicted: {len(pred_events)}, GT: {len(gt_events)})")
        else:
            print(f"  F1 score: 0.000 (No GT events found in window)")

        results.append({
            "Audiofilename": Path(query_file).name,
            "F1_score": f1
        })

    results_df = pd.DataFrame(results)
    final_csv_path = out_dir / "final_f1_scores.csv"
    results_df.to_csv(final_csv_path, index=False)

    avg_f1 = results_df["F1_score"].mean()
    print(f"\n=======================")
    print(f"Average F1 score over {len(results)} files: {avg_f1:.4f}")
    print(f"Individual F1 scores saved to {final_csv_path}")


if __name__ == "__main__":
    main()

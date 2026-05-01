"""
Cross-corpus evaluation: run RAVDESS-trained models on CREMA-D without
any retraining. Tests whether SER models generalize beyond their
training distribution.

CREMA-D label mapping to RAVDESS:
    ANG -> angry
    DIS -> disgust
    FEA -> fearful
    HAP -> happy
    NEU -> neutral
    SAD -> sad

CREMA-D has 6 emotions, RAVDESS has 8. The two RAVDESS-only emotions
(calm, surprised) are excluded from this evaluation. We report:
    1. Overall accuracy on the 6-class subset
    2. Per-emotion recall
    3. The accuracy drop from in-domain (RAVDESS) to out-of-domain (CREMA-D)

CREMA-D filename format: ActorID_SentenceID_Emotion_Intensity.wav
    e.g. 1001_DFA_ANG_XX.wav
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoFeatureExtractor

from data_loader import EMOTION_TO_ID, ID_TO_EMOTION
from model import SERClassifier, RECOMMENDED_MODELS


# CREMA-D emotion code -> RAVDESS emotion name
CREMA_TO_RAVDESS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Subset of RAVDESS emotion IDs that overlap with CREMA-D
OVERLAP_EMOTIONS = list(CREMA_TO_RAVDESS.values())
OVERLAP_IDS = [EMOTION_TO_ID[e] for e in OVERLAP_EMOTIONS]


def parse_crema_filename(filepath):
    """Parse CREMA-D filename and return emotion code (or None if invalid)."""
    name = os.path.basename(filepath).replace(".wav", "")
    parts = name.split("_")
    if len(parts) < 4:
        return None
    return parts[2]  # ANG, DIS, FEA, HAP, NEU, SAD


def build_crema_manifest(data_dir):
    """Scan CREMA-D directory and return a DataFrame."""
    import glob
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    rows = []
    for f in sorted(files):
        code = parse_crema_filename(f)
        if code is None or code not in CREMA_TO_RAVDESS:
            continue
        emotion = CREMA_TO_RAVDESS[code]
        rows.append({
            "filepath": f,
            "emotion_code": code,
            "emotion": emotion,
            "emotion_id": EMOTION_TO_ID[emotion],
        })
    df = pd.DataFrame(rows)
    print(f"Found {len(df)} CREMA-D clips matching 6 overlapping emotions")
    print(df["emotion"].value_counts())
    return df


def load_audio(filepath, target_sr=16000, max_duration=5.0):
    """Load and pad/truncate audio to fixed length."""
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    waveform = waveform.squeeze(0)
    target_len = int(target_sr * max_duration)
    if waveform.shape[0] > target_len:
        waveform = waveform[:target_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
    return waveform


def evaluate_model(model_key, model_dir, df, device, batch_size=16):
    """Run inference for one trained model on CREMA-D."""
    pretrained_name = RECOMMENDED_MODELS[model_key]
    processor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    model = SERClassifier(
        pretrained_name=pretrained_name,
        num_labels=len(EMOTION_TO_ID),
    ).to(device)

    weights_path = Path(model_dir) / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    paths = df["filepath"].tolist()
    labels = df["emotion_id"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc=f"{model_key} on CREMA-D"):
        batch_paths = paths[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        waveforms = [load_audio(p).numpy() for p in batch_paths]
        inputs = processor(
            waveforms, sampling_rate=16000, return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            logits = model(input_values)

        # Mask out the 2 non-overlap emotions (calm=1, surprised=7) by setting
        # their logits to -inf so the model can't predict them on this subset.
        for i_ in range(len(EMOTION_TO_ID)):
            if i_ not in OVERLAP_IDS:
                logits[:, i_] = float("-inf")

        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch_labels)

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true, y_pred, label_names_full):
    """Compute accuracy, F1, confusion matrix; restrict to overlap emotions."""
    overall_acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(label_names_full)))
    )
    report = classification_report(
        y_true, y_pred,
        labels=OVERLAP_IDS,
        target_names=OVERLAP_EMOTIONS,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": overall_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crema_dir", type=str, required=True,
                        help="Path to CREMA-D AudioWAV/ directory")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Where to find trained models (e.g. ../results/wavlm-base/best_model.pt)")
    parser.add_argument("--output_dir", type=str, default="../results/cross_corpus")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Build CREMA-D manifest ----
    print(f"\nScanning CREMA-D at {args.crema_dir}...")
    df = build_crema_manifest(args.crema_dir)
    if len(df) == 0:
        print("ERROR: no CREMA-D files found")
        return

    # ---- Load in-domain RAVDESS test results for comparison ----
    in_domain_results = {}
    for m in ["wav2vec2-base", "hubert-base", "wavlm-base"]:
        in_path = Path(args.results_dir) / m / "results.json"
        if in_path.exists():
            with open(in_path) as f:
                r = json.load(f)
            in_domain_results[m] = r["test_metrics"]["accuracy"]

    # ---- Run all 3 models on CREMA-D ----
    label_names_full = [ID_TO_EMOTION[i] for i in range(len(ID_TO_EMOTION))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cross_corpus_results = {}
    for model_key in ["wav2vec2-base", "hubert-base", "wavlm-base"]:
        model_dir = Path(args.results_dir) / model_key
        if not (model_dir / "best_model.pt").exists():
            print(f"SKIP {model_key}: no checkpoint")
            continue

        preds, labels = evaluate_model(
            model_key, model_dir, df, device, batch_size=args.batch_size,
        )
        metrics = compute_metrics(labels, preds, label_names_full)

        in_domain_acc = in_domain_results.get(model_key, None)
        drop = (in_domain_acc - metrics["accuracy"]) if in_domain_acc else None

        cross_corpus_results[model_key] = {
            "in_domain_accuracy_ravdess": in_domain_acc,
            "out_of_domain_accuracy_crema_d": metrics["accuracy"],
            "absolute_drop": drop,
            "f1_macro_crema_d": metrics["f1_macro"],
            "f1_weighted_crema_d": metrics["f1_weighted"],
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class_recall": {
                e: metrics["classification_report"].get(e, {}).get("recall", 0.0)
                for e in OVERLAP_EMOTIONS
            },
            "n_samples": len(labels),
        }

        print(f"\n=== {model_key} on CREMA-D ===")
        print(f"  RAVDESS test acc:  {in_domain_acc:.4f}" if in_domain_acc else "  RAVDESS test acc: N/A")
        print(f"  CREMA-D acc:       {metrics['accuracy']:.4f}")
        if drop is not None:
            print(f"  Absolute drop:     {drop:.4f}")
        print(f"  CREMA-D macro F1:  {metrics['f1_macro']:.4f}")

    # ---- Save ----
    out_path = output_dir / "cross_corpus_results.json"
    with open(out_path, "w") as f:
        json.dump(cross_corpus_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("CROSS-CORPUS SUMMARY (RAVDESS-trained, CREMA-D zero-shot)")
    print("=" * 70)
    print(f"{'Model':<16} {'RAVDESS':>10} {'CREMA-D':>10} {'Drop':>10}")
    print("-" * 50)
    for m, r in cross_corpus_results.items():
        ravdess = r["in_domain_accuracy_ravdess"]
        crema = r["out_of_domain_accuracy_crema_d"]
        drop = r["absolute_drop"]
        print(f"{m:<16} {ravdess:>10.4f} {crema:>10.4f} {drop:>10.4f}")


if __name__ == "__main__":
    main()

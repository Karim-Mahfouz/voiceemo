"""
Classical baseline: MFCC features + Logistic Regression.

This anchors the absolute numbers — without it, the pre-trained model
accuracies float in space. With it, we can report:
    "Self-supervised pre-training adds X absolute accuracy points over
     hand-crafted MFCC features."

Runs on CPU in a few minutes; no GPU required.

Usage:
    python mfcc_baseline.py --data_dir ../data/RAVDESS --output_dir ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from data_loader import (
    build_ravdess_manifest,
    speaker_independent_split,
    EMOTION_TO_ID, ID_TO_EMOTION,
)


def extract_mfcc_features(filepath, n_mfcc=40, target_sr=16000, max_duration=5.0):
    """Extract a fixed-size MFCC feature vector from one audio file.
    
    Returns a vector of shape (n_mfcc * 4,) — mean, std, min, max of each MFCC
    coefficient over time. This is a standard "statistical pooling" baseline.
    """
    waveform, sr = librosa.load(filepath, sr=target_sr, duration=max_duration)
    # Pad if shorter than max_duration
    target_len = int(target_sr * max_duration)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    
    mfcc = librosa.feature.mfcc(y=waveform, sr=target_sr, n_mfcc=n_mfcc)
    # Statistical pooling over time
    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc.min(axis=1),
        mfcc.max(axis=1),
    ])
    return features


def featurize_split(df, desc):
    """Extract MFCC features for every file in a split."""
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            feat = extract_mfcc_features(row["filepath"])
            features.append(feat)
            labels.append(row["emotion_id"])
        except Exception as e:
            print(f"Skipping {row['filepath']}: {e}")
    return np.array(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # ---- Load and split ----
    print("Building manifest...")
    df = build_ravdess_manifest(args.data_dir)
    train_df, val_df, test_df = speaker_independent_split(df)
    
    # ---- Feature extraction ----
    print("\nExtracting MFCC features (this takes ~3 minutes on CPU)...")
    X_train, y_train = featurize_split(train_df, "train")
    X_val, y_val = featurize_split(val_df, "val")
    X_test, y_test = featurize_split(test_df, "test")
    
    print(f"\nFeature shape: {X_train.shape}")
    
    # ---- Standardize ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ---- Train classifier ----
    print("\nTraining Logistic Regression...")
    # Stack train + val to train final model (small dataset)
    X_train_full = np.vstack([X_train_scaled, X_val_scaled])
    y_train_full = np.concatenate([y_train, y_val])
    
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=args.seed,
        solver="lbfgs",
    )
    clf.fit(X_train_full, y_train_full)
    
    # ---- Evaluate ----
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(ID_TO_EMOTION))))
    
    label_names = [ID_TO_EMOTION[i] for i in range(len(ID_TO_EMOTION))]
    report = classification_report(y_test, y_pred, target_names=label_names,
                                    output_dict=True, zero_division=0)
    
    print(f"\n=== MFCC + LogReg baseline results ===")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  F1 (macro):    {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    
    # ---- Save ----
    output_dir = Path(args.output_dir) / "mfcc-baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": "mfcc-baseline",
        "pretrained_name": "librosa-mfcc-40",
        "config": {
            "model": "mfcc-baseline",
            "n_mfcc": 40,
            "classifier": "LogisticRegression(C=1.0, lbfgs)",
            "feature_pooling": "mean+std+min+max",
            "seed": args.seed,
        },
        "history": {  # No epochs for LogReg, but keep schema-compatible
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1_macro": [],
        },
        "test_metrics": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "loss": 0.0,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "label_names": label_names,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()

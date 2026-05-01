"""
Layer-wise probing of a pre-trained speech encoder.

For each transformer layer in the encoder, extract features and train a
linear probe to predict emotion. This shows which layers carry the most
emotion-relevant information.

The key analysis: for each layer, compute accuracy on TWO subsets of
emotion pairs:
    - Arousal-distinct pairs (e.g., angry vs sad — far on arousal axis)
    - Valence-distinct pairs (e.g., neutral vs calm — far on valence axis)

If arousal accuracy saturates earlier than valence accuracy, that's
direct evidence that the model encodes arousal more robustly than
valence — the central claim of the report.

Usage:
    python layer_probe.py --data_dir ../data/RAVDESS --model wavlm-base \\
                          --output_dir ../results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoFeatureExtractor

from data_loader import (
    build_ravdess_manifest,
    speaker_independent_split,
    EMOTION_TO_ID, ID_TO_EMOTION,
)
from model import RECOMMENDED_MODELS


# Russell's circumplex — coarse arousal/valence assignments for RAVDESS emotions
# (1 = high, 0 = low) — used for the dimensional analysis
EMOTION_DIMENSIONS = {
    "angry":     {"arousal": 1, "valence": 0},
    "calm":      {"arousal": 0, "valence": 1},
    "disgust":   {"arousal": 0, "valence": 0},
    "fearful":   {"arousal": 1, "valence": 0},
    "happy":     {"arousal": 1, "valence": 1},
    "neutral":   {"arousal": 0, "valence": 1},  # treat as mildly positive baseline
    "sad":       {"arousal": 0, "valence": 0},
    "surprised": {"arousal": 1, "valence": 1},
}


def extract_layerwise_features(model, processor, df, device, max_duration=5.0,
                                target_sr=16000):
    """Extract per-layer features for every file in df.
    
    Returns a dict {layer_idx: np.ndarray of shape (N, hidden_dim)} and
    a corresponding labels array.
    """
    target_len = int(target_sr * max_duration)
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    
    layer_features = {i: [] for i in range(n_layers)}
    labels = []
    
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting"):
        waveform, sr = torchaudio.load(row["filepath"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] > target_len:
            waveform = waveform[:target_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
        
        inputs = processor(waveform.numpy(), sampling_rate=target_sr,
                           return_tensors="pt")
        input_values = inputs["input_values"].to(device)
        
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
        
        # outputs.hidden_states is tuple of (n_layers+1) tensors of
        # shape (1, seq_len, hidden_dim). Index 0 is embeddings, 1..N are
        # transformer layers.
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            layer_features[layer_idx].append(pooled)
        
        labels.append(row["emotion_id"])
    
    layer_features = {k: np.array(v) for k, v in layer_features.items()}
    return layer_features, np.array(labels)


def evaluate_pair_subset(y_true, y_pred, label_names, dimension, distinct_value):
    """Compute accuracy on samples where the true label has dimension=distinct_value.
    
    For arousal analysis: accuracy on samples whose emotion has the chosen
    arousal value (e.g., evaluate how well we identify HIGH-arousal emotions).
    """
    mask = np.array([
        EMOTION_DIMENSIONS[label_names[y]][dimension] == distinct_value
        for y in y_true
    ])
    if mask.sum() == 0:
        return None
    return accuracy_score(y_true[mask], y_pred[mask])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="wavlm-base",
                        choices=list(RECOMMENDED_MODELS.keys()))
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---- Load model (frozen, for feature extraction) ----
    pretrained_name = RECOMMENDED_MODELS[args.model]
    print(f"Loading {pretrained_name}...")
    processor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    model = AutoModel.from_pretrained(pretrained_name).to(device)
    model.eval()
    
    n_layers = model.config.num_hidden_layers + 1
    print(f"Model has {n_layers} layers (embedding + {n_layers-1} transformer)")
    
    # ---- Data ----
    print("\nBuilding manifest...")
    df = build_ravdess_manifest(args.data_dir)
    train_df, val_df, test_df = speaker_independent_split(df)
    
    # Combine train + val for the probe (no hyperparameter tuning)
    train_full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    
    # ---- Feature extraction ----
    print("\nExtracting train features per layer...")
    train_features, y_train = extract_layerwise_features(
        model, processor, train_full_df, device,
    )
    print("\nExtracting test features per layer...")
    test_features, y_test = extract_layerwise_features(
        model, processor, test_df, device,
    )
    
    # ---- Probe each layer ----
    label_names = [ID_TO_EMOTION[i] for i in range(len(ID_TO_EMOTION))]
    
    layer_results = []
    print(f"\nTraining linear probes on {n_layers} layers...")
    for layer_idx in range(n_layers):
        X_train = train_features[layer_idx]
        X_test = test_features[layer_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed,
                                  solver="lbfgs")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        
        overall_acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        
        # Dimensional analysis
        high_arousal_acc = evaluate_pair_subset(
            y_test, y_pred, label_names, "arousal", 1
        )
        low_arousal_acc = evaluate_pair_subset(
            y_test, y_pred, label_names, "arousal", 0
        )
        positive_valence_acc = evaluate_pair_subset(
            y_test, y_pred, label_names, "valence", 1
        )
        negative_valence_acc = evaluate_pair_subset(
            y_test, y_pred, label_names, "valence", 0
        )
        
        layer_results.append({
            "layer": layer_idx,
            "accuracy": float(overall_acc),
            "f1_macro": float(macro_f1),
            "high_arousal_acc": float(high_arousal_acc),
            "low_arousal_acc": float(low_arousal_acc),
            "positive_valence_acc": float(positive_valence_acc),
            "negative_valence_acc": float(negative_valence_acc),
        })
        
        print(f"  Layer {layer_idx:2d}: acc={overall_acc:.4f}  "
              f"hi-arousal={high_arousal_acc:.3f}  "
              f"lo-arousal={low_arousal_acc:.3f}  "
              f"pos-valence={positive_valence_acc:.3f}  "
              f"neg-valence={negative_valence_acc:.3f}")
    
    # ---- Save ----
    output_dir = Path(args.output_dir) / f"probing-{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": args.model,
        "pretrained_name": pretrained_name,
        "n_layers": n_layers,
        "label_names": label_names,
        "emotion_dimensions": EMOTION_DIMENSIONS,
        "layer_results": layer_results,
    }
    
    with open(output_dir / "probing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/probing_results.json")
    
    # ---- Plot ----
    try:
        import matplotlib.pyplot as plt
        
        layers = [r["layer"] for r in layer_results]
        overall = [r["accuracy"] for r in layer_results]
        hi_arousal = [r["high_arousal_acc"] for r in layer_results]
        lo_arousal = [r["low_arousal_acc"] for r in layer_results]
        pos_val = [r["positive_valence_acc"] for r in layer_results]
        neg_val = [r["negative_valence_acc"] for r in layer_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(layers, overall, "o-", color="#21295C", linewidth=2.5,
                     label="Overall accuracy", markersize=8)
        axes[0].set_xlabel("Layer index (0 = input embedding)")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"{args.model}: Layer-wise probing accuracy")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        axes[1].plot(layers, hi_arousal, "o-", color="#C62828", linewidth=2,
                     label="High-arousal emotions", markersize=7)
        axes[1].plot(layers, lo_arousal, "s-", color="#1565C0", linewidth=2,
                     label="Low-arousal emotions", markersize=7)
        axes[1].plot(layers, pos_val, "^--", color="#2E7D32", linewidth=2,
                     label="Positive-valence emotions", markersize=7, alpha=0.8)
        axes[1].plot(layers, neg_val, "v--", color="#F57C00", linewidth=2,
                     label="Negative-valence emotions", markersize=7, alpha=0.8)
        axes[1].set_xlabel("Layer index")
        axes[1].set_ylabel("Recall on subset")
        axes[1].set_title("Where does arousal vs. valence emerge?")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        figures_dir = Path(args.output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / f"layer_probing_{args.model}.png",
                    dpi=150, bbox_inches="tight")
        print(f"Plot saved to {figures_dir}/layer_probing_{args.model}.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()

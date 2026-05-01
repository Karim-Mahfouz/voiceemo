"""
Evaluation and visualization for VoiceEmo.

After training, run this to produce:
  - Confusion matrix plots
  - Per-class accuracy bar charts
  - Model comparison tables
  - Error analysis
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, labels, title, output_path, normalize=True):
    """Plot a confusion matrix."""
    cm = np.array(cm)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_normalized = cm.astype(float) / row_sums
        fmt = ".2f"
        data = cm_normalized
    else:
        fmt = "d"
        data = cm
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Emotion", fontsize=12)
    ax.set_ylabel("True Emotion", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_history(history, model_name, output_path):
    """Plot training and validation curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name}: Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name}: Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_accuracy(cm, labels, title, output_path):
    """Plot per-class accuracy as a bar chart."""
    cm = np.array(cm)
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.bar(labels, per_class, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Recall (per-class accuracy)")
    ax.set_xlabel("Emotion")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    
    for bar, value in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{value:.2f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def compare_models(results_list, output_dir):
    """Compare multiple models' results."""
    rows = []
    for results in results_list:
        rows.append({
            "Model": results["model"],
            "Accuracy": results["test_metrics"]["accuracy"],
            "F1 (macro)": results["test_metrics"]["f1_macro"],
            "F1 (weighted)": results["test_metrics"]["f1_weighted"],
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nModel comparison:\n{df.to_string(index=False)}")
    print(f"Saved: {csv_path}")
    
    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["Accuracy"], width, label="Accuracy", color="#3498db")
    ax.bar(x, df["F1 (macro)"], width, label="F1 (macro)", color="#e67e22")
    ax.bar(x + width, df["F1 (weighted)"], width, label="F1 (weighted)", color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on RAVDESS Test Set", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    
    for i, row in df.iterrows():
        ax.text(i - width, row["Accuracy"] + 0.01, f"{row['Accuracy']:.3f}",
                ha="center", fontsize=9)
        ax.text(i, row["F1 (macro)"] + 0.01, f"{row['F1 (macro)']:.3f}",
                ha="center", fontsize=9)
        ax.text(i + width, row["F1 (weighted)"] + 0.01, f"{row['F1 (weighted)']:.3f}",
                ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/model_comparison.png")
    
    return df


def error_analysis(results, output_path):
    """Identify the most-confused emotion pairs."""
    cm = np.array(results["test_metrics"]["confusion_matrix"])
    labels = results["label_names"]
    
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                row_total = cm[i].sum() or 1
                confusions.append({
                    "true": labels[i],
                    "predicted": labels[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / row_total),
                })
    
    confusions.sort(key=lambda x: x["count"], reverse=True)
    
    with open(output_path, "w") as f:
        f.write(f"Error Analysis: {results['model']}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Top 10 most-confused pairs (true -> predicted):\n\n")
        for c in confusions[:10]:
            f.write(f"  {c['true']:>10} -> {c['predicted']:<10}  "
                    f"count={c['count']:>3}  rate={c['rate']:.2%}\n")
    
    print(f"Saved: {output_path}")
    return confusions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing per-model subdirectories with results.json")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all results.json files
    all_results = []
    for json_path in sorted(results_dir.rglob("results.json")):
        with open(json_path) as f:
            results = json.load(f)
        all_results.append(results)
        
        model_name = results["model"]
        labels = results["label_names"]
        cm = results["test_metrics"]["confusion_matrix"]
        
        # Confusion matrix
        plot_confusion_matrix(
            cm, labels,
            f"Confusion Matrix: {model_name}",
            output_dir / f"confusion_{model_name}.png",
            normalize=True,
        )
        
        # Training history
        plot_training_history(
            results["history"], model_name,
            output_dir / f"history_{model_name}.png",
        )
        
        # Per-class accuracy
        plot_per_class_accuracy(
            cm, labels,
            f"Per-Class Accuracy: {model_name}",
            output_dir / f"per_class_{model_name}.png",
        )
        
        # Error analysis
        error_analysis(
            results, output_dir / f"errors_{model_name}.txt"
        )
    
    if len(all_results) > 1:
        compare_models(all_results, output_dir)
    
    print(f"\nDone. {len(all_results)} model(s) analyzed.")


if __name__ == "__main__":
    main()

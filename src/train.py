"""
Training loop for VoiceEmo SER model.

Usage:
    python train.py --data_dir /path/to/RAVDESS --model wav2vec2-base --epochs 10
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data_loader import (
    build_ravdess_manifest,
    speaker_independent_split,
    RAVDESSDataset,
    collate_fn,
    EMOTION_TO_ID,
    ID_TO_EMOTION,
)
from model import SERClassifier, RECOMMENDED_MODELS


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="train")
    for batch in pbar:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_values)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="eval"):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        logits = model(input_values)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the RAVDESS dataset root")
    parser.add_argument("--model", type=str, default="wav2vec2-base",
                        choices=list(RECOMMENDED_MODELS.keys()),
                        help="Which pre-trained model to use")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the encoder; only train the classification head")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_duration", type=float, default=5.0,
                        help="Max audio duration in seconds")
    args = parser.parse_args()
    
    set_seed(args.seed)
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---- Data ----
    print("\nBuilding manifest...")
    df = build_ravdess_manifest(args.data_dir)
    train_df, val_df, test_df = speaker_independent_split(df)
    
    # ---- Model ----
    pretrained_name = RECOMMENDED_MODELS[args.model]
    print(f"\nLoading model: {pretrained_name}")
    from transformers import AutoFeatureExtractor
    processor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    model = SERClassifier(
        pretrained_name=pretrained_name,
        num_labels=len(EMOTION_TO_ID),
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    
    # Freeze the convolutional feature encoder for stable fine-tuning.
    # This is standard practice (see Baevski et al. 2020, fairseq examples)
    # and prevents the feature encoder from collapsing during fine-tuning.
    # Particularly important for wav2vec2 which is more unstable than HuBERT/WavLM.
    if hasattr(model.encoder, "feature_extractor"):
        for param in model.encoder.feature_extractor.parameters():
            param.requires_grad = False
        print("Froze convolutional feature_extractor (standard SER fine-tuning practice)")
    
    trainable, total = model.get_num_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # ---- Datasets ----
    train_ds = RAVDESSDataset(train_df, processor, max_duration_sec=args.max_duration)
    val_ds = RAVDESSDataset(val_df, processor, max_duration_sec=args.max_duration)
    test_ds = RAVDESSDataset(test_df, processor, max_duration_sec=args.max_duration)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    
    # ---- Optimizer ----
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    # ---- Training ----
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1_macro": []}
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        
        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"Val:   loss={val_metrics['loss']:.4f} "
              f"acc={val_metrics['accuracy']:.4f} "
              f"f1_macro={val_metrics['f1_macro']:.4f}")
        
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"Saved best model (val_acc={best_val_acc:.4f})")
    
    # ---- Test ----
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest results for {args.model}:")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    
    # Per-class report
    label_names = [ID_TO_EMOTION[i] for i in range(len(ID_TO_EMOTION))]
    report = classification_report(
        test_metrics["labels"], test_metrics["predictions"],
        target_names=label_names, output_dict=True,
    )
    
    # Save everything
    results = {
        "model": args.model,
        "pretrained_name": pretrained_name,
        "config": vars(args),
        "history": history,
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "f1_macro": test_metrics["f1_macro"],
            "f1_weighted": test_metrics["f1_weighted"],
            "loss": test_metrics["loss"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": report,
        },
        "label_names": label_names,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()

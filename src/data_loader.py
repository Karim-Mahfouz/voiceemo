"""
RAVDESS Dataset Loader for Speech Emotion Recognition

The RAVDESS dataset filename encoding:
    Filename example: 03-01-06-01-02-01-12.wav
    
    Position 1: Modality (01=full-AV, 02=video-only, 03=audio-only)
    Position 2: Vocal channel (01=speech, 02=song)
    Position 3: Emotion (01=neutral, 02=calm, 03=happy, 04=sad,
                         05=angry, 06=fearful, 07=disgust, 08=surprised)
    Position 4: Emotional intensity (01=normal, 02=strong)
    Position 5: Statement (01="Kids are talking by the door",
                           02="Dogs are sitting by the door")
    Position 6: Repetition (01=1st, 02=2nd)
    Position 7: Actor (01-24, odd=male, even=female)
"""

import os
import glob
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# Emotion label mapping for RAVDESS
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Numeric labels for training
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(sorted(set(EMOTION_MAP.values())))}
ID_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_ID.items()}


def parse_ravdess_filename(filename):
    """Parse a RAVDESS filename and return metadata."""
    base = os.path.basename(filename).replace(".wav", "")
    parts = base.split("-")
    if len(parts) != 7:
        return None
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "emotion": EMOTION_MAP.get(parts[2], "unknown"),
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": int(parts[6]),
        "gender": "male" if int(parts[6]) % 2 == 1 else "female",
    }


def build_ravdess_manifest(data_dir, output_csv=None):
    """Scan a RAVDESS directory and build a manifest CSV.
    
    Args:
        data_dir: Path to RAVDESS root containing Actor_01, Actor_02, etc.
        output_csv: If provided, save manifest to this path.
    
    Returns:
        pandas DataFrame with columns: filepath, emotion, emotion_id, actor, gender, ...
    """
    pattern = os.path.join(data_dir, "**", "*.wav")
    files = glob.glob(pattern, recursive=True)
    
    rows = []
    for filepath in sorted(files):
        meta = parse_ravdess_filename(filepath)
        if meta is None:
            continue
        # Only audio-only speech (modality 03, vocal_channel 01)
        if meta["modality"] != "03" or meta["vocal_channel"] != "01":
            continue
        rows.append({
            "filepath": filepath,
            "emotion": meta["emotion"],
            "emotion_id": EMOTION_TO_ID[meta["emotion"]],
            "actor": meta["actor"],
            "gender": meta["gender"],
            "intensity": meta["intensity"],
            "statement": meta["statement"],
        })
    
    df = pd.DataFrame(rows)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Manifest saved to {output_csv}: {len(df)} files")
    
    return df


def speaker_independent_split(df, test_actors=None, val_actors=None):
    """Split RAVDESS by actor to ensure speaker independence.
    
    By default, uses actors 21-24 for test, 17-20 for val, 1-16 for train.
    This is a strong split since the model never sees test speakers during training.
    """
    if test_actors is None:
        test_actors = list(range(21, 25))  # 21, 22, 23, 24
    if val_actors is None:
        val_actors = list(range(17, 21))   # 17, 18, 19, 20
    
    train = df[~df["actor"].isin(test_actors + val_actors)].reset_index(drop=True)
    val = df[df["actor"].isin(val_actors)].reset_index(drop=True)
    test = df[df["actor"].isin(test_actors)].reset_index(drop=True)
    
    print(f"Train: {len(train)} samples, {train['actor'].nunique()} actors")
    print(f"Val:   {len(val)} samples, {val['actor'].nunique()} actors")
    print(f"Test:  {len(test)} samples, {test['actor'].nunique()} actors")
    
    return train, val, test


class RAVDESSDataset(Dataset):
    """PyTorch Dataset for RAVDESS audio files."""
    
    def __init__(self, manifest_df, processor, target_sr=16000, max_duration_sec=5.0):
        """
        Args:
            manifest_df: DataFrame with filepath and emotion_id columns
            processor: HuggingFace feature extractor (e.g., Wav2Vec2FeatureExtractor)
            target_sr: Target sample rate (Wav2Vec2/HuBERT use 16kHz)
            max_duration_sec: Truncate/pad audio to this duration
        """
        self.df = manifest_df.reset_index(drop=True)
        self.processor = processor
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["filepath"])
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Truncate or pad to fixed length
        waveform = waveform.squeeze(0)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            pad_size = self.max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        
        # Process with the model's feature extractor
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )
        
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "label": torch.tensor(row["emotion_id"], dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {"input_values": input_values, "labels": labels}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        df = build_ravdess_manifest(data_dir, output_csv="manifest.csv")
        print("\nClass distribution:")
        print(df["emotion"].value_counts())
        print("\nSplit:")
        train, val, test = speaker_independent_split(df)
    else:
        print("Usage: python data_loader.py <RAVDESS_directory>")

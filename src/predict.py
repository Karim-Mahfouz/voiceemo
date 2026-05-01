"""
Inference script — predict emotion from a single audio file.

Usage:
    python predict.py --audio path/to/audio.wav --model_dir results/wav2vec2-base
"""

import argparse
import torch
import torchaudio
from pathlib import Path

from data_loader import ID_TO_EMOTION, EMOTION_TO_ID
from model import SERClassifier, RECOMMENDED_MODELS
from transformers import AutoFeatureExtractor


def predict(audio_path, model_dir, model_key="wav2vec2-base", device=None):
    """Predict the emotion of a single audio file."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pretrained_name = RECOMMENDED_MODELS[model_key]
    processor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    model = SERClassifier(
        pretrained_name=pretrained_name,
        num_labels=len(EMOTION_TO_ID),
    ).to(device)
    
    # Load trained weights
    weights_path = Path(model_dir) / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    
    # Pad/truncate to 5 seconds
    target_len = 16000 * 5
    waveform = waveform.squeeze(0)
    if waveform.shape[0] > target_len:
        waveform = waveform[:target_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
    
    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    
    with torch.no_grad():
        logits = model(input_values)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    
    pred_idx = probs.argmax().item()
    pred_emotion = ID_TO_EMOTION[pred_idx]
    
    print(f"\nFile: {audio_path}")
    print(f"Predicted emotion: {pred_emotion} ({probs[pred_idx]:.2%})")
    print(f"\nAll probabilities:")
    for idx, prob in enumerate(probs):
        bar = "#" * int(prob.item() * 40)
        print(f"  {ID_TO_EMOTION[idx]:>10}: {prob.item():.2%}  {bar}")
    
    return pred_emotion, probs.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_key", type=str, default="wav2vec2-base",
                        choices=list(RECOMMENDED_MODELS.keys()))
    args = parser.parse_args()
    predict(args.audio, args.model_dir, args.model_key)

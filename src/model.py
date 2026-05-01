"""
Speech Emotion Recognition Model

Adds a classification head on top of pre-trained Wav2Vec2 / HuBERT.
The architecture: pre-trained encoder -> mean-pool over time -> dense -> classifier.
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    AutoFeatureExtractor,
    AutoModel,
)


class SERClassifier(nn.Module):
    """Speech emotion recognition classifier built on a pre-trained encoder."""
    
    def __init__(
        self,
        pretrained_name="facebook/wav2vec2-base",
        num_labels=8,
        freeze_encoder=False,
        dropout=0.1,
        hidden_dim=256,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_labels = num_labels
        
        # Load the pre-trained encoder
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        
        # Optionally freeze the encoder (faster training, less data needed)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        encoder_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(encoder_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, input_values, attention_mask=None):
        # Encoder produces (batch, seq_len, hidden_dim)
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Mean pool over the time dimension to get (batch, hidden_dim)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / count
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Classification head
        x = self.dropout(pooled)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits
    
    def get_num_parameters(self):
        """Return (trainable, total) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total


def load_processor_and_model(pretrained_name, num_labels=8, freeze_encoder=False):
    """Convenience function to load a feature extractor + model pair."""
    processor = AutoFeatureExtractor.from_pretrained(pretrained_name)
    model = SERClassifier(
        pretrained_name=pretrained_name,
        num_labels=num_labels,
        freeze_encoder=freeze_encoder,
    )
    return processor, model


# Recommended models for SER comparison
RECOMMENDED_MODELS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "hubert-base": "facebook/hubert-base-ls960",
    "wavlm-base": "microsoft/wavlm-base",
}


if __name__ == "__main__":
    # Quick smoke test
    processor, model = load_processor_and_model(
        "facebook/wav2vec2-base",
        num_labels=8,
        freeze_encoder=True,
    )
    trainable, total = model.get_num_parameters()
    print(f"Model: facebook/wav2vec2-base")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable percentage: {100*trainable/total:.2f}%")
    
    # Dummy forward pass
    dummy_input = torch.randn(2, 16000)  # 2 samples of 1 second at 16kHz
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Output shape: {logits.shape}  (expected: [2, 8])")

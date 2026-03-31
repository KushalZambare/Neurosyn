import torch
import torch.nn as nn

class MetadataEncoder(nn.Module):
    """
    A simple Fully Connected neural network to encode patient metadata.
    """
    def __init__(self, metadata_dim=2, embed_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.fc(x)

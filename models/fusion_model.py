import torch
import torch.nn as nn
from models.vit_encoder import VisionTransformer
from models.metadata_encoder import MetadataEncoder

class MultiModalFusionModel(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, img_embed_dim=128, metadata_dim=2, metadata_embed_dim=64, n_classes=4):
        super().__init__()
        
        # Initialize image encoder 
        self.img_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=img_embed_dim
        )
        
        # Initialize metadata encoder 
        self.metadata_encoder = MetadataEncoder(
            metadata_dim=metadata_dim, embed_dim=metadata_embed_dim
        )
        
        # Final classifier head following feature concatenation
        self.classifier = nn.Sequential(
            nn.Linear(img_embed_dim + metadata_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, img, meta):
        
        
        # Extract features
        img_features = self.img_encoder(img)  
        meta_features = self.metadata_encoder(meta)
        
        # Concatenate features
        fused_features = torch.cat((img_features, meta_features), dim=1)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits


class MultiModalDiagnosisModel(MultiModalFusionModel):
    pass

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and projections them into an embedding space.
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Projection layer: treats patch as a convolution
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2) 
        x = x.transpose(1, 2)  
        return x

class VisionTransformer(nn.Module):
    """
    A custom Vision Transformer (ViT) encoder implemented from scratch using nn.TransformerEncoder.
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=1, embed_dim=128, depth=4, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Special class token for sequence representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        img_feature = x[:, 0]
        return img_feature

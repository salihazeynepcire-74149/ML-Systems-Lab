import torch
import torch.nn as nn

class SalihaViT(nn.Module):
    """
    Vision Transformer with Dedicated Learnable [CLS] Token
    Image: 28x28 -> Non-overlapping 7x7 patches (16 tokens)
    Token dimension: 64
    Prepend: 1 learnable [CLS] token -> sequence length = 17
    Position Embedding: Learnable (1, 17, 64)
    Encoder: 2 Transformer Blocks with 4 attention heads
    Classification: Extracted exclusively from [CLS] token representation
    """
    def __init__(self, in_channels=1, patch_size=7, embed_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2  # 16 patches
        patch_dim = in_channels * patch_size * patch_size  # 49
        
        self.patch_projection = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_bias = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # Unfold into 16 patches of 7x7
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, 1, 4, 4, 7, 7)
        patches = patches.contiguous().view(B, self.num_patches, -1)  # (B, 16, 49)
        
        # Project patches
        x_proj = self.patch_projection(patches)  # (B, 16, 64)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        tokens = torch.cat((cls_tokens, x_proj), dim=1)  # (B, 17, 64)
        tokens = tokens + self.position_bias
        
        encoded = self.transformer_encoder(tokens)
        encoded = self.norm(encoded)
        
        # Isolate [CLS] token at index 0
        cls_rep = encoded[:, 0]
        return self.classification_head(cls_rep)

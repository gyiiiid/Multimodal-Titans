import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        in_channels=3, 
        dim=768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, 
            dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)
        # (batch_size, dim, h', w')
        x = rearrange(x, 'b c h w -> b (h w) c')
        # (batch_size, num_patches, dim)
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        dim=768,
        depth=12,
        heads=12,
        dim_head=64,
        mlp_dim=3072,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=dim
        )
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches + 1, dim))
        
        # Transformer encoder
        self.transformer_encoder = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_encoder.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout)
            ]))
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, img):
        # img: (batch_size, channels, height, width)
        x = self.patch_embedding(img)
        b, n, _ = x.shape
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embedding[:, :(n + 1)]
        
        # Apply transformer encoder
        for norm1, attn, norm2, ff in self.transformer_encoder:
            x_norm = norm1(x)
            x_norm = rearrange(x_norm, 'b n d -> n b d')
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            attn_out = rearrange(attn_out, 'n b d -> b n d')
            x = x + attn_out
            
            x_norm = norm2(x)
            x = x + ff(x_norm)
        
        # Final normalization
        x = self.norm(x)
        
        # Return CLS token as image representation
        return x[:, 0]  # (batch_size, dim)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class ImageProcessor:
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def preprocess(self, image):
        """
        预处理图像，调整大小并归一化
        
        Args:
            image: PIL Image或Tensor
            
        Returns:
            Tensor: 形状为(3, image_size, image_size)的预处理图像
        """
        if not isinstance(image, torch.Tensor):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
            return transform(image)
        else:
            # 假设已经是Tensor，只需调整大小和归一化
            image = F.interpolate(image.unsqueeze(0), size=(self.image_size, self.image_size), 
                                 mode='bilinear', align_corners=False)[0]
            return (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                   torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) 
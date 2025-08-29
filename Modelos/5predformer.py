# Modelos/5predformer.py
# Implementação avançada, baseada na arquitetura de pesquisa com atenção alternada.

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# ==============================================================================
# BLOCOS DE CONSTRUÇÃO (HELPER MODULES)
# ==============================================================================

class PreNorm(nn.Module):
    """Normalização antes de aplicar a função principal (atenção ou feedforward)."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """Camada FeedForward padrão do Transformer."""
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

class Attention(nn.Module):
    """Mecanismo de Multi-Head Self-Attention."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    """Bloco Transformer com atenção e feedforward."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

# ==============================================================================
# ARQUITETURA PRINCIPAL DO PRED FORMER (VERSÃO AVANÇADA E CONFIGURÁVEL)
# ==============================================================================

class PredFormer(nn.Module):
    def __init__(self, past_frames, future_frames, 
                 image_size=(21, 29), patch_size=(7, 7), 
                 d_model=128, nhead=4, num_encoder_layers=4, dropout_rate=0.1, **kwargs):
        super().__init__()

        self.past_frames = past_frames
        self.future_frames = future_frames
        
        # Garante que os parâmetros de tupla/lista do JSON sejam convertidos corretamente
        if isinstance(patch_size, list): patch_size = tuple(patch_size)
        if isinstance(image_size, list): image_size = tuple(image_size)
        
        # Parâmetros fixos da arquitetura avançada
        self.num_channels = 1
        self.dim_head = 64 
        
        # Parâmetros derivados
        self.image_h, self.image_w = image_size
        self.patch_h, self.patch_w = patch_size
        
        padded_h = (self.image_h + self.patch_h - 1) // self.patch_h * self.patch_h
        padded_w = (self.image_w + self.patch_w - 1) // self.patch_w * self.patch_w
        self.num_patches = (padded_h // self.patch_h) * (padded_w // self.patch_w)
        patch_dim = self.num_channels * self.patch_h * self.patch_w

        # --- Camadas do Modelo ---
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w),
            nn.Linear(patch_dim, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.past_frames, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            temporal_transformer = TransformerBlock(d_model, nhead, self.dim_head, d_model * 2, dropout_rate)
            spatial_transformer = TransformerBlock(d_model, nhead, self.dim_head, d_model * 2, dropout_rate)
            self.transformer_layers.append(nn.ModuleList([temporal_transformer, spatial_transformer]))

        self.to_future_proj = nn.Linear(past_frames, future_frames)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim)
        )

    def forward(self, x):
        B, T_in, H, W = x.shape
        x = x.unsqueeze(2)

        pad_h = (self.patch_h - H % self.patch_h) % self.patch_h
        pad_w = (self.patch_w - W % self.patch_w) % self.patch_w
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
        padded_H, padded_W = x.shape[-2], x.shape[-1]

        # Encoder
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)

        for temporal_attn, spatial_attn in self.transformer_layers:
            x_temp = rearrange(x, 'b t n d -> (b n) t d')
            x_temp = temporal_attn(x_temp)
            x = rearrange(x_temp, '(b n) t d -> b t n d', n=self.num_patches)
            
            x_spat = rearrange(x, 'b t n d -> (b t) n d')
            x_spat = spatial_attn(x_spat)
            x = rearrange(x_spat, '(b t) n d -> b t n d', t=T_in)

        # Decoder
        x = rearrange(x, 'b t n d -> b d n t')
        x = self.to_future_proj(x)
        x = rearrange(x, 'b d n t -> b t n d')
        
        x = self.mlp_head(x)
        
        # Reconstrução
        output = rearrange(x, 'b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)', 
                           h=padded_H // self.patch_h, 
                           w=padded_W // self.patch_w, 
                           p1=self.patch_h, p2=self.patch_w, 
                           c=self.num_channels)

        output = output[:, :, :, :H, :W]
        output = output.squeeze(2)

        return output
# Modelos/5predformer.py
# Implementação da variante Fac-T-S (Factorized Temporal-Spatial) do PredFormer,
# baseada no paper "Video Prediction Transformers without Recurrence or Convolution".

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

class Attention(nn.Module):
    """Mecanismo de Multi-Head Self-Attention."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SwiGLU(nn.Module):
    """FeedForward com ativação SwiGLU, como descrito no paper."""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # A projeção inicial é para o dobro da dimensão oculta para o gate
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.act = nn.SiLU() # Ativação Swish
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = self.act(gate) * x # Elemento chave do SwiGLU
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GatedTransformerBlock(nn.Module):
    """Bloco Transformer com Atenção e SwiGLU (GTB do paper)."""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ffn = PreNorm(dim, SwiGLU(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

# ==============================================================================
# ARQUITETURA PRINCIPAL DO PRED FORMER (FAC-T-S)
# ==============================================================================

class PredFormer(nn.Module):
    def __init__(self, past_frames, future_frames,
                 image_size=(21, 29), patch_size=(7, 7),
                 d_model=128, nhead=4,
                 num_temporal_layers=4, num_spatial_layers=4,
                 dropout_rate=0.1, **kwargs):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames

        if isinstance(patch_size, list): patch_size = tuple(patch_size)
        if isinstance(image_size, list): image_size = tuple(image_size)

        self.num_channels = 1
        self.dim_head = 64 # Dimensão por cabeça de atenção (padrão)

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

        # Usando positional encoding absoluto, como recomendado no paper
        self.pos_embedding = nn.Parameter(torch.randn(1, self.past_frames, self.num_patches, d_model))

        # --- Blocos Transformer Fatorizados ---
        # 1. Transformer Temporal
        self.temporal_transformer = nn.Sequential(*[
            GatedTransformerBlock(d_model, nhead, self.dim_head, d_model * 2, dropout_rate)
            for _ in range(num_temporal_layers)
        ])

        # 2. Transformer Espacial
        self.spatial_transformer = nn.Sequential(*[
            GatedTransformerBlock(d_model, nhead, self.dim_head, d_model * 2, dropout_rate)
            for _ in range(num_spatial_layers)
        ])

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

        # 1. Embedding
        x = self.to_patch_embedding(x)
        x += self.pos_embedding

        # 2. Atenção Temporal (Fac-T)
        x = rearrange(x, 'b t n d -> (b n) t d') # Trata cada patch como um item de batch
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b n) t d -> b t n d', n=self.num_patches)

        # 3. Atenção Espacial (Fac-S)
        x = rearrange(x, 'b t n d -> (b t) n d') # Trata cada frame como um item de batch
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b t) n d -> b t n d', t=T_in)

        # 4. Projeção para o Futuro e Reconstrução
        x = rearrange(x, 'b t n d -> b d n t')
        x = self.to_future_proj(x)
        x = rearrange(x, 'b d n t -> b t n d')
        x = self.mlp_head(x)

        output = rearrange(x, 'b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)',
                           h=padded_H // self.patch_h, w=padded_W // self.patch_w,
                           p1=self.patch_h, p2=self.patch_w, c=self.num_channels)

        output = output[:, :, :, :H, :W].squeeze(2)
        return output
# Modelos/5predformer.py
# Implementação da variante Fac-T-S (Factorized Temporal-Spatial) do PredFormer,
# baseada no paper "Video Prediction Transformers without Recurrence or Convolution".
import math
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
    """Mecanismo de Multi-Head Self-Attention com dropout na atenção."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_dropout)
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
        attn = self.attn_drop(attn)
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
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0.1):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, attn_dropout=attn_dropout))
        self.ffn = PreNorm(dim, SwiGLU(dim, mlp_dim, dropout=dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x

# ==============================================================================
# POSICIONAIS SENOIDAIS (1D/2D) – fatorizados
# ==============================================================================
def sinusoidal_encoding_1d(length, dim, device):
    """Senoidal padrão Transformer (1D)."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # [L,1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, D]

def sinusoidal_encoding_2d(h, w, dim, device):
    """Posicional 2D fatorizado: metade do dim para eixo H e metade para W."""
    assert dim % 2 == 0, "dim deve ser par para PE 2D fatorizado"
    dim_h = dim // 2
    dim_w = dim - dim_h
    pe_h = sinusoidal_encoding_1d(h, dim_h, device)  # [H, dim_h]
    pe_w = sinusoidal_encoding_1d(w, dim_w, device)  # [W, dim_w]
    pe_h = pe_h.unsqueeze(1).expand(h, w, dim_h)     # [H, W, dim_h]
    pe_w = pe_w.unsqueeze(0).expand(h, w, dim_w)     # [H, W, dim_w]
    pe_2d = torch.cat([pe_h, pe_w], dim=-1)          # [H, W, dim]
    pe_2d = pe_2d.view(h * w, dim)                   # [N, dim]
    return pe_2d

# ==============================================================================
# ARQUITETURA PRINCIPAL DO PRED FORMER (FAC-T-S)
# ==============================================================================
class PredFormer(nn.Module):
    def __init__(self, past_frames, future_frames,
                 image_size=(21, 29), patch_size=(7, 7),
                 d_model=128, nhead=4,
                 num_temporal_layers=4, num_spatial_layers=4,
                 dropout_rate=0.1,
                 # Novos kwargs opt-in para retrocompatibilidade:
                 num_channels=1, dim_head=None, attn_dropout=0.1,
                 use_learned_positional=False, **kwargs):
        super().__init__()
        # Parâmetros-base
        self.past_frames = past_frames
        self.future_frames = future_frames
        if isinstance(patch_size, list): patch_size = tuple(patch_size)
        if isinstance(image_size, list): image_size = tuple(image_size)

        # Canais e heads
        self.num_channels = int(num_channels)
        # head_dim padrão alinhado a d_model/nhead, pode ser sobrescrito via JSON
        self.dim_head = int(dim_head) if dim_head is not None else max(1, d_model // nhead)

        # Tamanhos de imagem/patch e grid acolchoado
        self.image_h, self.image_w = image_size
        self.patch_h, self.patch_w = patch_size
        padded_h = (self.image_h + self.patch_h - 1) // self.patch_h * self.patch_h
        padded_w = (self.image_w + self.patch_w - 1) // self.patch_w * self.patch_w
        self.grid_h = padded_h // self.patch_h
        self.grid_w = padded_w // self.patch_w
        self.num_patches = self.grid_h * self.grid_w

        patch_dim = self.num_channels * self.patch_h * self.patch_w

        # --- Camadas do Modelo ---
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w),
            nn.Linear(patch_dim, d_model),
        )

        # Posicionais: por padrão senoidal no forward; learned apenas se ativado
        self.use_learned_positional = bool(use_learned_positional)
        if self.use_learned_positional:
            # learned PE preso a [past_frames, num_patches, d_model]
            self.pos_embedding = nn.Parameter(torch.randn(1, self.past_frames, self.num_patches, d_model))
        else:
            self.register_parameter('pos_embedding', None)

        # Blocos Transformer Fatorizados
        self.temporal_transformer = nn.Sequential(*[
            GatedTransformerBlock(d_model, nhead, self.dim_head, d_model * 2, dropout_rate, attn_dropout)
            for _ in range(num_temporal_layers)
        ])
        self.spatial_transformer = nn.Sequential(*[
            GatedTransformerBlock(d_model, nhead, self.dim_head, d_model * 2, dropout_rate, attn_dropout)
            for _ in range(num_spatial_layers)
        ])

        # Projeção temporal T_in -> T_out
        self.to_future_proj = nn.Linear(past_frames, future_frames)

        # Recuperação de patches
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim)
        )

    def forward(self, x):
        # x pode vir como [B, T, H, W] (C=1) ou [B, T, C, H, W]
        if x.dim() == 4:
            B, T_in, H, W = x.shape
            C = 1
            x = x.unsqueeze(2)  # -> [B, T, 1, H, W]
        else:
            B, T_in, C, H, W = x.shape
            assert C == self.num_channels, "num_channels incompatível com a entrada"

        # T precisa bater com a projeção temporal
        assert T_in == self.past_frames, "T_in deve igualar past_frames"

        # Padding espacial (para múltiplos do patch)
        pad_h = (self.patch_h - H % self.patch_h) % self.patch_h
        pad_w = (self.patch_w - W % self.patch_w) % self.patch_w
        x = F.pad(x, (0, pad_w, 0, pad_h))  # [B, T, C, H', W']

        padded_H, padded_W = x.shape[-2], x.shape[-1]
        gh, gw = padded_H // self.patch_h, padded_W // self.patch_w
        N = gh * gw

        # Se learned PE estiver ligado, checa compatibilidade do grid; senão, usará senoidal
        if self.use_learned_positional:
            assert N == self.num_patches, "image_size/patch_size do __init__ não batem com HxW de entrada quando use_learned_positional=True"

        # 1) Patch embedding -> [B, T, N, D]
        x = self.to_patch_embedding(x)  # D = d_model

        # 2) Posicionais
        if self.use_learned_positional:
            # learned PE exige N fixo; soma direta
            x = x + self.pos_embedding
        else:
            device = x.device
            D = x.size(-1)
            pe_t = sinusoidal_encoding_1d(T_in, D, device).view(1, T_in, 1, D)    # [1,T,1,D]
            pe_s = sinusoidal_encoding_2d(gh, gw, D, device).view(1, 1, N, D)     # [1,1,N,D]
            x = x + pe_t + pe_s

        # 3) Fac-T: atenção temporal por patch
        x = rearrange(x, 'b t n d -> (b n) t d')
        x = self.temporal_transformer(x)
        x = rearrange(x, '(b n) t d -> b t n d', n=N)

        # 4) Fac-S: atenção espacial por frame
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b t) n d -> b t n d', t=T_in)

        # 5) Projeção T->T' e reconstrução
        x = rearrange(x, 'b t n d -> b d n t')
        x = self.to_future_proj(x)              # [B, D, N, T_out]
        x = rearrange(x, 'b d n t -> b t n d')  # [B, T_out, N, D]
        x = self.mlp_head(x)                    # [B, T_out, N, patch_dim]

        output = rearrange(
            x, 'b t (h w) (p1 p2 c) -> b t c (h p1) (w p2)',
            h=gh, w=gw, p1=self.patch_h, p2=self.patch_w, c=self.num_channels
        )
        # recorta para tamanho original
        output = output[:, :, :, :H, :W]
        # mantém compatibilidade anterior (saída [B, T_out, H, W] se C=1)
        if self.num_channels == 1:
            output = output.squeeze(2)
        return output

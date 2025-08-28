# Modelos/2fcnn.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==============================================================================
# DEFINIÇÃO DA ARQUITETURA DO MODELO (ADAPTADO PARA 1D)
# ==============================================================================

class FCNN_1D(nn.Module):
    def __init__(self, past_frames, future_frames, kernel_size=3):
        super().__init__()
        
        # O padding 'same' garante que o comprimento da sequência não mude após a convolução
        p = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=kernel_size, padding=p)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=p)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=p)
        self.bn3 = nn.BatchNorm1d(64)
        
        # A última camada convolucional mapeia para um único canal de saída
        self.conv_final = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=kernel_size, padding=p)
        
        # Camada linear para mapear do comprimento da janela de entrada para o comprimento da previsão
        self.linear = nn.Linear(past_frames, future_frames)
        
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Entrada x tem shape [Batch, Timesteps, Features]
        # Conv1d espera [Batch, Features, Timesteps], então precisamos permutar
        x = x.permute(0, 2, 1)

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(self.activation(self.bn2(self.conv2(x))))
        x = self.activation(self.bn3(self.conv3(x)))
        
        x = self.conv_final(x)
        
        # Remove a dimensão de canal (features) que agora é 1
        x = x.squeeze(1)
        
        # Aplica a camada linear para obter o número correto de passos de previsão
        x = self.linear(x)
        
        return x


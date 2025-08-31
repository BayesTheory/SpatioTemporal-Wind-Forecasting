# Modelos/2fcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN5(nn.Module):
    """
    Versão Dinâmica da Rede Neural Totalmente Constróida (FCNN).
    A arquitetura da rede é construída dinamicamente com base em uma lista de
    tamanhos de camadas ocultas fornecida durante a inicialização.
    """
    def __init__(self, past_frames, future_frames, height=21, width=29, 
                 camadas_ocultas=[2048, 1024, 512, 1024], dropout_rate=0.3, **kwargs):
        """
        Construtor da FCNN dinâmica.

        Args:
            past_frames (int): Número de frames de entrada.
            future_frames (int): Número de frames de saída.
            height (int): Altura da imagem.
            width (int): Largura da imagem.
            camadas_ocultas (list[int]): Uma lista com o número de neurônios para cada camada oculta.
            dropout_rate (float): A taxa de dropout a ser aplicada.
            **kwargs: Permite receber outros argumentos do JSON sem causar erro.
        """
        super(FCNN5, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.height = height
        self.width = width
        
        input_dim = past_frames * height * width
        output_dim = future_frames * height * width
        
        # Cria a lista de camadas dinamicamente
        self.camadas = nn.ModuleList()
        dim_anterior = input_dim
        
        for dim_camada in camadas_ocultas:
            self.camadas.append(nn.Linear(dim_anterior, dim_camada))
            dim_anterior = dim_camada # A saída da camada atual é a entrada da próxima
        
        # Camada final de saída
        self.camada_saida = nn.Linear(dim_anterior, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Achata a entrada para o formato (batch, features)
        x = x.view(x.shape[0], -1)
        
        # Passa por todas as camadas ocultas dinâmicas
        for camada in self.camadas:
            x = self.dropout(F.relu(camada(x)))
            
        # Passa pela camada de saída
        x = self.camada_saida(x)
        
        # Remodela a saída de volta para o formato de vídeo (batch, frames, H, W)
        return x.view(x.shape[0], self.future_frames, self.height, self.width)
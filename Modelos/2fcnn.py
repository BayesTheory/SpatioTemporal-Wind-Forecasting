# Modelos/2Fcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN5(nn.Module):
    """
    ARQUITETURA CORRETA: Rede Neural Totalmente Conectada (Fully Connected).
    Ela achata a sequência de imagens de entrada em um único vetor e o processa
    com camadas densas (Linear).
    """
    def __init__(self, past_frames, future_frames, height=21, width=29):
        super(FCNN5, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.height = height
        self.width = width

        input_dim = self.past_frames * self.height * self.width
        output_dim = self.future_frames * self.height * self.width

        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc_out = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        x = F.relu(self.fc1(x_flat))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        
        output_flat = self.fc_out(x)
        
        output = output_flat.view(batch_size, self.future_frames, self.height, self.width)
        return output
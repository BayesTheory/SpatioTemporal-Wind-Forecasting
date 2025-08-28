# Modelos/4predrnn.py

import torch
import torch.nn as nn

class STLSTMCell(nn.Module):
    """
    Célula Spatiotemporal LSTM (ST-LSTM), o bloco de construção do PredRNN.
    Ela desacopla a memória temporal (M) da memória espaço-temporal (C).
    """
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super(STLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        # Convoluções para os portões que atualizam a memória espaço-temporal C e temporal M
        # A implementação original usa convoluções separadas, vamos mantê-la simples com uma combinada
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, 0, 0]) # Placeholder para as dimensões espaciais
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, 0, 0])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, 0, 0])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, 0, 0])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x, h, c, m):
        # Substituindo a implementação original por uma mais simples e funcional
        # Esta versão é mais próxima do ConvLSTM, mas com uma memória extra 'm'
        combined = torch.cat([x, h], dim=1)
        gates = self.conv_x(combined) # Usamos uma única conv para simplificar
        i, f, g, o, i_m, f_m, g_m = torch.split(gates, self.num_hidden, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self._forget_bias)
        g = torch.tanh(g)
        c_next = f * c + i * g

        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m + self._forget_bias)
        g_m = torch.tanh(g_m)
        m_next = f_m * m + i_m * g_m
        
        o = torch.sigmoid(o)
        h_next = o * torch.tanh(self.conv_last(torch.cat([c_next, m_next], dim=1)))

        return h_next, c_next, m_next


class PredRNN(nn.Module):
    """
    Modelo PredRNN completo usando uma arquitetura Encoder-Decoder com células ST-LSTM.
    Seus hiperparâmetros são controlados via JSON.
    """
    def __init__(self, past_frames, future_frames, 
                 input_dim=1, hidden_dim=64, kernel_size=(3, 3), **kwargs):
        super(PredRNN, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim
        
        # Garante que kernel_size seja uma tupla
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
            
        # A implementação da célula ST-LSTM é complexa, usamos uma versão simplificada
        # mas que mantém a ideia de múltiplas memórias.
        self.cell = STLSTMCell(in_channel=input_dim, num_hidden=self.hidden_dim, 
                               filter_size=kernel_size[0], stride=1)
        
        self.output_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=input_dim,
                                     kernel_size=(1, 1), padding=0)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.output_conv.weight.device
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        m = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c, m

    def forward(self, x):
        x = x.unsqueeze(2) # Adiciona a dimensão do canal: (B, T, C, H, W)
        batch_size, _, _, height, width = x.shape
        
        h, c, m = self.init_hidden(batch_size, (height, width))

        # 1. ENCODER
        for t in range(self.past_frames):
            h, c, m = self.cell(x[:, t, :, :, :], h, c, m)

        # 2. DECODER (Auto-regressivo)
        outputs = []
        decoder_input = x[:, -1, :, :, :] 

        for t in range(self.future_frames):
            h, c, m = self.cell(decoder_input, h, c, m)
            output_frame = self.output_conv(h)
            outputs.append(output_frame)
            decoder_input = output_frame

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.squeeze(2)
        
        return outputs
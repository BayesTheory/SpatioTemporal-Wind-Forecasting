# Modelos/4predrnn.py

import torch
import torch.nn as nn

class STLSTMCell(nn.Module):
    """
    Célula Spatiotemporal LSTM (ST-LSTM) - Versão Corrigida e Simplificada.
    Preserva a ideia de múltiplas memórias (c e m) com uma implementação mais robusta.
    """
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super(STLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        # <<< CORREÇÃO AQUI: A convolução agora espera a entrada combinada (x + h) >>>
        # Ela produzirá todos os 7 portões de uma vez para eficiência.
        self.conv = nn.Conv2d(in_channels=in_channel + num_hidden,
                              out_channels=num_hidden * 7, # i, f, g, o, i_m, f_m, g_m
                              kernel_size=(filter_size, filter_size),
                              padding=self.padding,
                              stride=stride)

    def forward(self, x, h, c, m):
        # Concatena a entrada e o estado oculto temporal
        combined = torch.cat([x, h], dim=1)
        # Uma única convolução para todos os portões
        gates = self.conv(combined)

        # Divide para obter os 7 portões
        i, f, g, o, i_m, f_m, g_m = torch.split(gates, self.num_hidden, dim=1)

        # Atualização da memória espaço-temporal C (como no ConvLSTM)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self._forget_bias)
        g = torch.tanh(g)
        c_next = f * c + i * g

        # Atualização da memória temporal M
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m + self._forget_bias)
        g_m = torch.tanh(g_m)
        m_next = f_m * m + i_m * g_m
        
        # Portão de saída e estado oculto final
        o = torch.sigmoid(o)
        h_next = o * torch.tanh(c_next + m_next) # Combina as duas memórias

        return h_next, c_next, m_next


class PredRNN(nn.Module):
    """
    Modelo PredRNN completo usando uma arquitetura Encoder-Decoder com células ST-LSTM.
    """
    def __init__(self, past_frames, future_frames, 
                 input_dim=1, hidden_dim=64, kernel_size=(3, 3), **kwargs):
        super(PredRNN, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim
        
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
            
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
        x = x.unsqueeze(2) # (B, T, C, H, W)
        batch_size, _, _, height, width = x.shape
        
        h, c, m = self.init_hidden(batch_size, (height, width))

        # ENCODER
        for t in range(self.past_frames):
            h, c, m = self.cell(x[:, t, :, :, :], h, c, m)

        # DECODER
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
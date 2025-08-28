# Modelos/4predrnn.py

import torch
import torch.nn as nn

class STLSTMCell(nn.Module):
    """
    Célula Spatiotemporal LSTM (ST-LSTM), o bloco de construção do PredRNN.
    Ela desacopla a memória temporal (M) da memória espaço-temporal (C).
    """
    def __init__(self, in_channel, num_hidden, filter_size, stride, layer_norm):
        super(STLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        # Convoluções para os portões que atualizam a memória espaço-temporal C
        self.conv_c = nn.Conv2d(in_channel + num_hidden, num_hidden * 4, filter_size, stride, self.padding)
        # Convoluções para os portões que atualizam a memória temporal M
        self.conv_m = nn.Conv2d(in_channel + num_hidden, num_hidden * 3, filter_size, stride, self.padding)

    def forward(self, x, h, c, m):
        # Concatena a entrada x e o estado oculto temporal h anterior
        combined = torch.cat((x, h), 1)
        
        # --- Atualização da Memória Espaço-Temporal C ---
        conv_c_out = self.conv_c(combined)
        i_c, f_c, g_c, o_c = torch.split(conv_c_out, self.num_hidden, dim=1)
        
        i_c = torch.sigmoid(i_c)
        f_c = torch.sigmoid(f_c + self._forget_bias)
        g_c = torch.tanh(g_c)
        
        c_next = f_c * c + i_c * g_c

        # --- Atualização da Memória Temporal M ---
        # A memória C recém-calculada (c_next) é usada na entrada
        combined_m = torch.cat((x, c_next), 1) 
        conv_m_out = self.conv_m(combined_m)
        i_m, f_m, g_m = torch.split(conv_m_out, self.num_hidden, dim=1)

        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m + self._forget_bias)
        g_m = torch.tanh(g_m)

        m_next = f_m * m + i_m * g_m

        # --- Estado Oculto Final H ---
        o_c = torch.sigmoid(o_c)
        h_next = o_c * torch.tanh(m_next)

        return h_next, c_next, m_next


class PredRNN(nn.Module):
    """
    Modelo PredRNN completo usando uma arquitetura Encoder-Decoder com células ST-LSTM.
    """
    def __init__(self, past_frames, future_frames, input_dim=1, hidden_dim=64, kernel_size=(3, 3), bias=True):
        super(PredRNN, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.hidden_dim = hidden_dim
        
        self.cell = STLSTMCell(in_channel=input_dim, num_hidden=self.hidden_dim, 
                               filter_size=kernel_size[0], stride=1, layer_norm=True)
        
        self.output_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=input_dim,
                                     kernel_size=(1, 1), padding=0)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.output_conv.weight.device
        # Inicializa todos os 3 estados: h (oculto), c (espaço-temporal), m (temporal)
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
        decoder_input = x[:, -1, :, :, :] # Usa o último frame de entrada como primeiro input

        for t in range(self.future_frames):
            h, c, m = self.cell(decoder_input, h, c, m)
            output_frame = self.output_conv(h)
            outputs.append(output_frame)
            decoder_input = output_frame # O próximo input é a previsão atual

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.squeeze(2) # Remove a dimensão do canal
        
        return outputs
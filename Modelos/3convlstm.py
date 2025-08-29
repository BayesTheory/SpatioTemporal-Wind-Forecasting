# Modelos/3convlstm.py

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    Célula base do ConvLSTM. Esta é a unidade de processamento fundamental que
    substitui as multiplicações de matrizes de uma LSTM padrão por convoluções.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    Modelo ConvLSTM completo, configurável via JSON e com regularização Dropout.
    """
    def __init__(self, past_frames, future_frames, 
                 input_dim=1, hidden_dim=64, kernel_size=(3, 3), 
                 dropout_rate=0.2, bias=True, **kwargs):
        """
        Construtor do modelo ConvLSTM Encoder-Decoder.

        Args:
            hidden_dim (int): Capacidade do modelo (nº de canais ocultos).
            kernel_size (tuple): Tamanho do filtro da convolução.
            dropout_rate (float): Taxa de dropout para regularização.
        """
        super(ConvLSTM, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)

        self.cell = ConvLSTMCell(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 bias=bias)
                                 
        self.output_conv = nn.Conv2d(in_channels=hidden_dim,
                                     out_channels=input_dim,
                                     kernel_size=(1, 1),
                                     padding=0)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(2)
        batch_size, _, _, height, width = x.shape
        
        h, c = self.cell.init_hidden(batch_size, (height, width))

        # Encoder
        for t in range(self.past_frames):
            h, c = self.cell(input_tensor=x[:, t, :, :, :], cur_state=[h, c])
            
        # Decoder (Auto-regressivo)
        outputs = []
        decoder_input = x[:, -1, :, :, :] 
        
        for t in range(self.future_frames):
            h, c = self.cell(input_tensor=decoder_input, cur_state=[h, c])
            
            # Aplica o Dropout no estado oculto antes da camada de saída
            h_dropout = self.dropout(h)
            
            output_frame = self.output_conv(h_dropout)
            outputs.append(output_frame)
            decoder_input = output_frame
            
        outputs = torch.stack(outputs, dim=1).squeeze(2)
        return outputs
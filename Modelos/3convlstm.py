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

        # A convolução que processa a entrada e o estado oculto anterior juntos.
        # Fazemos uma única convolução para todos os portões para eficiência.
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # 4 para os portões i, f, o, g
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatena a entrada atual com o estado oculto anterior no canal de dimensão
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Aplica a convolução
        combined_conv = self.conv(combined)
        
        # Divide o tensor resultante para cada um dos quatro portões
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Aplica as funções de ativação
        i = torch.sigmoid(cc_i)  # Portão de entrada (input gate)
        f = torch.sigmoid(cc_f)  # Portão de esquecimento (forget gate)
        o = torch.sigmoid(cc_o)  # Portão de saída (output gate)
        g = torch.tanh(cc_g)     # Portão de célula (cell gate)

        # Calcula o próximo estado da célula e o próximo estado oculto
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # Retorna um tensor de zeros para o estado oculto e para o estado da célula
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Modelo ConvLSTM completo usando uma arquitetura Encoder-Decoder.
    Este modelo é o primeiro da nossa hierarquia a entender a estrutura espacial
    dos dados, tratando cada passo de tempo como uma imagem e não como um vetor.
    """
    def __init__(self, past_frames, future_frames, input_dim=1, hidden_dim=64, kernel_size=(3, 3), bias=True):
        """
        Construtor do modelo ConvLSTM Encoder-Decoder.

        Args:
            past_frames (int): Número de frames na sequência de entrada (T_in).
            future_frames (int): Número de frames na sequência de saída (T_out).
            input_dim (int): Número de canais nos frames de entrada (1 para ws100).
            hidden_dim (int): Número de canais nos estados ocultos.
            kernel_size (tuple): Tamanho do kernel da convolução.
            bias (bool): Se a camada de convolução usa bias.
        """
        super(ConvLSTM, self).__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        
        # Célula ConvLSTM que será usada tanto no encoder quanto no decoder
        self.cell = ConvLSTMCell(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 bias=bias)
                                 
        # Camada de convolução final para mapear o estado oculto de volta para 1 canal de saída
        self.output_conv = nn.Conv2d(in_channels=hidden_dim,
                                     out_channels=input_dim,
                                     kernel_size=(1, 1),
                                     padding=0)

    def forward(self, x):
        """
        Define a passagem para a frente (forward pass).

        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch, past_frames, height, width).

        Returns:
            torch.Tensor: Tensor de saída com as previsões, com shape (batch, future_frames, height, width).
        """
        # Adiciona a dimensão do canal, esperado pelo Conv2d: (batch, past, 1, H, W)
        # O .unsqueeze(2) adiciona uma dimensão de tamanho 1 na posição 2.
        x = x.unsqueeze(2)
        
        batch_size, _, _, height, width = x.shape
        
        # 1. ENCODER
        # Inicializa os estados oculto (h) e de célula (c)
        h, c = self.cell.init_hidden(batch_size, (height, width))
        
        # Itera sobre a sequência de entrada para "codificar" a informação
        for t in range(self.past_frames):
            h, c = self.cell(input_tensor=x[:, t, :, :, :], cur_state=[h, c])
            
        # Neste ponto, 'h' e 'c' contêm a representação do "conhecimento" sobre a sequência passada.

        # 2. DECODER
        # Usa os estados finais do encoder como estados iniciais do decoder.
        # O decoder irá gerar 'future_frames' quadros.
        outputs = []
        
        # O primeiro input para o decoder será o último frame visto pelo encoder
        decoder_input = x[:, -1, :, :, :] 
        
        for t in range(self.future_frames):
            h, c = self.cell(input_tensor=decoder_input, cur_state=[h, c])
            
            # O estado oculto 'h' tem 'hidden_dim' canais.
            # Usamos uma conv 1x1 para mapeá-lo de volta para 1 canal de saída.
            output_frame = self.output_conv(h)
            outputs.append(output_frame)
            
            # Importante: o próximo input do decoder é o frame que ele acabou de gerar.
            # Isso torna o modelo auto-regressivo.
            decoder_input = output_frame
            
        # Concatena a lista de frames de saída em um único tensor
        outputs = torch.stack(outputs, dim=1)
        
        # Remove a dimensão do canal para retornar ao formato esperado (batch, future, H, W)
        outputs = outputs.squeeze(2)
        
        return outputs
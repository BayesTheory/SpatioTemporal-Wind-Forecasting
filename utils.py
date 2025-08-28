# utils.py

import mlflow
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xarray as xr

# ==============================================================================
# FUNÇÕES DE LOGGING (MLFLOW)
# ==============================================================================

def iniciar_experimento_mlflow(nome_experimento: str, nome_run: str):
    """Inicia um experimento e uma run no MLflow."""
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment(nome_experimento)
    mlflow.start_run(run_name=nome_run)
    print(f"MLflow: Experimento '{nome_experimento}' e Run '{nome_run}' iniciados.")

def logar_parametros_mlflow(params: dict):
    """Registra um dicionário de parâmetros na run ativa do MLflow."""
    mlflow.log_params(params)

def logar_metricas_mlflow(metrics: dict):
    """Registra um dicionário de métricas na run ativa do MLflow."""
    mlflow.log_metrics(metrics)

# ==============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO E JANELAMENTO
# ==============================================================================

def formatar_janelas_video(dados_grade: xr.DataArray, janela_entrada: int, horizonte_previsao: int):
    """
    Cria janelas deslizantes a partir de dados em grade (vídeo).

    Args:
        dados_grade (xr.DataArray): Dados com shape (time, height, width).
        janela_entrada (int): Número de frames de entrada (X).
        horizonte_previsao (int): Número de frames de saída (y).

    Returns:
        tuple[np.array, np.array]: Tupla contendo os arrays X e y.
            X: (n_amostras, janela_entrada, height, width)
            y: (n_amostras, horizonte_previsao, height, width)
    """
    X, y = [], []
    n_tempos = len(dados_grade['valid_time'])
    tamanho_total = janela_entrada + horizonte_previsao

    for i in range(n_tempos - tamanho_total + 1):
        # Define a fatia de entrada e a fatia de saída
        fatia_x = dados_grade[i : i + janela_entrada]
        fatia_y = dados_grade[i + janela_entrada : i + tamanho_total]
        
        X.append(fatia_x.values)
        y.append(fatia_y.values)
    
    # Empilha as listas de arrays em um único array numpy
    return np.stack(X), np.stack(y)


# ==============================================================================
# FUNÇÕES DE NORMALIZAÇÃO (SCALING)
# ==============================================================================

def criar_e_treinar_scaler_grade(dados_4d: np.array) -> MinMaxScaler:
    """
    Cria e treina um scaler MinMaxScaler em dados de 4D.
    O scaler é treinado com todos os valores de todos os pixels e tempos.
    
    Args:
        dados_4d (np.array): Dados com shape (amostras, tempo, altura, largura).
    
    Returns:
        MinMaxScaler: O objeto scaler treinado.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape para 2D (n_valores, 1) para que o scaler possa ser treinado
    dados_2d = dados_4d.reshape(-1, 1)
    scaler.fit(dados_2d)
    return scaler

def aplicar_scaler_grade(dados_4d: np.array, scaler: MinMaxScaler) -> np.array:
    """Aplica um scaler treinado a dados de 4D."""
    original_shape = dados_4d.shape
    dados_2d = dados_4d.reshape(-1, 1)
    dados_norm_2d = scaler.transform(dados_2d)
    # Retorna ao shape original
    return dados_norm_2d.reshape(original_shape)

def desnormalizar_dados_grade(dados_norm_4d: np.array, scaler: MinMaxScaler) -> np.array:
    """Aplica a transformação inversa de um scaler a dados de 4D."""
    original_shape = dados_norm_4d.shape
    dados_norm_2d = dados_norm_4d.reshape(-1, 1)
    dados_reais_2d = scaler.inverse_transform(dados_norm_2d)
    # Retorna ao shape original
    return dados_reais_2d.reshape(original_shape)


# ==============================================================================
# FUNÇÕES DE AVALIAÇÃO (MÉTRICAS)
# ==============================================================================

def calcular_metricas(y_real: np.array, y_previsto: np.array) -> dict:
    """
    Calcula métricas de regressão (RMSE e MAE).
    
    Args:
        y_real (np.array): Valores verdadeiros.
        y_previsto (np.array): Valores previstos pelo modelo.
        
    Returns:
        dict: Dicionário contendo os nomes e valores das métricas.
    """
    # Achata os arrays para calcular as métricas sobre todos os pontos
    y_real_flat = y_real.flatten()
    y_previsto_flat = y_previsto.flatten()
    
    rmse = np.sqrt(mean_squared_error(y_real_flat, y_previsto_flat))
    mae = mean_absolute_error(y_real_flat, y_previsto_flat)
    
    return {'rmse': rmse, 'mae': mae}
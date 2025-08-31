# utils.py (atualizado)
import time
from typing import Dict, Tuple

import mlflow
import numpy as np
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# FUNÇÕES DE UTILIDADE GERAL
# ==============================================================================
def print_timestamp(mensagem: str):
    """Imprime uma mensagem com o timestamp atual no formato do log."""
    print(f"{mensagem} (Timestamp: {time.ctime()})")

# ==============================================================================
# FUNÇÕES DE LOGGING (MLFLOW)
# ==============================================================================
def iniciar_experimento_mlflow(nome_experimento: str, nome_run: str):
    """Inicia/renova experimento e run no MLflow de forma segura."""
    try:
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_experiment(nome_experimento)
        mlflow.start_run(run_name=nome_run)
        print(f"MLflow: Experimento '{nome_experimento}' e Run '{nome_run}' iniciados.")
    except Exception as e:
        print(f"MLflow: Falha ao iniciar experimento/run. Detalhes: {e}")

def logar_parametros_mlflow(params: Dict):
    """Registra um dicionário de parâmetros na run ativa do MLflow."""
    try:
        if mlflow.active_run():
            mlflow.log_params(params)
    except Exception as e:
        print(f"MLflow: Falha ao logar parâmetros. {e}")

def logar_metricas_mlflow(metrics: Dict, step: int = None):
    """Registra um dicionário de métricas na run ativa do MLflow."""
    try:
        if mlflow.active_run():
            if step is None:
                mlflow.log_metrics(metrics)
            else:
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v), step=step)
    except Exception as e:
        print(f"MLflow: Falha ao logar métricas. {e}")

# ==============================================================================
# FUNÇÕES DE PRÉ-PROCESSAMENTO E JANELAMENTO
# ==============================================================================
def formatar_janelas_video(dados_grade: xr.DataArray,
                           janela_entrada: int,
                           horizonte_previsao: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria janelas deslizantes a partir de dados em grade (vídeo) usando indexação explícita por dimensão.
    Retorna:
      - X: (amostras, janela_entrada, lat, lon)
      - y: (amostras, horizonte_previsao, lat, lon)
    """
    X, y = [], []
    n_tempos = dados_grade.sizes['valid_time']
    tamanho_total = janela_entrada + horizonte_previsao

    for i in range(n_tempos - tamanho_total + 1):
        fatia_x = dados_grade.isel(valid_time=slice(i, i + janela_entrada))
        fatia_y = dados_grade.isel(valid_time=slice(i + janela_entrada, i + tamanho_total))
        X.append(fatia_x.values)
        y.append(fatia_y.values)

    return np.stack(X), np.stack(y)

# ==============================================================================
# FUNÇÕES DE NORMALIZAÇÃO (SCALING)
# ==============================================================================
def criar_e_treinar_scaler_grade(dados_4d: np.ndarray) -> MinMaxScaler:
    """
    Cria e treina um MinMaxScaler em dados 4D (amostras, tempo, lat, lon),
    achatando para (-1, 1) para garantir consistência de features.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_2d = dados_4d.reshape(-1, 1)
    scaler.fit(dados_2d)
    return scaler

def aplicar_scaler_grade(dados_4d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Aplica o scaler treinado a dados 4D preservando a forma original."""
    original_shape = dados_4d.shape
    dados_2d = dados_4d.reshape(-1, 1)
    dados_norm_2d = scaler.transform(dados_2d)
    return dados_norm_2d.reshape(original_shape)

def desnormalizar_dados_grade(dados_norm_4d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Aplica a transformação inversa do scaler a dados 4D normalizados."""
    original_shape = dados_norm_4d.shape
    dados_norm_2d = dados_norm_4d.reshape(-1, 1)
    dados_reais_2d = scaler.inverse_transform(dados_norm_2d)
    return dados_reais_2d.reshape(original_shape)

# ==============================================================================
# FUNÇÕES DE AVALIAÇÃO (MÉTRICAS)
# ==============================================================================
def calcular_metricas(y_real: np.ndarray, y_previsto: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas básicas de regressão (RMSE e MAE) achatando o espaço-tempo.
    """
    y_real_flat = y_real.ravel()
    y_previsto_flat = y_previsto.ravel()
    rmse = float(np.sqrt(mean_squared_error(y_real_flat, y_previsto_flat)))
    mae = float(mean_absolute_error(y_real_flat, y_previsto_flat))
    return {'rmse': rmse, 'mae': mae}

def calcular_metricas_completas(y_real: np.ndarray, y_previsto: np.ndarray) -> Dict[str, float]:
    """
    Métricas completas: RMSE, MAE, R², média do erro (viés) e desvio-padrão do erro, achatando o espaço-tempo.
    """
    y_real_flat = y_real.ravel()
    y_previsto_flat = y_previsto.ravel()
    resid = y_previsto_flat - y_real_flat
    rmse = float(np.sqrt(mean_squared_error(y_real_flat, y_previsto_flat)))
    mae = float(mean_absolute_error(y_real_flat, y_previsto_flat))
    r2 = float(r2_score(y_real_flat, y_previsto_flat))
    media_erro = float(np.mean(resid))
    dp_erro = float(np.std(resid))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'media_erro': media_erro, 'dp_erro': dp_erro}

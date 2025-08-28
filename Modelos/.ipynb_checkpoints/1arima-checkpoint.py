# Modelos/1arima.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def executar_modelo_cv_arima(dados_treino, passos_previsao, ordem):
    # (Função original para ponto único - mantida para referência, mas não usada no fluxo principal)
    serie_treino = dados_treino['velocidade_do_vento']
    try:
        model = ARIMA(serie_treino, order=ordem)
        model_fit = model.fit()
        previsoes = model_fit.forecast(steps=passos_previsao)
        return previsoes.values
    except Exception as e:
        print(f"  AVISO: Erro ao treinar ARIMA com ordem {ordem}. Erro: {e}")
        return np.zeros(passos_previsao)

def executar_arima_para_grade(dados_treino_grade: pd.DataFrame, passos_previsao: int, ordem: tuple) -> np.ndarray:
    """
    Executa um modelo ARIMA para cada ponto da grade espacial.
    """
    # <<< CORREÇÃO AQUI: Reordena os níveis do índice para (latitude, longitude, tempo) >>>
    # Isso permite que a seleção com .loc[(lat, lon)] funcione corretamente.
    dados_treino_grade = dados_treino_grade.reorder_levels(['latitude', 'longitude', 'valid_time'])
    
    latitudes = dados_treino_grade.index.get_level_values('latitude').unique()
    longitudes = dados_treino_grade.index.get_level_values('longitude').unique()
    
    altura = len(latitudes)
    largura = len(longitudes)
    total_pontos = altura * largura

    previsoes_grade = np.zeros((passos_previsao, altura, largura))

    print(f"Iniciando treino de {total_pontos} modelos ARIMA individuais. Isso pode ser demorado...")
    
    pbar = tqdm(total=total_pontos, desc="Progresso ARIMA na Grade")

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            try:
                # Agora esta linha vai funcionar corretamente
                serie_ponto = dados_treino_grade.loc[(lat, lon)]['ws100']
                
                modelo = ARIMA(serie_ponto, order=ordem, enforce_stationarity=False, enforce_invertibility=False)
                modelo_fit = modelo.fit()
                previsao_ponto = modelo_fit.forecast(steps=passos_previsao)
                
                previsoes_grade[:, i, j] = previsao_ponto
            except Exception:
                previsoes_grade[:, i, j] = 0
            
            pbar.update(1)

    pbar.close()
    print("Previsões da grade com ARIMA concluídas.")
    return previsoes_grade
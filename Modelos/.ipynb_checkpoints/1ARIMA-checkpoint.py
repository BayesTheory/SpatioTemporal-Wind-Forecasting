# Modelos/1ARIMA.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
from tqdm import tqdm # Importa o tqdm para a barra de progresso

# Ignorar avisos comuns do statsmodels
warnings.filterwarnings("ignore")

def executar_modelo_cv_arima(dados_treino, passos_previsao, ordem):
    # (Esta é a sua função original para ponto único, mantenha ela aqui)
    serie_treino = dados_treino['velocidade_do_vento']
    try:
        model = ARIMA(serie_treino, order=ordem)
        model_fit = model.fit()
        previsoes = model_fit.forecast(steps=passos_previsao)
        return previsoes.values
    except Exception as e:
        print(f"  AVISO: Erro ao treinar ARIMA com ordem {ordem}. Erro: {e}")
        return np.zeros(passos_previsao)


# <<< NOVA FUNÇÃO ABAIXO >>>
def executar_arima_para_grade(dados_treino_grade, passos_previsao, ordem):
    """
    Executa um modelo ARIMA para cada ponto da grade espacial.

    Args:
        dados_treino_grade (pd.DataFrame): DataFrame com MultiIndex (latitude, longitude, valid_time).
        passos_previsao (int): Número de passos de tempo para prever.
        ordem (tuple): A ordem (p, d, q) do modelo ARIMA.

    Returns:
        np.array: Um array com as previsões no formato (passos_previsao, altura, largura).
    """
    latitudes = dados_treino_grade.index.get_level_values('latitude').unique()
    longitudes = dados_treino_grade.index.get_level_values('longitude').unique()
    
    altura = len(latitudes)
    largura = len(longitudes)
    total_pontos = altura * largura

    # Array para armazenar os resultados
    previsoes_grade = np.zeros((passos_previsao, altura, largura))

    print(f"Iniciando treino de {total_pontos} modelos ARIMA individuais. Isso pode ser demorado...")
    
    # Usa tqdm para criar uma barra de progresso
    pbar = tqdm(total=total_pontos, desc="Progresso ARIMA na Grade")

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            # Extrai a série temporal para um único ponto (lat, lon)
            serie_ponto = dados_treino_grade.loc[(lat, lon)]['ws100']
            
            # Treina o modelo ARIMA e faz a previsão para este ponto
            try:
                modelo = ARIMA(serie_ponto, order=ordem, enforce_stationarity=False, enforce_invertibility=False)
                modelo_fit = modelo.fit()
                previsao_ponto = modelo_fit.forecast(steps=passos_previsao)
                
                # Armazena a previsão no array de resultados
                previsoes_grade[:, i, j] = previsao_ponto
            except Exception:
                # Se um ponto falhar, preenche com zeros e continua
                previsoes_grade[:, i, j] = 0
            
            pbar.update(1)

    pbar.close()
    print("Previsões da grade com ARIMA concluídas.")
    return previsoes_grade
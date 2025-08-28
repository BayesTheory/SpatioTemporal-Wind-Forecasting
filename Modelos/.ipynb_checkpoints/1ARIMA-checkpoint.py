# Modelos/1ARIMA.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignorar avisos comuns do statsmodels sobre convergência, etc.
warnings.filterwarnings("ignore")

def executar_modelo_cv_arima(dados_treino, passos_previsao, ordem):
    """
    Treina um modelo ARIMA e prevê os próximos passos.

    Args:
        dados_treino (pd.DataFrame): DataFrame contendo a série temporal de treino.
                                     Deve ter uma coluna 'velocidade_do_vento'.
        passos_previsao (int): O número de passos de tempo para prever no futuro.
        ordem (tuple): A ordem (p, d, q) do modelo ARIMA.

    Returns:
        np.array: Um array NumPy com as previsões.
    """
    
    # 1. EXTRAIR A SÉRIE TEMPORAL
    # O ARIMA trabalha com uma série univariada (apenas uma coluna de dados)
    serie_treino = dados_treino['velocidade_do_vento']

    # 2. TREINAR O MODELO ARIMA
    try:
        # Instancia o modelo com os dados de treino e a ordem (p,d,q)
        model = ARIMA(serie_treino, order=ordem)
        
        # Treina o modelo. 'disp=0' desativa os logs de convergência.
        model_fit = model.fit()

        # 3. FAZER AS PREVISÕES
        # Usa o método .forecast() para prever fora da amostra de treino
        previsoes = model_fit.forecast(steps=passos_previsao)

        # 4. RETORNAR AS PREVISÕES COMO UM ARRAY NUMPY
        return previsoes.values

    except Exception as e:
        print(f"  AVISO: Erro ao treinar ARIMA com ordem {ordem}. Erro: {e}")
        print("  Retornando um array de zeros como previsão.")
        # Se o modelo falhar (ex: dados não estacionários), retorna um array de zeros
        # para não quebrar o pipeline principal.
        return np.zeros(passos_previsao)
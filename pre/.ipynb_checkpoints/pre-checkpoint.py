# pre/pre.py

import xarray as xr
import numpy as np
import pandas as pd
from functools import lru_cache

# ==============================================================================
# FUNÇÃO PARA O BASELINE ARIMA (OTIMIZADA)
# ==============================================================================

def preparar_dados_ponto_unico(caminho_arquivo_nc: str, lat: float, lon: float) -> pd.DataFrame:
    """
    [PARA O BASELINE ARIMA]
    Carrega dados para um único ponto geográfico de forma otimizada e retorna um DataFrame.
    """
    print("--- Pré-processamento: Carregando dados de PONTO ÚNICO para o baseline ---")
    try:
        ds = xr.open_dataset(caminho_arquivo_nc, engine='netcdf4')
        
        # OTIMIZAÇÃO: Seleciona o ponto ANTES de fazer cálculos pesados.
        # Isso evita calcular a velocidade do vento para a grade inteira.
        ponto_selecionado = ds[['u100', 'v100']].sel(latitude=lat, longitude=lon, method='nearest')
        
        # Agora, calcula a velocidade do vento apenas para a série temporal do ponto.
        ws100_ponto = np.sqrt(ponto_selecionado['u100']**2 + ponto_selecionado['v100']**2)
        
        # Converte para DataFrame e formata a saída
        df = ws100_ponto.to_dataframe(name='velocidade_do_vento').reset_index()
        df = df.rename(columns={'valid_time': 'data_hora'})
        
        print("DataFrame de ponto único preparado com sucesso.")
        return df[['data_hora', 'velocidade_do_vento', 'latitude', 'longitude']]

    except Exception as e:
        print(f"ERRO ao preparar dados de ponto único: {e}")
        return None

# ==============================================================================
# FUNÇÃO PARA MODELOS DL (ROBUSTA E COM CACHE)
# ==============================================================================

# PERFORMANCE: O cache acelera a execução de múltiplos modelos, pois o arquivo
# não precisa ser lido e processado do disco toda vez.
@lru_cache(maxsize=2)
def preparar_dados_grade(caminho_arquivo_nc: str) -> xr.DataArray:
    """
    [PARA MODELOS DE DEEP LEARNING]
    Carrega, processa e armazena em cache a grade espaço-temporal completa.
    1. Abre o arquivo NetCDF.
    2. Calcula a velocidade do vento (ws100).
    3. Verifica e trata valores ausentes (NaN).
    4. Garante o tipo de dado float32.
    5. Retorna um xarray.DataArray com as dimensões (valid_time, latitude, longitude).
    """
    print("--- Pré-processamento: Carregando GRADE espaço-temporal para modelos DL ---")
    try:
        ds = xr.open_dataset(caminho_arquivo_nc, engine='netcdf4')
        ws100 = np.sqrt(ds['u100']**2 + ds['v100']**2)
        
        # ROBUSTEZ: Verifica e trata valores ausentes (NaN)
        if ws100.isnull().any():
            print("  Aviso: Valores NaN detectados. Preenchendo com interpolação linear.")
            # Interpola ao longo do tempo. fill_value="extrapolate" lida com NaNs nas bordas.
            ws100 = ws100.interpolate_na(dim='valid_time', method='linear', fill_value="extrapolate")

        ws100.attrs['units'] = 'm/s'
        ws100.attrs['long_name'] = 'Wind speed at 100m'
        
        # EFICIÊNCIA: Garante que os dados estejam em float32 para DL
        ws100 = ws100.astype('float32')
        
        print("Grade ws100 calculada e processada com sucesso.")
        return ws100

    except Exception as e:
        print(f"ERRO ao preparar dados da grade: {e}")
        return None
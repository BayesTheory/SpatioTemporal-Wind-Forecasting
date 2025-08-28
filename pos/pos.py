# pos/pos.py

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.metrics import r2_score

# ==============================================================================
# FUNÇÃO 1: VISUALIZAÇÃO DE UMA ÚNICA PREVISÃO (MAPAS)
# ==============================================================================

def visualizar_previsao_e_erro(
    y_real: np.ndarray, 
    y_predito: np.ndarray, 
    dados_grade_base: xr.DataArray,
    sample_idx: int = 0, 
    horizon_idx: int = 0,
    save_path: str = None
):
    """
    Gera uma visualização lado a lado de (Real, Previsto, Erro) para uma única
    amostra e horizonte de previsão.

    Args:
        y_real (np.ndarray): Array com todos os valores reais do conj. de teste.
                             Shape: (n_amostras, horizonte, altura, largura).
        y_predito (np.ndarray): Array com as previsões do modelo. Shape igual a y_real.
        dados_grade_base (xr.DataArray): DataArray original (saída de pre.py) para
                                         obter as coordenadas de lat/lon.
        sample_idx (int): O índice da amostra de teste a ser plotada.
        horizon_idx (int): O passo de tempo no futuro a ser plotado (ex: 0 para 3h, 1 para 6h).
        save_path (str, optional): Caminho para salvar a imagem. Se None, apenas exibe.
    """
    # 1. Extrair as coordenadas e os frames 2D específicos
    lat_coords = dados_grade_base.latitude.values
    lon_coords = dados_grade_base.longitude.values
    
    real_frame = y_real[sample_idx, horizon_idx, :, :]
    predito_frame = y_predito[sample_idx, horizon_idx, :, :]
    erro_frame = real_frame - predito_frame

    # 2. Calcular métricas para ESTA IMAGEM específica
    rmse_frame = np.sqrt(np.mean(erro_frame ** 2))
    mae_frame = np.mean(np.abs(erro_frame))
    r2_frame = r2_score(real_frame.flatten(), predito_frame.flatten())

    # 3. Configurar a figura e os eixos
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    fig.suptitle(f"Comparação de Previsão (Amostra: {sample_idx}, Horizonte: +{(horizon_idx+1)*3}h)", fontsize=16)

    # Definir limites de cor consistentes para Real e Previsto
    vmin = min(real_frame.min(), predito_frame.min())
    vmax = max(real_frame.max(), predito_frame.max())
    
    # 4. Plotar o mapa Real (Ground Truth)
    im1 = axes[0].pcolormesh(lon_coords, lat_coords, real_frame, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Campo Real\n(Velocidade do Vento m/s)', fontsize=12)
    fig.colorbar(im1, ax=axes[0], orientation='vertical', label='m/s')

    # 5. Plotar o mapa Previsto
    im2 = axes[1].pcolormesh(lon_coords, lat_coords, predito_frame, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Campo Previsto\nR² (imagem): {r2_frame:.3f}', fontsize=12)
    fig.colorbar(im2, ax=axes[1], orientation='vertical', label='m/s')

    # 6. Plotar o mapa de Erro
    # Usar um colormap divergente para o erro, centrado em zero
    limite_erro = np.max(np.abs(erro_frame))
    im3 = axes[2].pcolormesh(lon_coords, lat_coords, erro_frame, cmap='coolwarm', vmin=-limite_erro, vmax=limite_erro)
    axes[2].set_title(f'Erro (Real - Previsto)\nRMSE (imagem): {rmse_frame:.3f} | MAE (imagem): {mae_frame:.3f}', fontsize=12)
    fig.colorbar(im3, ax=axes[2], orientation='vertical', label='Diferença (m/s)')
    
    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualização salva em: {save_path}")
        
    plt.show()


# ==============================================================================
# FUNÇÃO 2: COMPARAÇÃO GERAL DE MÉTRICAS ENTRE MODELOS (MLFLOW)
# ==============================================================================

def plotar_comparacao_metricas_modelos(
    mlflow_experiment_name: str,
    metricas: list = ['teste_rmse', 'teste_mae'],
    save_path: str = None
):
    """
    Busca os resultados de um experimento no MLflow e gera gráficos de barra
    comparando as métricas finais de teste dos modelos.

    Args:
        mlflow_experiment_name (str): Nome do experimento principal no MLflow.
        metricas (list): Lista de nomes das métricas a serem plotadas.
        save_path (str, optional): Caminho para salvar a imagem. Se None, apenas exibe.
    """
    print(f"--- Pós-processamento: Buscando dados do experimento '{mlflow_experiment_name}' no MLflow ---")
    try:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if not experiment:
            print(f"ERRO: Experimento '{mlflow_experiment_name}' não encontrado.")
            return
            
        # Busca todas as runs do experimento
        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time"])
    except Exception as e:
        print(f"ERRO ao conectar ou buscar dados do MLflow. Verifique se o servidor MLflow está ativo. Detalhes: {e}")
        return

    # Filtra apenas as runs 'pai' (não os folds de CV)
    parent_runs_df = runs_df[runs_df['tags.mlflow.parentRunId'].isna()].copy()

    if parent_runs_df.empty:
        print("Nenhuma run principal encontrada no experimento. Nada para plotar.")
        return

    # Extrai o nome do modelo do nome da run
    parent_runs_df['model_name'] = parent_runs_df['tags.mlflow.runName'].apply(
        lambda name: name.split('_')[1] if 'Final_' in name else name
    )
    
    # Renomeia as colunas de métricas para nomes mais amigáveis
    metricas_plot = [f'metrics.{m}' for m in metricas]
    df_plot = parent_runs_df[['model_name'] + metricas_plot].set_index('model_name')
    df_plot.columns = [col.replace('metrics.teste_', '').upper() for col in df_plot.columns]
    
    print("Resultados encontrados para os modelos:", df_plot.index.tolist())

    # Gera um subplot para cada métrica
    num_metricas = len(df_plot.columns)
    fig, axes = plt.subplots(num_metricas, 1, figsize=(10, 5 * num_metricas), sharex=True)
    if num_metricas == 1:
        axes = [axes] # Garante que 'axes' seja sempre uma lista
        
    fig.suptitle('Comparação de Desempenho dos Modelos (Conjunto de Teste)', fontsize=16, y=1.02)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))

    for i, metrica in enumerate(df_plot.columns):
        bars = axes[i].bar(df_plot.index, df_plot[metrica], color=colors)
        axes[i].set_title(f'Métrica: {metrica}')
        axes[i].set_ylabel(metrica)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adiciona os valores no topo das barras
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.xlabel('Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de comparação salvo em: {save_path}")

    plt.show()


# ==============================================================================
# BLOCO DE EXEMPLO (PARA TESTAR O MÓDULO ISOLADAMENTE)
# ==============================================================================

if __name__ == '__main__':
    # Para testar, você pode criar dados falsos ou carregar resultados salvos
    print("--- Testando o módulo de pós-processamento ---")

    # --- Teste da função de visualização ---
    # Carrega os dados de grade para obter as coordenadas
    from pre import pre # Supondo que você pode importar o módulo 'pre'
    grade_base = pre.preparar_dados_grade('../pre/vento turco.nc')
    
    if grade_base is not None:
        # Cria dados falsos para y_real e y_predito
        H, W = len(grade_base.latitude), len(grade_base.longitude)
        y_real_fake = np.random.rand(10, 2, H, W) * 15 # 10 amostras, 2h de horizonte
        y_predito_fake = y_real_fake + np.random.randn(10, 2, H, W) * 1.5
        
        print("\nGerando visualização de previsão com dados falsos...")
        visualizar_previsao_e_erro(
            y_real=y_real_fake,
            y_predito=y_predito_fake,
            dados_grade_base=grade_base,
            sample_idx=0,
            horizon_idx=0,
            save_path='exemplo_previsao.png'
        )

    # --- Teste da função de comparação de modelos ---
    # Substitua pelo nome real do seu experimento principal no MLflow
    NOME_EXPERIMENTO_MLFLOW = "Previsao_Vento_Turco_DL_Completo"
    
    print(f"\nGerando gráfico de comparação de modelos do experimento '{NOME_EXPERIMENTO_MLFLOW}'...")
    plotar_comparacao_metricas_modelos(
        mlflow_experiment_name=NOME_EXPERIMENTO_MLFLOW,
        save_path='comparacao_modelos.png'
    )
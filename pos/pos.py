# pos/pos.py
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

# ==============================================================================
# FUNÇÃO PRINCIPAL DE RELATÓRIO (CHAMADA PELO MAIN.PY)
# ==============================================================================

def gerar_relatorio_final_consolidado(
    df_resultados: pd.DataFrame,
    dict_previsoes: dict,
    y_real: np.ndarray,
    dados_grade_base: xr.DataArray
):
    """
    Gera um relatório final completo com tabela, gráfico e visões individuais.
    Esta função é chamada ao final da execução do main.py.
    """
    # 1. Cria o diretório para os resultados finais
    output_dir = os.path.join("resultados", "resultados_finais")
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*80)
    print(f"== GERANDO RELATÓRIO FINAL CONSOLIDADO EM: {output_dir} ==")
    print("="*80)

    # 2. Salva a tabela de resultados em CSV
    caminho_tabela = os.path.join(output_dir, "tabela_de_resultados.csv")
    df_resultados_sorted = df_resultados.sort_values(by=df_resultados.columns[0], ascending=True)
    df_resultados_sorted.to_csv(caminho_tabela, float_format="%.5f")
    print(f"> Tabela de resultados salva em: {caminho_tabela}")

    # 3. Gera e salva o gráfico de barras comparativo
    caminho_grafico = os.path.join(output_dir, "grafico_comparativo.png")
    _plotar_grafico_comparativo(df_resultados_sorted, save_path=caminho_grafico)

    # 4. Gera e salva a visualização individual para cada modelo
    for model_name, y_predito in dict_previsoes.items():
        print(f"\n--- Gerando relatório visual para o modelo: {model_name.upper()} ---")
        _gerar_relatorio_visual_modelo(
            y_real=y_real, y_predito=y_predito, dados_grade_base=dados_grade_base,
            model_name=model_name, output_dir=output_dir
        )
    print("\n" + "="*80)
    print("== RELATÓRIO FINAL GERADO COM SUCESSO ==")
    print("="*80)

# ==============================================================================
# FUNÇÃO AUTÔNOMA DE COMPARAÇÃO (VIA MLFLOW)
# ==============================================================================

def gerar_grafico_comparativo_do_mlflow(
    mlflow_experiment_name: str,
    metricas: list = ['teste_rmse', 'teste_mae'],
    save_path: str = None
):
    """
    Busca os resultados de um experimento no MLflow e gera gráficos de barra
    comparando as métricas finais de teste dos modelos.
    Pode ser executada a qualquer momento, de forma independente.
    """
    print(f"\n--- Buscando dados do experimento '{mlflow_experiment_name}' no MLflow ---")
    try:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if not experiment:
            print(f"ERRO: Experimento '{mlflow_experiment_name}' não encontrado.")
            return
        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time"])
    except Exception as e:
        print(f"ERRO ao conectar ou buscar dados do MLflow. Detalhes: {e}")
        return

    parent_runs_df = runs_df[runs_df['tags.mlflow.parentRunId'].isna()].copy()
    if parent_runs_df.empty:
        print("Nenhuma run principal encontrada no experimento. Nada para plotar.")
        return

    parent_runs_df['model_name'] = parent_runs_df['tags.mlflow.runName'].apply(
        lambda name: name.split('_')[1] if 'Final_' in name else name
    )
    
    metricas_plot = [f'metrics.{m}' for m in metricas]
    df_plot = parent_runs_df[['model_name'] + metricas_plot].set_index('model_name')
    df_plot.columns = [col.replace('metrics.teste_', '') for col in df_plot.columns]
    
    print("Resultados encontrados para os modelos:", df_plot.index.tolist())
    _plotar_grafico_comparativo(df_plot, save_path)
    plt.show()

# ==============================================================================
# FUNÇÕES AUXILIARES (INTERNAS)
# ==============================================================================

def _plotar_grafico_comparativo(df_plot: pd.DataFrame, save_path: str):
    """Gera um gráfico de barras a partir de um DataFrame de resultados."""
    # (Lógica de plotagem interna, sem alterações)
    df_plot_sorted = df_plot.sort_values(by=df_plot.columns[0], ascending=True)
    num_metricas = len(df_plot_sorted.columns)
    fig, axes = plt.subplots(num_metricas, 1, figsize=(10, 5 * num_metricas), sharex=True)
    if num_metricas == 1: axes = [axes]
    fig.suptitle('Comparação de Desempenho dos Modelos (Conjunto de Teste)', fontsize=16, y=1.02)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot_sorted)))
    for i, metrica in enumerate(df_plot_sorted.columns):
        bars = axes[i].bar(df_plot_sorted.index, df_plot_sorted[metrica], color=colors)
        axes[i].set_title(f'Métrica: {metrica.upper()}')
        axes[i].set_ylabel(metrica.upper())
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
    plt.xlabel('Modelo')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"> Gráfico comparativo salvo em: {save_path}")
    plt.close(fig)

def _gerar_relatorio_visual_modelo(y_real, y_predito, dados_grade_base, model_name, output_dir):
    """Gera e salva as 3 imagens de amostra para um único modelo."""
    # (Lógica interna, sem alterações)
    indices_para_plotar = [0, len(y_real) // 2, len(y_real) - 1]
    for sample_idx in indices_para_plotar:
        save_path = os.path.join(output_dir, f"previsao_{model_name}_amostra_{sample_idx}.png")
        _plotar_comparacao_individual(
            y_real=y_real, y_predito=y_predito, dados_grade_base=dados_grade_base,
            model_name=model_name, sample_idx=sample_idx, horizon_idx=0,
            save_path=save_path
        )

def _plotar_comparacao_individual(y_real, y_predito, dados_grade_base, model_name, sample_idx, horizon_idx, save_path):
    """Gera a imagem de 3 painéis (Real, Previsto, Erro)."""
    # (Lógica de plotagem interna, sem alterações)
    lat_coords = dados_grade_base.latitude.values
    lon_coords = dados_grade_base.longitude.values
    real_frame = y_real[sample_idx, horizon_idx, :, :]
    predito_frame = y_predito[sample_idx, horizon_idx, :, :]
    erro_frame = real_frame - predito_frame
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    fig.suptitle(f"Comparativo - Modelo: {model_name.upper()} (Amostra: {sample_idx})", fontsize=16)
    vmin = min(real_frame.min(), predito_frame.min())
    vmax = max(real_frame.max(), predito_frame.max())
    im1 = axes[0].pcolormesh(lon_coords, lat_coords, real_frame, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Campo Real (GT)')
    fig.colorbar(im1, ax=axes[0], label='Velocidade (m/s)')
    im2 = axes[1].pcolormesh(lon_coords, lat_coords, predito_frame, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Campo Previsto')
    fig.colorbar(im2, ax=axes[1], label='Velocidade (m/s)')
    limite_erro = np.max(np.abs(erro_frame))
    im3 = axes[2].pcolormesh(lon_coords, lat_coords, erro_frame, cmap='coolwarm', vmin=-limite_erro, vmax=limite_erro)
    rmse_frame = np.sqrt(np.mean(erro_frame ** 2))
    axes[2].set_title(f'Erro (Real - Previsto)\nRMSE na Imagem: {rmse_frame:.3f} m/s')
    fig.colorbar(im3, ax=axes[2], label='Diferença (m/s)')
    for ax in axes:
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  > Relatório visual para amostra {sample_idx} salvo em: {save_path}")

# ==============================================================================
# BLOCO DE EXEMPLO (PARA TESTAR O MÓDULO ISOLADAMENTE)
# ==============================================================================

if __name__ == '__main__':
    # Este bloco só é executado se você rodar 'python pos/pos.py' diretamente
    NOME_EXPERIMENTO_MLFLOW = "Previsao_Vento_Turco_DL_Completo"
    
    print(f"--- Testando a função autônoma do MLflow para o experimento '{NOME_EXPERIMENTO_MLFLOW}' ---")
    gerar_grafico_comparativo_do_mlflow(
        mlflow_experiment_name=NOME_EXPERIMENTO_MLFLOW,
        save_path='comparacao_modelos_mlflow.png'
    )
# pos/pos.py (corrigido)
import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from typing import Dict, Optional

def _metricas_completas(y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    resid = y_pred - y_real
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_real - y_pred)))
    from sklearn.metrics import r2_score
    r2 = float(r2_score(y_real.ravel(), y_pred.ravel()))
    media_erro = float(np.mean(resid))
    dp_erro = float(np.std(resid))
    return {"rmse": rmse, "mae": mae, "r2": r2, "media_erro": media_erro, "dp_erro": dp_erro}

def gerar_relatorio_final_consolidado(
    df_resultados: pd.DataFrame,
    dict_previsoes: Dict[str, np.ndarray],
    y_real: np.ndarray,
    dados_grade_base: Optional[xr.DataArray] = None,
    sample_idx: int = 0,
    horizon_idx: int = 0,
    mostrar: bool = False
):
    output_dir = os.path.join("resultados", "resultados_finais")
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*80)
    print(f"== GERANDO RELATÓRIO FINAL CONSOLIDADO EM: {output_dir} ==")
    print("="*80)

    # 1) Tabela ordenada e enriquecida
    caminho_tabela = os.path.join(output_dir, "tabela_de_resultados.csv")
    coluna_ordem = "rmse" if "rmse" in df_resultados.columns else df_resultados.columns
    df_ord = df_resultados.sort_values(by=coluna_ordem, ascending=True)
    df_ord.to_csv(caminho_tabela, float_format="%.5f")
    print(f"> Tabela de resultados salva em: {caminho_tabela}")

    linhas = []
    for modelo, y_pred in dict_previsoes.items():
        met = _metricas_completas(y_real, y_pred); met["modelo"] = modelo; linhas.append(met)
    df_extras = pd.DataFrame(linhas).set_index("modelo") if linhas else pd.DataFrame()

    if not df_extras.empty:
        colunas_candidatas = ["rmse", "mae", "r2", "media_erro", "dp_erro"]
        faltantes = [c for c in colunas_candidatas if c not in df_ord.columns and c in df_extras.columns]
        df_completa = df_ord.join(df_extras[faltantes], how="left") if faltantes else df_ord.copy()
    else:
        df_completa = df_ord.copy()

    caminho_tabela_completa = os.path.join(output_dir, "tabela_de_resultados_completa.csv")
    df_completa.to_csv(caminho_tabela_completa, float_format="%.5f")
    try:
        if mlflow.active_run():
            mlflow.log_artifact(caminho_tabela, artifact_path="relatorio_final")
            mlflow.log_artifact(caminho_tabela_completa, artifact_path="relatorio_final")
    except Exception:
        pass

    # 2) Gráfico de barras
    caminho_grafico = os.path.join(output_dir, "grafico_comparativo.png")
    _plotar_grafico_comparativo(df_ord, save_path=caminho_grafico)
    try:
        if mlflow.active_run():
            mlflow.log_artifact(caminho_grafico, artifact_path="relatorio_final")
    except Exception:
        pass

    # 3) Figura com todos os modelos
    caminho_fig_todos = os.path.join(output_dir, f"comparativo_todos_modelos_s{sample_idx}_h{horizon_idx}.png")
    _figura_todos_modelos_um_dia(
        dict_previsoes=dict_previsoes,
        y_real=y_real,
        dados_grade_base=dados_grade_base,
        sample_idx=sample_idx,
        horizon_idx=horizon_idx,
        save_path=caminho_fig_todos,
        mostrar=mostrar
    )
    try:
        if mlflow.active_run():
            mlflow.log_artifact(caminho_fig_todos, artifact_path="relatorio_final")
    except Exception:
        pass

    # 4) Visualizações individuais
    for model_name, y_predito in dict_previsoes.items():
        print(f"\n--- Gerando relatório visual para o modelo: {model_name.upper()} ---")
        _gerar_relatorio_visual_modelo(
            y_real=y_real, y_predito=y_predito, dados_grade_base=dados_grade_base,
            model_name=model_name, output_dir=output_dir
        )

    print("\n" + "="*80)
    print("== RELATÓRIO FINAL GERADO COM SUCESSO ==")
    print("="*80)

def gerar_grafico_comparativo_do_mlflow(
    mlflow_experiment_name: str,
    metricas: list = ['teste_rmse', 'teste_mae'],
    save_path: str = None,
    mostrar: bool = False
):
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

    def _nome_modelo(row):
        rn = str(row.get('tags.mlflow.runName', ''))
        if rn.startswith("Final_"):
            return rn.replace("Final_", "")
        p = row.get('params.modelo')
        if isinstance(p, str) and len(p) > 0:
            return p
        return rn

    parent_runs_df['model_name'] = parent_runs_df.apply(_nome_modelo, axis=1)
    metricas_plot = [f'metrics.{m}' for m in metricas if f'metrics.{m}' in parent_runs_df.columns]
    cols = ['model_name'] + metricas_plot
    df_plot = parent_runs_df[cols].set_index('model_name')
    df_plot.columns = [c.replace('metrics.teste_', '') for c in df_plot.columns]

    print("Resultados encontrados para os modelos:", df_plot.index.tolist())
    _plotar_grafico_comparativo(df_plot, save_path)
    if mostrar:
        plt.show()

def _plotar_grafico_comparativo(df_plot: pd.DataFrame, save_path: Optional[str]):
    coluna_ordem = 'rmse' if 'rmse' in df_plot.columns else df_plot.columns
    df_plot_sorted = df_plot.sort_values(by=coluna_ordem, ascending=True)
    num_metricas = len(df_plot_sorted.columns)
    fig, axes = plt.subplots(num_metricas, 1, figsize=(10, 5 * num_metricas), sharex=True)
    if num_metricas == 1:
        axes = [axes]
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

def _figura_todos_modelos_um_dia(
    dict_previsoes: Dict[str, np.ndarray],
    y_real: np.ndarray,
    dados_grade_base: Optional[xr.DataArray],
    sample_idx: int,
    horizon_idx: int,
    save_path: str,
    mostrar: bool = False
):
    modelos = list(dict_previsoes.keys())
    ncols = len(modelos) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(4*ncols, 8), sharey=True)
    if dados_grade_base is not None:
        lat = np.asarray(dados_grade_base.latitude.values)
        lon = np.asarray(dados_grade_base.longitude.values)
    else:
        h, w = y_real.shape[-2], y_real.shape[-1]
        lat = np.arange(h); lon = np.arange(w)
    gt = y_real[sample_idx, horizon_idx]
    vmin, vmax = float(np.min(gt)), float(np.max(gt))
    im0 = axes[0, 0].pcolormesh(lon, lat, gt, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("GT")
    c0 = fig.colorbar(im0, ax=axes[0, 0]); c0.set_label('Velocidade (m/s)')
    axes[1, 0].axis("off")
    for j, m in enumerate(modelos, start=1):
        pred = dict_previsoes[m][sample_idx, horizon_idx]
        im1 = axes[0, j].pcolormesh(lon, lat, pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, j].set_title(m)
        c1 = fig.colorbar(im1, ax=axes[0, j]); c1.set_label('Velocidade (m/s)')
        err = gt - pred
        el = float(np.max(np.abs(err))) if np.isfinite(err).any() else 0.0
        im2 = axes[1, j].pcolormesh(lon, lat, err, cmap='coolwarm', vmin=-el, vmax=+el)
        axes[1, j].set_title(f"Erro {m}")
        c2 = fig.colorbar(im2, ax=axes[1, j]); c2.set_label('Diferença (m/s)')
    for ax in axes.ravel():
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if mostrar: plt.show()
    plt.close(fig)
    print(f"> Figura comparativa (todos os modelos) salva em: {save_path}")

def _gerar_relatorio_visual_modelo(y_real, y_predito, dados_grade_base, model_name, output_dir):
    indices_para_plotar = [0, len(y_real) // 2, len(y_real) - 1]
    for sample_idx in indices_para_plotar:
        save_path = os.path.join(output_dir, f"previsao_{model_name}_amostra_{sample_idx}.png")
        _plotar_comparacao_individual(
            y_real=y_real, y_predito=y_predito, dados_grade_base=dados_grade_base,
            model_name=model_name, sample_idx=sample_idx, horizon_idx=0, save_path=save_path
        )

# pos/pos.py (função corrigida)

def _plotar_comparacao_individual(y_real, y_predito, dados_grade_base, model_name, sample_idx, horizon_idx, save_path):
    if dados_grade_base is not None:
        lat_coords = np.asarray(dados_grade_base.latitude.values)
        lon_coords = np.asarray(dados_grade_base.longitude.values)
    else:
        h, w = y_real.shape[-2], y_real.shape[-1]
        lat_coords = np.arange(h); lon_coords = np.arange(w)

    real_frame = y_real[sample_idx, horizon_idx]
    predito_frame = y_predito[sample_idx, horizon_idx]
    erro_frame = real_frame - predito_frame

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
    fig.suptitle(f"Comparativo - Modelo: {model_name.upper()} (Amostra: {sample_idx})", fontsize=16)

    vmin = float(min(real_frame.min(), predito_frame.min()))
    vmax = float(max(real_frame.max(), predito_frame.max()))

    # --- CORREÇÃO APLICADA AQUI ---
    # O objeto 'axes' é um array. Devemos indexá-lo para acessar cada subplot.
    # Painel 1: GT (Ground Truth)
    im1 = axes[0].pcolormesh(lon_coords, lat_coords, real_frame, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Campo Real (GT)')
    cb1 = fig.colorbar(im1, ax=axes[0])
    cb1.set_label('Velocidade (m/s)')

    # Painel 2: Previsto (já estava correto)
    im2 = axes[1].pcolormesh(lon_coords, lat_coords, predito_frame, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Campo Previsto')
    cb2 = fig.colorbar(im2, ax=axes[1])
    cb2.set_label('Velocidade (m/s)')

    # Painel 3: Erro (já estava correto)
    limite_erro = float(np.max(np.abs(erro_frame))) if np.isfinite(erro_frame).any() else 1.0
    im3 = axes[2].pcolormesh(lon_coords, lat_coords, erro_frame, cmap='coolwarm', vmin=-limite_erro, vmax=limite_erro)
    rmse_frame = float(np.sqrt(np.mean(erro_frame ** 2)))
    axes[2].set_title(f'Erro (Real - Previsto)\nRMSE na Imagem: {rmse_frame:.3f} m/s')
    cb3 = fig.colorbar(im3, ax=axes[2])
    cb3.set_label('Diferença (m/s)')

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  > Relatório visual para amostra {sample_idx} salvo em: {save_path}")

if __name__ == '__main__':
    NOME_EXPERIMENTO_MLFLOW = "Previsao_Vento_Turco_DL_Completo"
    print(f"--- Testando a função autônoma do MLflow para o experimento '{NOME_EXPERIMENTO_MLFLOW}' ---")
    gerar_grafico_comparativo_do_mlflow(
        mlflow_experiment_name=NOME_EXPERIMENTO_MLFLOW,
        save_path='comparacao_modelos_mlflow.png',
        mostrar=False
    )

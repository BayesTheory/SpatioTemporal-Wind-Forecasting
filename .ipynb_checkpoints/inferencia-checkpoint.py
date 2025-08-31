# inferencia.py
import os, mlflow, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mlflow.tracking import MlflowClient

EXPERIMENTO = "Previsao_Vento_Turco_DL_Completo"  # ou outro
OUT_DIR = os.path.join("resultados", "resultados_inferencia")
os.makedirs(OUT_DIR, exist_ok=True)

def buscar_runs_pai(experiment_name):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        raise RuntimeError(f"Experimento '{experiment_name}' não encontrado")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time"])
    pais = df[df["tags.mlflow.parentRunId"].isna()].copy()
    return exp, pais

def baixar_artefato(run_id, path_local, artifact_path_rel):
    client = MlflowClient()
    os.makedirs(path_local, exist_ok=True)
    client.download_artifacts(run_id, artifact_path_rel, path_local)
    return os.path.join(path_local, artifact_path_rel)

def calcular_metricas_extras(y_true, y_pred):
    # Erro = predito - real
    resid = y_pred - y_true
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true.ravel(), y_pred.ravel())
    media_erro = float(np.mean(resid))
    dp_erro = float(np.std(resid))
    return {"rmse": rmse, "mae": mae, "r2": r2, "media_erro": media_erro, "dp_erro": dp_erro}

def montar_tabela(exp_name):
    exp, pais = buscar_runs_pai(exp_name)
    # Espera métricas 'teste_rmse' e 'teste_mae' já logadas nas runs finais
    cols_m = ["metrics.teste_rmse","metrics.teste_mae"]
    disponiveis = [c for c in cols_m if c in pais.columns]
    df_tab = pais[["tags.mlflow.runName"] + disponiveis].copy()
    df_tab["modelo"] = df_tab["tags.mlflow.runName"].str.replace(r"^Final_","",regex=True)
    df_tab = df_tab.set_index("modelo").drop(columns=["tags.mlflow.runName"])
    # Enriquecer com métricas extras se artefatos de y/preds estiverem disponíveis
    extras = []
    for _, row in pais.iterrows():
        run_id = row["run_id"]
        model_name = row["tags.mlflow.runName"].replace("Final_","")
        try:
            # Convenção sugerida: artifacts/y_test.npy e artifacts/preds.npy
            y_path = baixar_artefato(run_id, os.path.join(OUT_DIR,"tmp",run_id), "y_test.npy")
            p_path = baixar_artefato(run_id, os.path.join(OUT_DIR,"tmp",run_id), "preds.npy")
            y = np.load(y_path); p = np.load(p_path)
            met = calcular_metricas_extras(y, p)
            met["modelo"] = model_name
            extras.append(met)
        except Exception:
            # Se não houver artefatos, segue só com métricas logadas
            pass
    if extras:
        df_ex = pd.DataFrame(extras).set_index("modelo")
        df_tab = df_tab.join(df_ex, how="left")
    # Renomear colunas canônicas
    df_tab = df_tab.rename(columns={
        "metrics.teste_rmse":"rmse",
        "metrics.teste_mae":"mae",
    })
    # Ordenar por rmse quando existir
    by = "rmse" if "rmse" in df_tab.columns else df_tab.columns
    df_tab = df_tab.sort_values(by=by)
    out_csv = os.path.join(OUT_DIR, "tabela_inferencia.csv")
    df_tab.to_csv(out_csv, float_format="%.5f")
    return df_tab, out_csv

def figura_comparativa_um_dia(df_tab, y_true, dict_preds, lat, lon, sample_idx=0, horizon_idx=0, save_path=None):
    # Monta grade com GT + cada modelo
    modelos = list(df_tab.index)
    n = len(modelos) + 1
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8), sharey=True)
    # GT
    gt = y_true[sample_idx, horizon_idx]
    vmin, vmax = float(np.min(gt)), float(np.max(gt))
    im = axes[0,0].pcolormesh(lon, lat, gt, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0,0].set_title("GT")
    fig.colorbar(im, ax=axes[0,0])
    axes[1,0].axis("off")
    # Modelos
    for j, m in enumerate(modelos, start=1):
        pred = dict_preds[m][sample_idx, horizon_idx]
        im1 = axes[0,j].pcolormesh(lon, lat, pred, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0,j].set_title(m)
        fig.colorbar(im1, ax=axes[0,j])
        err = gt - pred
        lim = float(np.max(np.abs(err)))
        im2 = axes[1,j].pcolormesh(lon, lat, err, cmap="coolwarm", vmin=-lim, vmax=+lim)
        axes[1,j].set_title(f"Erro {m}")
        fig.colorbar(im2, ax=axes[1,j])
    for ax in axes.ravel():
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return save_path

if __name__ == "__main__":
    # Exemplo de uso: buscar runs, montar tabela e tentar baixar y/preds para figura
    df, csv_path = montar_tabela(EXPERIMENTO)
    print(f"Tabela salva em: {csv_path}")
    # Opcional: caso y/preds tenham sido logados, carregar e montar figura para sample_idx=0
    # Usuário deve preparar dict_preds e y_true se quiser a figura sem baixar do MLflow.

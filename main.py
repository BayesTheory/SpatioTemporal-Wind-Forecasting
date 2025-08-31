# main.py
# Versão Final 3.1 - Modos: treino e relatorio_mlflow (com salvamento local para inferência)
import importlib
import warnings
import argparse
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
import torch
import json
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

# Importa módulos do projeto
import utils
import train
from pre import pre
from pos.pos import gerar_relatorio_final_consolidado, gerar_grafico_comparativo_do_mlflow

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

def carregar_config_modelos(caminho_json: str) -> dict:
    """Carrega as configurações dos modelos a partir de um arquivo JSON (chaves para minúsculo)."""
    try:
        with open(caminho_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # Converte chaves principais para minúsculo para consistência
        config_lower = {k.lower(): v for k, v in config.items()}
        # Converte a ordem do ARIMA para uma tupla, se existir
        if '1arima' in config_lower and 'ordem' in config_lower['1arima']:
            config_lower['1arima']['ordem'] = tuple(config_lower['1arima']['ordem'])
        return config_lower
    except Exception as e:
        print(f"ERRO ao carregar ou processar o arquivo de configuração: {e}")
        return {}

def calcular_metricas_completas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula um conjunto completo de métricas: RMSE, MAE, R², média e DP do erro."""
    resid = y_pred - y_real
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_real - y_pred)))
    r2 = float(r2_score(y_real.ravel(), y_pred.ravel())) # Achatamento para R²
    media_erro = float(np.mean(resid))
    dp_erro = float(np.std(resid))
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'media_erro': media_erro, 'dp_erro': dp_erro}

# ==============================================================================
# EXECUÇÃO DE UM MODELO (CV, Treino Final, Teste)
# ==============================================================================

def executar_modelo(model_name: str, model_config: dict,
                    dados_treino_cv: xr.DataArray, dados_teste: xr.DataArray,
                    passos_janela: int, passos_prever: int,
                    freq_horas: int, horizonte_horas: int):
    """
    Executa o fluxo completo para um modelo: CV (para DL), treino final, teste.
    Salva artefatos no MLflow e localmente, e retorna métricas e previsões.
    """
    print("=" * 60)
    print(f"== EXECUTANDO FLUXO COMPLETO PARA {model_name.upper()} ==")
    print("=" * 60)

    # Define o dispositivo (GPU se disponível, senão CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define nomes de experimento e run principal no MLflow
    exp_name = "Previsao_Vento_Turco_Baseline" if model_name.lower() == '1arima' else "Previsao_Vento_Turco_DL_Completo"
    run_name_pai = f"Final_{model_name}_prev{horizonte_horas}h"
    
    # Inicia o experimento e a run principal no MLflow
    utils.iniciar_experimento_mlflow(exp_name, run_name_pai)

    # Pasta unificada para salvamentos locais de modelos DL
    MODELOS_SALVOS_DIR = "modelos_salvos"

    try:
        # Log de parâmetros gerais
        utils.logar_parametros_mlflow({
            "modelo": model_name,
            "passos_janela": passos_janela,
            "passos_previsao": passos_prever,
            "freq_horas": freq_horas,
            "horizonte_horas": horizonte_horas
        })
        # Log de parâmetros específicos do modelo
        utils.logar_parametros_mlflow(model_config)

        # Prepara os dados de teste com contexto para formar as janelas deslizantes
        dados_contexto = dados_treino_cv.isel(valid_time=slice(-passos_janela, None))
        dados_teste_com_contexto = xr.concat([dados_contexto, dados_teste], dim="valid_time")
        X_teste, y_teste_real = utils.formatar_janelas_video(dados_teste_com_contexto, passos_janela, passos_prever)

        if model_name.lower() == '1arima':
            print(f"\n--- {model_name.upper()} | TREINO E PREVISÃO EM GRADE ---")
            df_treino_grade = dados_treino_cv.to_dataframe(name='ws100')
            
            # Importa dinamicamente o módulo do ARIMA
            modulo_arima = importlib.import_module("Modelos.1arima")
            previsoes_continuas = modulo_arima.executar_arima_para_grade(
                df_treino_grade, len(dados_teste), model_config['ordem']
            )
            
            # Formata as previsões no mesmo formato de janela dos modelos DL
            previsoes_reais_teste = []
            for i in range(len(y_teste_real)):
                fatia = previsoes_continuas[i: i + passos_prever]
                previsoes_reais_teste.append(fatia)
            previsoes_reais_teste = np.array(previsoes_reais_teste)

        else: # Modelos de Deep Learning
            # Cross-Validation Temporal
            tscv = TimeSeriesSplit(n_splits=N_FOLDS_CV)
            for fold, (train_index, val_index) in enumerate(tscv.split(dados_treino_cv.valid_time)):
                print(f"\n--- {model_name.upper()} | FOLD CV [{fold + 1}/{N_FOLDS_CV}] ---")
                dados_treino_fold = dados_treino_cv.isel(valid_time=train_index)
                dados_validacao_fold = dados_treino_cv.isel(valid_time=val_index)

                with mlflow.start_run(run_name=f"CV_Fold_{fold+1}", nested=True):
                    X_treino, y_treino = utils.formatar_janelas_video(dados_treino_fold, passos_janela, passos_prever)
                    X_val, y_val_real = utils.formatar_janelas_video(dados_validacao_fold, passos_janela, passos_prever)

                    # Checa se há amostras suficientes para treinar e validar no fold
                    if len(X_treino) == 0 or len(X_val) == 0:
                        print("  Aviso: Fold pulado por falta de amostras suficientes.")
                        continue
                    
                    # Normalização dos dados
                    scaler = utils.criar_e_treinar_scaler_grade(X_treino)
                    X_treino_n = utils.aplicar_scaler_grade(X_treino, scaler)
                    y_treino_n = utils.aplicar_scaler_grade(y_treino, scaler)
                    X_val_n = utils.aplicar_scaler_grade(X_val, scaler)
                    y_val_n = utils.aplicar_scaler_grade(y_val_real, scaler)

                    # Importa a classe do modelo dinamicamente
                    modulo = importlib.import_module(f"Modelos.{model_name}")
                    model_class = getattr(modulo, model_config['class_name'])
                    
                    model_config_completo = {
                        'past_frames': passos_janela,
                        'future_frames': passos_prever,
                        **model_config,
                        'models_out_dir': MODELOS_SALVOS_DIR,
                    }
                    
                    # Executa o treino (o salvamento do melhor/último modelo é feito dentro de train.py)
                    train.executar_treino_e_previsao(
                        model_class=model_class, model_config=model_config_completo,
                        X_treino=X_treino_n, y_treino=y_treino_n, X_pred=X_val_n,
                        device=device, X_val=X_val_n, y_val=y_val_n
                    )

            # Treino Final com todos os dados de treino/CV e avaliação no conjunto de teste
            print(f"\n--- {model_name.upper()} | TREINO FINAL E AVALIAÇÃO EM TESTE ---")
            X_treino_final, y_treino_final = utils.formatar_janelas_video(dados_treino_cv, passos_janela, passos_prever)
            
            # Normalização final
            scaler_final = utils.criar_e_treinar_scaler_grade(X_treino_final)
            X_treino_final_n = utils.aplicar_scaler_grade(X_treino_final, scaler_final)
            y_treino_final_n = utils.aplicar_scaler_grade(y_treino_final, scaler_final)
            X_teste_n = utils.aplicar_scaler_grade(X_teste, scaler_final)

            modulo = importlib.import_module(f"Modelos.{model_name}")
            model_class = getattr(modulo, model_config['class_name'])
            model_config_completo = {
                'past_frames': passos_janela,
                'future_frames': passos_prever,
                **model_config,
                'models_out_dir': MODELOS_SALVOS_DIR,
            }

            previsoes_n = train.executar_treino_e_previsao(
                model_class=model_class, model_config=model_config_completo,
                X_treino=X_treino_final_n, y_treino=y_treino_final_n,
                X_pred=X_teste_n, device=device
            )
            previsoes_reais_teste = utils.desnormalizar_dados_grade(previsoes_n, scaler_final)

            # Salva o scaler para inferência futura
            os.makedirs("artefatos", exist_ok=True)
            scaler_path = os.path.join("artefatos", f"scaler_{model_name}.pkl")
            joblib.dump(scaler_final, scaler_path)
            try:
                mlflow.log_artifact(scaler_path, artifact_path="preproc")
            except Exception:
                pass

        # Métricas finais no conjunto de teste
        metricas_teste_final = calcular_metricas_completas(y_teste_real, previsoes_reais_teste)
        print(f"\n--- MÉTRICAS FINAIS DE {model_name.upper()} NO TESTE ---")
        print(metricas_teste_final)

        # Log das métricas no MLflow
        utils.logar_metricas_mlflow({f"teste_{k}": v for k, v in metricas_teste_final.items()})

        # Salva previsões e valores reais para relatórios offline
        os.makedirs("inferencia", exist_ok=True)
        y_path = os.path.join("inferencia", "y_test.npy")
        p_path = os.path.join("inferencia", f"preds_{model_name}.npy")
        np.save(y_path, y_teste_real)
        np.save(p_path, previsoes_reais_teste)
        try:
            mlflow.log_artifact(y_path, artifact_path="inferencia")
            mlflow.log_artifact(p_path, artifact_path="inferencia")
        except Exception:
            pass
        
        return metricas_teste_final, previsoes_reais_teste

    except Exception as e:
        print(f"\nERRO: A execução do modelo {model_name} falhou. {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        # Garante que a run do MLflow seja finalizada
        if mlflow.active_run():
            mlflow.end_run()
        print(f"\n{'=' * 60}\n== PROCESSO PARA {model_name.upper()} FINALIZADO ==\n{'=' * 60}")

# ==============================================================================
# Modo relatório via MLflow (sem treinar)
# ==============================================================================

def gerar_relatorio_do_mlflow(experiment_name: str, saida_dir: str, modelos_esperados: list = None):
    """
    Busca runs 'pai' no experimento do MLflow, baixa os artefatos (previsões e
    valores reais) e chama a função para gerar o relatório consolidado.
    """
    from mlflow.tracking import MlflowClient
    
    print(f"\n--- Modo relatório_mlflow para experimento '{experiment_name}' ---")
    os.makedirs(saida_dir, exist_ok=True)
    
    # Busca o experimento pelo nome
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"Experimento '{experiment_name}' não encontrado.")
        return

    # Busca todas as runs do experimento, ordenadas por tempo
    runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time"])
    # Filtra apenas as runs principais (que não são aninhadas)
    pais = runs_df[runs_df['tags.mlflow.parentRunId'].isna()].copy()

    if pais.empty:
        print("Nenhuma run principal encontrada.")
        return

    client = MlflowClient()
    dict_previsoes, y_real = {}, None
    modelos = []

    for _, row in pais.iterrows():
        run_id = row["run_id"]
        run_name = row["tags.mlflow.runName"]
        
        # Processa apenas runs que seguem o padrão de nome "Final_..."
        if not str(run_name).startswith("Final_"):
            continue
            
        model_name = run_name.split("_prev")[0].replace("Final_", "") # Extrai o nome do modelo
        
        if (modelos_esperados is not None) and (model_name not in modelos_esperados):
            continue
            
        try:
            # Baixa os artefatos de inferência
            tmp_dir = os.path.join(saida_dir, "tmp", run_id)
            os.makedirs(tmp_dir, exist_ok=True)
            y_local = client.download_artifacts(run_id, "inferencia/y_test.npy", tmp_dir)
            p_local = client.download_artifacts(run_id, f"inferencia/preds_{model_name}.npy", tmp_dir)
            
            y_arr = np.load(y_local)
            p_arr = np.load(p_local)
            
            if y_real is None:
                y_real = y_arr
                
            dict_previsoes[model_name] = p_arr
            modelos.append(model_name)
        except Exception as e:
            print(f"  Aviso: Artefatos não encontrados para {model_name} (run_id: {run_id}): {e}")
            continue

    if not dict_previsoes or y_real is None:
        print("Não foi possível montar previsões/y_real a partir dos artefatos.")
        return

    # Recalcula as métricas a partir dos dados baixados
    linhas = []
    for m, preds in dict_previsoes.items():
        met = calcular_metricas_completas(y_real, preds)
        met["modelo"] = m
        linhas.append(met)
        
    df_resultados = pd.DataFrame(linhas).set_index("modelo").sort_values(by="rmse")
    
    # Gera o relatório final
    gerar_relatorio_final_consolidado(
        df_resultados=df_resultados,
        dict_previsoes=dict_previsoes,
        y_real=y_real,
        dados_grade_base=None # Não temos a grade base neste modo
    )
    print("Relatório a partir do MLflow gerado com sucesso.")


# ==============================================================================
# PONTO DE ENTRADA (MAIN)
# ==============================================================================

if __name__ == "__main__":
    # Configuração dos argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Pipeline de previsão espaço-temporal (treino/relatório).")
    parser.add_argument("--modo", choices=["treino", "relatorio_mlflow"], default="treino",
                        help="treino: executa CV/treino/teste e gera relatório; relatorio_mlflow: somente relatório a partir de runs existentes.")
    parser.add_argument("--config", default="modelos.json", help="Caminho do JSON de configuração dos modelos.")
    parser.add_argument("--exp_relatorio", default="Previsao_Vento_Turco_DL_Completo", help="Nome do experimento para usar no modo relatorio_mlflow.")
    args = parser.parse_args()

    utils.print_timestamp("Iniciando pipeline...")
    print("=" * 80)
    
    # --- PARÂMETROS GLOBAIS DO PIPELINE ---
    FREQUENCIA_DADOS_HORAS = 3      # Frequência dos dados (ex: a cada 3 horas)
    JANELA_ENTRADA_HORAS = 24       # Quantas horas de dados usar para prever
    HORIZONTE_PREVISAO_HORAS = 6    # Quantas horas no futuro prever
    
    # Converte horas para "passos" ou "frames"
    PASSOS_JANELA_ENTRADA = JANELA_ENTRADA_HORAS // FREQUENCIA_DADOS_HORAS
    PASSOS_A_PREVER = HORIZONTE_PREVISAO_HORAS // FREQUENCIA_DADOS_HORAS
    
    N_FOLDS_CV = 5                  # Número de folds para a validação cruzada temporal
    TEST_SIZE = 0.2                 # Proporção do conjunto de dados a ser usado para teste final
    
    # ----------------------------------------
    
    # Se o modo for 'relatorio_mlflow', executa a função correspondente e encerra
    if args.modo == "relatorio_mlflow":
        gerar_relatorio_do_mlflow(experiment_name=args.exp_relatorio,
                                  saida_dir=os.path.join("resultados", "resultados_finais"))
        print("=" * 80)
        utils.print_timestamp("Concluído (modo relatorio_mlflow).")
        exit(0)

    # Lógica para o modo 'treino'
    CONFIG_MODELOS = carregar_config_modelos(args.config)

    print("--- Pré-processamento: Carregando GRADE espaço-temporal ---")
    dados_grade_completos = pre.preparar_dados_grade(caminho_arquivo_nc='pre/vento turco.nc')

    if dados_grade_completos is not None:
        print("Grade ws100 calculada e processada com sucesso.")
        
        # Divisão dos dados em treino/CV e teste
        n_tempos_total = len(dados_grade_completos.valid_time)
        n_teste = int(n_tempos_total * TEST_SIZE)
        n_treino_cv = n_tempos_total - n_teste
        
        dados_treino_cv = dados_grade_completos.isel(valid_time=slice(0, n_treino_cv))
        dados_teste = dados_grade_completos.isel(valid_time=slice(n_treino_cv, None))
        
        print(f"Dados divididos: {len(dados_treino_cv.valid_time)} para Treino/CV e {len(dados_teste.valid_time)} para Teste Final.")

        # Filtra apenas os modelos marcados como "ativo" no JSON
        modelos_a_executar = {nome: cfg for nome, cfg in CONFIG_MODELOS.items() if cfg.get("ativo", False)}
        resultados_finais = []
        previsoes_finais = {}

        if not modelos_a_executar:
            print("Nenhum modelo ativo no arquivo 'modelos.json'. Encerrando.")
        else:
            print(f"Modelos a serem executados: {list(modelos_a_executar.keys())}")
            # Loop para treinar e avaliar cada modelo ativo
            for nome_modelo, config_modelo in modelos_a_executar.items():
                metricas, previsoes = executar_modelo(
                    nome_modelo, config_modelo, dados_treino_cv, dados_teste,
                    PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER,
                    FREQUENCIA_DADOS_HORAS, HORIZONTE_PREVISAO_HORAS
                )
                if metricas and previsoes is not None:
                    resultados_finais.append({'modelo': nome_modelo, **metricas})
                    previsoes_finais[nome_modelo] = previsoes

        # Geração do relatório final consolidado após todos os modelos rodarem
        if resultados_finais:
            df_resultados = pd.DataFrame(resultados_finais).set_index('modelo')
            print("\n\n" + "=" * 80)
            print("== TABELA COMPARATIVA FINAL - RESULTADOS NO CONJUNTO DE TESTE ==")
            print("=" * 80)
            print(df_resultados.sort_values(by='rmse').to_string(float_format="%.5f"))
            print("=" * 80)
            
            # Obtém o y_real do conjunto de teste para o relatório
            dados_contexto = dados_treino_cv.isel(valid_time=slice(-PASSOS_JANELA_ENTRADA, None))
            dados_teste_com_contexto = xr.concat([dados_contexto, dados_teste], dim="valid_time")
            _, y_teste_real = utils.formatar_janelas_video(dados_teste_com_contexto, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
            
            gerar_relatorio_final_consolidado(
                df_resultados=df_resultados,
                dict_previsoes=previsoes_finais,
                y_real=y_teste_real,
                dados_grade_base=dados_grade_completos
            )
    else:
        print("Falha ao carregar os dados. Execução interrompida.")

    print("=" * 80)
    utils.print_timestamp("Execução do pipeline principal concluída.")
# main.py
# Versão Final 2.2 - Com passagem de parâmetros de treino avançado

import importlib
import warnings
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
import torch
import json
import os
from sklearn.model_selection import TimeSeriesSplit

# Importa os módulos do nosso projeto
import utils
import train
from pre import pre
from pos import pos

warnings.filterwarnings("ignore")

# ==============================================================================
# FUNÇÕES DE CONFIGURAÇÃO E EXECUÇÃO
# ==============================================================================

def carregar_config_modelos(caminho_json: str) -> dict:
    """Carrega as configurações dos modelos a partir de um arquivo JSON."""
    try:
        with open(caminho_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config_lower = {k.lower(): v for k, v in config.items()}
        if '1arima' in config_lower and 'ordem' in config_lower['1arima']:
            config_lower['1arima']['ordem'] = tuple(config_lower['1arima']['ordem'])
        return config_lower
    except Exception as e:
        print(f"ERRO ao carregar ou processar o arquivo de configuração: {e}")
        return {}

def executar_modelo(model_name: str, model_config: dict, dados_treino_cv: xr.DataArray, dados_teste: xr.DataArray):
    """
    Executa o fluxo completo (CV, Treino Final, Teste) para um único modelo.
    """
    print("="*60)
    print(f"== EXECUTANDO FLUXO COMPLETO PARA {model_name.upper()} ==")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = "Previsao_Vento_Turco_Baseline" if model_name.lower() == '1arima' else "Previsao_Vento_Turco_DL_Completo"
    run_name_pai = f"Final_{model_name}_prev{HORIZONTE_PREVISAO_HORAS}h"
    utils.iniciar_experimento_mlflow(exp_name, run_name_pai)
    
    try:
        parametros_gerais = {"modelo": model_name, "passos_janela": PASSOS_JANELA_ENTRADA, "passos_previsao": PASSOS_A_PREVER}
        utils.logar_parametros_mlflow(parametros_gerais)
        utils.logar_parametros_mlflow(model_config)

        dados_contexto = dados_treino_cv.isel(valid_time=slice(-PASSOS_JANELA_ENTRADA, None))
        dados_teste_com_contexto = xr.concat([dados_contexto, dados_teste], dim="valid_time")
        X_teste, y_teste_real = utils.formatar_janelas_video(dados_teste_com_contexto, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)

        if model_name.lower() == '1arima':
            print(f"\n--- {model_name.upper()} | TREINO E PREVISÃO EM GRADE ---")
            df_treino_grade = dados_treino_cv.to_dataframe(name='ws100')
            modulo_arima = importlib.import_module("Modelos.1arima")
            previsoes_continuas = modulo_arima.executar_arima_para_grade(df_treino_grade, len(dados_teste), model_config['ordem'])
            previsoes_reais_teste = []
            for i in range(len(y_teste_real)):
                fatia = previsoes_continuas[i : i + PASSOS_A_PREVER]
                previsoes_reais_teste.append(fatia)
            previsoes_reais_teste = np.array(previsoes_reais_teste)

        else: # Modelos de Deep Learning
            # --- VALIDAÇÃO CRUZADA ---
            tscv = TimeSeriesSplit(n_splits=N_FOLDS_CV)
            for fold, (train_index, val_index) in enumerate(tscv.split(dados_treino_cv.valid_time)):
                print(f"\n--- {model_name.upper()} | FOLD CV [{fold+1}/{N_FOLDS_CV}] ---")
                dados_treino_fold = dados_treino_cv.isel(valid_time=train_index)
                dados_validacao_fold = dados_treino_cv.isel(valid_time=val_index)
                with mlflow.start_run(run_name=f"CV_Fold_{fold+1}", nested=True):
                    X_treino, y_treino = utils.formatar_janelas_video(dados_treino_fold, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
                    X_val, y_val_real = utils.formatar_janelas_video(dados_validacao_fold, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
                    if X_treino.shape[0] < 1 or X_val.shape[0] < 1:
                        print("  Aviso: Fold pulado por falta de amostras suficientes.")
                        continue
                    
                    scaler = utils.criar_e_treinar_scaler_grade(X_treino)
                    X_treino_norm = utils.aplicar_scaler_grade(X_treino, scaler)
                    y_treino_norm = utils.aplicar_scaler_grade(y_treino, scaler)
                    X_val_norm = utils.aplicar_scaler_grade(X_val, scaler)

                    modulo = importlib.import_module(f"Modelos.{model_name}")
                    model_class = getattr(modulo, model_config['class_name'])
                    model_params = {'past_frames': PASSOS_JANELA_ENTRADA, 'future_frames': PASSOS_A_PREVER, **model_config}
                    
                    train.executar_treino_e_previsao(
                        model_class, model_params, X_treino_norm, y_treino_norm, X_val_norm,
                        model_config['epocas'], model_config['batch_size'], model_config['learning_rate'], device,
                        X_val=X_val_norm, y_val=utils.aplicar_scaler_grade(y_val_real, scaler),
                        patience=model_config.get('patience', 15),
                        optimizer_name=model_config.get('optimizer', 'adam'),
                        weight_decay=model_config.get('weight_decay', 0),
                        scheduler_name=model_config.get('scheduler', 'plateau'),
                        scheduler_patience=model_config.get('scheduler_patience', 7),
                        scheduler_factor=model_config.get('scheduler_factor', 0.5),
                        warmup_epochs=model_config.get('warmup_epochs', 0)
                    )

            # --- TREINO FINAL E PREVISÃO NO TESTE ---
            print(f"\n--- {model_name.upper()} | TREINO FINAL E AVALIAÇÃO EM TESTE ---")
            X_treino_final, y_treino_final = utils.formatar_janelas_video(dados_treino_cv, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
            scaler_final = utils.criar_e_treinar_scaler_grade(X_treino_final)
            X_treino_final_norm = utils.aplicar_scaler_grade(X_treino_final, scaler_final)
            y_treino_final_norm = utils.aplicar_scaler_grade(y_treino_final, scaler_final)
            X_teste_norm = utils.aplicar_scaler_grade(X_teste, scaler_final)
            
            modulo = importlib.import_module(f"Modelos.{model_name}")
            model_class = getattr(modulo, model_config['class_name'])
            model_params = {'past_frames': PASSOS_JANELA_ENTRADA, 'future_frames': PASSOS_A_PREVER, **model_config}

            previsoes_norm_teste = train.executar_treino_e_previsao(
                model_class, model_params, X_treino_final_norm, y_treino_final_norm, X_teste_norm,
                model_config['epocas'], model_config['batch_size'], model_config['learning_rate'], device,
                patience=model_config.get('patience', 15),
                optimizer_name=model_config.get('optimizer', 'adam'),
                weight_decay=model_config.get('weight_decay', 0),
                scheduler_name=model_config.get('scheduler', 'plateau'),
                scheduler_patience=model_config.get('scheduler_patience', 7),
                scheduler_factor=model_config.get('scheduler_factor', 0.5),
                warmup_epochs=model_config.get('warmup_epochs', 0)
            )
            previsoes_reais_teste = utils.desnormalizar_dados_grade(previsoes_norm_teste, scaler_final)
        
        # --- MÉTRICAS E PÓS-PROCESSAMENTO (PARA TODOS OS MODELOS) ---
        metricas_teste_final = utils.calcular_metricas(y_teste_real, previsoes_reais_teste)
        print(f"\n--- MÉTRICAS FINAIS DE {model_name.upper()} NO CONJUNTO DE TESTE ---")
        print(metricas_teste_final)
        utils.logar_metricas_mlflow({f"teste_{k}": v for k, v in metricas_teste_final.items()})

        print(f"\n--- {model_name.upper()} | GERANDO RELATÓRIO VISUAL ---")
        caminho_relatorio = os.path.join("resultados", model_name)
        os.makedirs(caminho_relatorio, exist_ok=True)
        pos.gerar_relatorio_visual(
            y_real=y_teste_real, y_predito=previsoes_reais_teste,
            dados_grade_base=dados_treino_cv, model_name=model_name,
            output_dir=caminho_relatorio
        )
        mlflow.log_artifacts(caminho_relatorio, artifact_path=model_name)
        
        return metricas_teste_final

    except Exception as e:
        print(f"\nERRO: A execução do modelo {model_name} falhou. {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if mlflow.active_run(): mlflow.end_run()
        print(f"\n{'='*60}\n== PROCESSO PARA {model_name.upper()} FINALIZADO ==\n{'='*60}")

# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # --- 1. CONFIGURAÇÕES GERAIS ---
    FREQUENCIA_DADOS_HORAS = 3
    JANELA_ENTRADA_HORAS = 24
    HORIZONTE_PREVISAO_HORAS = 6
    PASSOS_JANELA_ENTRADA = JANELA_ENTRADA_HORAS // FREQUENCIA_DADOS_HORAS
    PASSOS_A_PREVER = HORIZONTE_PREVISAO_HORAS // FREQUENCIA_DADOS_HORAS
    N_FOLDS_CV = 5
    TEST_SIZE = 0.2

    # --- 2. CARREGAMENTO E DIVISÃO DOS DADOS ---
    CONFIG_MODELOS = carregar_config_modelos('modelos.json')
    dados_grade_completos = pre.preparar_dados_grade(caminho_arquivo_nc='pre/vento turco.nc')
    
    if dados_grade_completos is not None:
        n_tempos_total = len(dados_grade_completos.valid_time)
        n_teste = int(n_tempos_total * TEST_SIZE)
        n_treino_cv = n_tempos_total - n_teste
        
        dados_treino_cv = dados_grade_completos.isel(valid_time=slice(0, n_treino_cv))
        dados_teste = dados_grade_completos.isel(valid_time=slice(n_treino_cv, None))

        print(f"Dados divididos: {len(dados_treino_cv.valid_time)} para Treino/CV e {len(dados_teste.valid_time)} para Teste Final.")

        # --- 3. LOOP DE EXECUÇÃO DOS MODELOS ATIVOS ---
        modelos_a_executar = {
            nome: config for nome, config in CONFIG_MODELOS.items() if config.get("ativo", False)
        }
        resultados_finais = []

        if not modelos_a_executar:
            print("Nenhum modelo ativo no 'modelos.json'.")
        else:
            print(f"Modelos a serem executados: {list(modelos_a_executar.keys())}")
            for nome_modelo, config_modelo in modelos_a_executar.items():
                metricas = executar_modelo(nome_modelo, config_modelo, dados_treino_cv, dados_teste)
                if metricas:
                    resultados_finais.append({'modelo': nome_modelo, **metricas})
        
        # --- 4. EXIBIÇÃO DA TABELA COMPARATIVA FINAL ---
        if resultados_finais:
            print("\n\n" + "="*80)
            print("== TABELA COMPARATIVA FINAL - RESULTADOS NO CONJUNTO DE TESTE ==")
            print("="*80)
            df_resultados = pd.DataFrame(resultados_finais).set_index('modelo').sort_values(by='rmse')
            print(df_resultados.to_string(float_format="%.5f"))
            print("="*80)

    else:
        print("Falha ao carregar os dados. Execução interrompida.")
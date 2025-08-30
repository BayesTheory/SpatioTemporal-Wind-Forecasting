# main.py
# Versão Final 2.4 - Com Relatório Final Consolidado

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
                    y_val_norm = utils.aplicar_scaler_grade(y_val_real, scaler)

                    modulo = importlib.import_module(f"Modelos.{model_name}")
                    model_class = getattr(modulo, model_config['class_name'])
                    model_config_completo = {'past_frames': PASSOS_JANELA_ENTRADA, 'future_frames': PASSOS_A_PREVER, **model_config}
                    
                    train.executar_treino_e_previsao(
                        model_class=model_class, model_config=model_config_completo,
                        X_treino=X_treino_norm, y_treino=y_treino_norm, X_pred=X_val_norm,
                        device=device, X_val=X_val_norm, y_val=y_val_norm
                    )

            print(f"\n--- {model_name.upper()} | TREINO FINAL E AVALIAÇÃO EM TESTE ---")
            X_treino_final, y_treino_final = utils.formatar_janelas_video(dados_treino_cv, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
            scaler_final = utils.criar_e_treinar_scaler_grade(X_treino_final)
            X_treino_final_norm = utils.aplicar_scaler_grade(X_treino_final, scaler_final)
            y_treino_final_norm = utils.aplicar_scaler_grade(y_treino_final, scaler_final)
            X_teste_norm = utils.aplicar_scaler_grade(X_teste, scaler_final)
            
            modulo = importlib.import_module(f"Modelos.{model_name}")
            model_class = getattr(modulo, model_config['class_name'])
            model_config_completo = {'past_frames': PASSOS_JANELA_ENTRADA, 'future_frames': PASSOS_A_PREVER, **model_config}

            previsoes_norm_teste = train.executar_treino_e_previsao(
                model_class=model_class, model_config=model_config_completo,
                X_treino=X_treino_final_norm, y_treino=y_treino_final_norm,
                X_pred=X_teste_norm, device=device
            )
            previsoes_reais_teste = utils.desnormalizar_dados_grade(previsoes_norm_teste, scaler_final)
        
        metricas_teste_final = utils.calcular_metricas(y_teste_real, previsoes_reais_teste)
        print(f"\n--- MÉTRICAS FINAIS DE {model_name.upper()} NO CONJunto DE TESTE ---")
        print(metricas_teste_final)
        utils.logar_metricas_mlflow({f"teste_{k}": v for k, v in metricas_teste_final.items()})
        
        # <<< MUDANÇA 1: Retorna as previsões junto com as métricas >>>
        return metricas_teste_final, previsoes_reais_teste

    except Exception as e:
        print(f"\nERRO: A execução do modelo {model_name} falhou. {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        if mlflow.active_run(): mlflow.end_run()
        print(f"\n{'='*60}\n== PROCESSO PARA {model_name.upper()} FINALIZADO ==\n{'='*60}")


# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    utils.print_timestamp("Iniciando a execução do pipeline principal...")
    print("="*80)
    
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
    print("--- Pré-processamento: Carregando GRADE espaço-temporal para modelos DL ---")
    dados_grade_completos = pre.preparar_dados_grade(caminho_arquivo_nc='pre/vento turco.nc')
    
    if dados_grade_completos is not None:
        print("Grade ws100 calculada e processada com sucesso.")
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
        
        # <<< MUDANÇA 2: Cria dicionários para armazenar todos os resultados >>>
        resultados_finais = []
        previsoes_finais = {}

        if not modelos_a_executar:
            print("Nenhum modelo ativo no 'modelos.json'.")
        else:
            print(f"Modelos a serem executados: {list(modelos_a_executar.keys())}")
            for nome_modelo, config_modelo in modelos_a_executar.items():
                # <<< MUDANÇA 3: Captura tanto as métricas quanto as previsões >>>
                metricas, previsoes = executar_modelo(nome_modelo, config_modelo, dados_treino_cv, dados_teste)
                if metricas and previsoes is not None:
                    resultados_finais.append({'modelo': nome_modelo, **metricas})
                    previsoes_finais[nome_modelo] = previsoes
        
        # --- 4. GERAÇÃO DO RELATÓRIO FINAL ---
        if resultados_finais:
            df_resultados = pd.DataFrame(resultados_finais).set_index('modelo')
            
            print("\n\n" + "="*80)
            print("== TABELA COMPARATIVA FINAL - RESULTADOS NO CONJUNTO DE TESTE ==")
            print("="*80)
            print(df_resultados.sort_values(by='rmse').to_string(float_format="%.5f"))
            print("="*80)

            # <<< MUDANÇA 4: Gera o y_teste_real e chama a nova função de relatório consolidado >>>
            dados_contexto = dados_treino_cv.isel(valid_time=slice(-PASSOS_JANELA_ENTRADA, None))
            dados_teste_com_contexto = xr.concat([dados_contexto, dados_teste], dim="valid_time")
            _, y_teste_real = utils.formatar_janelas_video(dados_teste_com_contexto, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)

            pos.gerar_relatorio_final_consolidado(
                df_resultados=df_resultados,
                dict_previsoes=previsoes_finais,
                y_real=y_teste_real,
                dados_grade_base=dados_grade_completos
            )
    else:
        print("Falha ao carregar os dados. Execução interrompida.")
    
    print("="*80)
    utils.print_timestamp("Execução do pipeline principal concluída.")
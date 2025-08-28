# main.py
# Versão Final - Orquestrador que executa múltiplos modelos ativos

import importlib
import warnings
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
import torch
import json

# Importa os módulos do nosso projeto
import utils
import train
from pre import pre

warnings.filterwarnings("ignore")

# ==============================================================================
# FUNÇÃO PARA CARREGAR CONFIGURAÇÕES
# ==============================================================================

def carregar_config_modelos(caminho_json: str) -> dict:
    """Carrega as configurações dos modelos a partir de um arquivo JSON."""
    try:
        with open(caminho_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if '1ARIMA' in config and 'ordem' in config['1ARIMA']:
            config['1ARIMA']['ordem'] = tuple(config['1ARIMA']['ordem'])
        return config
    except FileNotFoundError:
        print(f"ERRO: Arquivo de configuração '{caminho_json}' não encontrado.")
        return {}
    except json.JSONDecodeError:
        print(f"ERRO: O arquivo '{caminho_json}' não é um JSON válido.")
        return {}

# ==============================================================================
# PAINEL DE CONTROLE E CONFIGURAÇÕES GERAIS
# ==============================================================================

# --- 1. Definições da Série Temporal ---
FREQUENCIA_DADOS_HORAS = 3
JANELA_ENTRADA_HORAS = 24
HORIZONTE_PREVISAO_HORAS = 6

# --- 2. Conversão de Horas para Passos de Tempo ---
PASSOS_JANELA_ENTRADA = JANELA_ENTRADA_HORAS // FREQUENCIA_DADOS_HORAS
PASSOS_A_PREVER = HORIZONTE_PREVISAO_HORAS // FREQUENCIA_DADOS_HORAS

# --- 3. Carregamento das Configurações ---
CONFIG_MODELOS = carregar_config_modelos('modelos.json')

# ==============================================================================
# FUNÇÕES DE EXECUÇÃO
# ==============================================================================

def executar_baseline_arima(model_name: str, model_config: dict):
    """Executa um fluxo simples de treino/teste para o ARIMA como baseline."""
    print("="*60)
    print(f"== EXECUTANDO BASELINE {model_name} ==")
    print("="*60)
    
    run_name = f"Baseline_{model_name}"
    utils.iniciar_experimento_mlflow("Previsao_Vento_Turco_Baseline", run_name)
    try:
        utils.logar_parametros_mlflow(model_config)
        
        todos_os_dados = pre.preparar_dados_ponto_unico(caminho_arquivo_nc='pre/vento turco.nc', lat=37.25, lon=26.00)
        
        dados_treino = todos_os_dados[todos_os_dados.index.month < 11]
        dados_teste = todos_os_dados[todos_os_dados.index.month >= 11]
        print(f"Dados de treino: {len(dados_treino)}. Dados de teste: {len(dados_teste)}.")
        mlflow.log_param("mes_inicio_teste", 11)

        modulo_arima = importlib.import_module("Modelos.1ARIMA")
        previsoes = modulo_arima.executar_modelo_cv_arima(
            dados_treino=dados_treino,
            passos_previsao=len(dados_teste),
            ordem=model_config['ordem']
        )
        y_reais = dados_teste['velocidade_do_vento'].values[:len(previsoes)]
        
        metricas_finais = utils.calcular_metricas(y_reais, previsoes)
        print(f"\n--- MÉTRICAS FINAIS DO BASELINE {model_name} ---"); print(metricas_finais)
        utils.logar_metricas_mlflow(metricas_finais)

    except Exception as e:
        print(f"\nERRO: A execução do baseline {model_name} falhou. {e}")
    finally:
        mlflow.end_run()
        print(f"\n{'='*60}\n== EXECUÇÃO DO {model_name} FINALIZADA ==\n{'='*60}")


def executar_cv_e_teste_dl(model_name: str, model_config: dict):
    """Executa o fluxo completo de Validação Cruzada e Teste para modelos de Deep Learning."""
    print("="*60)
    print(f"== EXECUTANDO FLUXO COMPLETO PARA {model_name} ==")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name_pai = f"Final_{model_name}_prev{HORIZONTE_PREVISAO_HORAS}h"
    utils.iniciar_experimento_mlflow("Previsao_Vento_Turco_DL_Completo", run_name_pai)
    
    try:
        parametros_gerais = {"modelo": model_name, "passos_janela": PASSOS_JANELA_ENTRADA, "passos_previsao": PASSOS_A_PREVER}
        utils.logar_parametros_mlflow(parametros_gerais)
        utils.logar_parametros_mlflow(model_config)

        dados_grade = pre.preparar_dados_grade(caminho_arquivo_nc='pre/vento turco.nc')
        if dados_grade is None: raise ValueError("Falha ao carregar dados da grade.")

        dados_para_cv = dados_grade.where(dados_grade.valid_time.dt.month < 11, drop=True)
        dados_teste_final = dados_grade.where(dados_grade.valid_time.dt.month >= 11, drop=True)
        print(f"Dados carregados: {len(dados_para_cv.valid_time)} timesteps para CV, {len(dados_teste_final.valid_time)} para Teste Final.")

        metricas_por_fold = []
        for mes_validacao in range(10, 11): # Valida de Fev (2) a Out (10)
            print(f"\n--- {model_name} | FOLD CV (Mês de Validação: {mes_validacao}) ---")
            with mlflow.start_run(run_name=f"CV_Fold_Val_Mes_{mes_validacao}", nested=True):
                dados_treino_fold = dados_para_cv.where(dados_para_cv.valid_time.dt.month < mes_validacao, drop=True)
                dados_validacao_fold = dados_para_cv.where(dados_para_cv.valid_time.dt.month == mes_validacao, drop=True)
                X_treino, y_treino = utils.formatar_janelas_video(dados_treino_fold, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
                X_val, y_val_real = utils.formatar_janelas_video(dados_validacao_fold, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
                if X_treino.shape[0] == 0 or X_val.shape[0] == 0: continue
                
                scaler = utils.criar_e_treinar_scaler_grade(X_treino)
                X_treino_norm = utils.aplicar_scaler_grade(X_treino, scaler)
                y_treino_norm = utils.aplicar_scaler_grade(y_treino, scaler)
                X_val_norm = utils.aplicar_scaler_grade(X_val, scaler)

                modulo = importlib.import_module(f"Modelos.{model_name}")
                model_class = getattr(modulo, model_config['class_name'])
                model_params = {'past_frames': PASSOS_JANELA_ENTRADA, 'future_frames': PASSOS_A_PREVER}
                
                previsoes_norm = train.executar_treino_e_previsao(
                    model_class, model_params,
                    X_treino=X_treino_norm, y_treino=y_treino_norm,
                    X_pred=X_val_norm,
                    epocas=model_config['epocas'],
                    batch_size=model_config['batch_size'],
                    learning_rate=model_config['learning_rate'],
                    device=device,
                    X_val=X_val_norm, y_val=utils.aplicar_scaler_grade(y_val_real, scaler)
                )
                
                previsoes_reais = utils.desnormalizar_dados_grade(previsoes_norm, scaler)
                metricas_fold = utils.calcular_metricas(y_val_real, previsoes_reais)
                utils.logar_metricas_mlflow(metricas_fold)
                metricas_por_fold.append(metricas_fold)

        print(f"\n--- {model_name} | TREINO FINAL E AVALIAÇÃO EM TESTE ---")
        X_treino_final, y_treino_final = utils.formatar_janelas_video(dados_para_cv, PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER)
        X_teste, y_teste_real = utils.formatar_janelas_video(
            xr.concat([dados_para_cv.isel(valid_time=slice(-PASSOS_JANELA_ENTRADA, None)), dados_teste_final], dim="valid_time"),
            PASSOS_JANELA_ENTRADA, PASSOS_A_PREVER
        )
        scaler_final = utils.criar_e_treinar_scaler_grade(X_treino_final)
        X_treino_final_norm = utils.aplicar_scaler_grade(X_treino_final, scaler_final)
        y_treino_final_norm = utils.aplicar_scaler_grade(y_treino_final, scaler_final)
        X_teste_norm = utils.aplicar_scaler_grade(X_teste, scaler_final)
        
        modulo = importlib.import_module(f"Modelos.{model_name}")
        model_class = getattr(modulo, model_config['class_name'])
        model_params = {'past_frames': PASSOS_JANELA_ENTRADA, 'future_frames': PASSOS_A_PREVER}
        previsoes_norm_teste = train.executar_treino_e_previsao(
            model_class, model_params,
            X_treino=X_treino_final_norm, y_treino=y_treino_final_norm,
            X_pred=X_teste_norm,
            epocas=model_config['epocas'],
            batch_size=model_config['batch_size'],
            learning_rate=model_config['learning_rate'],
            device=device,
        )
        previsoes_reais_teste = utils.desnormalizar_dados_grade(previsoes_norm_teste, scaler_final)
        metricas_teste_final = utils.calcular_metricas(y_teste_real, previsoes_reais_teste)
        print(f"\n--- MÉTRICAS FINAIS DE {model_name} NO CONJUNTO DE TESTE ---")
        print(metricas_teste_final)
        utils.logar_metricas_mlflow({f"teste_{k}": v for k, v in metricas_teste_final.items()})

    except Exception as e:
        print(f"\nERRO: A execução do modelo {model_name} falhou. {e}")
        import traceback
        traceback.print_exc()
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        print(f"\n{'='*60}\n== PROCESSO PARA {model_name} FINALIZADO ==\n{'='*60}")

# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    
    modelos_a_executar = {
        nome: config for nome, config in CONFIG_MODELOS.items() if config.get("ativo", False)
    }

    if not modelos_a_executar:
        print("Nenhum modelo está marcado como 'ativo' no arquivo 'modelos.json'. Finalizando.")
    else:
        print(f"Modelos a serem executados: {list(modelos_a_executar.keys())}")
        
        for nome_modelo, config_modelo in modelos_a_executar.items():
            try:
                if nome_modelo == '1ARIMA':
                    executar_baseline_arima(nome_modelo, config_modelo)
                else:
                    importlib.import_module(f"Modelos.{nome_modelo}")
                    getattr(importlib.import_module(f"Modelos.{nome_modelo}"), config_modelo['class_name'])
                    executar_cv_e_teste_dl(nome_modelo, config_modelo)
            
            except ImportError:
                print(f"ERRO FATAL: O arquivo do modelo 'Modelos/{nome_modelo}.py' não foi encontrado. Pulando este modelo.")
            except AttributeError:
                print(f"ERRO FATAL: A classe '{config_modelo['class_name']}' não foi encontrada em 'Modelos/{nome_modelo}.py'. Pulando este modelo.")
            except Exception as e:
                print(f"ERRO FATAL inesperado ao tentar executar {nome_modelo}: {e}. Pulando este modelo.")
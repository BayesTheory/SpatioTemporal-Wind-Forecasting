# train.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

# Funções de métricas que operam diretamente em tensores PyTorch para eficiência
def rmse_torch(y_true, y_pred):
    """Calcula o RMSE para tensores PyTorch."""
    return torch.sqrt(torch.mean((y_true - y_pred)**2))

def mae_torch(y_true, y_pred):
    """Calcula o MAE para tensores PyTorch."""
    return torch.mean(torch.abs(y_true - y_pred))


def executar_treino_e_previsao(
    model_class,
    model_params,
    X_treino,
    y_treino,
    X_pred,
    epocas,
    batch_size,
    learning_rate,
    device,
    X_val=None,
    y_val=None,
    patience=10
):
    """
    Função genérica para treinar um modelo PyTorch e fazer previsões,
    com logging de métricas e Early Stopping.
    """
    # --- 1. Preparação ---
    modelo = model_class(**model_params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

    X_treino_tensor = torch.from_numpy(X_treino).float()
    y_treino_tensor = torch.from_numpy(y_treino).float()
    X_pred_tensor = torch.from_numpy(X_pred).float()

    train_dataset = TensorDataset(X_treino_tensor, y_treino_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).float().to(device)
    
    # Variáveis para controlar o Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"  Iniciando treino por até {epocas} épocas (Patience: {patience}) no dispositivo: {device.type.upper()}")
    print("  Métricas (RMSE, MAE) são exibidas nos dados normalizados (escala 0-1).")
    start_time = time.time()
    
    # --- 2. Loop de Treino ---
    for epoch in range(epocas):
        modelo.train()
        
        epoch_loss, epoch_train_rmse, epoch_train_mae = 0.0, 0.0, 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = modelo(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            with torch.no_grad():
                epoch_train_rmse += rmse_torch(y_batch, outputs).item()
                epoch_train_mae += mae_torch(y_batch, outputs).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_rmse = epoch_train_rmse / len(train_loader)
        avg_train_mae = epoch_train_mae / len(train_loader)

        # Lógica de validação e Early Stopping
        val_metrics_str = ""
        if X_val is not None and y_val is not None:
            modelo.eval()
            with torch.no_grad():
                val_outputs = modelo(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_rmse = rmse_torch(y_val_tensor, val_outputs).item()
                val_mae = mae_torch(y_val_tensor, val_outputs).item()
                val_metrics_str = f"| Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} | Val MAE: {val_mae:.5f}"
            
            # Lógica do Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = modelo.state_dict().copy() # Salva o estado do melhor modelo
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"    --- Parada Antecipada na Época {epoch+1}! A perda de validação não melhora há {patience} épocas. ---")
                break # Sai do loop de treino

        if (epoch + 1) % 10 == 0 or epoch == epocas - 1:
             print(f"    Época [{epoch+1:>{len(str(epocas))}}/{epocas}] | Train Loss: {avg_train_loss:.5f} {val_metrics_str}")

    end_time = time.time()
    print(f"  Treino concluído em {end_time - start_time:.2f} segundos.")

    # Carrega o melhor modelo salvo pelo Early Stopping antes da previsão
    if best_model_state:
        print("  Carregando o melhor estado do modelo encontrado durante a validação.")
        modelo.load_state_dict(best_model_state)

    # --- 3. Previsão ---
    print("  Iniciando previsão com o modelo final...")
    modelo.eval()
    all_predictions = []
    pred_dataset = TensorDataset(X_pred_tensor)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size*2, shuffle=False)
    
    with torch.no_grad():
        for X_batch in pred_loader:
            predictions_batch = modelo(X_batch[0].to(device))
            all_predictions.append(predictions_batch.cpu().numpy())

    final_predictions = np.concatenate(all_predictions, axis=0)
    print("  Previsão concluída.")
    
    return final_predictions
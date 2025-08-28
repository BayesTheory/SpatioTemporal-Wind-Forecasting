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
    y_val=None
):
    """
    Função genérica para treinar um modelo PyTorch e fazer previsões,
    com logging de métricas (RMSE, MAE) por época.
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


    # --- 2. Loop de Treino ---
    print(f"  Iniciando treino por {epocas} épocas no dispositivo: {device.type.upper()}")
    print("  Métricas (RMSE, MAE) são exibidas nos dados normalizados (escala 0-1).")
    start_time = time.time()
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

        val_metrics_str = ""
        if X_val is not None and y_val is not None:
            modelo.eval()
            with torch.no_grad():
                val_outputs = modelo(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_rmse = rmse_torch(y_val_tensor, val_outputs).item()
                val_mae = mae_torch(y_val_tensor, val_outputs).item()
                val_metrics_str = f"| Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} | Val MAE: {val_mae:.5f}"
            modelo.train()
        
        if (epoch + 1) % 10 == 0 or epoch == epocas - 1:
             print(f"    Época [{epoch+1:>{len(str(epocas))}}/{epocas}] "
                   f"| Train Loss: {avg_train_loss:.5f} "
                   f"| Train RMSE: {avg_train_rmse:.5f} "
                   f"| Train MAE: {avg_train_mae:.5f} "
                   f"{val_metrics_str}")

    end_time = time.time()
    print(f"  Treino concluído em {end_time - start_time:.2f} segundos.")


    # --- 3. Previsão ---
    print("  Iniciando previsão...")
    modelo.eval()
    
    all_predictions = []
    pred_dataset = TensorDataset(X_pred_tensor)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size*2, shuffle=False)
    
    with torch.no_grad():
        for X_batch in pred_loader:
            X_batch = X_batch[0].to(device)
            predictions_batch = modelo(X_batch)
            all_predictions.append(predictions_batch.cpu().numpy())

    final_predictions = np.concatenate(all_predictions, axis=0)
    print("  Previsão concluída.")
    
    return final_predictions
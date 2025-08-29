# train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import time

def rmse_torch(y_true, y_pred):
    """Calcula o RMSE para tensores PyTorch."""
    return torch.sqrt(torch.mean((y_true - y_pred)**2))

def mae_torch(y_true, y_pred):
    """Calcula o MAE para tensores PyTorch."""
    return torch.mean(torch.abs(y_true - y_pred))

def executar_treino_e_previsao(
    model_class, model_params,
    X_treino, y_treino, X_pred,
    epocas, batch_size, learning_rate, device,
    X_val=None, y_val=None,
    patience=10,
    scheduler_patience=5, scheduler_factor=0.5):

    modelo = model_class(**model_params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        patience=scheduler_patience,
        factor=scheduler_factor,
        verbose=True
    )

    X_treino_tensor = torch.from_numpy(X_treino).float()
    y_treino_tensor = torch.from_numpy(y_treino).float()
    X_pred_tensor = torch.from_numpy(X_pred).float()
    train_dataset = TensorDataset(X_treino_tensor, y_treino_tensor)

    if X_val is None or y_val is None:
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        indices = list(range(len(train_dataset)))
        train_subset = Subset(train_dataset, indices[:train_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        X_val_tensor = X_treino_tensor[indices[train_size:]].to(device)
        y_val_tensor = y_treino_tensor[indices[train_size:]].to(device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"  Iniciando treino por até {epocas} épocas (Patience ES: {patience}, Patience LR: {scheduler_patience}) no dispositivo: {device.type.upper()}")
    start_time = time.time()
    
    for epoch in range(epocas):
        modelo.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            # ESTA É A LINHA CORRETA
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = modelo(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        modelo.eval()
        with torch.no_grad():
            val_outputs = modelo(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_rmse = rmse_torch(y_val_tensor, val_outputs).item()
            val_mae = mae_torch(y_val_tensor, val_outputs).item()
            val_metrics_str = f"| Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} | Val MAE: {val_mae:.5f}"

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = modelo.state_dict().copy()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"    --- Parada Antecipada na Época {epoch+1}! A perda de validação não melhora há {patience} épocas. ---")
            break

        if (epoch + 1) % 10 == 0 or epoch == epocas - 1:
             print(f"    Época [{epoch+1:>{len(str(epocas))}}/{epocas}] | Train Loss: {avg_train_loss:.5f} {val_metrics_str}")

    end_time = time.time()
    print(f"  Treino concluído em {end_time - start_time:.2f} segundos.")

    if best_model_state:
        print("  Carregando o melhor estado do modelo encontrado durante a validação.")
        modelo.load_state_dict(best_model_state)

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
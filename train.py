# train.py (Versão 3.0 - Estado-da-Arte Flexível)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
    patience=15,
    # Parâmetros para o scheduler Plateau (padrão)
    scheduler_patience=7, scheduler_factor=0.5,
    # Parâmetros para o modo avançado
    optimizer_name='adam',
    weight_decay=0,
    scheduler_name='plateau',
    warmup_epochs=0
):
    """
    Função de treino avançada com otimizadores e schedulers configuráveis.
    Por padrão, usa Adam + ReduceLROnPlateau. Se especificado no JSON,
    pode usar AdamW + Cosine Annealing + Warmup.
    """
    modelo = model_class(**model_params).to(device)
    criterion = nn.MSELoss()

    # --- 1. Preparação do Otimizador e Scheduler ---
    
    # Otimizador Configurável (Adam vs AdamW)
    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else: # Padrão é Adam
        optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

    # Scheduler Configurável (Plateau vs Cosine)
    if scheduler_name.lower() == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epocas - warmup_epochs if epocas > warmup_epochs else epocas)
    else: # Padrão é ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=scheduler_factor, verbose=True)

    # Lógica de validação automática
    train_dataset = TensorDataset(torch.from_numpy(X_treino).float(), torch.from_numpy(y_treino).float())
    if X_val is None or y_val is None:
        print("  Aviso: Nenhum conjunto de validação fornecido. Separando os últimos 10% do treino para Early Stopping.")
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        indices = list(range(len(train_dataset)))
        train_subset = Subset(train_dataset, indices[:train_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        X_val_tensor = train_dataset.tensors[0][indices[train_size:]].to(device)
        y_val_tensor = train_dataset.tensors[1][indices[train_size:]].to(device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).float().to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    print(f"  Iniciando treino (Otimizador: {optimizer_name.upper()}, Scheduler: {scheduler_name.upper()})...")
    start_time = time.time()
    
    # --- 2. Loop de Treino ---
    for epoch in range(epocas):
        modelo.train()
        epoch_loss = 0.0
        
        # Lógica de Warmup
        if epoch < warmup_epochs and scheduler_name.lower() == 'cosine':
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_scale
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = modelo(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Avaliação da época
        modelo.eval()
        with torch.no_grad():
            val_outputs = modelo(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_rmse = rmse_torch(y_val_tensor, val_outputs).item()
            val_mae = mae_torch(y_val_tensor, val_outputs).item()
            val_metrics_str = f"| Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} | Val MAE: {val_mae:.5f}"
        
        # Passo do scheduler
        if scheduler_name.lower() == 'cosine':
            if epoch >= warmup_epochs:
                scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Lógica do Early Stopping
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Época [{epoch+1:>{len(str(epocas))}}/{epocas}] | Train Loss: {avg_train_loss:.5f} {val_metrics_str} | LR: {current_lr:.1e}")

    end_time = time.time()
    print(f"  Treino concluído em {end_time - start_time:.2f} segundos.")

    if best_model_state:
        print("  Carregando o melhor estado do modelo encontrado durante a validação.")
        modelo.load_state_dict(best_model_state)

    # --- 3. Previsão ---
    print("  Iniciando previsão com o modelo final...")
    modelo.eval()
    all_predictions = []
    pred_dataset = TensorDataset(torch.from_numpy(X_pred).float())
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size*2, shuffle=False)
    
    with torch.no_grad():
        for X_batch in pred_loader:
            predictions_batch = modelo(X_batch[0].to(device))
            all_predictions.append(predictions_batch.cpu().numpy())

    final_predictions = np.concatenate(all_predictions, axis=0)
    print("  Previsão concluída.")
    
    return final_predictions
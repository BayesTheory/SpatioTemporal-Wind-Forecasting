import os
import time
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

# =========================
# MÉTRICAS EM TORCH
# =========================
def rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calcula o Root Mean Squared Error."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mae_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calcula o Mean Absolute Error."""
    return torch.mean(torch.abs(y_true - y_pred))

# =========================
# OTIMIZADOR & SCHEDULER
# =========================
def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Cria um otimizador com base na configuração."""
    lr = float(config.get('learning_rate', 1e-3))
    wd = float(config.get('weight_decay', 0.0))
    name = str(config.get('optimizer', 'adam')).lower()
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], epochs: int):
    """Cria um scheduler de learning rate com base na configuração."""
    sched = str(config.get('scheduler', 'plateau')).lower()
    warm = int(config.get('warmup_epochs', 0))
    if sched == 'cosine':
        t_max = epochs - warm if epochs > warm else epochs
        return CosineAnnealingLR(optimizer, T_max=t_max)
    patience = int(config.get('scheduler_patience', 10))
    factor = float(config.get('scheduler_factor', 0.5))
    return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, verbose=False)

# =========================
# SALVAMENTO LOCAL
# =========================
def _save_checkpoint_local(path: str,
                           model: nn.Module,
                           optimizer: torch.optim.Optimizer,
                           scheduler,
                           scaler: Optional[GradScaler],
                           model_config: Dict[str, Any],
                           best_val_loss: float,
                           epoch: int,
                           state_override: Optional[Dict[str, torch.Tensor]] = None):
    """Salva um checkpoint do modelo e estado do treino localmente."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = state_override if state_override is not None else model.state_dict()
    ckpt = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": getattr(scheduler, "state_dict", lambda: {})(),
        "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
        "model_config": model_config,
    }
    torch.save(ckpt, path)

# =========================
# LOOP DE TREINO GENÉRICO
# =========================
def executar_treino_e_previsao(model_class,
                               model_config: Dict[str, Any],
                               X_treino: np.ndarray,
                               y_treino: np.ndarray,
                               X_pred: np.ndarray,
                               device: torch.device,
                               X_val: Optional[np.ndarray] = None,
                               y_val: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Motor de treino com AMP, grad clipping, acumulação de gradiente, salvamento local (best/last) e previsão.
    Retorna as previsões (np.ndarray).
    """
    # Configurações
    epocas = int(model_config.get('epocas', 250))
    batch_size = int(model_config.get('batch_size', 16))
    patience = int(model_config.get('patience', 15))
    warmup_epochs = int(model_config.get('warmup_epochs', 0))
    target_lr = float(model_config.get('learning_rate', 1e-3))
    num_workers = int(model_config.get('num_workers', 2))
    grad_clip = float(model_config.get('grad_clip', 0.0))
    grad_accum_steps = int(model_config.get('grad_accum_steps', 1))
    drop_last = bool(model_config.get('drop_last', False))
    use_amp = bool(model_config.get('amp', True) and device.type == 'cuda')

    # Pastas de saída
    model_name_for_files = str(model_config.get('class_name', 'model'))
    models_out_dir = str(model_config.get('models_out_dir', 'modelos_salvos'))
    save_dir = os.path.join(models_out_dir, model_name_for_files)
    os.makedirs(save_dir, exist_ok=True)

    # Modelo e otimizadores
    modelo = model_class(**model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(modelo, model_config)
    scheduler = get_scheduler(optimizer, model_config, epocas)
    scaler = GradScaler(enabled=use_amp)

    # DataLoaders
    pin = (device.type == 'cuda')
    train_dataset = TensorDataset(torch.from_numpy(X_treino).float(),
                                  torch.from_numpy(y_treino).float())
    
    # Lógica de Validação e Criação dos DataLoaders
    if X_val is None or y_val is None:
        print("  Aviso: Nenhum conjunto de validação fornecido. Separando os últimos 10% do treino.")
        val_fraction = 0.1
        num_samples = len(train_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * (1 - val_fraction))
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin,
                                  persistent_workers=(num_workers > 0), drop_last=drop_last)
                                  
        val_loader = DataLoader(val_subset, batch_size=batch_size * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=pin,
                                persistent_workers=(num_workers > 0))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin,
                                  persistent_workers=(num_workers > 0), drop_last=drop_last)
        
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False,
                                num_workers=num_workers, pin_memory=pin,
                                persistent_workers=(num_workers > 0))

    # Estado de melhor
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    # Histórico
    hist_train, hist_val, hist_rmse, hist_mae, hist_lr = [], [], [], [], []

    print(f"  Iniciando treino (Opt: {model_config.get('optimizer','adam').upper()}, "
          f"Sched: {model_config.get('scheduler','plateau').upper()}, AMP: {use_amp})...")
    start_time = time.time()

    for epoch in range(epocas):
        modelo.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_train_loss = 0.0

        # Warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr_scale = (epoch + 1) / max(1, warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = target_lr * lr_scale

        # Batches de Treino
        for bidx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = modelo(X_batch)
                loss = criterion(outputs, y_batch)

            # Acumulação de Gradiente
            loss_to_scale = loss / max(1, grad_accum_steps)
            scaler.scale(loss_to_scale).backward()

            ready_step = ((bidx + 1) % grad_accum_steps == 0) or (bidx == len(train_loader) - 1)
            if ready_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(modelo.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))

        # Loop de Validação
        modelo.eval()
        val_loss, val_rmse, val_mae = 0.0, 0.0, 0.0
        with torch.no_grad(), autocast(enabled=use_amp):
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device, non_blocking=True)
                y_val_batch = y_val_batch.to(device, non_blocking=True)
                
                val_outputs = modelo(X_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()
                val_rmse += rmse_torch(y_val_batch, val_outputs).item()
                val_mae += mae_torch(y_val_batch, val_outputs).item()

        # Calcula a média das métricas de validação
        num_val_batches = max(1, len(val_loader))
        val_loss /= num_val_batches
        val_rmse /= num_val_batches
        val_mae /= num_val_batches

        # Scheduler
        if epoch >= warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Melhor estado (salva best imediatamente)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            _save_checkpoint_local(os.path.join(save_dir, "best.pth"),
                                   modelo, optimizer, scheduler, scaler,
                                   model_config, best_val_loss, epoch + 1,
                                   state_override=best_model_state)
        else:
            epochs_no_improve += 1

        # LR atual
        current_lr = optimizer.param_groups[0]['lr']
        hist_train.append(avg_train_loss)
        hist_val.append(val_loss)
        hist_rmse.append(val_rmse)
        hist_mae.append(val_mae)
        hist_lr.append(current_lr)

        if ((epoch + 1) % 10 == 0) or (epoch == epocas - 1) or (epoch < warmup_epochs):
            patience_str = f"Patience: {epochs_no_improve}/{patience}"
            print(f"    Época [{epoch+1}/{epocas}] | Train Loss: {avg_train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} | "
                  f"LR: {current_lr:.1e} | {patience_str}")

        # Parada antecipada
        if epochs_no_improve >= patience:
            print(f"    --- Parada Antecipada na Época {epoch+1}! "
                  f"A perda de validação não melhora há {patience} épocas. ---")
            break

    dur = time.time() - start_time
    print(f"  Treino concluído em {dur:.2f} segundos.")

    # Carrega o melhor para inferência e salva "last"
    if best_model_state is not None:
        print("  Carregando o melhor estado do modelo encontrado durante a validação.")
        modelo.load_state_dict(best_model_state)

    _save_checkpoint_local(os.path.join(save_dir, "last.pth"),
                           modelo, optimizer, scheduler, scaler,
                           model_config, best_val_loss, epoch + 1, state_override=None)

    # Histórico CSV (na mesma pasta)
    try:
        hist_df = pd.DataFrame({
            "epoch": list(range(1, len(hist_train) + 1)),
            "train_loss": hist_train,
            "val_loss": hist_val,
            "val_rmse": hist_rmse,
            "val_mae": hist_mae,
            "lr": hist_lr,
        })
        hist_df.to_csv(os.path.join(save_dir, f"{model_name_for_files}_history.csv"), index=False)
    except Exception:
        pass

    # Previsão final
    print("  Iniciando previsão com o modelo final...")
    modelo.eval()
    all_predictions = []
    pred_dataset = TensorDataset(torch.from_numpy(X_pred).float())
    pred_loader = DataLoader(pred_dataset,
                             batch_size=batch_size * 2,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin,
                             persistent_workers=(num_workers > 0))
    with torch.no_grad(), autocast(enabled=use_amp):
        for (X_batch,) in pred_loader:
            preds = modelo(X_batch.to(device, non_blocking=True))
            all_predictions.append(preds.cpu().numpy())
    final_predictions = np.concatenate(all_predictions, axis=0)
    print("  Previsão concluída.")

    return final_predictions
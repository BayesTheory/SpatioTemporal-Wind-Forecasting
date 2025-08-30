# train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time

def rmse_torch(y_true, y_pred):
    """Calcula o RMSE para tensores PyTorch."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mae_torch(y_true, y_pred):
    """Calcula o MAE para tensores PyTorch."""
    return torch.mean(torch.abs(y_true - y_pred))

def get_optimizer(model, config):
    """Cria o otimizador com base na configuração do JSON."""
    lr = config.get('learning_rate', 0.001)
    wd = config.get('weight_decay', 0.0)
    optimizer_name = config.get('optimizer', 'adam').lower()
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr)

def get_scheduler(optimizer, config, epochs):
    """Cria o scheduler com base na configuração do JSON."""
    scheduler_name = config.get('scheduler', 'plateau').lower()
    warmup_epochs = config.get('warmup_epochs', 0)

    if scheduler_name == 'cosine':
        t_max = epochs - warmup_epochs if epochs > warmup_epochs else epochs
        return CosineAnnealingLR(optimizer, T_max=t_max)
    patience = config.get('scheduler_patience', 10)
    factor = config.get('scheduler_factor', 0.5)
    return ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, verbose=False)

def executar_treino_e_previsao(model_class, model_config,
                                 X_treino, y_treino, X_pred, device,
                                 X_val=None, y_val=None):
    """
    Motor de treino genérico com AMP, grad clipping e snapshot seguro do melhor modelo.
    """
    # 1) Configuração e objetos
    epocas = int(model_config.get('epocas', 250))
    batch_size = int(model_config.get('batch_size', 16))
    patience = int(model_config.get('patience', 15))
    warmup_epochs = int(model_config.get('warmup_epochs', 0))
    target_lr = float(model_config.get('learning_rate', 0.001))
    num_workers = int(model_config.get('num_workers', 2))
    use_amp = bool(model_config.get('amp', True) and device.type == 'cuda')
    grad_clip = float(model_config.get('grad_clip', 0.0))

    modelo = model_class(**model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(modelo, model_config)
    scheduler = get_scheduler(optimizer, model_config, epocas)
    scaler = GradScaler(enabled=use_amp)

    # 2) DataLoaders otimizados
    pin = (device.type == 'cuda')
    train_dataset = TensorDataset(torch.from_numpy(X_treino).float(),
                                  torch.from_numpy(y_treino).float())

    if X_val is None or y_val is None:
        print("  Aviso: Nenhum conjunto de validação fornecido. Separando os últimos 10% do treino.")
        val_size = max(1, int(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - val_size
        indices = list(range(len(train_dataset)))

        train_loader = DataLoader(
            Subset(train_dataset, indices[:train_size]),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin
        )
        X_val_tensor = train_dataset.tensors[0][indices[train_size:]].to(device, non_blocking=True)
        # <<< CORREÇÃO 2: O índice correto para y_val_tensor é 1, não 12. >>>
        y_val_tensor = train_dataset.tensors[1][indices[train_size:]].to(device, non_blocking=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin
        )
        X_val_tensor = torch.from_numpy(X_val).float().to(device, non_blocking=True)
        y_val_tensor = torch.from_numpy(y_val).float().to(device, non_blocking=True)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    print(f"  Iniciando treino (Opt: {model_config.get('optimizer','adam').upper()}, "
          f"Sched: {model_config.get('scheduler','plateau').upper()}, AMP: {use_amp})...")
    start_time = time.time()

    # 3) Loop de treino
    for epoch in range(epocas):
        modelo.train()
        epoch_train_loss = 0.0

        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = target_lr * lr_scale

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = modelo(X_batch)
                loss = criterion(outputs, y_batch)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(modelo.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))

        # Validação
        modelo.eval()
        with torch.no_grad(), autocast(enabled=use_amp):
            val_outputs = modelo(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_rmse = rmse_torch(y_val_tensor, val_outputs).item()
            val_mae = mae_torch(y_val_tensor, val_outputs).item()

        if epoch >= warmup_epochs:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"    --- Parada Antecipada na Época {epoch+1}! A perda de validação não melhora há {patience} épocas. ---")
            break

        if (epoch + 1) % 10 == 0 or epoch == epocas - 1 or epoch < warmup_epochs:
            # <<< CORREÇÃO 1: Acessa o lr com o índice [0] >>>
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Época [{epoch+1}/{epocas}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} | LR: {current_lr:.1e}")

    dur = time.time() - start_time
    print(f"  Treino concluído em {dur:.2f} segundos.")

    if best_model_state is not None:
        print("  Carregando o melhor estado do modelo encontrado durante a validação.")
        modelo.load_state_dict(best_model_state)

    # 4) Previsão final
    print("  Iniciando previsão com o modelo final...")
    modelo.eval()
    all_predictions = []
    pred_dataset = TensorDataset(torch.from_numpy(X_pred).float())
    pred_loader = DataLoader(pred_dataset,
                             batch_size=batch_size * 2,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin)

    with torch.no_grad(), autocast(enabled=use_amp):
        for (X_batch,) in pred_loader:
            preds = modelo(X_batch.to(device, non_blocking=True))
            all_predictions.append(preds.cpu().numpy())

    final_predictions = np.concatenate(all_predictions, axis=0)
    print("  Previsão concluída.")
    return final_predictions
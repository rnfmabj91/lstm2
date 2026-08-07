"""
===========================================================================
 训练模块 v2 — 混合精度 + 梯度累积 + cudnn benchmark
===========================================================================
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from .config import cfg
from .model import active_region_mask

# CUDA 加速配置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train(model: nn.Module,
          X_train: torch.Tensor,
          X_val: torch.Tensor) -> dict:
    """
    训练 CNN-LSTM 自编码器 (仅正常数据).

    - 混合精度 (AMP) 加速 GPU 计算
    - 梯度累积 (accumulation_steps=4) 减少同步开销
    - cudnn benchmark 自动调优卷积算法
    """
    device = next(model.parameters()).device
    use_amp = (device.type == 'cuda')   # AMP 仅 CUDA 可用, CPU 下退化为 fp32
    print('\n' + '=' * 50)
    print('训练 (混合精度 + 梯度累积)')
    print('=' * 50)

    model_dir = os.path.join(cfg.save_dir, cfg.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    n = len(X_train)
    batch_sz = cfg.train.batch_size
    accum_steps = 4  # 梯度累积步数, 等效 batch = batch_sz × 4

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs
    )
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None  # 混合精度缩放器

    best_val = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    patience = cfg.train.early_stop_patience

    history = {
        'train_losses': [],
        'val_losses': [],
        'recon_losses': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
    }

    # loss_accum: GPU 端累加, 每轮只同步一次到 CPU
    #   loss_accum   — 完整训练损失 (重构MSE + λ_nc×NormErr, 用于反向)
    #   recon_accum  — 纯重构 masked-MSE (与验证损失同口径, 用于日志对比)
    loss_accum  = torch.zeros(1, device=device)
    recon_accum = torch.zeros(1, device=device)

    for epoch in range(1, cfg.train.epochs + 1):
        # --- 训练 (手动batch, 梯度累积) ---
        model.train()
        loss_accum.zero_()
        recon_accum.zero_()
        n_batch = 0

        indices = torch.randperm(n, device=device)
        optimizer.zero_grad()

        for i in range(0, n, batch_sz):
            bx = X_train[indices[i:i+batch_sz]]

            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    if hasattr(model, 'training_loss'):
                        # 重构 + 可选辅助对齐损失 + FNM正常性损失 (FSCA Context-Alignment)
                        loss, recon = model.training_loss(bx, criterion)
                    else:
                        recon = model(bx)
                        loss = criterion(recon, bx)
            else:
                if hasattr(model, 'training_loss'):
                    # 重构 + 可选辅助对齐损失 + FNM正常性损失 (FSCA Context-Alignment)
                    loss, recon = model.training_loss(bx, criterion)
                else:
                    recon = model(bx)
                    loss = criterion(recon, bx)

            # 纯重构 masked-MSE (与验证同口径; FNM 只进总损失反向, 不进此日志)
            with torch.no_grad():
                _mask = active_region_mask(bx)
                recon_accum += (((recon - bx) ** 2 * _mask).sum() / _mask.sum()).detach()

            # 梯度累积: 每 accum_steps 步更新一次参数
            scaled_loss = loss / accum_steps
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (i // batch_sz + 1) % accum_steps == 0 or (i + batch_sz) >= n:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            loss_accum += loss.detach()   # GPU 上累加, 不触发同步
            n_batch += 1

        avg_train   = (loss_accum / n_batch).item()    # 完整训练损失 (含FNM项)
        avg_recon   = (recon_accum / n_batch).item()   # 纯重构 masked-MSE
        history['train_losses'].append(avg_train)
        history['recon_losses'] = history.get('recon_losses', []) + [avg_recon]

        # --- 验证 (每轮都算, 确保早停信号准确) ---
        model.eval()
        with torch.no_grad():
            # 分批验证: 避免全量前向 OOM
            recon_parts = []
            for j in range(0, len(X_val), batch_sz):
                recon_parts.append(model(X_val[j:j + batch_sz]))
            recon_val = torch.cat(recon_parts, dim=0)
            # 与训练一致: 掩码重构损失 (排除尾部零填充, 消除长度伪影)
            mask_val = active_region_mask(X_val)
            val_loss = (((recon_val - X_val) ** 2 * mask_val).sum()
                        / mask_val.sum()).item()
        history['val_losses'].append(val_loss)

        # --- 保存最佳 ---
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(cfg.save_dir, cfg.model_dir, 'best_model.pt'),
            )
        else:
            early_stop_counter += 1

        # 每轮打印 (跟踪训练进度)
        print(f'  Epoch [{epoch:3d}/{cfg.train.epochs}] '
              f'Train总: {avg_train:.6f} (重构: {avg_recon:.6f}) | '
              f'Val: {val_loss:.6f} | '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')

        if early_stop_counter >= patience:
            print(f'  早停于 epoch {epoch}, 最佳 {best_val:.6f} (epoch {best_epoch})')
            break

        # 余弦退火: 每轮结束衰减学习率 (AdamW 基础 lr = 1e-3 → ~0, T_max=epochs)
        scheduler.step()

    history['best_val_loss'] = best_val
    history['best_epoch'] = best_epoch
    print(f'  训练完成! 最佳验证损失: {best_val:.6f} (epoch {best_epoch})')

    model.load_state_dict(
        torch.load(os.path.join(cfg.save_dir, cfg.model_dir, 'best_model.pt'),
                   weights_only=True)
    )
    return history

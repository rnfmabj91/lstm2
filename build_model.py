# -*- coding: utf-8 -*-
"""
lstm2 — 最终模型构建 (2026-08-07 确认架构: xalign + 检测侧仅正权5个剪枝)

镜像原项目 _train_xalign_blocking.py 的构造方式:
  时域通路 (5×ResConv, 3→256)  +  频域通路 (rFFT 保留相位, 6→256)
  → CrossAttentionAlign 双向交叉注意力对齐 (TimeCMA 思想, 默认开启)
  → 拼接融合 → BiLSTM → ConvDecoder → 重构 (B, 3, 800)

用法:
    from build_model import build_final_model
    model = build_final_model()
"""
import torch
from src.config import cfg
from src.model import CNNLSTM_Autoencoder


def build_final_model(device=None, xalign=True):
    """构建最终架构模型 (xalign 交叉注意力对齐默认开启)."""
    model = CNNLSTM_Autoencoder(
        in_channels=cfg.model.in_channels,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        dropout=cfg.model.dropout,
        freq_pathway=cfg.fusion.enabled,
        aux_features=cfg.model.aux_features,
        aux_hidden=cfg.model.aux_hidden,
        xalign_enabled=xalign,
        normalcy_enabled=cfg.model.normalcy_enabled,
        normalcy_bottleneck=cfg.model.normalcy_bottleneck,
        normalcy_weight=cfg.model.normalcy_weight,
        seq_len=cfg.data.total_pts,
        domain_phys=cfg.model.domain_phys,
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == '__main__':
    m = build_final_model()
    n = sum(p.numel() for p in m.parameters())
    n_xa = sum(p.numel() for p in m.xalign.parameters()) if m.xalign_enabled else 0
    print(f'xalign_enabled={m.xalign_enabled} | 总参数={n:,} '
          f'({n/1e6:.2f}M, 其中 CrossAttentionAlign +{n_xa:,})')
    x = torch.randn(2, 3, cfg.data.total_pts)
    with torch.no_grad():
        y = m(x)
    print(f'前向 OK | {tuple(x.shape)} → {tuple(y.shape)}')

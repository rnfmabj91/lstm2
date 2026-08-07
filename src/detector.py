"""
===========================================================================
 异常检测模块 v2 (相位感知)

 核心:
   anomaly_score = w_MSE × MSE × scale_mse
                 + α × PeakErr × scale_peak
                 + β × PhaseErr × scale_phase
                 + γ × SpectralErr × scale_spectral
                 + δ × NormErr × scale_normalcy

  各分量自动缩放到训练集上的中位数量级, 确保平衡贡献 (权重见 cfg.detect).
  PhaseErr 捕捉峰值时间偏移、启动能量畸变等相位结构异常.
  SpectralErr 捕捉频谱结构偏差 (v2频谱特征 z-score, 参考 FSCA 频域表示).
  NormErr 为可学习频域正常性模型 (FNM) 的谱重构误差.

 流程:
   1. 在训练集上计算各分量的缩放系数
   2. 计算训练/验证/测试三集异常分数
   3. 基于验证集分位数确定阈值
   4. 输出检测指标
===========================================================================
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn

from .config import cfg
from .early_warning import (WarningSystem, RelErrEarlyWarning,
                            plot_machine_overview,
                            WARNING_GREEN, WARNING_YELLOW,
                            WARNING_ORANGE, WARNING_RED, LEVEL_LABELS)
from .phase_features import auto_scale_phase, batch_phase_err
from .model import active_region_mask, _band_bins, _BAND_EDGES_HZ_V2


@torch.no_grad()
def compute_aux_features_v2(x: torch.Tensor, x_fft: torch.Tensor = None) -> torch.Tensor:
    """
    在线计算 v2 频谱特征 (48 维) — 检测侧 SpectralErr 用 (从模型侧迁移至此).

    原属于 model.py, 模型精简后移除, 检测侧频谱分量仍依赖它:
      fine_psd 24 (8细频带 log1p 功率) + shape 9 (质心/平坦度/滚降)
      + autocorr 6 + peak_amp/time/rms 9
    检测侧只取 [:33] (fine_psd + shape).
    """
    B, C, T = x.shape
    orig_dtype = x.dtype
    xf = x.float()  # AMP: half cuFFT 不支持非 2 的幂长度 800, 强制 float32
    eps = 1e-8
    n_freq = T // 2 + 1                              # 800→401

    if x_fft is None:
        x_fft = torch.fft.rfft(xf, dim=-1)
    power = x_fft.real ** 2 + x_fft.imag ** 2            # (B, 3, n_freq)

    band_edges = _band_bins(T, _BAND_EDGES_HZ_V2)
    if band_edges[-1] > n_freq:
        band_edges[-1] = n_freq
    fine_psd = torch.stack(
        [power[..., a:b].sum(-1) for a, b in zip(band_edges[:-1], band_edges[1:])],
        dim=-1,
    )                                                   # (B, 3, 8)
    fine_psd = torch.log1p(fine_psd)

    freqs = torch.arange(n_freq, device=x.device, dtype=power.dtype)
    psum = power.sum(-1, keepdim=True)
    centroid = (power * freqs).sum(-1) / (psum.squeeze(-1) + eps) / (n_freq - 1)
    flatness = torch.exp((power + eps).log().mean(-1)) / (power.mean(-1) + eps)
    cum = torch.cumsum(power, -1) / (psum + eps)
    rolloff = torch.argmax((cum >= 0.85).to(torch.uint8), -1).float() / (n_freq - 1)
    shape = torch.stack([centroid, flatness, rolloff], -1)  # (B, 3, 3)

    c = torch.fft.rfft(xf, dim=-1)
    corr = torch.fft.irfft(c * torch.conj(c), n=T, dim=-1)
    corr_n = corr[..., 1:] / (corr[..., 0:1].abs() + eps)
    _, lag_idx = torch.topk(corr_n, 2, dim=-1)
    lags = lag_idx.float() / (T - 1)

    peak_amp  = xf.abs().max(-1).values
    peak_time = xf.argmax(-1).float() / (T - 1)
    rms       = torch.sqrt((xf ** 2).mean(-1))

    return torch.cat([
        fine_psd.reshape(B, C * 8), shape.reshape(B, C * 3),
        lags.reshape(B, C * 2), peak_amp, peak_time, rms,
    ], dim=-1).to(orig_dtype)                                # (B, 48)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.covariance import EmpiricalCovariance
from sklearn.cluster import KMeans


@torch.no_grad()
def batched_model_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """分批推理完整数据集, 避免 OOM"""
    B = len(x)
    recons = []
    for i in range(0, B, cfg.train.batch_size):
        bx = x[i:i + cfg.train.batch_size]
        recons.append(model(bx))
    return torch.cat(recons, dim=0)


# ============================================================
#  时域相区权重 (按采样率动态生成, 8s 窗口物理阶段)
#   区1: 启动峰值 (0-0.8s)    — 权重偏低, 正常波动大
#   区2: 转换段   (0.8-4.0s)  — 权重最高, 正常方差最小
#   区3: 缓放段   (4.0-6.4s)  — 权重适中
#   区4: 落零段   (6.4-8.0s)  — 权重最低, 噪声区
# ============================================================
PHASE_WEIGHTS = torch.tensor(
    [0.8]*int(0.8*cfg.data.fs) + [2.0]*int(3.2*cfg.data.fs)
    + [1.0]*int(2.4*cfg.data.fs) + [0.5]*int(1.6*cfg.data.fs),
    dtype=torch.float32
).view(1, 1, -1)  # (1, 1, total_pts) 便于广播

_phase_weights_cache = {}  # 按设备缓存, 避免重复 .to(device)

# 峰值点误差窗口半径 (采样点): ±80ms, 25Hz→2, 100Hz→8
PEAK_HALF_WIN = max(2, round(0.08 * cfg.data.fs))


def _get_pw(device: torch.device) -> torch.Tensor:
    """获取已缓存到指定设备的相区权重"""
    if device not in _phase_weights_cache:
        _phase_weights_cache[device] = PHASE_WEIGHTS.to(device)
    return _phase_weights_cache[device]


@torch.no_grad()
def compute_scores(model: nn.Module, x: torch.Tensor,
                   mse_scale: float = 1.0, peak_scale: float = 1.0,
                   alpha: float = 1.0, mse_weight: float = 1.0):
    """
    计算异常分数 (含时域小算法):
      score = mse_weight × PW-MSE × scale_mse
            + α × PeakErr × scale_peak

    时域算法:
      1. PW-MSE: 相区加权MSE — 转换段权重2.0, 噪声段0.5
      2. DS-Err:  一阶差分形态误差 (单独返回, 可选融合)

    Args:
        model: CNN-LSTM 自编码器
        x: (B, 3, T) 输入电流 (100Hz → T=800)
        mse_scale: MSE 分量缩放系数
        peak_scale: 峰值误差缩放系数
        alpha: 峰值误差权重 (0 时峰值分量不贡献)
        mse_weight: MSE 显式权重 (cfg.detect.mse_weight)

    Returns:
        scores: (B,) 异常分数
        recon:  (B, 3, T) 重构曲线
        extras: dict 含 pw_mse, ds_err 分量
    """
    model.eval()
    recon = batched_model_forward(model, x)
    B, C, T = x.shape

    pw = _get_pw(x.device)

    # ============================================================
    #  时域算法1: 相区加权 MSE (PW-MSE)
    #  转换段(20-100)权重2.0, 因为该段正常方差最小, 异常最易暴露
    # ============================================================
    err = (recon - x) ** 2                          # (B, 3, T)
    # 有效区掩码: 排除尾部零填充 (消除曲线长度作为免费判别信号)
    mask = active_region_mask(x)                    # (B, 1, T)
    n_active = (mask.sum(dim=-1) * C).clamp(min=1).view(-1)  # (B,) 有效采样点 × 通道数
    pw_mse = (err * pw * mask).sum(dim=[1, 2]) / n_active  # (B,)

    # ============================================================
    #  时域算法2: 一阶差分形态误差 (DS-Err)
    #  捕捉上升/下降斜率差异, 对峰值偏移和形态畸变敏感
    #  异常样本差分值比正常高 +39%, 有很好区分度
    # ============================================================
    ds_recon = recon[:, :, 1:] - recon[:, :, :-1]   # (B, 3, T-1)
    ds_x     = x[:, :, 1:]     - x[:, :, :-1]       # (B, 3, T-1)
    m_ds     = mask[..., 1:]                        # (B, 1, T-1)
    n_ds     = (m_ds.sum(dim=-1) * C).clamp(min=1).view(-1)  # (B,)
    ds_err   = ((ds_recon - ds_x) ** 2 * m_ds).sum(dim=[1, 2]) / n_ds  # (B,)

    # --- 峰值点相对误差分量 (窗口化) ---
    # 使用 ±PEAK_HALF_WIN 窗口避免单点 argmax 对相位偏移敏感 (随 fs 缩放)
    HALF_WIN = PEAK_HALF_WIN
    peak_errs = []
    for ch in range(C):
        x_ch = x[:, ch, :]                                 # (B, T)
        r_ch = recon[:, ch, :]
        peak_idx = x_ch.argmax(dim=1)                      # (B,)
        ch_max   = x_ch.abs().max(dim=1)[0]                # (B,) 通道幅度最大值
        shifts = torch.arange(-HALF_WIN, HALF_WIN + 1, device=x.device)
        idxs = torch.clamp(peak_idx[:, None] + shifts[None, :], 0, T - 1)  # (B, W)
        br = torch.arange(B, device=x.device)[:, None]
        x_win = x_ch[br, idxs]                             # (B, W)
        r_win = r_ch[br, idxs]
        se = ((x_win - r_win) / (ch_max[:, None] + 1e-6)) ** 2  # (B, W)
        peak_errs.append(se.min(dim=1)[0])                 # (B,) 取窗口内最小误差

    peak_err = torch.stack(peak_errs).mean(dim=0)         # (B,)

    # --- 综合 (用 PW-MSE 替代原始 MSE, 显式 mse_weight 降权过重分量) ---
    scores = mse_weight * pw_mse * mse_scale + alpha * peak_err * peak_scale

    extras = {
        'pw_mse': pw_mse.cpu().numpy(),
        'ds_err': ds_err.cpu().numpy(),
    }

    return scores.cpu().numpy(), recon, extras


@torch.no_grad()
def compute_phase_scores(x: torch.Tensor,
                          phase_scale: float, phase_stats: dict,
                          beta: float) -> np.ndarray:
    """
    计算相位偏差分数分量 (不依赖模型).

    Args:
        x: (B, 3, 125) 输入电流
        phase_scale: PhaseErr 缩放系数
        phase_stats: 训练集相位特征统计量
        beta: 相位偏差权重

    Returns:
        (B,) 相位偏差分量
    """
    x_np = x.cpu().numpy()
    phase_errs = batch_phase_err(x_np, phase_stats)
    return phase_errs * phase_scale * beta


# ============================================================
#  频谱结构偏差 (SpectralErr) — 参考 FSCA 频域表示
#  对 v2 频谱特征 (fine_psd + 谱形状, 33维) 做训练集 z-score,
#  衡量样本频谱结构与正常分布的偏离. 与 PhaseErr 互补.
# ============================================================
# v2 特征切片: [0:24]=8细频带PSD, [24:33]=谱质心/平坦度/滚降,
# [33:39]=自相关滞后(该信号无周期, 判别力≈随机, 排除), [39:48]=时域峰值(已被模型覆盖)
_SPECTRAL_SLICE = slice(0, 33)


def _batch_v2_features(x: torch.Tensor) -> torch.Tensor:
    """批量计算 v2 频谱特征 (取判别力最强的 fine_psd+谱形状)."""
    feats = []
    for i in range(0, len(x), cfg.train.batch_size):
        bx = x[i:i + cfg.train.batch_size]
        feats.append(compute_aux_features_v2(bx)[:, _SPECTRAL_SLICE])
    return torch.cat(feats, dim=0)


@torch.no_grad()
def compute_spectral_stats(x: torch.Tensor) -> dict:
    """在训练集上计算 v2 频谱特征的均值/标准差."""
    feats = _batch_v2_features(x)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0).clamp(min=1e-6)
    return {'mean': mean, 'std': std}


@torch.no_grad()
def batch_spectral_err(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """批量计算频谱偏差分数 (均方 z-score)."""
    feats = _batch_v2_features(x)
    z = (feats - stats['mean']) / stats['std']
    return torch.mean(z ** 2, dim=1)


@torch.no_grad()
def auto_scale_spectral(train_x: torch.Tensor) -> tuple:
    """自动确定 SpectralErr 缩放系数 (中位数归一化)."""
    stats = compute_spectral_stats(train_x)
    train_err = batch_spectral_err(train_x, stats)
    med = max(train_err.median().item(), 1e-10)
    scale = 1.0 / med
    print(f'  [频谱] SpectralErr中位数={med:.4f} → scale={scale:.2f}')
    return scale, stats


@torch.no_grad()
def compute_spectral_scores(x: torch.Tensor,
                            spectral_scale: float, spectral_stats: dict,
                            gamma: float) -> np.ndarray:
    """计算频谱偏差分数分量."""
    errs = batch_spectral_err(x, spectral_stats)
    return (errs * spectral_scale * gamma).cpu().numpy()


# ============================================================
#  频域正常性偏差 (NormErr) — 可学习 SpectralErr 升级版
#  FNM 子自编码器在正常数据上学到正常频谱流形, 对 log功率谱做重构.
#  异常频谱偏离正常流形 → 重构误差升高. 与 SpectralErr 互补 (非线性 vs z-score).
# ============================================================
@torch.no_grad()
def _batched_normalcy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """分批计算 NormErr, 避免 800 点下全量前向 OOM (LSTM 100步激活)"""
    B = cfg.train.batch_size
    parts = [model.normalcy_errors(x[i:i + B]) for i in range(0, len(x), B)]
    return torch.cat(parts)


@torch.no_grad()
def auto_scale_normalcy(model: nn.Module, train_x: torch.Tensor) -> float:
    """自动确定 NormErr 缩放系数 (中位数归一化)."""
    errs = _batched_normalcy(model, train_x)
    med = max(errs.median().item(), 1e-10)
    scale = 1.0 / med
    print(f'  [正常性] NormErr中位数={med:.4f} → scale={scale:.2f}')
    return scale


@torch.no_grad()
def compute_normalcy_scores(model: nn.Module, x: torch.Tensor,
                            normalcy_scale: float, delta: float) -> np.ndarray:
    """计算频域正常性偏差分数分量."""
    errs = _batched_normalcy(model, x)
    return (errs * normalcy_scale * delta).cpu().numpy()


# ============================================================
#  latent 空间偏差 (LatentErr) — 马氏距离 vs 正常流形
#  难样本 (无缓放台阶/功率不足/启动冲击过高) 时域频域重构误差均小,
#  但 LSTM 潜在表示已偏离正常流形 (诊断: 马氏距离 AUC 0.999/0.810/0.663).
#  利用 LSTM 输出 (B, T, 128) 时间轴均值池化 → 128 维,
#  在参考集 (默认 val, 含阻力增大/季节漂移) 上拟合协方差后计算马氏距离.
#  用 val 而非 train 拟合: train 是早期纯正常, 漂移本身被误判为异常
#  (实测 val P99.9 重尾 778→526, 功率不足 AUC 0.714→0.840). 与重构误差互补.
# ============================================================
@torch.no_grad()
def _batched_latent_full(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """分批提取 LSTM 潜在表示, 保留时间轴 → (B, T, D). 供时段锚切片."""
    B = cfg.train.batch_size
    parts = []
    for i in range(0, len(x), B):
        bx = x[i:i + B]
        captured = {}
        def hook(m, inp, out):
            captured['lstm'] = out[0]   # (B, T, D)
        h = model.lstm.register_forward_hook(hook)
        model(bx)
        h.remove()
        parts.append(captured['lstm'].float().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _batched_latent(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """分批提取 LSTM 潜在表示 (时间轴均值池化 → (B, D)).

    按块先对时间轴取均值再拼接: 避免攒整集 (N,T,D) (训练集 57037×15×128
    ≈ 438MB, 与 phase/rel 等全量副本叠加会耗尽系统内存导致假死).
    """
    B = cfg.train.batch_size
    parts = []
    for i in range(0, len(x), B):
        bx = x[i:i + B]
        captured = {}
        def hook(m, inp, out):
            captured['lstm'] = out[0]   # (B, T, D)
        h = model.lstm.register_forward_hook(hook)
        model(bx)
        h.remove()
        parts.append(captured['lstm'].float().cpu().mean(dim=1))  # (B, D)
    return torch.cat(parts, dim=0)


@torch.no_grad()
def compute_latent_stats(model: nn.Module, ref_x: torch.Tensor) -> dict:
    """在参考集 (正常流形, 默认 val 含漂移) 上拟合 LSTM latent 的多元高斯."""
    lat = _batched_latent(model, ref_x)
    cov = EmpiricalCovariance().fit(lat.numpy())
    return {'mean': cov.location_, 'precision': cov.precision_, 'cov': cov.covariance_}


@torch.no_grad()
def auto_scale_latent(model: nn.Module, ref_x: torch.Tensor) -> tuple:
    """自动确定 LatentErr 缩放系数 (参考集马氏距离中位数归一化).

    ref_x: 正常流形参考集. 用 val (阻力增大/漂移后) 拟合比 train 更贴近
    检测时的正常分布, 压缩重尾、降低误报 (诊断实测).
    """
    stats = compute_latent_stats(model, ref_x)
    ref_err = batch_latent_err(_batched_latent(model, ref_x), stats)
    med = max(np.median(ref_err), 1e-10)
    scale = 1.0 / med
    print(f'  [LatentErr] 马氏距离中位数={med:.2f} → scale={scale:.4f} (参考集 len={len(ref_x)})')
    return scale, stats


def batch_latent_err(lat: torch.Tensor, stats: dict) -> np.ndarray:
    """由 latent 矩阵 (B, 128) 计算马氏距离."""
    d = (lat.numpy() - stats['mean'])
    return np.einsum('ni,ij,nj->n', d, stats['precision'], d)


@torch.no_grad()
def compute_latent_scores(model: nn.Module, x: torch.Tensor,
                          latent_scale: float, latent_stats: dict,
                          epsilon: float) -> np.ndarray:
    """计算 latent 空间偏差分数分量."""
    lat = _batched_latent(model, x)
    errs = batch_latent_err(lat, latent_stats)
    return errs * latent_scale * epsilon


# ============================================================
#  时段锚 latent (SAL) — YOLOv2 锚框思想的时序映射
#  难样本的 latent 偏移是局部的 (启动延迟→启动段, 启动冲击→落零段,
#  功率不足→转换段), 全局均值池化把局部偏移稀释 (实测时段马氏:
#  启动延迟 0.975 vs 全局 0.922, 启动冲击 0.930 vs 0.865).
#  对每个物理时段锚在 val 上拟合独立马氏, 样本取最异常时段锚的马氏.
# ============================================================
SEG_ANCHOR_SECS = [(0.0, 0.8), (0.8, 4.0), (4.0, 6.4), (6.4, 8.0)]
SEG_ANCHOR_NAMES = ['启动', '转换', '缓放', '落零']


def _seg_anchor_bounds(T: int) -> list:
    """物理时段 (秒) → latent 步边界 [(a, b), ...]."""
    return [(int(a / 8.0 * T), int(b / 8.0 * T)) for a, b in SEG_ANCHOR_SECS]


@torch.no_grad()
def _probe_latent_T(model: nn.Module) -> int:
    """探测 LSTM 潜在表示的时间步数 T (时段锚边界换算用).

    用一个最小样本前向, 从 lstm 输出钩子取 (1, T, D) 的 T. 输入长度需与
    模型 seq_len 一致 (cfg.data.total_pts), 仅一次廉价前向.
    """
    dev = next(model.parameters()).device
    probe = torch.zeros(1, cfg.model.in_channels, cfg.data.total_pts, device=dev)
    captured = {}
    def hook(m, inp, out):
        captured['lstm'] = out[0]   # (1, T, D)
    h = model.lstm.register_forward_hook(hook)
    model(probe)
    h.remove()
    return captured['lstm'].shape[1]


@torch.no_grad()
def _batched_latent_anchors(model: nn.Module, x: torch.Tensor,
                            bounds: list) -> list:
    """分批提取每时段锚子向量的时间均值 → [每锚 (N, D)].

    单次前向, 按块取各锚子序列均值, 不攒整集 (N,T,D):
    避免 SegLatent 在训练集上一次性分配 (57037,15,128) ≈ 438MB.
    """
    B = cfg.train.batch_size
    n_anchor = len(bounds)
    parts = [[] for _ in range(n_anchor)]
    for i in range(0, len(x), B):
        bx = x[i:i + B]
        captured = {}
        def hook(m, inp, out):
            captured['lstm'] = out[0]   # (B, T, D)
        h = model.lstm.register_forward_hook(hook)
        model(bx)
        h.remove()
        lat = captured['lstm'].float().cpu()   # (B, T, D)
        for a_i, (a, b) in enumerate(bounds):
            parts[a_i].append(lat[:, a:b, :].mean(dim=1))   # (B, D)
    return [torch.cat(p, dim=0) for p in parts]


@torch.no_grad()
def auto_scale_seg_latent(model: nn.Module, ref_x: torch.Tensor) -> tuple:
    """对每个时段锚在参考集 (val, 含漂移) 上拟合 latent 子向量马氏.

    Returns:
        seg_scales: [每时段锚缩放系数]
        seg_stats:  [{'bounds','mean','precision'} × 时段锚]
    """
    bounds = _seg_anchor_bounds(_probe_latent_T(model))
    subs = _batched_latent_anchors(model, ref_x, bounds)   # [每锚 (V, D)]
    scales, stats = [], []
    for (a, b), name, sub in zip(bounds, SEG_ANCHOR_NAMES, subs):
        sub = sub.numpy()                                   # (V, D)
        cov = EmpiricalCovariance().fit(sub)
        d = np.einsum('ni,ij,nj->n', sub - cov.location_, cov.precision_, sub - cov.location_)
        med = max(np.median(d), 1e-10)
        scales.append(1.0 / med)
        stats.append({'bounds': (a, b), 'mean': cov.location_, 'precision': cov.precision_})
        print(f'  [时段锚] {name}({a}:{b}步) 马氏中位={med:.2f} → scale={1.0 / med:.4f}')
    return scales, stats


@torch.no_grad()
def compute_seg_latent_scores(model: nn.Module, x: torch.Tensor,
                              seg_scales: list, seg_stats: list,
                              omega: float) -> np.ndarray:
    """每时段锚马氏 × 缩放, 样本取最异常时段锚 (max) → (B,)."""
    N = len(x)
    bounds = [st['bounds'] for st in seg_stats]
    subs = _batched_latent_anchors(model, x, bounds)   # [每锚 (N, D)]
    seg_errs = np.zeros((N, len(seg_stats)))
    for i, (scale, st) in enumerate(zip(seg_scales, seg_stats)):
        sub = subs[i].numpy()                                    # (N, D)
        d = np.einsum('ni,ij,nj->n', sub - st['mean'], st['precision'], sub - st['mean'])
        seg_errs[:, i] = d * scale
    return seg_errs.max(axis=1) * omega


@torch.no_grad()
def compute_seg_anchor_scores(model: nn.Module, x: torch.Tensor,
                              seg_scales: list, seg_stats: list) -> np.ndarray:
    """逐时段锚马氏 × 缩放 → (N, n_anchor). 供独立阈值触发用."""
    N = len(x)
    bounds = [st['bounds'] for st in seg_stats]
    subs = _batched_latent_anchors(model, x, bounds)   # [每锚 (N, D)]
    out = np.zeros((N, len(seg_stats)))
    for i, (scale, st) in enumerate(zip(seg_scales, seg_stats)):
        sub = subs[i].numpy()                                    # (N, D)
        d = np.einsum('ni,ij,nj->n', sub - st['mean'], st['precision'], sub - st['mean'])
        out[:, i] = d * scale
    return out


# ============================================================
#  机簇锚 latent (ClusterLatent) — 幅度类异常"相对自身基线"检测
#  全局 latent 高斯混入跨机 scale 方差 (低基础机卡阻 ≈ 高基础机正常),
#  使绝对水平型异常 (卡阻/启动冲击/启动延迟) 的 latent 偏移被稀释.
#  在训练集正常 latent 上按机聚类 (KMeans), 再对每簇拟合独立马氏,
#  样本取最匹配簇 (min) 的马氏 → 与自身机簇基线比较, 放大幅度类局部偏移.
#  实测 (2026-08-05): 卡阻难样本子集 AUC 0.950→0.994; 全局综合 0.962→0.985;
#  代价: 功率不足单分量 0.852→0.817 (min-over-clusters 让全局低幅样本
#  可匹配低基础簇), 故作附加分量与全局 LatentErr 互补, 不替代.
# ============================================================
def auto_scale_cluster_latent(model: nn.Module, ref_x: torch.Tensor,
                              k: int = 20) -> tuple:
    """在参考集 (训练集正常 latent) 上 KMeans 聚类 + 每簇独立马氏.

    Returns:
        cluster_stats: [{'mean','precision','scale','n'} × 簇]  (在标准化空间)
        z_mean, z_std: 训练集 latent 逐维标准化参数
    """
    lat = _batched_latent(model, ref_x).numpy()
    z_mean = lat.mean(axis=0)
    z_std = lat.std(axis=0) + 1e-8
    zs = (lat - z_mean) / z_std
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(zs)
    stats = []
    for c in range(k):
        sub = zs[km.labels_ == c]
        cov = EmpiricalCovariance().fit(sub)
        d = np.einsum('ni,ij,nj->n', sub - cov.location_, cov.precision_,
                      sub - cov.location_)
        stats.append({'mean': cov.location_, 'precision': cov.precision_,
                      'scale': 1.0 / max(np.median(d), 1e-10), 'n': len(sub)})
        print(f'  [机簇锚] 簇{c} n={len(sub)} 马氏中位={np.median(d):.2f} '
              f'→ scale={stats[-1]["scale"]:.4f}')
    return stats, z_mean, z_std


@torch.no_grad()
def compute_cluster_latent_scores(model: nn.Module, x: torch.Tensor,
                                  cluster_stats: list, z_mean: np.ndarray,
                                  z_std: np.ndarray, omega: float) -> np.ndarray:
    """样本取最匹配簇 (min) 的缩放马氏 → (N,)."""
    lat = _batched_latent(model, x).numpy()
    zs = (lat - z_mean) / z_std
    out = np.full(len(zs), np.inf)
    for st in cluster_stats:
        d = np.einsum('ni,ij,nj->n', zs - st['mean'], st['precision'],
                      zs - st['mean'])
        out = np.minimum(out, d * st['scale'])
    return out * omega


# ============================================================
#  相对物理特征 RelErr — 攻击低基础机卡阻漏检 (波形级, 无需重训)
#  latent 族 (全局/SAL/机簇) 对卡阻在 FPR<1% 封顶 ~77% 召回: 残余 FN 全是
#  低基础机样本, 其 latent 读作正常 P99~P99.9. 区分信号在波形本身:
#  卡阻只抬升转换段 (峰值/解锁不变) → 转换/峰值、转换/解锁 比值升高;
#  中段尖峰 → 转换段内 std 升高. 对功率不足 (整体等比例缩放, 比值不变) 天然免疫.
#  RelErr = Σ (z∈[0,clip])², 仅取"偏高"方向, clip=3 控制重尾 (clip=6 重尾
#  抬 val 阈值致 P99.5 召回回退, 实测 clip3 卡阻中位 1.35×valP99.5).
#  (2026-08-05: blocking P99.5 召回 0.759→0.866, 救回 12 个低基础机 FN)
# ============================================================
_REL_CONV_WIN = (1.5, 4.0)      # 转换段固定窗 (秒), 与检测特征一致
_REL_UNLOCK_OFF = (0.15, 1.15)  # 峰值后解锁段估计窗 (秒)


def _batch_rel_feats(x: torch.Tensor) -> np.ndarray:
    """A相相对特征 (B, 3): [转换/峰值, 转换/解锁, 转换段内std].

    按块处理: 原实现对整集 (训练 57037×3×800) 一次 .cpu().numpy() 拷贝
    547MB, 与其它分量叠加耗尽内存导致假死. 分块后单块拷贝 ~1MB.
    """
    fs = cfg.data.fs
    B = cfg.train.batch_size
    outs = []
    for i in range(0, len(x), B):
        A = x[i:i + B].cpu().numpy()[:, 0, :]   # 小块 (chunk, T)
        T = A.shape[1]
        a = max(0, int(_REL_CONV_WIN[0] * fs))
        b = min(T, int(_REL_CONV_WIN[1] * fs))
        conv = np.median(A[:, a:b], axis=1)
        fluct = A[:, a:b].std(axis=1)
        peak = A.max(axis=1)
        u0 = int(_REL_UNLOCK_OFF[0] * fs)
        u1 = int(_REL_UNLOCK_OFF[1] * fs)
        pi = A.argmax(axis=1)
        unlock = np.array([np.median(A[i, max(0, pi[i]+u0):min(T, pi[i]+u1)])
                           for i in range(len(A))])
        outs.append(np.stack([conv / (peak + 1e-6),
                              conv / (unlock + 1e-6), fluct], axis=1))
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def auto_scale_rel(x_train: torch.Tensor) -> tuple:
    """拟合相对特征均值/标准差 + 训练集中位缩放系数."""
    F = _batch_rel_feats(x_train)
    mu = F.mean(axis=0)
    sd = F.std(axis=0) + 1e-8
    z = np.clip((F - mu) / sd, 0, cfg.detect.rel_clip) ** 2
    err = z.sum(axis=1)
    med = max(np.median(err), 1e-10)
    scale = 1.0 / med
    print(f'  [RelErr] Σz²中位={med:.2f} → scale={scale:.4f} (clip={cfg.detect.rel_clip})')
    return {'mu': mu, 'sd': sd}, scale


@torch.no_grad()
def compute_rel_scores(x: torch.Tensor, rel_stats: dict, rel_scale: float,
                       omega: float) -> np.ndarray:
    """RelErr = Σ (z∈[0,clip])² × scale × ω → (N,)."""
    F = _batch_rel_feats(x)
    z = np.clip((F - rel_stats['mu']) / rel_stats['sd'], 0, cfg.detect.rel_clip) ** 2
    return z.sum(axis=1) * rel_scale * omega


# ============================================================
#  条件对齐残差 (AlignResidual) — TimeCMA 跨模态对齐思想
#  latent z (解耦但弱: 难样本读作正常) 提供"这台机正常应怎样"的上下文,
#  在正常参考集上学跨模态关系 f ≈ g(z) (f=幅度时域+相对抬升特征),
#  残差 ‖f − g(z)‖ 作异常度. 与 PhysErr 的区别: PhysErr 全局 z-score 受
#  机基线方差污染 (重尾抬阈值, ζ=0 被弃); Align 以 z 为条件消除基线方差 →
#  低基础机卡阻/功率不足的 f 偏移在残差里放大. 免重训, 检测侧拟合 (同 amp_head).
# ============================================================
def _batch_align_feats(x: torch.Tensor) -> np.ndarray:
    """对齐特征 f (B, 12) = PhysErr 幅度时域集 [:9] + RelErr 相对抬升集 [3].
    覆盖文档化难样本签名: 功率不足/启动冲击→幅度, 低基础机卡阻→相对抬升."""
    # 注意: x 可能是 CUDA 张量, _batch_phys_feats 返回保持设备 → 必须 .cpu() 才能 .numpy()
    # (bug: 之前 AlignResidual 默认 weight=0 从不执行, 此错潜伏未暴露)
    phys = _batch_phys_feats(x).cpu().numpy()[:, :9]   # peak_amp, rms, peak_time
    rel = _batch_rel_feats(x)                    # r_cp, r_cu, conv_fluct
    return np.concatenate([phys, rel], axis=1)


class _AlignConditionalMap(nn.Module):
    """条件映射 g: z → f (在标准化空间拟合). mlp 默认 / linear 稳健性对照."""
    def __init__(self, z_dim: int, f_dim: int, map_type: str = 'mlp'):
        super().__init__()
        if map_type == 'linear':
            self.net = nn.Linear(z_dim, f_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(z_dim, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, f_dim),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def auto_scale_align(model: nn.Module, ref_x: torch.Tensor,
                     map_type: str = 'mlp', fit_n: int = 0,
                     epochs: int = 100, seed: int = 42) -> dict:
    """在正常参考集上拟合条件映射 + 缩放系数 (参考集残差中位数归一化).

    NOTE: 不能加 @torch.no_grad() — 内部 fit 映射 g 需要 loss.backward().
    (bug: 之前 AlignResidual 默认 weight=0 从不执行, no_grad 下 backward 失败潜伏未暴露;
     latent/特征提取本身已是 no_grad, 仅 g 的训练需要梯度)

    映射拟合在参考集内 90/10 早停 (防过拟合; 10ep 无改进即停).
    Returns stats: {'g','z_mean','z_std','f_mean','f_std','scale'}
    """
    z = _batched_latent(model, ref_x).numpy()      # (N, 256)
    f = _batch_align_feats(ref_x)                  # (N, 12)
    if fit_n and len(z) > fit_n:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(z), fit_n, replace=False)
        z, f = z[idx], f[idx]
    z_mean, z_std = z.mean(0), z.std(0) + 1e-8
    f_mean, f_std = f.mean(0), f.std(0) + 1e-6
    dev = next(model.parameters()).device
    zs = torch.FloatTensor((z - z_mean) / z_std).to(dev)
    fs = torch.FloatTensor((f - f_mean) / f_std).to(dev)
    g = _AlignConditionalMap(z.shape[1], f.shape[1], map_type).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    perm = np.random.RandomState(seed).permutation(len(z))
    nf = int(len(z) * 0.9)
    zt, ft = zs[perm[:nf]], fs[perm[:nf]]
    zv, fv = zs[perm[nf:]], fs[perm[nf:]]
    best, best_state, stop = float('inf'), None, 0
    for _ in range(epochs):
        g.train(); opt.zero_grad()
        loss = crit(g(zt), ft); loss.backward(); opt.step()
        g.eval()
        with torch.no_grad():
            vloss = crit(g(zv), fv).item()
        if vloss < best:
            best, best_state, stop = vloss, copy.deepcopy(g.state_dict()), 0
        else:
            stop += 1
            if stop >= 10:
                break
    g.load_state_dict(best_state); g.eval()
    # fs 是 CUDA 张量, g 输出在 CUDA → 统一转 CPU numpy 再算残差
    res = ((fs.cpu().numpy() - g(zs).detach().cpu().numpy()) ** 2).mean(axis=1)
    scale = 1.0 / max(np.median(res), 1e-10)
    print(f'  [AlignResidual] g({map_type}) 早停valMSE={best:.4f}, '
          f'参考集残差中位={np.median(res):.4f} → scale={scale:.2f}')
    return {'g': g, 'z_mean': z_mean, 'z_std': z_std,
            'f_mean': f_mean, 'f_std': f_std, 'scale': scale}


@torch.no_grad()
def compute_align_scores(model: nn.Module, x: torch.Tensor,
                         stats: dict, weight: float) -> np.ndarray:
    """AlignResidual = Σ(f_std − g(z_std))² × scale × ω → (N,)."""
    z = _batched_latent(model, x).numpy()
    f = _batch_align_feats(x)
    dev = next(model.parameters()).device
    zs = torch.FloatTensor((z - stats['z_mean']) / stats['z_std']).to(dev)
    pred = stats['g'](zs).cpu().numpy()
    fs = (f - stats['f_mean']) / stats['f_std']
    res = ((fs - pred) ** 2).mean(axis=1)
    return res * stats['scale'] * weight


# ============================================================
#  物理特征偏差 (PhysErr) — 精选物理量 z-score
#  难样本中 功率不足(整体幅度低)/启动延迟(峰值时间偏)/启动冲击(峰值偏高)
#  在重构误差与 latent 中均弱, 但物理量直接反映幅度/峰时/时长偏移
#  (诊断实测 AUC: 功率不足 0.867, 启动延迟 0.906, 启动冲击 0.671).
#  注意: 对无缓放台阶不利(0.440) — 其 LatentErr 已是 1.000, 由低权重 zeta 保护.
#  与 SpectralErr 相同: 训练集 z-score 均方 + 中位数归一化.
# ============================================================
_PHYS_FEAT_DIM = 3 + 3 + 3 + 1 + 3   # peak_amp 3 + rms 3 + peak_time 3 + active_len 1 + zero_rate 3


def _batch_phys_feats(x: torch.Tensor) -> torch.Tensor:
    """批量计算精选物理特征 (B, 13): 峰幅3 + RMS3 + 峰时3 + 有效长度1 + 零率3."""
    xf = x.float()
    T = xf.shape[-1]
    a = xf.abs()
    peak_amp = a.max(dim=-1).values                        # (B, 3) ∈[0,1]
    rms = torch.sqrt((xf ** 2).mean(dim=-1))               # (B, 3) ∈[0,1]
    peak_time = xf.argmax(dim=-1).float() / (T - 1)        # (B, 3) ∈[0,1]
    # 有效长度: 任一相 > eps 的活跃点数占比 (归一化, 对短曲线敏感)
    active = (a.max(dim=1, keepdim=True).values > 1e-3)    # (B,1,T)
    al = active.float().mean(dim=-1)                       # (B, 1)
    zero_rate = (a < 1e-4).float().mean(dim=-1)            # (B, 3)
    return torch.cat([peak_amp, rms, peak_time, al, zero_rate], dim=-1)  # (B, 13)


@torch.no_grad()
def compute_phys_stats(x: torch.Tensor) -> dict:
    """在训练集上计算物理特征的均值/标准差."""
    feats = _batch_phys_feats(x)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0).clamp(min=1e-6)
    return {'mean': mean, 'std': std}


@torch.no_grad()
def auto_scale_phys(x: torch.Tensor) -> tuple:
    """自动确定 PhysErr 缩放系数 (训练集 z-score 均方中位数归一化)."""
    stats = compute_phys_stats(x)
    train_err = batch_phys_err(x, stats)
    med = max(train_err.median().item(), 1e-10)
    scale = 1.0 / med
    print(f'  [PhysErr] 物理特征z-score中位数={med:.4f} → scale={scale:.2f}')
    return scale, stats


def batch_phys_err(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """批量计算物理特征偏差分数 (均方 z-score)."""
    feats = _batch_phys_feats(x)
    z = (feats - stats['mean']) / stats['std']
    return torch.mean(z ** 2, dim=1)


@torch.no_grad()
def compute_phys_scores(x: torch.Tensor,
                        phys_scale: float, phys_stats: dict,
                        zeta: float) -> np.ndarray:
    """计算物理特征偏差分数分量."""
    errs = batch_phys_err(x, phys_stats)
    return (errs * phys_scale * zeta).cpu().numpy()


# ============================================================
#  幅度解码头 (AmpHead) — latent 重构 peak/RMS 的误差
#  难样本中 启动冲击过高(峰值高12-30%)/功率不足(整体幅度低) 在所有
#  重构/频谱/latent 分量中均静默, 但 latent 已含幅度信息 —
#  用一个小 MLP 在 X_train(全正常) 上拟合 latent→(peak,rms),
#  异常样本幅度偏离正常流形 → 重构误差升高.
#  (诊断: 启动冲击 AUC 0.867 / 功率不足 0.823, 均优于 z-score)
#  头只在正常样本上训练 — 符合"训练只用完全正常样本"原则.
# ============================================================
def _amp_targets(x: torch.Tensor) -> torch.Tensor:
    """由电流曲线提取幅度目标 (B, 6): peak_amp 3 + rms 3. 保持输入设备."""
    xf = x.float()
    peak_amp = xf.abs().max(dim=-1).values
    rms = torch.sqrt((xf ** 2).mean(dim=-1))
    return torch.cat([peak_amp, rms], dim=-1).to(x.device)


def fit_amp_head(model: nn.Module, train_x: torch.Tensor,
                 hidden: int = 128, epochs: int = 200,
                 lr: float = 1e-3) -> nn.Module:
    """在 X_train(全正常) 上拟合幅度解码头 latent→(peak,rms).

    Returns:
        head: 训练好的 MLP (latent 128 → 6 维幅度), eval 态
    """
    # 固定 head 初始化/训练的随机性: 否则每次 run_detection 的 AmpHead 不同,
    # 综合分分布漂移 → FP/召回不可复现 (实测同配置 FP 波动 185→232)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    dev = next(model.parameters()).device
    lat = _batched_latent(model, train_x).float().to(dev)
    tgt = _amp_targets(train_x).float()
    head = nn.Sequential(
        nn.Linear(lat.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, tgt.shape[1]),
    ).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.MSELoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(head(lat), tgt)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def compute_amp_scores(model: nn.Module, head: nn.Module,
                       x: torch.Tensor) -> np.ndarray:
    """计算幅度重构误差 (latent → head → 与真实 peak/rms 的 MSE)."""
    lat = _batched_latent(model, x).float().to(next(head.parameters()).device)
    tgt = _amp_targets(x).float()
    pred = head(lat)
    return ((pred - tgt) ** 2).mean(dim=1).cpu().numpy()


def auto_scale_amp_head(model: nn.Module, train_x: torch.Tensor) -> tuple:
    """拟合幅度头并确定缩放 (训练集误差中位数归一化).

    内部含训练 (fit_amp_head backward), 不能包 torch.no_grad().
    头只在 X_train 全正常样本上拟合 — 符合"训练只用正常样本"原则.
    """
    head = fit_amp_head(model, train_x)
    train_err = compute_amp_scores(model, head, train_x)
    med = max(np.median(train_err), 1e-10)
    scale = 1.0 / med
    print(f'  [AmpHead] 幅度重构误差中位数={med:.5f} → scale={scale:.1f} (头在正常样本上拟合)')
    return head, scale


def auto_scale_components(model: nn.Module, x_ref: torch.Tensor):
    """
    自动确定两个分量的缩放系数, 使它们在训练集上量级相当.

    策略: 中位数归一化 (对各分量除以各自的中位数)

    注: 使用分批推理避免 OOM (双通路模型内存需求翻倍)
    """
    model.eval()
    with torch.no_grad():
        recon = batched_model_forward(model, x_ref)
        B, C, T = x_ref.shape
        # 使用相区加权 MSE 做缩放基准 — 必须与 compute_scores 一致地加
        # active_region_mask, 否则 scale 基准(全窗均值)与实际分数(mask后)
        # 不一致, 导致 MSE 有效权重漂移 (实测 1.0 → 0.884).
        pw = _get_pw(x_ref.device)
        err = (recon - x_ref) ** 2
        mask = active_region_mask(x_ref)
        n_active = (mask.sum(dim=-1) * C).clamp(min=1).view(-1)
        pw_mse = (err * pw * mask).sum(dim=[1, 2]) / n_active

        HALF_WIN = PEAK_HALF_WIN
        peak_errs = []
        for ch in range(C):
            x_ch = x_ref[:, ch, :]
            r_ch = recon[:, ch, :]
            peak_idx = x_ch.argmax(dim=1)
            ch_max = x_ch.abs().max(dim=1)[0]
            shifts = torch.arange(-HALF_WIN, HALF_WIN + 1, device=x_ref.device)
            idxs = torch.clamp(peak_idx[:, None] + shifts[None, :], 0, T - 1)
            br = torch.arange(B, device=x_ref.device)[:, None]
            x_win = x_ch[br, idxs]
            r_win = r_ch[br, idxs]
            se = ((x_win - r_win) / (ch_max[:, None] + 1e-6)) ** 2
            peak_errs.append(se.min(dim=1)[0])
        peak_err = torch.stack(peak_errs).mean(dim=0)

        pw_mse_med = max(pw_mse.median().item(), 1e-10)
        peak_med = max(peak_err.median().item(), 1e-10)

        mse_scale  = 1.0 / pw_mse_med
        peak_scale = 1.0 / peak_med

    print(f'    分量缩放: PW-MSE中位数={pw_mse_med:.6f} → scale={mse_scale:.1f}, '
          f'Peak中位数={peak_med:.6f} → scale={peak_scale:.1f}')

    return mse_scale, peak_scale


def choose_threshold(val_scores: np.ndarray) -> float:
    """
    基于验证集确定异常阈值.

    使用 P{threshold_percentile} 分位数 — 因为异常率仅约 0.1%,
    用 P99.9 使理论 FP 率 ≈ 0.1% (与异常率相当).

    注意: 不再使用 mean+3σ 作为备选, 因为异常分数分布高度右偏,
          mean+3σ 可能对应远低于 P{threshold_percentile} 的分位点, 导致阈值过于宽松.
    """
    p = cfg.detect.threshold_percentile
    threshold = np.percentile(val_scores, p)

    # 估算理论虚警数
    n_val = len(val_scores)
    est_fp_rate = (100 - p) / 100
    print(f'    验证集 P{p:.1f}分位数: {threshold:.6f}')
    print(f'    最终阈值:             {threshold:.6f}')
    print(f'    (理论虚警率 ≈ {est_fp_rate:.1%}, '
          f'即验证集 ~{int(est_fp_rate * n_val)} 条)')

    return threshold


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """
    在测试集上评估检测性能.

    Returns:
        {tp, fp, tn, fn, precision, recall, f1, auc_roc, auc_pr, predictions}
    """
    predictions = (scores > threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-10)
    auc_roc  = roc_auc_score(labels, scores)
    auc_pr   = average_precision_score(labels, scores)
    fpr      = fp / max(fp + tn, 1)       # 虚警率 (FPR) = FP / (FP + TN)
    fdr      = fp / max(tp + fp, 1)       # 误报率 (FDR) = FP / (TP + FP)

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'auc_roc': auc_roc, 'auc_pr': auc_pr,
        'fpr': fpr,   # 虚警率 = FP/(FP+TN)
        'fdr': fdr,   # 误报率 = FP/(TP+FP)
        'predictions': predictions,
    }


def evaluate_hybrid(scores: np.ndarray, labels: np.ndarray,
                    threshold: float, anchor_flags: np.ndarray) -> dict:
    """
    weighted 综合分 + 时段锚独立触发 的混合评估.

    predictions = (综合分 > 阈值) | (任一时段锚马氏 > 该锚独立阈值)
    AUC 用连续综合分 (scores); 精确/召回/F1/FPR 用混合判定.
    """
    pred = ((scores > threshold) | anchor_flags).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr = fp / max(fp + tn, 1)
    fdr = fp / max(tp + fp, 1)
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'fpr': fpr, 'fdr': fdr,
        'auc_roc': roc_auc_score(labels, scores),
        'auc_pr': average_precision_score(labels, scores),
        'predictions': pred,
    }


def evaluate_or(components: dict, val_scores_map: dict,
                labels: np.ndarray, percentile: float = 99.9) -> dict:
    """
    OR 规则评估: 每个分量在验证集(正常)上独立取分位阈值,
    任一分量超过其阈值即判异常.

    Args:
        components:  {'名': 测试集分量分数 (n,)}
        val_scores_map: {'名': 验证集同分量分数 (n,)}
        labels: 测试集标签
        percentile: 各分量独立阈值分位数

    Returns:
        results dict (同 evaluate 结构, 但 auc_* 基于 OR 概率代理)
    """
    n = len(labels)
    thresholds = {}
    pred = np.zeros(n, dtype=bool)
    for name, te in components.items():
        th = np.percentile(val_scores_map[name], percentile)
        thresholds[name] = th
        pred |= (te > th)

    # OR 规则是离散判定, 无连续分数 → 用 "是否超过自身阈值" 构造 0/1 代理分
    proxy = pred.astype(float)

    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    tn = int(np.sum(~pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-10)
    fpr       = fp / max(fp + tn, 1)
    fdr       = fp / max(tp + fp, 1)

    results = {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'auc_roc': roc_auc_score(labels, proxy),
        'auc_pr': average_precision_score(labels, proxy),
        'fpr': fpr, 'fdr': fdr,
        'predictions': pred.astype(int),
        'or_thresholds': thresholds,
    }
    return results


def run_detection(model: nn.Module,
                  X_train_t: torch.Tensor,
                  X_val_t: torch.Tensor,
                  X_test_t: torch.Tensor,
                  labels_test: np.ndarray) -> tuple:
    """
    完整异常检测流程.

    Returns:
        (results dict, scores dict)
    """
    alpha = cfg.detect.alpha
    beta  = cfg.detect.beta
    gamma = cfg.detect.gamma
    delta = cfg.detect.delta
    epsilon = cfg.detect.epsilon
    zeta = cfg.detect.zeta
    mse_weight = cfg.detect.mse_weight
    align_weight = cfg.detect.align_residual_weight
    align_enabled = (cfg.detect.align_residual_enabled
                     and align_weight > 0)

    # 频域正常性分量是否启用: 模型构建了 FNM 且 checkpoint 加载到了权重
    nc_enabled = (getattr(model, 'normalcy_enabled', False)
                  and getattr(model, 'normalcy_ready', True)
                  and hasattr(model, 'normalcy_errors'))

    formula = ('score = w_MSE×MSE×scale + α×PeakErr×scale + β×PhaseErr×scale '
               '+ γ×SpectralErr×scale' + (' + δ×NormErr×scale' if nc_enabled else '')
               + f' + ε×LatentErr×scale + ω×SegLatent + ω_c×ClusterLatent + ω_r×RelErr '
               + f'+ ζ×PhysErr×scale' + (f' + ω_a×AlignResidual×scale' if align_enabled else '')
               + f' (ε={epsilon}, ζ={zeta})')
    print('\n' + '=' * 50)
    print(f'异常检测: {formula}')
    print(f'          w_MSE={mse_weight}, α={alpha}, β={beta}, γ={gamma}'
          + (f', δ={delta}' if nc_enabled else ' [δ分量关闭]')
          + f', ε={epsilon}, ζ={zeta}'
          + (f', ω_a={align_weight}' if align_enabled else ' [AlignResidual关闭]'))
    print('=' * 50)

    n_comp = 12  # 步骤槽位: MSE+Peak/Phase/Spectral/综合/NormErr?/Latent/Phys/AmpHead/SegLatent/ClusterLatent/RelErr/Align
                 # (NormErr 关闭时跳过第5步, 其余编号不变)

    # 1. 自动缩放 (MSE + PeakErr)
    print(f'  [分量1/{n_comp}] MSE + PeakErr 缩放...')
    mse_scale, peak_scale = auto_scale_components(model, X_train_t)

    # 2. 相位特征缩放 (PhaseErr)
    print(f'  [分量2/{n_comp}] PhaseErr 缩放 (相位结构偏差)...')
    X_train_np = X_train_t.cpu().numpy()
    X_val_np   = X_val_t.cpu().numpy()
    phase_scale, phase_stats = auto_scale_phase(X_train_np, X_val_np)
    del X_train_np, X_val_np   # 用后即弃: 各占 547/149MB, 留到函数末会与后续分量叠加

    # 3. 频谱特征缩放 (SpectralErr, v2频谱 z-score)
    print(f'  [分量3/{n_comp}] SpectralErr 缩放 (频谱结构偏差, 参考FSCA)...')
    spectral_scale, spectral_stats = auto_scale_spectral(X_train_t)

    # 4. 计算三集综合分数
    print(f'  [分量4/{n_comp}] 计算综合异常分数...')
    # PW-MSE + PeakErr 分量 (compute_scores 返回 scores, recon, extras)
    train_mp, _, train_extras = compute_scores(model, X_train_t, mse_scale, peak_scale, alpha, mse_weight)
    val_mp,   _, val_extras   = compute_scores(model, X_val_t,   mse_scale, peak_scale, alpha, mse_weight)
    test_mp, recon, test_extras = compute_scores(model, X_test_t, mse_scale, peak_scale, alpha, mse_weight)

    # PhaseErr 分量
    train_ph = compute_phase_scores(X_train_t, phase_scale, phase_stats, beta)
    val_ph   = compute_phase_scores(X_val_t,   phase_scale, phase_stats, beta)
    test_ph  = compute_phase_scores(X_test_t,  phase_scale, phase_stats, beta)

    # SpectralErr 分量
    train_sp = compute_spectral_scores(X_train_t, spectral_scale, spectral_stats, gamma)
    val_sp   = compute_spectral_scores(X_val_t,   spectral_scale, spectral_stats, gamma)
    test_sp  = compute_spectral_scores(X_test_t,  spectral_scale, spectral_stats, gamma)

    # DS-Err 一阶差分形态误差 (单独评估, 暂不融入主分数)
    train_ds = train_extras['ds_err']
    val_ds   = val_extras['ds_err']
    test_ds  = test_extras['ds_err']

    # NormErr 分量 (可学习频域正常性模型)
    if nc_enabled:
        print(f'  [分量5/{n_comp}] NormErr 缩放 (可学习频域正常性模型)...')
        normalcy_scale = auto_scale_normalcy(model, X_train_t)
        train_nc = compute_normalcy_scores(model, X_train_t, normalcy_scale, delta)
        val_nc   = compute_normalcy_scores(model, X_val_t,   normalcy_scale, delta)
        test_nc  = compute_normalcy_scores(model, X_test_t,  normalcy_scale, delta)
    else:
        train_nc = np.zeros(len(X_train_t))
        val_nc   = np.zeros(len(X_val_t))
        test_nc  = np.zeros(len(X_test_t))
        normalcy_scale = None

    # LatentErr 分量 (LSTM latent 马氏距离, 捕捉时频重构误差均小的难样本)
    # 参考集用 val (含阻力增大/季节漂移), 压缩重尾、降低误报 (诊断实测)
    print(f'  [分量6/{n_comp}] LatentErr 缩放 (latent 马氏距离, val漂移校准)...')
    latent_scale, latent_stats = auto_scale_latent(model, X_val_t)
    train_lt = compute_latent_scores(model, X_train_t, latent_scale, latent_stats, epsilon)
    val_lt   = compute_latent_scores(model, X_val_t,   latent_scale, latent_stats, epsilon)
    test_lt  = compute_latent_scores(model, X_test_t,  latent_scale, latent_stats, epsilon)

    # PhysErr 分量 (精选物理特征 z-score, 捕捉幅度/时长偏移类难样本)
    print(f'  [分量7/{n_comp}] PhysErr 缩放 (物理特征 z-score)...')
    phys_scale, phys_stats = auto_scale_phys(X_train_t)
    train_phz = compute_phys_scores(X_train_t, phys_scale, phys_stats, zeta)
    val_phz   = compute_phys_scores(X_val_t,   phys_scale, phys_stats, zeta)
    test_phz  = compute_phys_scores(X_test_t,  phys_scale, phys_stats, zeta)

    # AmpHead 分量 (幅度解码头, 在 X_train 全正常样本上拟合)
    if cfg.detect.amp_head_enabled:
        print(f'  [分量8/{n_comp}] AmpHead 缩放 (幅度解码头, 正常样本拟合)...')
        amp_head, amp_scale = auto_scale_amp_head(model, X_train_t)
        train_am = compute_amp_scores(model, amp_head, X_train_t) * amp_scale
        val_am   = compute_amp_scores(model, amp_head, X_val_t)   * amp_scale
        test_am  = compute_amp_scores(model, amp_head, X_test_t)  * amp_scale
    else:
        amp_head = None
        train_am = np.zeros(len(X_train_t))
        val_am   = np.zeros(len(X_val_t))
        test_am  = np.zeros(len(X_test_t))

    # SAL 时段锚 latent 分量 (YOLOv2 锚框时序映射: 时段马氏取max, 难样本局部偏移)
    # 独立阈值方案已弃: val 分位在测试集分布漂移下失效, 穷举4锚分位无组合满足FPR<1%
    print(f'  [分量9/{n_comp}] SegLatent 缩放 (时段锚马氏max, 参考val)...')
    seg_scales, seg_stats = auto_scale_seg_latent(model, X_val_t)
    train_sg = compute_seg_latent_scores(model, X_train_t, seg_scales, seg_stats, cfg.detect.seg_weight)
    val_sg   = compute_seg_latent_scores(model, X_val_t,   seg_scales, seg_stats, cfg.detect.seg_weight)
    test_sg  = compute_seg_latent_scores(model, X_test_t,  seg_scales, seg_stats, cfg.detect.seg_weight)

    # ClusterLatent 机簇锚 latent 分量 (幅度类异常"相对自身机簇基线"检测)
    # 在训练集正常 latent 上按机聚类 + 每簇独立马氏, 样本取最匹配簇 (min)
    cluster_weight = cfg.detect.cluster_weight
    if cluster_weight > 0:
        print(f'  [分量10/{n_comp}] ClusterLatent 缩放 (机簇锚马氏min, k={cfg.detect.cluster_k}, 参考train)...')
        cluster_stats, cl_mean, cl_std = auto_scale_cluster_latent(
            model, X_train_t, cfg.detect.cluster_k)
        train_cl = compute_cluster_latent_scores(model, X_train_t, cluster_stats,
                                                 cl_mean, cl_std, cluster_weight)
        val_cl   = compute_cluster_latent_scores(model, X_val_t,   cluster_stats,
                                                 cl_mean, cl_std, cluster_weight)
        test_cl  = compute_cluster_latent_scores(model, X_test_t,  cluster_stats,
                                                 cl_mean, cl_std, cluster_weight)
    else:
        train_cl = np.zeros(len(X_train_t))
        val_cl   = np.zeros(len(X_val_t))
        test_cl  = np.zeros(len(X_test_t))

    # RelErr 相对物理特征分量 (转换/峰值、转换/解锁、转换段波动 — 低基础机卡阻的波形级信号)
    rel_weight = cfg.detect.rel_weight
    if rel_weight > 0:
        print(f'  [分量11/{n_comp}] RelErr 缩放 (相对物理特征 Σz², 参考train)...')
        rel_stats, rel_scale = auto_scale_rel(X_train_t)
        train_rel = compute_rel_scores(X_train_t, rel_stats, rel_scale, rel_weight)
        val_rel   = compute_rel_scores(X_val_t,   rel_stats, rel_scale, rel_weight)
        test_rel  = compute_rel_scores(X_test_t,  rel_stats, rel_scale, rel_weight)
    else:
        rel_stats = None
        train_rel = np.zeros(len(X_train_t))
        val_rel   = np.zeros(len(X_val_t))
        test_rel  = np.zeros(len(X_test_t))

    # AlignResidual 条件对齐残差分量 (TimeCMA 跨模态对齐: 学 f≈g(z) 正常关系, 残差作异常度)
    if align_enabled:
        ref_set = X_val_t if cfg.detect.align_residual_fit_set == 'val' else X_train_t
        print(f'  [分量12/{n_comp}] AlignResidual 缩放 (条件对齐残差, '
              f'参考{cfg.detect.align_residual_fit_set}, map={cfg.detect.align_residual_map})...')
        align_stats = auto_scale_align(model, ref_set,
                                       map_type=cfg.detect.align_residual_map,
                                       fit_n=cfg.detect.align_residual_fit_n,
                                       epochs=cfg.detect.align_residual_epochs)
        train_al = compute_align_scores(model, X_train_t, align_stats, align_weight)
        val_al   = compute_align_scores(model, X_val_t,   align_stats, align_weight)
        test_al  = compute_align_scores(model, X_test_t,  align_stats, align_weight)
    else:
        align_stats = None
        train_al = np.zeros(len(X_train_t))
        val_al   = np.zeros(len(X_val_t))
        test_al  = np.zeros(len(X_test_t))

    # 综合 (含 SAL max + 机簇锚 min + RelErr + AlignResidual)
    # 分量剪枝 (cfg.detect.pruned_components): 融合时跳过 (置零), 不影响各分量单独评估.
    # 卡阻"仅正权5个" = ['sp','nc','lt','sg','am'] → 只留 PW-MSE/Phase/Cluster/RelErr
    pruned = set(getattr(cfg.detect, 'pruned_components', []))
    if pruned:
        print(f'  [剪枝] 融合跳过分量: {sorted(pruned)} (其余正常参与)')
    def _fuse(*named):
        return sum(arr for name, arr in named if name not in pruned)
    train_scores = _fuse(('mp', train_mp), ('ph', train_ph), ('sp', train_sp),
                         ('nc', train_nc), ('lt', train_lt), ('sg', train_sg),
                         ('cl', train_cl), ('rel', train_rel), ('phz', train_phz),
                         ('am', train_am), ('al', train_al))
    val_scores   = _fuse(('mp', val_mp),   ('ph', val_ph),   ('sp', val_sp),
                         ('nc', val_nc),   ('lt', val_lt),   ('sg', val_sg),
                         ('cl', val_cl),   ('rel', val_rel), ('phz', val_phz),
                         ('am', val_am),   ('al', val_al))
    test_scores  = _fuse(('mp', test_mp),  ('ph', test_ph),  ('sp', test_sp),
                         ('nc', test_nc),  ('lt', test_lt),  ('sg', test_sg),
                         ('cl', test_cl),  ('rel', test_rel), ('phz', test_phz),
                         ('am', test_am),  ('al', test_al))

    print(f'    PW-MSE+Peak:     训练均值={np.mean(train_mp):.4f}')
    print(f'    DS-Err(差分):    训练均值={np.mean(train_ds):.4f}')
    print(f'    PhaseErr(相位):  训练均值={np.mean(train_ph):.4f}')
    print(f'    SpectralErr(频谱): 训练均值={np.mean(train_sp):.4f}')
    if nc_enabled:
        print(f'    NormErr(正常性): 训练均值={np.mean(train_nc):.4f}')
    print(f'    LatentErr(latent): 训练均值={np.mean(train_lt):.4f}')
    print(f'    SegLatent(时段锚): 训练均值={np.mean(train_sg):.4f}')
    if cluster_weight > 0:
        print(f'    ClusterLatent(机簇锚): 训练均值={np.mean(train_cl):.4f}')
    if rel_weight > 0:
        print(f'    RelErr(相对特征): 训练均值={np.mean(train_rel):.4f}')
    print(f'    PhysErr(物理):   训练均值={np.mean(train_phz):.4f}')
    print(f'    AmpHead(幅度):  训练均值={np.mean(train_am):.4f}')
    if align_enabled:
        print(f'    AlignResidual(对齐): 训练均值={np.mean(train_al):.4f}')
    print(f'    综合分数:       训练均值={np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}')
    print(f'                    验证均值={np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}')

    # 4. 阈值 + 评估 (weighted 加权和 vs or 各分量独立)
    fusion_mode = cfg.detect.fusion_mode
    if fusion_mode == 'or':
        print(f'  [融合] OR 规则: 各分量独立 P{cfg.detect.or_percentile:.1f} 阈值, 任一超即异常')
        components = {
            'PW-MSE+Peak': test_mp, 'PhaseErr': test_ph,
            'SpectralErr': test_sp, 'NormErr': test_nc,
            'LatentErr': test_lt, 'PhysErr': test_phz,
            'AmpHead': test_am,
        }
        val_map = {
            'PW-MSE+Peak': val_mp, 'PhaseErr': val_ph,
            'SpectralErr': val_sp, 'NormErr': val_nc,
            'LatentErr': val_lt, 'PhysErr': val_phz,
            'AmpHead': val_am,
        }
        results = evaluate_or(components, val_map, labels_test,
                              percentile=cfg.detect.or_percentile)
        threshold = results['or_thresholds']      # dict: 各分量独立阈值
        # OR 模式无单一综合阈值: 供可视化/诊断脚本的参考阈值取各分量阈值中位数,
        # 真实判定以 results['predictions'] (各分量独立阈值 OR) 为准
        main_threshold = float(np.median(list(threshold.values())))
    else:
        threshold = choose_threshold(val_scores)
        main_threshold = threshold
        results = evaluate(test_scores, labels_test, threshold)

    n_test = len(labels_test)
    n_anom = int(labels_test.sum())
    n_norm = n_test - n_anom
    print(f'\n  检测 (测试集: 正常 {n_norm} / 异常 {n_anom}):')
    print(f'    TP={results["tp"]}  FP={results["fp"]}  '
          f'TN={results["tn"]}  FN={results["fn"]}')
    print(f'    AUC-ROC: {results["auc_roc"]:.4f}  |  AUC-PR: {results["auc_pr"]:.4f}  ← 不平衡数据看这个!')
    print(f'    精确率:  {results["precision"]:.4f}  |  召回率:  {results["recall"]:.4f}')
    print(f'    F1:       {results["f1"]:.4f}')
    print(f'    虚警率(FPR): {results["fpr"]:.4f} ({results["fpr"]*100:.2f}%)')
    print(f'    误报率(FDR): {results["fdr"]:.4f} ({results["fdr"]*100:.2f}%)')

    # 6. 各分量贡献对比
    test_mp_only, _, _ = compute_scores(model, X_test_t, mse_scale, peak_scale, 0.0, mse_weight)
    mp_auc  = roc_auc_score(labels_test, test_mp)
    ph_auc  = roc_auc_score(labels_test, test_ph)
    sp_auc  = roc_auc_score(labels_test, test_sp)
    ds_auc  = roc_auc_score(labels_test, test_ds)
    full_auc = results['auc_roc']
    print(f'\n  分量贡献 (AUC-ROC):')
    print(f'    PW-MSE (w_MSE={mse_weight}):        {mp_auc:.4f}')
    if alpha > 0:
        test_peak_only, _, _ = compute_scores(model, X_test_t, 0.0, peak_scale, alpha, 1.0)
        print(f'    PeakErr (α={alpha}):                {roc_auc_score(labels_test, test_peak_only):.4f}')
    print(f'    DS-Err (差分形态):              {ds_auc:.4f}')
    print(f'    PhaseErr (相位结构, β={beta}):  {ph_auc:.4f}')
    print(f'    SpectralErr (频谱结构, γ={gamma}): {sp_auc:.4f}')
    if nc_enabled:
        nc_auc = roc_auc_score(labels_test, test_nc)
        print(f'    NormErr (正常性, δ={delta}):    {nc_auc:.4f}')
    lt_auc = roc_auc_score(labels_test, test_lt)
    print(f'    LatentErr (latent, ε={epsilon}):  {lt_auc:.4f}')
    sg_auc = roc_auc_score(labels_test, test_sg)
    print(f'    SegLatent (时段锚, ω={cfg.detect.seg_weight}): {sg_auc:.4f}')
    if cluster_weight > 0:
        cl_auc = roc_auc_score(labels_test, test_cl)
        print(f'    ClusterLatent (机簇锚, ω={cluster_weight}): {cl_auc:.4f}')
    if rel_weight > 0:
        rel_auc = roc_auc_score(labels_test, test_rel)
        print(f'    RelErr (相对特征, ω={rel_weight}): {rel_auc:.4f}')
    phz_auc = roc_auc_score(labels_test, test_phz)
    print(f'    PhysErr (物理, ζ={zeta}):      {phz_auc:.4f}')
    am_auc = roc_auc_score(labels_test, test_am)
    print(f'    AmpHead (幅度, {amp_head is not None}):   {am_auc:.4f}')
    if align_enabled:
        al_auc = roc_auc_score(labels_test, test_al)
        print(f'    AlignResidual (对齐, ω_a={align_weight}): {al_auc:.4f}')
    print(f'    综合:                            {full_auc:.4f}')

    # 7. 退化趋势预警分析 (仅 weighted 模式有单一阈值)
    warning_summary = None
    if (cfg.early_warning.enabled and fusion_mode == 'weighted'
            and len(test_scores) > cfg.early_warning.min_trend_days):
        print('\n' + '=' * 50)
        print('退化趋势预警分析 (逐台机器监测, 真实天数)')
        print('=' * 50)
        # 加载真实监测天数 + 机器编号; 缺失时回退为单序列
        days_test = None
        machines_test = None
        try:
            from scipy import io as sio
            days_path = os.path.join(cfg.data.data_dir, 'days_test.mat')
            mach_path = os.path.join(cfg.data.data_dir, 'machines_test.mat')
            if os.path.exists(days_path):
                days_test = sio.loadmat(days_path)['days_test'].ravel().astype(np.int64)
            if os.path.exists(mach_path):
                machines_test = sio.loadmat(mach_path)['machines_test'].ravel().astype(np.int64)
        except Exception:
            days_test = None
            machines_test = None

        if machines_test is None or len(machines_test) != len(test_scores):
            # 无机器信息 → 单序列预警
            ws = WarningSystem(train_scores)
            ws.feed_sequence(test_scores, days=days_test)
            warning_summary = ws.summary()
            ws.plot_trend(os.path.join(cfg.save_dir, '07_warning_trend.png'))
        else:
            # 逐机器监测: 每台机器独立 WarningSystem (滚动窗口从零开始),
            # 阈值统一用训练集分位; 收集各机器预警摘要 + 日最大分数曲线供总览图
            machine_summaries = {}
            machine_series = []
            m_ids = np.unique(machines_test)
            print(f'  [逐机器] {len(m_ids)} 台机器, 共用训练阈值')
            for m in m_ids:
                mask = machines_test == m
                m_scores = test_scores[mask]
                m_days = days_test[mask]
                m_lab = labels_test[mask]
                ws_m = WarningSystem(train_scores)
                ws_m.feed_sequence(m_scores, days=m_days)
                n_alarm = sum(1 for r in ws_m.records
                              if r.level >= WARNING_ORANGE)
                max_level = max((r.level for r in ws_m.records),
                                default=WARNING_GREEN)
                max_day = next((r.day for r in ws_m.records
                                if r.level == max_level), None)
                machine_summaries[int(m)] = {
                    'n': int(len(m_scores)),
                    'n_fault': int(m_lab.sum()),
                    'n_alarm': n_alarm,
                    'max_level': int(max_level),
                    'max_day': int(max_day) if max_day is not None else None,
                }
                days_u = np.unique(m_days)
                dmax = np.array([m_scores[m_days == d].max()
                                 for d in days_u])
                fmask = m_lab == 1
                machine_series.append({
                    'machine': int(m),
                    'days': days_u,
                    'scores': dmax,
                    'fault_days': m_days[fmask],
                    'fault_scores': m_scores[fmask],
                })
            warning_summary = {'machine_summaries': machine_summaries}
            print('  机器 | 样本 | 故障 | 预警次数 | 最高级别(天)')
            for m, sm in sorted(machine_summaries.items()):
                lvl = LEVEL_LABELS.get(sm['max_level'], '?')
                lvl = lvl.split(']')[-1].strip()
                line = (f'    M{m:<3d}| {sm["n"]:>5d} | {sm["n_fault"]:>3d} | '
                        f'{sm["n_alarm"]:>5d} | {lvl}')
                if sm['max_day']:
                    line += f' (第{sm["max_day"]}天)'
                print(line)
            plot_machine_overview(machine_series, main_threshold,
                                  os.path.join(cfg.save_dir,
                                               '07_warning_trend.png'))

        # RelErr 卡阻退化预警: 训练P99.5直接阈值 + 连续越线确认
        # (完整分的 latent/马氏分量对逐日带噪监测过敏感 → 用 RelErr 纯波形信号)
        if rel_weight > 0:
            rew = RelErrEarlyWarning(train_rel)
            rew.feed_sequence(test_rel)
            if rew.alarm_day is not None:
                n_samples = len(test_rel)
                print(f'  [RelErr预警] 首次橙警: 第 {rew.alarm_day} 样本 '
                      f'(RelErr={test_rel[rew.alarm_day]:.1f}, 阈值={rew.threshold:.1f}) | '
                      f'监测{n_samples}样本')
                warning_summary = warning_summary or {}
                warning_summary['relerr_alarm_day'] = rew.alarm_day
                warning_summary['relerr_threshold'] = rew.threshold

    scores = {
        'train': train_scores,
        'val': val_scores,
        'test': test_scores,
        'recon': recon,
        'pw_mse': test_extras['pw_mse'],
        'ds_err': test_ds,
        'phase_err': test_ph,
        'spectral_err': test_sp,
        'normalcy_err': test_nc,
        'latent_err': test_lt,
        'seg_latent': test_sg,
        'cluster_latent': test_cl,
        'rel_err': test_rel,
        'phys_err': test_phz,
        'amp_err': test_am,
        'align_err': test_al,
        'mse_only_test': test_mp_only,
        'phase_only_test': test_ph,
        'spectral_only_test': test_sp,
        'threshold': main_threshold,
        'or_thresholds': threshold if fusion_mode == 'or' else None,
    }

    return results, scores

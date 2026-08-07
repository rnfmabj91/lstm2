"""
===========================================================================
 相位特征提取模块 — 基于转辙机动作电流物理特性的结构特征

 功能:
   1. 自动分割电流曲线的 4 个物理阶段 (启动→解锁→转换→缓放)
   2. 提取 3 类高区分度相位特征 (峰值时间、启动能量、转换波动)
   3. 计算相位偏差分数 PhaseErr, 补充 MSE/PeakErr 未捕捉的信号

 相位特征 (每相 3 类):
   - peak_time:   峰值出现时间 (采样点)     ← 区分度 ~4.0σ, 最强信号
   - startup_energy: 启动段能量 (0~峰值+5点) ← 区分度 ~2.1σ
   - conv_fluct:  转换段波动 (峰峰值)       ← 区分度 ~1.1σ

 PhaseErr = Σ(z²(peak_time) + z²(startup_energy) + z²(conv_fluct)) / 9
===========================================================================
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from .config import cfg


# 采样点级常量随采样率缩放 (25Hz基准: 5点=0.2s; 100Hz→20点)
_SMOOTH_SIZE = max(5, int(0.2 * cfg.data.fs))   # 平滑窗 (200ms)
_START_OFF   = max(5, int(0.2 * cfg.data.fs))   # 峰值后启动能量窗宽 (200ms)

# 批量分块大小: 整集一次性转 float64 + 多个 (N,3,800) 临时数组可占数 GB
# (训练集 5.7 万条 × 800 点, float64 单数组 ~1.1GB, 系统内存不足会溢出到页面文件导致假死)
# 按块循环后每块 float64 峰值 ~80MB (4096×3×800×8B), 累计输出仅 (N,9) 微不足道
_PHASE_CHUNK = 4096


def extract_phase_features(signal: np.ndarray) -> np.ndarray:
    """
    提取单样本 3 相电流的相位结构特征.

    Args:
        signal: (3, T) 归一化电流曲线 (T=200)

    Returns:
        (9,) 特征向量: [A_peak_time, A_startup_energy, A_conv_fluct,
                        B_peak_time, B_startup_energy, B_conv_fluct,
                        C_peak_time, C_startup_energy, C_conv_fluct]
    """
    T = signal.shape[-1]
    feats = []
    for ch in range(3):
        s = signal[ch, :]

        # --- 1. 峰值时间 (采样点位置) ---
        # 平滑后找峰值, 避免噪声导致 argmax 跳变
        sm = uniform_filter1d(s.astype(np.float64), size=_SMOOTH_SIZE)
        peak_time = int(np.argmax(sm))

        # --- 2. 启动能量 (0 ~ 峰值后+200ms) ---
        end = min(peak_time + _START_OFF + 1, T)
        startup_energy = float(np.sum(sm[:end] ** 2))

        # --- 3. 转换波动 (峰值后0.4s ~ 缓放前4.6s) ---
        # 物理常量按采样率缩放 (25Hz基准: 10点=0.4s, 100点=4.0s, 115点=4.6s)
        sc = cfg.data.fs / 25.0
        start = min(peak_time + int(10*sc), int(100*sc))
        conv_seg = sm[start:int(115*sc)] if start < int(115*sc) else sm[start:]
        conv_fluct = float(np.max(conv_seg) - np.min(conv_seg)) if len(conv_seg) > 0 else 0.0

        feats.extend([float(peak_time), startup_energy, conv_fluct])

    return np.array(feats, dtype=np.float64)


def _batch_extract_features(signals: np.ndarray) -> np.ndarray:
    """
    向量化批量提取相位特征.

    Args:
        signals: (N, 3, T) 归一化电流曲线 (T=200)

    Returns:
        feats: (N, 9) 特征矩阵: [A_pt, A_se, A_cf, B_pt, B_se, B_cf, C_pt, C_se, C_cf]
    """
    N = len(signals)
    if N == 0:
        return np.zeros((0, 9), dtype=np.float64)

    T = signals.shape[-1]
    sm = uniform_filter1d(signals.astype(np.float64), size=_SMOOTH_SIZE, axis=2)  # (N, 3, T)
    peak_time = np.argmax(sm, axis=2).astype(int)                     # (N, 3)

    t_idx = np.arange(T)[None, None, :]  # (1, 1, T)

    # startup_energy: 0 ~ peak_time + 200ms
    mask_start = t_idx <= (peak_time[:, :, None] + _START_OFF)
    startup_energy = np.sum(sm ** 2 * mask_start, axis=2)  # (N, 3)

    # conv_fluct: 峰值后0.4s ~ 缓放前4.6s (物理常量按 fs 缩放)
    sc = cfg.data.fs / 25.0
    conv_start = np.minimum(peak_time + int(10*sc), int(100*sc))
    mask_conv = (t_idx >= conv_start[:, :, None]) & (t_idx < int(115*sc))
    masked = np.where(mask_conv, sm, np.nan)
    with np.errstate(all='ignore'):
        conv_fluct = np.nan_to_num(
            np.nanmax(masked, axis=2) - np.nanmin(masked, axis=2), nan=0.0
        )  # (N, 3)

    # 拼成 (N, 9): [A_pt, A_se, A_cf, B_pt, ...]
    return np.stack([peak_time, startup_energy, conv_fluct], axis=2).reshape(N, 9)


def compute_phase_stats(train_signals: np.ndarray) -> dict:
    """
    在训练集上计算相位特征的均值/标准差.

    Args:
        train_signals: (N, 3, 125) 训练集电流曲线 (仅正常样本)

    Returns:
        {'mean': (9,) 均值, 'std': (9,) 标准差}
    """
    # 按块提取: 全量调用时 _batch_extract_features 内部会生成多个 (N,3,800) float64
    # 临时数组 (训练集 ~1.1GB/个), 分块后每块 ~80MB; 特征输出 (N,9) 累积仅 ~4MB
    n = len(train_signals)
    feats_list = []
    for i in range(0, n, _PHASE_CHUNK):
        feats_list.append(_batch_extract_features(train_signals[i:i + _PHASE_CHUNK]))
    all_feats = np.concatenate(feats_list, axis=0)

    mean = np.mean(all_feats, axis=0)
    std = np.std(all_feats, axis=0)
    std = np.clip(std, 1e-6, None)  # 防止除零

    print(f'  [相位] 特征均值: {np.round(mean, 3)}')
    print(f'  [相位] 特征标准差: {np.round(std, 3)}')

    return {'mean': mean, 'std': std}


def compute_phase_err(signal: np.ndarray, stats: dict) -> float:
    """
    计算单样本的相位偏差分数.

    PhaseErr = 均方 z-score (9维特征向量的卡方距离)

    Args:
        signal: (3, 125) 电流曲线
        stats: 训练集统计量 (mean, std)

    Returns:
        phase_err: 标量, 越大表示相位结构越异常
    """
    feats = extract_phase_features(signal)
    z = (feats - stats['mean']) / stats['std']
    # 卡方距离: 均方 z-score
    phase_err = float(np.mean(z ** 2))
    return phase_err


def batch_phase_err(signals: np.ndarray, stats: dict) -> np.ndarray:
    """
    批量计算相位偏差分数 (向量化实现, 避免 Python 循环).

    Args:
        signals: (N, 3, 125) 电流曲线
        stats: 训练集统计量 {'mean': (9,), 'std': (9,)}

    Returns:
        (N,) 相位偏差分数
    """
    n = len(signals)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, _PHASE_CHUNK):
        feats = _batch_extract_features(signals[i:i + _PHASE_CHUNK])  # (chunk, 9)
        z = (feats - stats['mean']) / stats['std']                     # (chunk, 9)
        out[i:i + _PHASE_CHUNK] = np.mean(z ** 2, axis=1)
    return out


def auto_scale_phase(train_signals: np.ndarray,
                     val_signals: np.ndarray) -> tuple:
    """
    自动确定 PhaseErr 的缩放系数 (中位数归一化).

    Args:
        train_signals: (N, 3, 125) 训练集
        val_signals:   (M, 3, 125) 验证集

    Returns:
        (phase_scale, phase_stats)
    """
    stats = compute_phase_stats(train_signals)
    train_err = batch_phase_err(train_signals, stats)  # 向量化后全量计算
    phase_med = max(np.median(train_err), 1e-10)
    phase_scale = 1.0 / phase_med

    print(f'  [相位] PhaseErr中位数={phase_med:.4f} → scale={phase_scale:.2f}')

    return phase_scale, stats

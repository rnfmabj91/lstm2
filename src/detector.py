"""
===========================================================================
 异常检测模块 v2 (精简: 仅 4 个参与分量)

 核心:
   anomaly_score = w_MSE × PW-MSE × scale
                 + β × PhaseErr × scale
                 + ω_c × ClusterLatent
                 + ω_r × RelErr

 各分量自动缩放到训练集上的中位数量级, 确保平衡贡献 (权重见 cfg.detect).
 PhaseErr 捕捉峰值时间偏移、启动能量畸变等相位结构异常.
 ClusterLatent 机簇锚 latent 马氏 (幅度类异常相对自身机簇基线).
 RelErr 相对物理特征 (低基础机卡阻的波形级信号).

 流程:
   1. 在训练集上计算各分量的缩放系数
   2. 计算训练/验证/测试三集异常分数
   3. 基于验证集分位数确定阈值
   4. 输出检测指标
===========================================================================
"""
import os
import numpy as np
import torch
import torch.nn as nn

from .config import cfg
from .early_warning import (WarningSystem, RelErrEarlyWarning,
                            plot_machine_overview,
                            WARNING_GREEN, WARNING_YELLOW,
                            WARNING_ORANGE, WARNING_RED, LEVEL_LABELS)
from .phase_features import auto_scale_phase, batch_phase_err
from .model import active_region_mask
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

def _get_pw(device: torch.device) -> torch.Tensor:
    """获取已缓存到指定设备的相区权重"""
    if device not in _phase_weights_cache:
        _phase_weights_cache[device] = PHASE_WEIGHTS.to(device)
    return _phase_weights_cache[device]


@torch.no_grad()
def compute_scores(model: nn.Module, x: torch.Tensor,
                   mse_scale: float = 1.0, mse_weight: float = 1.0):
    """
    计算 PW-MSE 异常分数 (时域算法):
      score = mse_weight × PW-MSE × scale_mse

    时域算法:
      1. PW-MSE: 相区加权MSE — 转换段权重2.0, 噪声段0.5
      2. DS-Err:  一阶差分形态误差 (单独返回, 可选融合)

    Args:
        model: CNN-LSTM 自编码器
        x: (B, 3, T) 输入电流 (100Hz → T=800)
        mse_scale: MSE 分量缩放系数
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

    # --- 综合 (显式 mse_weight 降权过重的重构误差) ---
    scores = mse_weight * pw_mse * mse_scale

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
#  latent 提取 — LSTM 潜在表示 (机簇锚 ClusterLatent 的共享基础设施)
# ============================================================
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


def auto_scale_components(model: nn.Module, x_ref: torch.Tensor) -> float:
    """
    自动确定 PW-MSE 分量缩放系数 (中位数归一化).

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

        pw_mse_med = max(pw_mse.median().item(), 1e-10)
        mse_scale = 1.0 / pw_mse_med

    print(f'    分量缩放: PW-MSE中位数={pw_mse_med:.6f} → scale={mse_scale:.1f}')

    return mse_scale


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
    完整异常检测流程 (4 个参与分量: PW-MSE / PhaseErr / ClusterLatent / RelErr).

    Returns:
        (results dict, scores dict)
    """
    beta = cfg.detect.beta
    mse_weight = cfg.detect.mse_weight
    cluster_weight = cfg.detect.cluster_weight
    rel_weight = cfg.detect.rel_weight

    formula = ('score = w_MSE×PW-MSE×scale + β×PhaseErr×scale '
               '+ ω_c×ClusterLatent + ω_r×RelErr')
    print('\n' + '=' * 50)
    print(f'异常检测: {formula}')
    print(f'          w_MSE={mse_weight}, β={beta}, ω_c={cluster_weight}, ω_r={rel_weight}')
    print('=' * 50)

    # 1. PW-MSE 自动缩放
    print('  [分量1/4] PW-MSE 缩放 (相区加权重构误差)...')
    mse_scale = auto_scale_components(model, X_train_t)

    # 2. PhaseErr 缩放 (相位结构偏差)
    print('  [分量2/4] PhaseErr 缩放 (相位结构偏差)...')
    X_train_np = X_train_t.cpu().numpy()
    X_val_np   = X_val_t.cpu().numpy()
    phase_scale, phase_stats = auto_scale_phase(X_train_np, X_val_np)
    del X_train_np, X_val_np   # 用后即弃: 各占 547/149MB, 留到函数末会与后续分量叠加

    # 3. 计算三集综合分数 (PW-MSE + PhaseErr)
    print('  [分量3/4] 计算综合异常分数...')
    train_mp, _, train_extras = compute_scores(model, X_train_t, mse_scale, mse_weight)
    val_mp,   _, val_extras   = compute_scores(model, X_val_t,   mse_scale, mse_weight)
    test_mp, recon, test_extras = compute_scores(model, X_test_t, mse_scale, mse_weight)

    train_ph = compute_phase_scores(X_train_t, phase_scale, phase_stats, beta)
    val_ph   = compute_phase_scores(X_val_t,   phase_scale, phase_stats, beta)
    test_ph  = compute_phase_scores(X_test_t,  phase_scale, phase_stats, beta)

    # DS-Err 一阶差分形态误差 (单独评估, 暂不融入主分数)
    train_ds = train_extras['ds_err']
    val_ds   = val_extras['ds_err']
    test_ds  = test_extras['ds_err']

    # 4. ClusterLatent (机簇锚马氏) + RelErr (相对物理特征)
    print('  [分量4/4] ClusterLatent + RelErr 缩放...')
    cluster_stats, cl_mean, cl_std = auto_scale_cluster_latent(
        model, X_train_t, cfg.detect.cluster_k)
    train_cl = compute_cluster_latent_scores(model, X_train_t, cluster_stats,
                                             cl_mean, cl_std, cluster_weight)
    val_cl   = compute_cluster_latent_scores(model, X_val_t,   cluster_stats,
                                             cl_mean, cl_std, cluster_weight)
    test_cl  = compute_cluster_latent_scores(model, X_test_t,  cluster_stats,
                                             cl_mean, cl_std, cluster_weight)

    rel_stats, rel_scale = auto_scale_rel(X_train_t)
    train_rel = compute_rel_scores(X_train_t, rel_stats, rel_scale, rel_weight)
    val_rel   = compute_rel_scores(X_val_t,   rel_stats, rel_scale, rel_weight)
    test_rel  = compute_rel_scores(X_test_t,  rel_stats, rel_scale, rel_weight)

    # 综合 = PW-MSE + PhaseErr + ClusterLatent + RelErr
    train_scores = train_mp + train_ph + train_cl + train_rel
    val_scores   = val_mp + val_ph + val_cl + val_rel
    test_scores  = test_mp + test_ph + test_cl + test_rel

    print(f'    PW-MSE:      训练均值={np.mean(train_mp):.4f}')
    print(f'    DS-Err(差分): 训练均值={np.mean(train_ds):.4f}')
    print(f'    PhaseErr(相位): 训练均值={np.mean(train_ph):.4f}')
    print(f'    ClusterLatent(机簇锚): 训练均值={np.mean(train_cl):.4f}')
    print(f'    RelErr(相对特征): 训练均值={np.mean(train_rel):.4f}')
    print(f'    综合分数:       训练均值={np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}')
    print(f'                    验证均值={np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}')

    # 5. 阈值 + 评估 (weighted 加权和 vs or 各分量独立)
    fusion_mode = cfg.detect.fusion_mode
    if fusion_mode == 'or':
        print(f'  [融合] OR 规则: 各分量独立 P{cfg.detect.or_percentile:.1f} 阈值, 任一超即异常')
        components = {
            'PW-MSE': test_mp, 'PhaseErr': test_ph,
            'ClusterLatent': test_cl, 'RelErr': test_rel,
        }
        val_map = {
            'PW-MSE': val_mp, 'PhaseErr': val_ph,
            'ClusterLatent': val_cl, 'RelErr': val_rel,
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

    # 6. 各分量贡献对比 (AUC-ROC)
    full_auc = results['auc_roc']
    print(f'\n  分量贡献 (AUC-ROC):')
    print(f'    PW-MSE (w_MSE={mse_weight}):        '
          f'{roc_auc_score(labels_test, test_mp):.4f}')
    print(f'    DS-Err (差分形态):              '
          f'{roc_auc_score(labels_test, test_ds):.4f}')
    print(f'    PhaseErr (相位结构, β={beta}):  '
          f'{roc_auc_score(labels_test, test_ph):.4f}')
    print(f'    ClusterLatent (机簇锚, ω={cluster_weight}): '
          f'{roc_auc_score(labels_test, test_cl):.4f}')
    print(f'    RelErr (相对特征, ω={rel_weight}): '
          f'{roc_auc_score(labels_test, test_rel):.4f}')
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
        'cluster_latent': test_cl,
        'rel_err': test_rel,
        'mse_only_test': test_mp,
        'phase_only_test': test_ph,
        'threshold': main_threshold,
        'or_thresholds': threshold if fusion_mode == 'or' else None,
    }

    return results, scores

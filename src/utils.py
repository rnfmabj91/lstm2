"""
===========================================================================
 工具模块 - 可视化 + 日志
===========================================================================
"""
import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import ConfusionMatrixDisplay

from .config import cfg

# ============================================================
#  中文字体初始化 (全局)
# ============================================================
def _setup_chinese_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC',
                  'WenQuanYi Micro Hei', 'PingFang SC']
    import matplotlib.font_manager as _fm
    avail = {f.name for f in _fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in avail), 'sans-serif')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    if chosen != 'sans-serif':
        print(f'  [字体] 中文字体: {chosen}')

_setup_chinese_font()
warnings.filterwarnings('ignore')


def plot_losses(history: dict, save_path: str):
    """训练 & 验证损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['train_losses'], 'b-', lw=1.5, label='训练')
    ax.plot(history['val_losses'], 'r-', lw=1.5, label='验证')
    best = history['best_epoch'] - 1
    ax.axvline(best, color='g', ls='--',
               label=f'最佳 epoch {history["best_epoch"]} '
                     f'({history["best_val_loss"]:.6f})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('训练 & 验证损失'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] 损失曲线 → {os.path.basename(save_path)}')


def plot_score_distribution(train_scores, val_scores, test_scores,
                            labels_test, threshold, save_path: str):
    """训练/验证/测试 三集分数分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(train_scores, bins=50, color='blue', alpha=0.6, edgecolor='white')
    axes[0].set_title(f'训练 (低阻力)\n均值={np.mean(train_scores):.4f}')
    axes[0].set_xlabel('异常分数'); axes[0].set_ylabel('频次'); axes[0].grid(True, alpha=0.3)

    axes[1].hist(val_scores, bins=50, color='magenta', alpha=0.6, edgecolor='white')
    axes[1].axvline(threshold, color='red', ls='--', lw=2, label=f'阈值={threshold:.4f}')
    axes[1].set_title(f'验证 (阻力↑)\n均值={np.mean(val_scores):.4f}')
    axes[1].set_xlabel('异常分数'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].hist(test_scores[labels_test == 0], bins=50, color='cyan', alpha=0.5, label='正常')
    axes[2].hist(test_scores[labels_test == 1], bins=50, color='red', alpha=0.7, label='异常')
    axes[2].axvline(threshold, color='red', ls='--', lw=2)
    axes[2].set_title(f'测试 ({np.sum(labels_test==0)}正常:{np.sum(labels_test==1)}异常)')
    axes[2].set_xlabel('异常分数'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] 分数分布 → {os.path.basename(save_path)}')


def plot_roc_pr(labels_test, scores, save_path: str):
    """ROC 和 PR 曲线"""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    fpr, tpr, _ = roc_curve(labels_test, scores)
    prec, rec, _ = precision_recall_curve(labels_test, scores)
    auc_roc = auc(fpr, tpr)
    auc_pr  = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC-ROC={auc_roc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.15, color='blue')
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC 曲线'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(rec, prec, 'r-', lw=2, label=f'AUC-PR={auc_pr:.4f}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('PR 曲线'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] ROC/PR → {os.path.basename(save_path)}')


def plot_recon_samples(X_train, X_val, X_test, labels_test,
                       test_scores, model, device, save_path: str):
    """原始 vs 重构: 训练/验证/异常 对比"""
    T = X_test.shape[-1]                     # 序列长度 (800→8s @100Hz)
    t_axis = np.arange(T) / cfg.data.fs
    model.eval()

    # 转为 Tensor
    if isinstance(X_train, np.ndarray):
        X_train_t = torch.FloatTensor(X_train).to(device)
        X_val_t   = torch.FloatTensor(X_val).to(device)
        X_test_t  = torch.FloatTensor(X_test).to(device)
    else:
        X_train_t, X_val_t, X_test_t = X_train, X_val, X_test

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    with torch.no_grad():
        r_train = model(X_train_t[:1])
        r_val   = model(X_val_t[:1])

    pairs = [
        ('Train (early, low R)', X_train_t[0].cpu(), r_train[0].cpu()),
        ('Val (mid, R↑)', X_val_t[0].cpu(), r_val[0].cpu()),
    ]

    for row, (title, orig, recon) in enumerate(pairs):
        for col, ch in enumerate(['A', 'B', 'C']):
            ax = axes[row, col]
            ax.plot(t_axis, orig[col].numpy(), 'b-', lw=1.2, label='Orig')
            ax.plot(t_axis, recon[col].numpy(), 'r--', lw=1.2, label='Recon')
            ax.set_title(f'{title} {ch}'); ax.set_xlabel('Time (s)')
            ax.set_ylabel('Current'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 异常样本
    fidx = np.where(labels_test == 1)[0]
    if len(fidx) > 0:
        f = fidx[0]
        with torch.no_grad():
            r_fault = model(X_test_t[f:f+1])
        for col, ch in enumerate(['A', 'B', 'C']):
            ax = axes[2, col]
            ax.plot(t_axis, X_test_t[f, col].cpu(), 'b-', lw=1.2, label='原始')
            ax.plot(t_axis, r_fault[0, col].cpu(), 'r--', lw=1.2, label='重构')
            ax.set_title(f'Fault #{f} {ch} (score={test_scores[f]:.4f})')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Current')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('原始 vs 重构: 训练 / 验证 / 异常', fontsize=14)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] 重构对比 → {os.path.basename(save_path)}')


def plot_confusion(labels_test, predictions, threshold, f1, save_path: str):
    """混淆矩阵"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        labels_test, predictions, display_labels=['正常', '异常'],
        cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(f'混淆矩阵 (阈值={threshold:.4f}, F1={f1:.4f})')
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] 混淆矩阵 → {os.path.basename(save_path)}')


def plot_score_decomposition(test_scores, mse_only_scores, labels_test,
                             threshold, alpha, save_path: str):
    """异常分数成分分解 (MSE vs PeakErr)"""
    idx_sort = np.argsort(test_scores)[::-1]
    fig, ax = plt.subplots(figsize=(12, 4))
    x_axis = np.arange(len(test_scores))
    ax.bar(x_axis, mse_only_scores[idx_sort], color='blue', alpha=0.5, label='MSE')
    ax.bar(x_axis, test_scores[idx_sort] - mse_only_scores[idx_sort],
           bottom=mse_only_scores[idx_sort], color='red', alpha=0.5,
           label=f'其余分量 (非MSE, α={alpha})')
    ax.axhline(threshold, color='green', ls='--', lw=2, label=f'阈值={threshold:.4f}')
    # 标记异常点
    for i in np.where(labels_test == 1)[0]:
        pos = np.where(idx_sort == i)[0][0]
        ax.plot(pos, test_scores[i], 'rv', markersize=10)
    ax.set_xlabel('样本 (降序)'); ax.set_ylabel('异常分数')
    ax.set_title(f'异常分数成分分解'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] 分数分解 → {os.path.basename(save_path)}')


def plot_val_trend(val_scores, X_val, threshold, save_path: str):
    """验证集: 异常分数 + 峰值趋势 vs 天数 (阻力增大效应)"""
    val_days = np.linspace(130, 160, len(val_scores))
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.scatter(val_days, val_scores, c='magenta', s=3, alpha=0.5, label='验证分数')
    ax1.axhline(threshold, color='red', ls='--', label=f'阈值={threshold:.4f}')
    ax1.set_xlabel('天数'); ax1.set_ylabel('异常分数', color='magenta')
    ax1.tick_params(axis='y', labelcolor='magenta')
    ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    trend = X_val.reshape(len(X_val), -1).max(axis=1)
    ax2.plot(val_days[:len(trend)], trend, 'c-', lw=2, alpha=0.7, label='峰值趋势')
    ax2.set_ylabel('峰值电流', color='cyan'); ax2.tick_params(axis='y', labelcolor='cyan')
    ax2.legend(loc='upper right')
    fig.suptitle('验证集: 异常分数 vs 天数 (阻力增大)')
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f'  [✓] 阻力趋势 → {os.path.basename(save_path)}')


def save_results(results: dict, history: dict, scores: dict,
                 labels_test: np.ndarray):
    """保存数值结果到文件"""
    np.save(os.path.join(cfg.save_dir, 'results.npy'), results)
    np.save(os.path.join(cfg.save_dir, 'test_scores.npy'), scores['test'])
    np.save(os.path.join(cfg.save_dir, 'predictions.npy'), results['predictions'])

    with open(os.path.join(cfg.save_dir, 'results_summary.txt'), 'w',
              encoding='utf-8') as f:
        f.write('==========================================\n')
        f.write('转辙机动作电流 双通路CNN-LSTM 异常检测结果\n')
        f.write('==========================================\n\n')
        f.write(f'score = w_MSE×PW-MSE×scale + β×PhaseErr×scale + ω_c×ClusterLatent + ω_r×RelErr,  '
                f'w_MSE={cfg.detect.mse_weight} β={cfg.detect.beta} '
                f'ω_c={cfg.detect.cluster_weight} ω_r={cfg.detect.rel_weight}\n')
        if cfg.fusion.enabled:
            f.write('时频融合: 拼接融合 (concat 双通路, 无 α 门控)\n')
        f.write('\n')
        f.write(f'训练: {history["train_losses"][-1]:.4f} (best {history["best_val_loss"]:.6f})\n')
        f.write(f'阈值: {scores["threshold"]:.6f}\n\n')
        f.write(f'AUC-ROC:  {results["auc_roc"]:.4f}\n')
        f.write(f'AUC-PR:   {results["auc_pr"]:.4f}\n')
        f.write(f'精确率:   {results["precision"]:.4f}\n')
        f.write(f'召回率:   {results["recall"]:.4f}\n')
        f.write(f'F1-Score: {results["f1"]:.4f}\n')
        f.write(f'TP={results["tp"]}  FP={results["fp"]}  '
                f'TN={results["tn"]}  FN={results["fn"]}\n')

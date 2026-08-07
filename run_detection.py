# -*- coding: utf-8 -*-
"""
lstm2 — 独立检测脚本 (最终架构)

加载 outputs/<model_dir>/best_model.pt, 在指定数据集上跑完整异常检测并报告指标.

用法:
    E:/anaconda/envs/DL1/python.exe run_detection.py
    E:/anaconda/envs/DL1/python.exe run_detection.py --model_dir models_blocking_xalign
    E:/anaconda/envs/DL1/python.exe run_detection.py --data_dir data/preprocessed --model_dir models_100hz
"""
import os, sys, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无头环境, 不弹交互窗口
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

_avail = {f.name for f in _fm.fontManager.ttflist}
_cn = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans SC'] if f in _avail), 'sans-serif')
plt.rcParams['font.sans-serif'] = [_cn, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import cfg
from src.model import CNNLSTM_Autoencoder
from src.detector import run_detection
from src.utils import (plot_score_distribution, plot_roc_pr, plot_recon_samples,
                       plot_confusion)
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data/raw_blocking/preprocessed',
                    help='检测数据目录 (默认 blocking)')
    ap.add_argument('--model_dir', type=str, default='models_blocking_xalign')
    ap.add_argument('--model_path', type=str, default=None, help='直接指定 checkpoint 路径')
    args = ap.parse_args()

    cfg.data.data_dir = args.data_dir
    cfg.model_dir = args.model_dir
    model_path = args.model_path or os.path.join('outputs', cfg.model_dir, 'best_model.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[设备] {device}')
    if not os.path.exists(model_path):
        print(f'[错误] 未找到模型: {model_path} — 先运行 train.py 或指定 --model_path')
        sys.exit(1)

    # ---------- 1. 数据 ----------
    print('\n' + '=' * 50)
    print('加载数据')
    print('=' * 50)
    from scipy import io as sio
    def load_mat(path, key):
        return sio.loadmat(os.path.join(cfg.data.data_dir, path))[key].astype(np.float32)
    X_train = load_mat('X_train.mat', 'X_train')
    X_val   = load_mat('X_val.mat',   'X_val')
    X_test  = load_mat('X_test.mat',  'X_test')
    labels  = load_mat('labels_test.mat', 'labels_test').ravel().astype(np.int64)
    n_fault = int(labels.sum())
    print(f'  训练: {X_train.shape[0]} | 验证: {X_val.shape[0]} | '
          f'测试: {X_test.shape[0]} (异常 {n_fault})')

    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t   = torch.from_numpy(X_val).to(device)
    X_test_t  = torch.from_numpy(X_test).to(device)

    # ---------- 2. 模型 ----------
    print('\n' + '=' * 50)
    print('加载模型参数')
    print('=' * 50)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = CNNLSTM_Autoencoder(
        in_channels=cfg.model.in_channels,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        dropout=cfg.model.dropout,
        freq_pathway=cfg.fusion.enabled,
        aux_features=cfg.model.aux_features,
        aux_hidden=cfg.model.aux_hidden,
        xalign_enabled=True,
        normalcy_enabled=cfg.model.normalcy_enabled,
        normalcy_bottleneck=cfg.model.normalcy_bottleneck,
        normalcy_weight=cfg.model.normalcy_weight,
        seq_len=cfg.data.total_pts,
        domain_phys=cfg.model.domain_phys,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # 兼容: 原项目 checkpoint 含 fusion_gate 占位 (已剔除), 其余须全匹配
    unexpected = [k for k in unexpected if 'fusion_gate' not in k]
    if missing:
        print(f'  [警告] 缺失权重 {len(missing)}: {missing[:5]}... → 用初始化值, 结果不可信')
    if unexpected:
        print(f'  [警告] 额外权重 {len(unexpected)}: {unexpected[:5]}... → 已忽略')
    model.eval()
    print(f'  参数量: {sum(p.numel() for p in model.parameters()):,}')

    # ---------- 3. 检测 ----------
    results, scores = run_detection(model, X_train_t, X_val_t, X_test_t, labels)

    # ---------- 4. 保存图 ----------
    os.makedirs(cfg.save_dir, exist_ok=True)
    plot_score_distribution(scores['train'], scores['val'], scores['test'],
                            labels, scores['threshold'],
                            os.path.join(cfg.save_dir, '02_score_dist.png'))
    plot_roc_pr(labels, scores['test'],
                os.path.join(cfg.save_dir, '03_roc_pr.png'))
    plot_recon_samples(X_train, X_val, X_test, labels, scores['test'],
                       model, device, os.path.join(cfg.save_dir, '04_recon_samples.png'))
    plot_confusion(labels, results['predictions'], scores['threshold'], results['f1'],
                   os.path.join(cfg.save_dir, '05_confusion.png'))
    print(f'[图] 已保存到 {cfg.save_dir}/ (02_score_dist / 03_roc_pr / 04_recon_samples / 05_confusion)')

    # ---------- 5. 摘要 ----------
    print(f'\n  {"=" * 50}')
    print(f'  检测完成!  数据: {cfg.data.data_dir}')
    print(f'  {"=" * 50}')
    print(f'  AUC-ROC: {results["auc_roc"]:.4f}  |  AUC-PR:  {results["auc_pr"]:.4f}')
    print(f'  精确率:  {results["precision"]:.4f}  |  召回率:  {results["recall"]:.4f}')
    print(f'  F1:      {results["f1"]:.4f}        |  虚警率(FPR): {results["fpr"]:.4f}')
    print(f'  检出:    {results["tp"]}/{results["tp"] + results["fn"]}  |  FP: {results["fp"]}')
    print(f'  {"=" * 50}')


if __name__ == '__main__':
    main()

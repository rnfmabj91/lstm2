"""
===========================================================================
 数据加载模块 — 加载预处理后的 .mat 文件
===========================================================================
"""
import os
import numpy as np
from scipy import io as sio
from .config import cfg


def load_mat(path, key):
    return sio.loadmat(os.path.join(cfg.data.data_dir, path))[key]


def load_all():
    """返回 {X_train, X_val, X_test, labels_test, PSD_train/val/test, peak_train, stats}"""
    print('=' * 50)
    print('加载预处理数据')
    print('=' * 50)

    X_train = load_mat('X_train.mat', 'X_train')
    X_val   = load_mat('X_val.mat',   'X_val')
    X_test  = load_mat('X_test.mat',  'X_test')
    labels  = load_mat('labels_test.mat', 'labels_test').ravel().astype(np.int64)

    # 注: 数据在 preprocess_data.m 中已做 min-max 归一化到 [0,1]
    #     此处不再重复标准化, 保持与 train_model.py / run_detection.py 一致

    # 加载辅助特征: PSD 频带功率
    try:
        PSD_train = load_mat('PSD_train.mat', 'PSD_train')
        PSD_val   = load_mat('PSD_val.mat',   'PSD_val')
        PSD_test  = load_mat('PSD_test.mat',  'PSD_test')
        print(f'  PSD: 训练 {PSD_train.shape} | 验证 {PSD_val.shape} | 测试 {PSD_test.shape}')
    except (FileNotFoundError, KeyError):
        print('  [警告] PSD 文件未找到, 返回 None')
        PSD_train = PSD_val = PSD_test = None

    # 加载时域峰值特征 (P_peak, T_peak, RMS 拼接)
    try:
        peak_train = load_mat('peak_train.mat', 'peak_train')
        print(f'  峰值特征: {peak_train.shape}')
    except (FileNotFoundError, KeyError):
        print('  [警告] peak_train.mat 未找到, 返回 None')
        peak_train = None

    stats = {
        'n_train': X_train.shape[0],
        'n_val':   X_val.shape[0],
        'n_test':  X_test.shape[0],
        'n_normal': int(np.sum(labels == 0)),
        'n_fault':  int(np.sum(labels == 1)),
    }

    print(f'  训练: {stats["n_train"]} | 验证: {stats["n_val"]} | '
          f'测试: {stats["n_test"]} (正常 {stats["n_normal"]}, 异常 {stats["n_fault"]})')

    return {
        'X_train': X_train.astype(np.float32),
        'X_val':   X_val.astype(np.float32),
        'X_test':  X_test.astype(np.float32),
        'labels_test': labels,
        'PSD_train': PSD_train.astype(np.float32) if PSD_train is not None else None,
        'PSD_val':   PSD_val.astype(np.float32)   if PSD_val   is not None else None,
        'PSD_test':  PSD_test.astype(np.float32)  if PSD_test  is not None else None,
        'peak_train': peak_train.astype(np.float32) if peak_train is not None else None,
        **stats,
    }

# -*- coding: utf-8 -*-
"""
lstm2 — 独立训练脚本 (最终架构: xalign + 拼接融合, 无 α)

默认在 blocking 数据集上训练 (最终模型同数据集), 输出 outputs/<model_dir>/best_model.pt

用法:
    E:/anaconda/envs/DL1/python.exe train.py                     # 默认 blocking, cfg.train.epochs
    E:/anaconda/envs/DL1/python.exe train.py --epochs 10 --batch 128
    E:/anaconda/envs/DL1/python.exe train.py --data_dir data/preprocessed --model_dir models_100hz
"""
import os, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from src.config import cfg
from src.data_loader import load_all
from build_model import build_final_model
from src.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None, help='覆盖 cfg.train.epochs')
    ap.add_argument('--batch', type=int, default=None, help='覆盖 batch_size')
    ap.add_argument('--data_dir', type=str, default='data/raw_blocking/preprocessed',
                    help='训练数据目录 (默认 blocking)')
    ap.add_argument('--model_dir', type=str, default='models_blocking_xalign')
    args = ap.parse_args()

    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch:
        cfg.train.batch_size = args.batch
    cfg.data.data_dir = args.data_dir
    cfg.model_dir = args.model_dir

    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {DEV} | data={cfg.data.data_dir} | 输出=outputs/{cfg.model_dir} '
          f'| epochs={cfg.train.epochs} | batch={cfg.train.batch_size}')

    data = load_all()
    X_train_t = torch.FloatTensor(data['X_train']).to(DEV)
    X_val_t   = torch.FloatTensor(data['X_val']).to(DEV)
    print(f'训练 {len(X_train_t)} | 验证 {len(X_val_t)}')

    model = build_final_model(device=DEV)
    print(f'  参数量: {sum(p.numel() for p in model.parameters()):,} '
          f'(xalign={model.xalign_enabled})')

    history = train(model, X_train_t, X_val_t)
    print(f'[训练完成] best_val={history["best_val_loss"]:.6f} @ epoch {history["best_epoch"]}')
    print(f'[保存] outputs/{cfg.model_dir}/best_model.pt')


if __name__ == '__main__':
    main()

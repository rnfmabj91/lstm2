# lstm2 — 转辙机动作电流异常检测与退化预警

基于 CNN-LSTM 自编码器的 **转辙机(道岔)动作电流** 异常检测与退化趋势预警框架。

输入三相动作电流波形(100Hz × 8s × 3 相),输出:逐样本异常分数 + 检测指标(召回/虚警率/AUC)+ **逐台机器的退化预警**。

- **最终确认架构 (2026-08-07)**:交叉注意力对齐(xalign)+ 检测侧"仅正权 5 分量剪枝"
- **数据**:卡阻(blocking)模拟数据集,20 台机器 × 180 天,故障率 0.1%
- **数据生成器**在 [`E:\py\PythonProject\lstm`](https://github.com/rnfmabj91/lstm)(MATLAB,未包含在本仓库)

---

## 1. 框架总览

```
                ┌────────────────────────────────────────────┐
                │            data/raw_blocking/preprocessed   │
                │  X_train/val/test (N,3,800) + labels + days │
                │  machines/hours  (逐行对齐的时序字段)         │
                └──────────────────┬─────────────────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                  src/model.py                        │
        │   CNNLSTM_Autoencoder                                │
        │   时域通路 5×ResConv(3→256)                           │
        │   频域通路 rFFT(6→256)                                │
        │   CrossAttentionAlign 双向交叉注意力对齐 (TimeCMA)    │
        │   → 拼接 → BiLSTM → ConvDecoder → 重构 (B,3,800)     │
        │   + AuxEncoder(PSD/峰值) + NormalcyModel(FNM)        │
        │   + 幅度解码头 + 时段判别头                            │
        └──────────────────────────┬──────────────────────────┘
                                   │ latent + 重构 + 多模态特征
        ┌──────────────────────────┴──────────────────────────┐
        │                 src/detector.py                      │
        │   run_detection: 12 个打分分量                       │
        │   PW-MSE/PeakErr · PhaseErr · SpectralErr · NormErr  │
        │   LatentErr · PhysErr · AmpHead · SegLatent          │
        │   ClusterLatent · RelErr · AlignResidual             │
        │   → weighted 加权和 (剪枝 5 分量)                     │
        │   → 验证集 P99.5 阈值 → 检测指标(AUC/F1/召回/虚警)    │
        └──────────────────────────┬──────────────────────────┘
                                   │ 测试集分数 + 机器/天数
        ┌──────────────────────────┴──────────────────────────┐
        │             src/early_warning.py                     │
        │   WarningSystem: 漂移补偿 + 趋势跟踪 + 分级阈值       │
        │   (绿/黄/橙/红)                                       │
        │   plot_machine_overview: 逐台机器退化预警总览图       │
        └────────────────────────────────────────────────────┘
```

### 模块职责

| 文件 | 职责 |
|---|---|
| `src/config.py` | 全部超参数集中管理(Data/Model/Train/Fusion/Detect/EarlyWarning) |
| `src/model.py` | `CNNLSTM_Autoencoder`:时频双通路 + 交叉注意力对齐 + 辅助特征 + 频域正常性模型 |
| `src/detector.py` | 检测管线:12 打分分量自动缩放 → 加权融合 → 阈值 → 评估;退化预警驱动 |
| `src/early_warning.py` | 退化预警:`WarningSystem`(漂移补偿/趋势跟踪/分级)、`RelErrEarlyWarning`、逐机器总览图 |
| `src/trainer.py` | 训练循环(early stop、best-val 保存) |
| `src/data_loader.py` | 加载预置 `.mat` 数据 |
| `src/phase_features.py` | 相位特征提取/统计/误差 |
| `src/utils.py` | 检测结果绘图(分数分布/ROC-PR/重构对比/混淆矩阵) |
| `build_model.py` | 构建最终架构模型 |
| `train.py` | 训练入口 |
| `run_detection.py` | 检测入口 |

---

## 2. 数据

**数据来源**:`E:\py\PythonProject\lstm` 的 MATLAB 生成器(`generate_switch_machine_data.m` + `preprocess_data.m`)。

- **采样率** 100Hz(满足 50Hz 工频电流抽样定理),单样本 8s = 800 点,三相 A/B/C
- **20 台机器 × 180 天**(2026-01-01 起),故障率 0.1%(全期仅 124 个故障,均匀散布)
- **预处理后**(卡阻集):训练 62,027 / 验证 16,916 / 测试 33,946(33,834 正常 + 112 故障)
- **严格时序**:测试集按机器分组(每台一个连续时间线,机器内按天+时排序),故障落在所属机器上;
  `days_*.mat / machines_*.mat / hours_*.mat` 与 `X_*.mat` 逐行对齐,供退化预警使用。

---

## 3. 模型架构(`src/model.py`)

`CNNLSTM_Autoencoder` — 时域-频域双通路自编码器:

| 组件 | 说明 |
|---|---|
| 时域通路 | 5×`ResidualConvBlock`(3→256),提取波形形态 |
| 频域通路 | `FreqEncoder`(rFFT 保留相位,6→256),提取频谱结构 |
| **CrossAttentionAlign** | 时域↔频域双向交叉注意力对齐(TimeCMA 思想),实证:精确率 29%→37%、虚警 0.70%→0.49% |
| 融合 | 双通路拼接 → BiLSTM(2 层,hidden 128) |
| 解码 | `ConvDecoder`(残差转置卷积)→ 重构 (B,3,800) |
| AuxEncoder | 在线辅助特征(PSD 频带 + 峰值)广播加至 LSTM 输出 |
| NormalcyModel | 频域正常性模型 FNM(瓶颈 16 维),重构误差作 NormErr 分量 |
| 幅度头 / 时段判别头 | 训练期辅助监督目标(latent→peak/RMS、时段损坏判别),强化 latent 表征 |

---

## 4. 异常检测管线(`src/detector.py`)

综合分数为各分量加权和(每个分量先在训练集上自动缩放):

```
score = w_MSE×PW-MSE + α×PeakErr + β×PhaseErr + γ×SpectralErr
      + δ×NormErr + ε×LatentErr + ζ×PhysErr + ω_s×SegLatent
      + ω_c×ClusterLatent + ω_r×RelErr + ω_a×AlignResidual
```

| 分量 | 含义 | 默认状态 |
|---|---|---|
| PW-MSE + PeakErr | 相区加权重构误差 + 峰值误差 | ✅ 参与 |
| PhaseErr (β=0.3) | 相位结构偏差(峰值时间偏移等形态畸变) | ✅ 参与 |
| SpectralErr (γ=0.5) | 频谱结构偏差 | 剪枝 |
| NormErr (δ=0.5) | 频域正常性重构误差 | 剪枝 |
| LatentErr (ε=0.7) | LSTM latent 马氏距离 | 剪枝 |
| PhysErr (ζ=0) | 物理特征 z-score | 关闭 |
| AmpHead | 幅度解码头 latent→peak/RMS | 剪枝 |
| SegLatent (ω=0.5) | 时段锚 latent(马氏 max) | 剪枝 |
| ClusterLatent (ω=1.0) | 机簇锚 latent(KMeans k=20,min 马氏) | ✅ 参与 |
| RelErr (ω=1.0) | 相对物理特征 Σz²(转换/峰值、转换/解锁、波动) | ✅ 参与 |
| AlignResidual (ω=0) | 条件对齐残差 | 关闭 |

- **融合**:`weighted`(加权和),**剪枝** `['sp','nc','lt','sg','am']`(默认"仅正权 5 分量")
- **省计算**:不参与最终评估的分量(剪枝 + 零权重)**直接跳过计算**(检测时打印 `[跳过]`),只算 PW-MSE/Phase/Cluster/RelErr 4 个参与分量
- **阈值**:验证集分数 **P99.5** 分位数(理论虚警率 ≈ 0.5%)
- **指标**:AUC-ROC / AUC-PR / 精确率 / 召回率 / F1 / 虚警率(FPR)

---

## 5. 退化预警(`src/early_warning.py`)

预警系统按 **逐台机器** 独立监测(每台机器自己的 180 天时间线):

- `DriftCompensator` — 滑动窗口 z-score,消除工况/季节漂移
- `TrendTracker` — 指数平滑 + 滑动窗口斜率(检测趋势加速)
- 分级阈值(基于训练集分数分位):绿(正常)→ 黄(关注,P95)→ 橙(预警,P99.5)→ 红(报警,P99.9)
- `plot_machine_overview` — 总览图:20 台机器共用 0–180 天轴,日最大分数曲线(对数轴)+ 红色 × 故障标记 + 阈值线

---

## 6. 使用方式

```bash
# 训练 (默认 blocking 数据集, 输出 outputs/<model_dir>/best_model.pt)
E:/anaconda/envs/DL1/python.exe train.py [--epochs 80] [--batch 128] \
    [--data_dir data/raw_blocking/preprocessed] [--model_dir models_blocking_xalign]

# 检测 (加载 best_model.pt, 输出指标 + 5 张图 + 逐机器预警总览)
E:/anaconda/envs/DL1/python.exe run_detection.py \
    [--data_dir data/raw_blocking/preprocessed] [--model_dir models_blocking_xalign]

# 模型前向验证
E:/anaconda/envs/DL1/python.exe build_model.py
```

**检测输出**(`outputs/`):`02_score_dist` 分数分布 / `03_roc_pr` ROC-PR / `04_recon_samples` 重构对比 / `05_confusion` 混淆矩阵 / `07_warning_trend` 逐机器退化预警总览。

---

## 7. 检测结果

最近一次检测(卡阻集,`models_blocking_xalign`):

| 指标 | 值 |
|---|---|
| AUC-ROC | 0.9921 |
| AUC-PR | 0.8443 |
| 召回率 | 85.7%(96/112) |
| 虚警率 (FPR) | 0.63%(212/33,834) |
| 精确率 | 31.2% |
| F1 | 0.457 |

> ℹ️ 由于不参与评估的分量(含未设种子的 AlignResidual/AmpHead 拟合)已被跳过,**最终融合分数只由确定性分量组成,指标可复现**。实测同配置两次运行 FP 一致(212)。

---

## 8. 主要依赖

- Python 3.10+ / PyTorch(含 CUDA)
- numpy / scipy / scikit-learn
- matplotlib

---

## 9. 备注

- **数据生成器不在本仓库**:卡阻数据由 `E:\py\PythonProject\lstm` 的 MATLAB 脚本生成并预处理。
- 故障率极低(0.1%),为极端类别不平衡场景;评估以 **AUC-PR / 召回率** 为主。

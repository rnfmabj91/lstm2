"""
===========================================================================
 配置文件 - 所有超参数集中管理
===========================================================================
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """数据相关参数"""
    data_dir: str = 'data/preprocessed'
    fs: int = 100                   # 采样率 Hz — 100Hz, 满足 50Hz 工频电流抽样定理 (≥2×50Hz)
    seg_len_sec: float = 8.0        # 有效段长度 (秒) — 覆盖正常动作全长 + 转换超时判据
    total_pts: int = 800            # seg_len_sec * fs (8s × 100Hz)
    test_ratio: float = 0.001       # 异常占比 ≈ 1/1000


@dataclass
class ModelConfig:
    """CNN-LSTM 模型架构参数"""
    in_channels: int = 3            # A/B/C 三相
    lstm_hidden: int = 128          # LSTM 隐层维度 (2026-08-03 扩展 latent 64→128: 难样本功率不足/启动冲击 latent 区分不足)
    lstm_layers: int = 2            # LSTM 层数 (15时间步用4层过深, 易过拟合)
    dropout: float = 0.2            # Dropout 比率
    domain_phys: bool = True        # 物理领域特征注入 (消融 no_phys 关闭; 主模型默认开)
    aux_features: bool = True       # 启用在线辅助特征 (PSD+峰值) 注入 (add 广播加至 LSTM 输出)
    aux_hidden: int = 64            # 辅助特征编码器隐层维度 (add 模式)
    normalcy_enabled: bool = True   # 频域正常性模型 FNM (拼接融合下不再拖垮训练; 12epoch验证 NormErr 0.042→0.023)
    normalcy_bottleneck: int = 16   # FNM 瓶颈维度 (z ∈ R^16)
    normalcy_weight: float = 0.1    # FNM 损失权重 λ_nc: loss = 重构MSE + λ_nc×NormErr
    # 幅度解码头 (B 阶段): latent → peak/RMS 辅助重构目标, 强制 latent 编码幅度信息
    # (诊断: 现有 latent 已含幅度信息, 幅度头 AUC 启动冲击 0.867/功率不足 0.823;
    #  训练时加此目标让 latent 更强调幅度 → 难样本分离更强)
    amp_head_train: bool = True     # 训练时启用幅度解码头
    amp_head_weight: float = 0.5    # 幅度头损失权重: loss = 重构MSE + λ_nc×NormErr + λ_amp×AmpErr
    # 时段判别头 (SegDisc): 自监督损坏判别, 替代检测侧手动时段马氏 (SAL/LatentErr)
    # 训练: 构造随机时段损坏样本(幅度×0.5/1.5), 判别头学"哪个时段被破坏" → 端到端学正常局部模式
    # 检测: 一次前向输出各时段损坏概率, 免检测侧马氏/z-score 现算 (提速 + 去特征工程)
    seg_disc_enabled: bool = True
    seg_disc_weight: float = 0.3    # 判别头损失权重: loss = 重构 + λ_amp + λ_nc×NormErr + λ_disc×DiscErr
    # 交叉注意力对齐 (TimeCMA 思想, 2026-08-06): 时域↔频域双分支互相检索上下文.
    # 实证 (卡阻): 精确率 29%→37%, FP 220→166, 虚警率 0.70%→0.49%, 召回 86.6% 持平
    # **2026-08-07 用户确认取 "xalign + 仅正权5个剪枝" 为最终模型** → 默认开启
    # (注: 原项目为兼容旧 checkpoint 默认关; 此处为 lstm2 最终架构提取, 无旧权重, 默认开)
    xalign_enabled: bool = True


@dataclass
class TrainConfig:
    """训练参数"""
    epochs: int = 80
    batch_size: int = 128    # 800点数据, 8GB显存下 128 稳妥
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 15


@dataclass
class FusionConfig:
    """FFT频域-时域双通路融合参数"""
    enabled: bool = True            # 拼接融合 (concat 双通路送LSTM), 无 α 门控; 8s/800点(100Hz)


@dataclass
class DetectConfig:
    """异常检测参数 (4 个参与分量)"""
    beta: float = 0.3               # 相位结构偏差权重: 捕捉峰值时间偏移等形态畸变
    cluster_k: int = 20            # 机簇锚 KMeans 簇数 (生成器 num_machines=20)
    cluster_weight: float = 1.0     # 机簇锚 latent 权重 (ClusterLatent, min-over-clusters 马氏):
                                    # 幅度类异常"相对自身机簇基线"检测, 与重构误差互补
    rel_clip: float = 3.0          # RelErr z 截断上界: clip=3 优于 6 (重尾受限, 卡阻中位>val P99.5 1.35×)
    rel_weight: float = 1.0        # 相对物理特征权重 (RelErr): Σ(z∈[0,3])², A相
                                    # 特征: r_cp=转换/峰值, r_cu=转换/解锁, conv_fluct=转换段std
                                    # 卡阻只抬转换段→比值升、功率不足整体等比例缩放→比值不变(天然免疫)
    mse_weight: float = 0.3         # PW-MSE 显式权重: 原隐含1.0过重 (占综合中位30.7%而AUC仅0.67, 稀释其它分量)
                                    # 网格搜索: MSE∈{0.3,0.5,0.7}均优于1.0, 0.3时综合 AUC-ROC 0.805→0.873
    threshold_percentile: float = 99.5  # 验证集分位数作为阈值 (weighted 融合下 P99.5 优先召回)
    fusion_mode: str = 'weighted'   # 分量融合: 'weighted'(加权和) / 'or'(各分量独立阈值, 任一超即异常)
    or_percentile: float = 99.95    # or 模式: 各分量的独立阈值分位数 (仅 or 模式用)


@dataclass
class EarlyWarningConfig:
    """退化趋势预警参数"""
    enabled: bool = True            # 启用趋势预警分析
    # 趋势跟踪
    trend_window: int = 7           # 滑动窗口大小 (样本)
    smooth_alpha: float = 0.3       # 指数平滑系数 (越大越关注近期)
    min_trend_days: int = 3         # 最小趋势计算样本数
    # 分级阈值 (基于训练集分数的分位数)
    p_green: float = 95.0           # 绿色→黄色 (关注)
    p_yellow: float = 99.5          # 黄色→橙色 (预警)
    p_orange: float = 99.9          # 橙色→红色 (报警)
    # 趋势加速阈值
    slope_warn: float = 0.05        # 斜率超过此值 → 橙色预警 (归一化分数斜率/样本)


@dataclass
class GlobalConfig:
    """全局配置"""
    seed: int = 42
    save_dir: str = 'outputs'
    model_dir: str = 'models_100hz'   # 模型参数子目录 (随采样率命名)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    early_warning: EarlyWarningConfig = field(default_factory=EarlyWarningConfig)


# 全局单例
cfg = GlobalConfig()

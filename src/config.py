"""
===========================================================================
 配置文件 - 所有超参数集中管理
===========================================================================
"""
from dataclasses import dataclass, field
from typing import List


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
    """异常检测参数"""
    alpha: float = 0.0              # 峰值点误差权重: score = w_MSE×MSE + α×PeakErr + β×PhaseErr + γ×SpectralErr + δ×NormErr + ε×LatentErr + ζ×PhysErr
                                    # α=0: 峰值分量实证反相关 (窗口min AUC 0.457, mean/max 重做最高仅 0.604), 已删除
    beta: float = 0.3               # 相位结构偏差权重: 捕捉峰值时间偏移等形态畸变
    gamma: float = 0.5              # 频谱结构偏差权重 (v2频谱特征 z-score; 网格搜索 0.5 优于 0.3)
    delta: float = 0.5              # 频域正常性偏差权重 (FNM子AE重构误差; 网格搜索 0.5 优于 0.3)
    epsilon: float = 0.7            # latent 空间马氏距离权重 (难样本: 无缓放台阶/功率不足/启动冲击; 诊断 latent AUC 0.999/0.810/0.663)
                                    # 网格搜索: ε∈{0.3,0.5,0.7} 单调提升 FPR<1%召回 56.3→62.1→67.0→68.9%, 取 0.7
    zeta: float = 0.0               # 物理特征 z-score 权重 (PhysErr) — 实测净负: z-score 重尾抬高 val P99.9 阈值(7.16→8.74),
                                    # 压过无缓放台阶的 LatentErr 信号 (9/13→3/13), 单分量 AUC 高但无法转综合召回. 置 0 禁用.
    seg_weight: float = 0.5         # 时段锚 latent 权重 (SAL, max 模式): 扫描 ω∈{0..0.5} 召回 77.7→81.6%, ω=0.5 最优
                                    # 注: 独立阈值模式 (anchor_percentiles) 下此权重不参与综合分, SAL 作独立触发
    cluster_k: int = 20            # 机簇锚 KMeans 簇数 (生成器 num_machines=20; 实测 k=20 AUC 卡阻0.997 难样本0.994)
    cluster_weight: float = 1.0     # 机簇锚 latent 权重 (ClusterLatent, min-over-clusters 马氏, 2026-08-05):
                                    # 主数据集(8故障) P99.5 召回 0.583→0.641(ω=0.5)→0.670(ω=1.0), 功率不足不变(全局LatentErr兜底);
                                    # blocking 卡阻 综合 AUC 0.962→0.985, 固定P99.5召回持平 85/112, FPR≈1%召回 0.759→0.777
                                    # 代价: 功率不足单分量 0.852→0.817 (min 让低幅样本匹配低基础簇), 与全局 LatentErr 互补
    rel_clip: float = 3.0          # RelErr z 截断上界: clip=3 优于 6 (重尾受限, 卡阻中位>val P99.5 1.35×)
    rel_weight: float = 1.0        # 相对物理特征权重 (RelErr, 2026-08-05): Σ(z∈[0,3])², A相
                                    # 特征: r_cp=转换/峰值, r_cu=转换/解锁, conv_fluct=转换段std
                                    # 卡阻只抬转换段→比值升、功率不足整体等比例缩放→比值不变(天然免疫)
                                    # **2026-08-05 用户决定默认开 1.0**: blocking 卡阻 P99.5召回 0.759→0.866 (97/112),
                                    # 注意主数据集(8故障)会回退 0.641→0.544 (val重尾抬阈值, PhysErr ζ=0 同理) —
                                    # 若要主模型优先, 改回 0.0
    # 时段锚独立阈值 (anchor_percentiles) 已弃: val 分位在测试集分布漂移下失效,
    # 穷举 4 锚分位 {99.3..99.95} 无任何组合满足 FPR<1% (锚触发 FP 过多). 保留 max 加权模式.
    mse_weight: float = 0.3         # MSE 显式权重: 原隐含1.0过重 (占综合中位30.7%而AUC仅0.67, 稀释Spectral/Norm)
                                    # 网格搜索: MSE∈{0.3,0.5,0.7}均优于1.0, 0.3时综合 AUC-ROC 0.805→0.873
    threshold_percentile: float = 99.5  # 验证集分位数作为阈值 (2026-08-03 weighted 融合下 P99.5 优先召回: 召回73.8%/FPR0.59%, 漏检32→27)
    fusion_mode: str = 'weighted'   # 分量融合: 'weighted'(加权和+P99.9) / 'or'(各分量独立P99.9, 任一超即异常)
                                    # 2026-08-03 lstm_hidden 64→128 后实测: weighted(含弱,ε=0.7)@P99.30 FPR<1%最高召回 76.7%
                                    # 优于 OR 全7(P99.95,72.8%) 与 OR 强3(73.8%); 维度扩展后 weighted 反超, 不再需要 OR 的独立阈值
    or_percentile: float = 99.95    # or 模式: 各分量的独立阈值分位数 (仅 or 模式用)
                                    # 实证(旧,lstm_hidden=64): 全7分量 P99.95 → 68.9% (0.51% FPR), 优于 P99.9(0.73%)
    amp_head_enabled: bool = True   # 幅度解码头 (latent→peak/RMS 重构误差, 在 X_train 正常样本上拟合)
                                    # 诊断: 启动冲击过高 AUC 0.867 / 功率不足 0.823, 全分量静默难样本的关键信号
    # 条件对齐残差 (AlignResidual) — TimeCMA 跨模态对齐思想 (2026-08-06):
    # 学正常跨模态关系 f ≈ g(z) (f=幅度时域+相对抬升特征, z=LSTM latent), 残差 ‖f−g(z)‖ 作异常度.
    # latent 提供"这台机正常应怎样"上下文 → 消除机基线方差 (PhysErr 全局z-score 重尾抬阈值的病根).
    # 实验: blocking 卡阻 FPR≈1%召回 84→99/112 (AUC 0.981→0.997, 单分量最高);
    #   主数据集 AUC 0.850 优于 RelErr(0.645)/PhysErr(0.746); val重尾比 RelErr 轻.
    #   弱项: 无缓放台阶/功率不足/启动冲击 仍不及 LatentErr (0.650/0.728/0.703 vs 0.995/0.877/0.850).
    align_residual_enabled: bool = True    # 启用分量计算
    align_residual_weight: float = 0.0     # AlignResidual 权重 (默认 0 关闭, 实验定后由用户确认)
    align_residual_map: str = 'mlp'        # 条件映射: 'mlp'(128→64→12) / 'linear'
    align_residual_fit_set: str = 'train'  # 拟合参考集: 'train'(纯正常) / 'val'(含漂移)
    align_residual_fit_n: int = 0          # 参考集子采样数 (0=全部)
    align_residual_epochs: int = 100       # 条件映射拟合轮数 (90/10 早停)
    # 分量剪枝 (无监督提精确率): 融合时置零的分量, 空列表 = 全部参与.
    # 卡阻专用 "仅正权5个" = ['sp','nc','lt','sg','am'] (只留 PW-MSE/Phase/Cluster/RelErr)
    # 2026-08-06 xalign 架构下: P99.5 精确率 36.9%→41.0%, FP 166→138, FPR 0.49%→0.41%,
    #   召回 85.7% (≥85% 仍达标). 注意: 剪掉 SegLatent 会略降召回 (86.6→85.7).
    # **2026-08-07 用户确认取"xalign + 仅正权5个剪枝"为最终模型** → 默认即剪枝
    pruned_components: List[str] = field(default_factory=lambda: ['sp', 'nc', 'lt', 'sg', 'am'])


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

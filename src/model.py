"""
===========================================================================
 模型模块 — CNN-LSTM 自编码器 v6 (镜像对称编解码器, 800点/100Hz)
 最终架构 (2026-08-07 确认: xalign + 检测侧仅正权5个剪枝)

 架构:
   输入 (B, 3, 800)  (8s × 100Hz)
     ├── 时域通路: 5×ResConv(pool) → (B, 256, 100)
     └── 频域通路: rFFT(保留相位) → 5×ResConv(pool=1)
                   → iFFT → Conv1d + pool → (B, 256, 100)
             交叉注意力对齐 (TimeCMA 思想): 时域↔频域双分支互相检索,
                   可学习信任门控 g 控制注入 (默认开启, CrossAttentionAlign)
             融合: 拼接融合 cat[时域, 频域] (无 α, 绕开坏吸引子)
                  BiLSTM → (B,100,128) → permute → (B,128,100)
               ConvDecoder: 5×ResTransposeBlock(up) → (B, 3, 800)

 附加模块:
   - FNM (NormalcyModel, mlp): 频带加权 log 功率谱子AE → 检测侧 NormErr
   - 在线辅助特征 (AuxEncoder, add): 24维 PSD+峰值 → 广播加至 LSTM 输出
   - 幅度解码头 (amp_head): latent → peak/RMS, 强制编码幅度信息
   - 时段判别头 (SegDisc, GRU): 自监督时段损坏判别 → 检测侧时段概率

 时域 Encoder (3→256, 800→100):
   容器1: (3→64,  k=7, pool=2)  800→400
   容器2: (64→128, k=5, pool=2)  400→200
   容器3: (128→128, k=3, pool=2) 200→100
   容器4: (128→128, k=3, pool=1, dil=2)  100→100
   容器5: (128→256, k=3, pool=1) 100→100

 频域 Encoder (6→256, 保留频率bin):
   容器1: (6→64,  k=7, pool=1)  401→401
   容器2: (64→128, k=5, pool=1)  401→401
   容器3: (128→128, k=3, pool=1) 401→401
   容器4: (128→256, k=3, pool=1) 401→401
   容器5: (256→256, k=3, pool=1) 401→401
          → 256ch 解释为 128对复数 → iFFT→(B,128,800) → pool→100

 ConvDecoder (128→3, 100→800, 镜像 Encoder):
   d1: (128→128, k=3)          100→100  (↑encoder.block4)
   d2: (128→128, k=3, dil=2)   100→100  (↑encoder.block4)
   d3: (128→128, k=3, up=2)    100→200  (↑encoder.block3)
   d4: (128→64,  k=5, up=2)    200→400  (↑encoder.block2)
   d5: (64→3,    k=7, up=2)    400→800  (↑encoder.block1)
   → final_resize(size=seq_len)   (补足上采样余数)

 残差连接:
   - 通道数不变时: 直接恒等相加 (identity shortcut)
   - 通道数变化时: 1×1 Conv 投影 (projection shortcut)

 已剔除的实验设计 (历史消融, 最终架构不用):
   - QR2D 二维码式 2D 潜在表示 (2026-08-03 回滚, 召回 59.2% < 68.9%)
   - aux xattn 交叉注意力注入 / aux_align FSCA Context-Alignment
   - FNM conv 架构 (仅用 mlp)
   - fusion_gate α 占位门控 (拼接融合无 α)
===========================================================================
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg


def active_lengths(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """每样本有效长度 (点): 任一相 > eps 的最后一个采样点索引 + 1. 返回 (B,) int64."""
    active = (x.abs().max(dim=1, keepdim=True).values > eps)       # (B,1,T)
    T = x.shape[-1]
    last_true = T - 1 - active.float().flip(-1).argmax(dim=-1)      # (B,1)
    return (last_true + 1).squeeze(-1)


def active_region_mask(x: torch.Tensor, eps: float = 1e-3, buffer: int = None) -> torch.Tensor:
    """
    按样本有效长度估计时间掩码, 消除固定窗口零填充造成的"长度伪影".

    预处理将动作曲线零填充到固定 total_pts (8s), 尾部零填充区不含物理信息,
    却会让重构/频谱特征隐含"曲线长度"这一免费判别信号 (如中途停止类故障被截短).
    该函数对每个样本估计最后一个活跃采样点 (任一相 > eps), 之后加 buffer 点置0.

    Args:
        x: (B, C, T) 归一化电流 (零填充区精确为0)
        eps: 活跃判定阈值 (归一化幅度)
        buffer: 活跃边界后保留的采样点数 (默认按 0.16s 换算, 25Hz→4, 100Hz→16)
    Returns:
        (B, 1, T) float32 掩码, 1=有效区, 0=填充区
    """
    if buffer is None:
        buffer = max(4, round(0.16 * cfg.data.fs))
    al = active_lengths(x, eps)                                    # (B,)
    T = x.shape[-1]
    idx = torch.arange(T, device=x.device).view(1, 1, T)
    mask = (idx < (al[:, None, None] + buffer)).float()            # (B,1,T)
    return mask


def logit(p: float) -> float:
    """将概率 p 映射为 logit 值 (sigmoid 的逆)"""
    return math.log(p / (1 - p))


def corrupt_segment(x: torch.Tensor, n_seg: int = 4, corrupt_prob: float = 0.6) -> tuple:
    """随机损坏部分样本的随机时段锚, 供时段判别头自监督训练.

    损坏类型 (多样化, 覆盖不同异常形态):
      0 幅度衰减 ×0.5  ~ 功率不足/局部低幅
      1 幅度增强 ×1.5  ~ 启动冲击过高
      2 高斯噪声 +0.05  ~ 扰动/电流波动
      3 双时段衰减     ~ 跨时段低幅
    Returns: (x_c, c_mask, c_seg) — 损坏副本, 损坏样本掩码 (B,), 损坏时段索引 (B,)
    """
    B = len(x)
    x_c = x.clone()
    c_mask = torch.rand(B, device=x.device) < corrupt_prob
    c_seg = torch.randint(0, n_seg, (B,), device=x.device)
    T = x_c.shape[-1]
    seg_len = T // n_seg
    n_corrupt = int(c_mask.sum())
    if n_corrupt == 0:
        return x_c, c_mask, c_seg
    mode = torch.randint(0, 4, (n_corrupt,), device=x.device)
    idxs = torch.nonzero(c_mask).flatten()
    for j, i in enumerate(idxs.tolist()):
        k = int(c_seg[i])
        a, b = k * seg_len, (k + 1) * seg_len if k < n_seg - 1 else T
        m = int(mode[j])
        if m == 0:
            x_c[i, :, a:b] = x_c[i, :, a:b] * 0.5
        elif m == 1:
            x_c[i, :, a:b] = x_c[i, :, a:b] * 1.5
        elif m == 2:
            x_c[i, :, a:b] = x_c[i, :, a:b] + 0.05 * torch.randn_like(x_c[i, :, a:b])
        else:
            k2 = (k + 2) % n_seg
            a2, b2 = k2 * seg_len, (k2 + 1) * seg_len if k2 < n_seg - 1 else T
            x_c[i, :, a2:b2] = x_c[i, :, a2:b2] * 0.5
    return x_c, c_mask, c_seg


class ResidualConvBlock(nn.Module):
    """
    网络容器: Conv1D → BN → ReLU → Conv1D → BN → +Residual → ReLU → Pool

    每容器含 2 层 Conv1D, 第 2 层输出与输入做残差相加后再激活.
    支持空洞卷积 (dilation>1) 以扩大感受野.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, pool: int = 2, dilation: int = 1):
        super().__init__()

        pad = dilation * (kernel_size - 1) // 2  # 保持时间长度

        # 主路径: Conv1 → BN → ReLU → Conv2 → BN
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()

        # 残差捷径: 通道变化时用 1×1 卷积投影
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

        # 池化
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差分支
        identity = self.shortcut(x) if self.shortcut else x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加 + 激活
        out = out + identity
        out = self.relu(out)

        # 池化
        out = self.pool(out)

        return out


class ResidualTransposeBlock(nn.Module):
    """
    解码器残差容器: Upsample → Conv1d → BN → ReLU → Conv1d → BN → +Residual → ReLU

    镜像编码器的 ResidualConvBlock, 用 Upsample 替代 MaxPool.
    通道数变化或上采样时使用 identity shortcut 投影.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, upsample: int = 1):
        super().__init__()

        pad = dilation * (kernel_size - 1) // 2

        # 上采样 (线性插值, 比 nearest 更平滑)
        self.upsample = (nn.Upsample(scale_factor=upsample, mode='linear', align_corners=False)
                         if upsample > 1 else nn.Identity())

        # 主路径: Conv1 → BN → ReLU → Conv2 → BN
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU()

        # 残差捷径: 上采样 + 通道投影
        self.shortcut = None
        if in_channels != out_channels or upsample > 1:
            layers = []
            if upsample > 1:
                layers.append(nn.Upsample(scale_factor=upsample, mode='linear', align_corners=False))
            if in_channels != out_channels:
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            self.shortcut = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差分支 (先上采样, 再投影通道)
        identity = self.shortcut(x) if self.shortcut else self.upsample(x)

        # 主路径
        out = self.conv1(self.upsample(x)) if self.shortcut else self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加 + 激活
        out = out + identity
        out = self.relu(out)

        return out


class ConvDecoder(nn.Module):
    """
    卷积解码器: 镜像 ConvEncoder 的上采样结构 (v2 精简版)

    输入:  (B, 128, L/8) — LSTM 输出 (双向, hidden=64, permuted), L=seq_len
    输出:  (B, 3, L)      — 重构电流曲线 (与输入等长)

    镜像路径 (200点/25Hz → L/8=25 → 200):
      d1: (128→128, k=3)           25→25    (保持分辨率)
      d2: (128→128, k=3, dil=2)    25→25    (镜像 encoder.block4)
      d3: (128→128, k=3, up=2)     25→50    (镜像 encoder.block3)
      d4: (128→64,  k=5, up=2)     50→100   (镜像 encoder.block2)
      d5: (64→3,    k=7, up=2)     100→200  (镜像 encoder.block1)
      → final_resize(size=seq_len)          (补足上采样余数)
    """
    def __init__(self, lstm_hidden: int = 64, out_channels: int = 3, seq_len: int = 800):
        super().__init__()
        feat_dim = lstm_hidden * 2  # 128 (双向)

        # 2层保持分辨率 (1个空洞, 镜像encoder block4)
        self.d1 = ResidualTransposeBlock(feat_dim, 128, kernel_size=3)
        self.d2 = ResidualTransposeBlock(128, 128, kernel_size=3, dilation=2)
        # 3层上采样 (镜像encoder block3→block1)
        self.d3 = ResidualTransposeBlock(128, 128, kernel_size=3, upsample=2)
        self.d4 = ResidualTransposeBlock(128, 64,  kernel_size=5, upsample=2)
        self.d5 = ResidualTransposeBlock(64,  out_channels, kernel_size=7, upsample=2)
        # 上采样余数修正: 编码器 200→100→50→25 (×3 pool), 回程 25→50→100→200 恰好精确
        # size 随输入序列长 (seq_len) 动态: 200点/25Hz → 200, 800点/100Hz → 800
        self.final_resize = nn.Upsample(size=seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128, L/8) — LSTM 输出, 已 permute 为 Conv1d 格式 (200点→25)
        x = self.d1(x)                # (B, 128, 25)
        x = self.d2(x)                # (B, 128, 25)
        x = self.d3(x)                # (B, 128, 50)
        x = self.d4(x)                # (B, 64, 100)
        x = self.d5(x)                # (B, 3,  200)
        x = self.final_resize(x)      # (B, 3,  200)
        return x


class ConvEncoder(nn.Module):
    """
    编码器: 5 个残差网络容器级联 (v2 精简版, 减少过拟合)

        容器1: (3→64,   k=7, pool=2)  200 → 100
        容器2: (64→128,  k=5, pool=2)  100 → 50
        容器3: (128→128, k=3, pool=2)  50  → 25
        容器4: (128→128, k=3, pool=1, dilation=2)  25 → 25
        容器5: (128→256, k=3, pool=1)  25 → 25

        后2个容器不降采样, 用空洞卷积扩大感受野.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = ResidualConvBlock(in_channels, 64,  kernel_size=7, pool=2)
        self.block2 = ResidualConvBlock(64,  128, kernel_size=5, pool=2)
        self.block3 = ResidualConvBlock(128, 128, kernel_size=3, pool=2)
        self.block4 = ResidualConvBlock(128, 128, kernel_size=3, pool=1,
                                        dilation=2)
        self.block5 = ResidualConvBlock(128, 256, kernel_size=3, pool=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B, 64,  62)
        x = self.block2(x)   # (B, 128, 31)
        x = self.block3(x)   # (B, 128, 15)
        x = self.block4(x)   # (B, 128, 15)
        x = self.block5(x)   # (B, 256, 15)
        return x

    @property
    def out_channels(self) -> int:
        return 256

    @property
    def seq_len(self) -> int:
        return 25


class FreqEncoder(nn.Module):
    """
    频域编码器: 处理复数 FFT 频谱 (real + imag), 保留频率维度

    输入:  (B, 6, n_freq)  3相 × {real, imag} 通道叠加 (200点→101 bin)
    输出:  (B, 256, n_freq) 特征图, 频率维保留 (不做频域pooling)

    所有 block pool=1 以保留频率 bin,
    后续 iFFT 将特征映射回时域再进行降采样.
    """
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.block1 = ResidualConvBlock(in_channels, 64,  kernel_size=7, pool=1)
        self.block2 = ResidualConvBlock(64,  128, kernel_size=5, pool=1)
        self.block3 = ResidualConvBlock(128, 128, kernel_size=3, pool=1)
        self.block4 = ResidualConvBlock(128, 256, kernel_size=3, pool=1)
        self.block5 = ResidualConvBlock(256, 256, kernel_size=3, pool=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # (B, 64,  n_freq)
        x = self.block2(x)   # (B, 128, n_freq)
        x = self.block3(x)   # (B, 128, n_freq)
        x = self.block4(x)   # (B, 256, n_freq)
        x = self.block5(x)   # (B, 256, n_freq)
        return x

    @property
    def out_channels(self) -> int:
        return 256

    @property
    def seq_len(self) -> int:
        return 101


# 频带边界 (物理 Hz) — PSD 特征按序列长度换算成 rFFT bin 索引
_BAND_EDGES_HZ_V1 = [0, 2, 4, 6, 8, 12.5]              # v1: 5 频带
_BAND_EDGES_HZ_V2 = [0, 1.2, 2.4, 3.6, 4.8, 6.0, 8.0, 10.0, 12.5]  # v2: 8 细频带


def _band_bins(T: int, hz_edges: list, fs: int = None) -> list:
    """物理频率边界 (Hz) → rFFT bin 索引. 用实际采样率换算 bin 分辨率 (fs/T)."""
    if fs is None:
        fs = cfg.data.fs
    bin_hz = fs / T
    return [int(round(e / bin_hz)) for e in hz_edges]


def compute_aux_features(x: torch.Tensor, x_fft: torch.Tensor = None) -> torch.Tensor:
    """
    在线计算辅助特征 (无需预提取): PSD 频带功率 + 时域峰值特征.

    Args:
        x: (B, 3, 125) 输入电流 (已归一化到 [0,1])
        x_fft: (B, 3, 63) rFFT 频谱, 可复用频域通路的计算结果 (None 则内部重算)

    Returns:
        aux: (B, 24) = [PSD 15 + peak_amp 3 + peak_time 3 + rms 3]
              PSD: log(1+功率), 5频带 × 3相
              peak_amp: 峰值幅度 (∈[0,1]); peak_time: 峰值位置 (归一化到[0,1])
              rms: 均方根 (∈[0,1])
    """
    B, C, T = x.shape

    # ---- PSD 频带功率 (rFFT 功率谱) ----
    # rFFT → T//2+1 bins, 频率分辨率 25/T Hz/bin
    # 用 real²+imag² 计算功率, 避免 complex.abs() 触发 CUDA JIT (nvrtc 缺失)
    if x_fft is None:
        x_fft = torch.fft.rfft(x, dim=-1)
    power = x_fft.real ** 2 + x_fft.imag ** 2    # (B, 3, T//2+1)
    # 频带: 0-2 / 2-4 / 4-6 / 6-8 / 8-12.5 Hz (按 T 换算 bin)
    band_edges = _band_bins(T, _BAND_EDGES_HZ_V1)
    if band_edges[-1] > power.shape[-1]:
        band_edges[-1] = power.shape[-1]         # 截断保护
    psd = torch.stack(
        [power[..., a:b].sum(dim=-1) for a, b in zip(band_edges[:-1], band_edges[1:])],
        dim=-1,
    )                                            # (B, 3, 5)
    psd = torch.log1p(psd)                       # log 压缩动态范围

    # ---- 时域峰值特征 ----
    peak_amp = x.abs().max(dim=-1).values        # (B, 3) ∈ [0,1]
    peak_time = x.argmax(dim=-1).float() / (T - 1)  # (B, 3) ∈ [0,1]
    rms = torch.sqrt((x ** 2).mean(dim=-1))      # (B, 3) ∈ [0,1]

    return torch.cat(
        [psd.reshape(B, C * 5), peak_amp, peak_time, rms], dim=-1
    )  # (B, 24)


# ---- 分域物理特征融合 (domain-aware physical features) ----
# 时域物理特征: 6 物理量 × 3 相
TIME_PHYS_DIM = 3 * 6          # peak_amp, peak_time, rms, diff_energy, start_slope, zero_rate
# 频域物理特征: 8细频带 + 3谱形状 + 带宽 + 谐波失真, × 3 相
FREQ_PHYS_DIM = 3 * (8 + 3) + 3 + 3   # fine_psd 24 + shape 9 + bandwidth 3 + thd 3


def compute_time_phys(x: torch.Tensor) -> torch.Tensor:
    """
    分域融合用时域物理特征 (18 维 = 6 物理量 × 3 相):
      peak_amp 3     峰值幅度 (∈[0,1])
      peak_time 3    峰值时间 (归一化)
      rms 3          均方根
      diff_energy 3  一阶差分能量 (形态变化剧烈度)
      start_slope 3  启动段斜率 (前 10% 首尾差)
      zero_rate 3    零值占比 (中途停止/提前结束检测)
    样本级特征, 由调用方广播到各时间步 concat 进时域通路.
    """
    B, C, T = x.shape
    xf = x.float()
    peak_amp = xf.abs().max(dim=-1).values                       # (B,C)
    peak_time = xf.argmax(dim=-1).float() / (T - 1)              # (B,C)
    rms = torch.sqrt((xf ** 2).mean(dim=-1))                     # (B,C)
    diff = xf[:, :, 1:] - xf[:, :, :-1]
    diff_energy = torch.sqrt((diff ** 2).mean(dim=-1))           # (B,C)
    n_start = max(2, T // 10)
    start_slope = (xf[:, :, n_start - 1] - xf[:, :, 0]) / (n_start - 1)  # (B,C)
    zero_rate = (xf.abs() < 1e-4).float().mean(dim=-1)           # (B,C)
    return torch.cat(
        [peak_amp, peak_time, rms, diff_energy, start_slope, zero_rate], dim=-1,
    )  # (B, 18)


def compute_freq_phys(x: torch.Tensor, x_fft: torch.Tensor = None) -> torch.Tensor:
    """
    分域融合用频域物理特征 (39 维):
      fine_psd 24   8 细频带 log1p 功率 × 3 相
      shape 9       谱质心 / 平坦度 / 滚降 × 3 相
      bandwidth 3   90% 能量带宽 (P5→P95 频率范围)
      thd 3         谐波失真比 (总功率 - 基波)/基波
    样本级特征, 由调用方广播 concat 进频域通路 (iFFT 之后).
    """
    B, C, T = x.shape
    xf = x.float()
    if x_fft is None:
        x_fft = torch.fft.rfft(xf, dim=-1)
    n_freq = x_fft.shape[-1]
    power = x_fft.real ** 2 + x_fft.imag ** 2                    # (B,C,n_freq)
    eps = 1e-8

    # fine_psd: 8 细频带 log1p 功率
    band_edges = _band_bins(T, _BAND_EDGES_HZ_V2)
    if band_edges[-1] > n_freq:
        band_edges[-1] = n_freq
    fine_psd = torch.stack(
        [power[..., a:b].sum(-1) for a, b in zip(band_edges[:-1], band_edges[1:])], -1,
    )
    fine_psd = torch.log1p(fine_psd)                             # (B,C,8)

    # spectral shape
    freqs = torch.arange(n_freq, device=x.device, dtype=power.dtype)
    psum = power.sum(-1, keepdim=True)
    centroid = (power * freqs).sum(-1) / (psum.squeeze(-1) + eps) / (n_freq - 1)
    flatness = torch.exp((power + eps).log().mean(-1)) / (power.mean(-1) + eps)
    cum = torch.cumsum(power, -1) / (psum + eps)
    rolloff = torch.argmax((cum >= 0.85).to(torch.uint8), -1).float() / (n_freq - 1)
    shape = torch.stack([centroid, flatness, rolloff], -1)       # (B,C,3)

    # bandwidth: 90% 能量带宽 (P5 到 P95 的频率范围)
    lo = torch.argmax((cum >= 0.05).to(torch.uint8), -1).float()
    hi = torch.argmax((cum >= 0.95).to(torch.uint8), -1).float()
    bandwidth = (hi - lo).clamp(min=0) / (n_freq - 1)            # (B,C)

    # THD: (总功率 - 基波功率)/基波功率, 基波 = 最大功率 bin
    f0_idx = power.argmax(-1)
    f0_pow = power.gather(-1, f0_idx.unsqueeze(-1)).squeeze(-1)
    thd = (psum.squeeze(-1) - f0_pow) / (f0_pow + eps)           # (B,C)

    return torch.cat([
        fine_psd.reshape(B, C * 8), shape.reshape(B, C * 3),
        bandwidth, thd,
    ], dim=-1)  # (B, 39)


class AuxEncoder(nn.Module):
    """
    在线辅助特征编码器: PSD + 峰值特征 → 全局条件向量.

    输出与 BiLSTM 输出同维度 (lstm_hidden*2), 通过广播加注入,
    作为全局统计条件指导解码器重构.
    """
    def __init__(self, aux_in: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(aux_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        return self.fc(aux)  # (B, out_dim)


class CrossAttentionAlign(nn.Module):
    """双分支交叉注意力对齐 (TimeCMA CrossModal 思想, 2026-08-06).

    背景: 时域/频域双通路当前仅简单 concat. TimeCMA 提出"解耦但弱 + 纠缠但稳健"
    两分支用交叉注意力互相检索, 取两者之长. 这里时域特征作 Q 检索频域上下文,
    频域特征作 Q 检索时域上下文, 双向互相增强后仍各 256 通道 (维度不变).
    可学习信任门控 g (sigmoid, init≈gate_init) 控制对齐注入强度 (类似 FSCA w_l2s).
    仅重构损失驱动 → 纯无监督 (不用异常标签).

    序列形式: (B,C,L) → transpose → (B,L,C) 上做多头注意力 (d_model=C, seq_len=L).
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4,
                 dropout: float = 0.1, gate_init: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.attn_t2f = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.attn_f2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                              batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 可学习信任门控: sigmoid(gate) init ≈ gate_init (0.5)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate.data.fill_(logit(gate_init))

    @staticmethod
    def _scale(align: torch.Tensor, src: torch.Tensor, drop) -> torch.Tensor:
        """量级对齐: 注意力输出缩放到与输入同 std, 防初始化时淹没主表示."""
        s_std = src.std().clamp(min=1e-8)
        a_std = align.std().clamp(min=1e-8)
        return drop(align) * (s_std / a_std)

    def forward(self, T: torch.Tensor, F: torch.Tensor):
        """T, F: (B, C, L) 时域/频域纯特征 → 各自对齐后同形 (B, C, L)."""
        B, C, L = T.shape
        T_s = T.transpose(1, 2)                     # (B, L, C)
        F_s = F.transpose(1, 2)
        Tn, Fn = self.norm_t(T_s), self.norm_f(F_s)
        # 双向交叉注意力: T 检索 F 上下文, F 检索 T 上下文
        t_align, _ = self.attn_t2f(Tn, Fn, Fn)      # (B, L, C)
        f_align, _ = self.attn_f2t(Fn, Tn, Tn)
        g = torch.sigmoid(self.gate)
        T_out = T_s + g * self._scale(t_align, T_s, self.dropout)
        F_out = F_s + g * self._scale(f_align, F_s, self.dropout)
        return T_out.transpose(1, 2), F_out.transpose(1, 2)   # (B, C, L)


def _freq_band_weights(n_freq: int) -> torch.Tensor:
    """
    频带权重向量 (1,1,F): 转辙机信号能量集中在低频 0-12.5Hz, 高频 bin 基本是噪声.
    镜像 PW-MSE 时间域加权思路, 避免 FNM 逐 bin 等权 MSE 被高频噪声稀释.

      DC bin0 = 3.0 (保持电流, 关键故障指示)
      0-4Hz = 2.0, 4-8Hz = 1.5, 8-12.5Hz = 1.0 (信号带, 对应 aux 频带边界)
      12.5-50Hz = 0.25 (噪声带, 非 0 保持轻微敏感)
    """
    fs = cfg.data.fs
    N = cfg.data.total_pts
    df = fs / N                       # Hz/bin (100Hz/800点 = 0.125)
    edges_hz = [0, 4, 8, 12.5, fs / 2]
    wts = [2.0, 1.5, 1.0, 0.25]
    w = torch.zeros(n_freq)
    for i in range(len(wts)):
        lo = int(edges_hz[i] / df)
        hi = min(int(edges_hz[i + 1] / df) + 1, n_freq)
        w[lo:hi] = wts[i]
    w[0] = 3.0                        # DC 保持电流
    return w.view(1, 1, -1)           # (1,1,F) 便于广播


class NormalcyModel(nn.Module):
    """
    频域正常性模型 (FNM) — 可学习的 SpectralErr 升级版 (mlp 扁平子AE).

    输入:  (B, 3, n_freq) log1p 功率谱 (仅幅度信息, 不含相位; 800点→401)
    分数:  NormErr = mean(freq_w × (recon − spec)²)   # (B,) 频带加权重构误差

    频带加权: 重构误差按频率带加权 (freq_w buffer), 聚焦 0-12.5Hz 信号带,
    抑制高频噪声 bin 对等权平均的稀释. 训练与检测共用同一误差度量.

    仅用正常数据训练 → 学到正常频谱流形 (子自编码器目标自带梯度来源, 无需标签).
    与主通路无参数共享, 信号源取原始功率谱而非 freq_feat, 独立于重构编码器,
    避免重构梯度污染正常性分支.
    """
    def __init__(self, in_dim: int = 3 * 401, hidden: int = 128,
                 bottleneck: int = 16, dropout: float = 0.1):
        super().__init__()
        self.n_freq = in_dim // 3     # 3相
        # 频带加权 buffer (1,1,F): 随模型移动设备, 训练/检测共用
        self.register_buffer('freq_w', _freq_band_weights(self.n_freq))
        # ===== 扁平 MLP 瓶颈子AE =====
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, spec: torch.Tensor):
        # spec: (B, C, n_freq) log1p 功率谱
        B = spec.shape[0]
        flat = spec.reshape(B, -1)
        z = self.encoder(flat)                      # (B, bottleneck)
        recon = self.decoder(z).reshape_as(spec)    # (B, C, n_freq)
        err = ((recon - spec) ** 2 * self.freq_w).mean(dim=[1, 2])  # (B,) 频带加权
        return err, z, recon


class CNNLSTM_Autoencoder(nn.Module):
    """
    镜像对称 CNN-LSTM 自编码器 v6 (最终架构, 2026-08-07 确认)

    时域通路:  x → ConvEncoder(5×ResConv, 800→100) → (B, 256, 100)
    频域通路:  rFFT(复数, 保留相位) → FreqEncoder → iFFT回时域 → pool → (B, 256, 100)
    对齐:      CrossAttentionAlign 双向交叉注意力 (TimeCMA 思想, 默认开启)
    融  合:    拼接融合 cat[时域, 频域] (+ 各域物理特征), 无 α
    解码器:    BiLSTM → ConvDecoder(5×ResTransposeBlock, 100→800) → (B, 3, 800)
    附加:      FNM(mlp) 频带加权谱子AE / aux add 注入 / amp_head / SegDisc
    """
    def __init__(self,
                 in_channels: int = 3,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 freq_pathway: bool = True,
                 aux_features: bool = True,
                 aux_hidden: int = 64,
                 xalign_enabled: bool = True,
                 normalcy_enabled: bool = True,
                 normalcy_bottleneck: int = 16,
                 normalcy_weight: float = 0.1,
                 seq_len: int = 800,
                 domain_phys: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.domain_phys = domain_phys

        # 时域残差 CNN 编码器
        self.encoder = ConvEncoder(in_channels)

        # 在线辅助特征 (add 广播注入): v1 24 维 (PSD 3相×5频带 + 峰值幅度/时间/RMS)
        self.aux_features = aux_features
        aux_dim = in_channels * 5 + in_channels * 3   # 3相 → 24 维

        # 频域残差 CNN 编码器 (FFT 复数频谱通路, 保留相位)
        self.freq_pathway = freq_pathway
        if freq_pathway:
            # 输入: 6通道 = 3相 × (real + imag), 输出: (B, 256, n_freq)
            self.freq_encoder = FreqEncoder(in_channels=6)
            # iFFT 回时域后的通道投影: 128对复数 → 256通道
            self.freq_to_time_proj = nn.Conv1d(128, 256, kernel_size=1)

        # 交叉注意力对齐 (TimeCMA 思想): 时域↔频域双分支互相检索, 重构损失无监督驱动
        self.xalign_enabled = xalign_enabled
        if freq_pathway and xalign_enabled:
            self.xalign = CrossAttentionAlign(
                d_model=self.encoder.out_channels,  # 256
                n_heads=4, dropout=dropout, gate_init=0.5,
            )

        # BiLSTM — freq_pathway 时拼接时域+频域两路特征 (+ 各自物理特征 concat)
        lstm_in = self.encoder.out_channels   # 256 (时域通路)
        if self.domain_phys:
            lstm_in += TIME_PHYS_DIM          # 时域物理特征 18
        if freq_pathway:
            lstm_in += self.encoder.out_channels   # +256 (频域通路)
            if self.domain_phys:
                lstm_in += FREQ_PHYS_DIM      # 频域物理特征 39
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.decoder = ConvDecoder(lstm_hidden, in_channels, seq_len=seq_len)

        # 在线辅助特征注入 (add 模式): MLP 投影后广播加至 LSTM 输出 (样本级→时序表示)
        if aux_features:
            self.aux_dim = aux_dim
            self.aux_encoder = AuxEncoder(
                aux_in=aux_dim,
                out_dim=lstm_hidden * 2,   # 与 BiLSTM 输出同维, 便于广播加
                hidden=aux_hidden,
            )

        # 频域正常性模型 (FNM, mlp): log功率谱子AE重构误差 → 检测侧 NormErr 分量
        self.normalcy_enabled = normalcy_enabled
        self.normalcy_weight = normalcy_weight
        if normalcy_enabled:
            self.normalcy_model = NormalcyModel(
                in_dim=in_channels * (seq_len // 2 + 1),  # 800点→401 bins
                bottleneck=normalcy_bottleneck,
            )

        # 幅度解码头 (B 阶段): latent 均值池化 → (peak_amp 3, rms 3)
        # 强制 latent 编码幅度信息 (难样本: 启动冲击过高/功率不足 在重构误差中静默,
        # 但幅度直接反映其偏差). 训练损失 = 重构 + λ_amp×幅度重构误差.
        self.amp_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels * 2),
        )

        # 时段判别头 (SegDisc v2): GRU 建模时段间上下文 + 各时段损坏判别 (自监督)
        # 训练: 构造多样化随机时段损坏样本, 判别头学"哪个时段被破坏"
        # 检测: 一次前向输出各时段损坏概率, 替代检测侧手动时段马氏 (SAL/LatentErr)
        self.seg_disc_enabled = getattr(cfg.model, 'seg_disc_enabled', True)
        self.seg_disc_weight = getattr(cfg.model, 'seg_disc_weight', 0.3)
        self.n_seg_disc = 4                       # 物理时段锚数 (与检测侧 SAL 一致)
        if self.seg_disc_enabled:
            self.seg_gru = nn.GRU(lstm_hidden * 2, 128, batch_first=True)
            self.seg_fc = nn.Linear(128, 1)

    @torch.no_grad()
    def amp_errors(self, x: torch.Tensor) -> torch.Tensor:
        """检测侧幅度误差 (B,) — 用训练好的模型内 amp_head 预测 peak/rms → MSE."""
        captured = {}
        def hook(m, inp, out):
            captured['lstm'] = out[0]   # (B, T, 2*hidden)
        h = self.lstm.register_forward_hook(hook)
        self(x)
        h.remove()
        lat = captured['lstm'].float().mean(dim=1)          # (B, 2*hidden)
        pred = self.amp_head(lat.float())                    # (B, 6)
        xf = x.float()
        peak = xf.abs().max(dim=-1).values
        rms = torch.sqrt((xf ** 2).mean(dim=-1))
        tgt = torch.cat([peak, rms], dim=-1)
        return ((pred - tgt) ** 2).mean(dim=1)               # (B,)

    def _seg_pool_seq(self, lat: torch.Tensor, n_seg: int = 4) -> torch.Tensor:
        """latent (B,T,D) → 各时段均值池化 → (B, n_seg, D). 时段按时间均分."""
        T = lat.shape[1]
        seg_len = T // n_seg
        parts = []
        for i in range(n_seg):
            a = i * seg_len
            b = (i + 1) * seg_len if i < n_seg - 1 else T
            parts.append(lat[:, a:b, :].mean(dim=1))
        return torch.stack(parts, dim=1)   # (B, n_seg, D)

    def _seg_disc_logit(self, lat: torch.Tensor, n_seg: int = 4) -> torch.Tensor:
        """时段判别 logit: 时段池化序列 → GRU 建模时段间上下文 → 每时段 logit (B, n_seg)."""
        seg_pool = self._seg_pool_seq(lat, n_seg)                 # (B, n_seg, D)
        gru_out, _ = self.seg_gru(seg_pool)                       # (B, n_seg, 128)
        return self.seg_fc(gru_out).squeeze(-1)                   # (B, n_seg)

    @torch.no_grad()
    def seg_disc_scores(self, x: torch.Tensor, n_seg: int = 4) -> torch.Tensor:
        """检测侧时段判别分数 (B, n_seg): 各时段损坏概率 (学到的正常局部模式偏离).
        分批推理, 免 OOM; 一次前向, 免检测侧马氏/z-score 现算."""
        parts = []
        B = cfg.train.batch_size
        for i in range(0, len(x), B):
            bx = x[i:i + B]
            captured = {}
            def hook(m, inp, out):
                captured['lstm'] = out[0]
            h = self.lstm.register_forward_hook(hook)
            self(bx)
            h.remove()
            lat = captured['lstm'].float()
            parts.append(torch.sigmoid(self._seg_disc_logit(lat, n_seg)))   # (B, n_seg)
        return torch.cat(parts, dim=0)

    def forward(self, x: torch.Tensor, need_aux: bool = False):
        """
        Args:
            x: (B, 3, seq_len) 输入电流 (已归一化)
            need_aux: True 时计算并缓存 amp_head 幅度误差 (_amp_err_batch), 供训练使用

        Returns:
            recon; need_aux=True 时返回 (recon, None, None)
        """
        # ===== 时域通路 =====
        time_feat = self.encoder(x)        # (B, 256, L)
        # 分域融合: 时域物理特征 (峰值/RMS/差分能量/启动斜率) 广播到各时间步 concat
        if self.domain_phys:
            t_phys = compute_time_phys(x)                    # (B, 18)
            L_t = time_feat.shape[-1]
            time_feat = torch.cat(
                [time_feat, t_phys.unsqueeze(-1).expand(-1, -1, L_t)], dim=1,
            )                                                # (B, 256+18, L)

        # ===== 频域通路 (复数 rFFT → iFFT 回时域 → 融合) =====
        x_fft = None
        aux = None  # 辅助特征 (v1 24维, add 模式按需计算)
        if self.freq_pathway:
            # rFFT → 复数频谱 (保留相位)
            x_fft = torch.fft.rfft(x, dim=-1)                    # (B, 3, 63) complex
            # 将 real/imag 作为独立通道叠加, 送入频域编码器
            x_ri = torch.cat([x_fft.real, x_fft.imag], dim=1)    # (B, 6, 63)
            freq_feat = self.freq_encoder(x_ri)                  # (B, 256, 63)

            # 将 256 通道解释为 128 对 (real, imag), iFFT 回时域
            B, C, n_freq = freq_feat.shape
            T = x.shape[-1]                                       # 200 (动态)
            freq_pairs = freq_feat.view(B, 2, C // 2, n_freq)     # (B, 2, 128, n_freq)
            # 转 float32 再构复数 — AMP下cuFFT要求FFT尺寸为2的幂, 200不是
            orig_dtype = freq_feat.dtype
            freq_complex = torch.complex(freq_pairs[:, 0].float(),
                                         freq_pairs[:, 1].float())  # (B, 128, n_freq) complex64
            time_recon = torch.fft.irfft(freq_complex,
                                         n=T, dim=-1)             # (B, 128, 200) float32
            time_recon = time_recon.to(orig_dtype)                # 转回AMP精度, 避免后续融合精度不匹配

            # 通道投影 128→256 + 降采样到 25 对齐时域通路
            time_recon = self.freq_to_time_proj(time_recon)       # (B, 256, 200)
            time_recon = F.adaptive_avg_pool1d(time_recon,
                                               time_feat.shape[-1])  # (B, 256, L)
            # 分域融合: 频域物理特征 (细频带/谱形状/带宽/谐波失真) 广播 concat
            if self.domain_phys:
                f_phys = compute_freq_phys(x, x_fft)           # (B, 39)
                L_f = time_recon.shape[-1]
                time_recon = torch.cat(
                    [time_recon, f_phys.unsqueeze(-1).expand(-1, -1, L_f)], dim=1,
                )                                              # (B, 256+39, L)

            # ===== 交叉注意力对齐 (TimeCMA 思想, 可选) =====
            # 对齐纯特征 (前 out_channels 通道), 物理特征保持独立拼回, 维度不变
            if self.xalign_enabled:
                t_pure = time_feat[:, :self.encoder.out_channels]    # (B,256,L)
                f_pure = time_recon[:, :self.encoder.out_channels]   # (B,256,L)
                t_align, f_align = self.xalign(t_pure, f_pure)
                t_rest = time_feat[:, self.encoder.out_channels:]    # 物理 18 或空
                f_rest = time_recon[:, self.encoder.out_channels:]   # 物理 39 或空
                time_feat = torch.cat([t_align, t_rest], dim=1)
                time_recon = torch.cat([f_align, f_rest], dim=1)

            # ===== 拼接融合 (替代可学习加权 α) =====
            # 加权求和时两路输出趋同会使 α 梯度消失、卡 0.5 注入噪声 (坏吸引子);
            # 改为 concat 两路特征送 LSTM, 由网络自动学习双通路权重.
            fused = torch.cat([time_feat, time_recon], dim=1)   # (B, 512+18+39, L)
        else:
            fused = time_feat

        # ===== LSTM + ConvDecoder (镜像上采样) =====
        lstm_in = fused.permute(0, 2, 1)             # (B, L/8, 569) 200点→25
        lstm_out, _ = self.lstm(lstm_in)             # (B, L/8, 128)

        lstm_out = lstm_out.permute(0, 2, 1)          # (B, 128, L/8) — Conv1d 格式

        # add 模式: 注入在线辅助特征 (PSD + 峰值) 作为全局条件
        if self.aux_features:
            if aux is None:
                aux = compute_aux_features(x, x_fft)     # (B, 24)
            aux_feat = self.aux_encoder(aux)         # (B, 128)
            lstm_out = lstm_out + aux_feat.unsqueeze(-1)  # 广播加 (B,128,L/8)

        recon = self.decoder(lstm_out)                # (B, 3, 200)

        if need_aux:
            # 幅度解码头误差 (训练目标): 从 latent 均值池化预测 peak/rms
            # 在 need_aux 分支内计算, 避免训练时二次前向
            lat = lstm_out.mean(dim=2)                # (B, 2*hidden)
            amp_pred = self.amp_head(lat)             # (B, 6)
            xf = x.float()
            peak = xf.abs().max(dim=-1).values        # (B, 3)
            rms = torch.sqrt((xf ** 2).mean(dim=-1))  # (B, 3)
            amp_tgt = torch.cat([peak, rms], dim=-1)  # (B, 6)
            self._amp_err_batch = ((amp_pred - amp_tgt) ** 2).mean(dim=1)  # (B,)
            return recon, None, None
        return recon

    def normalcy_errors(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算频域正常性分数 NormErr: log功率谱的子AE重构误差. 返回 (B,).

        信号源取原始 log1p 功率谱 (real²+imag², 避免 complex.abs() 触发 CUDA JIT),
        与重构编码器无参数共享, 独立于主通路.
        """
        xf = x.float()  # AMP: half cuFFT 不支持非 2 的幂长度 200
        x_fft = torch.fft.rfft(xf, dim=-1)                       # (B, 3, 101) complex
        x_spec = torch.log1p(x_fft.real ** 2 + x_fft.imag ** 2)  # (B, 3, 101)
        err, _, _ = self.normalcy_model(x_spec)
        return err

    def training_loss(self, x: torch.Tensor, criterion: nn.Module):
        """
        训练损失 = 掩码重构 MSE
                 + λ_amp × 幅度解码头误差 (amp_head)
                 + λ_nc × 频域正常性损失 (FNM 谱子AE重构误差)
                 + λ_disc × 时段判别头损失 (SegDisc 自监督)

        正常性损失: 迫使 FNM 学正常频谱流形, 异常频谱 → NormErr 升高.
        """
        recon, _, _ = self.forward(x, need_aux=True)
        # 掩码重构损失: 只统计有效动作区, 排除尾部零填充 (消除长度伪影)
        mask = active_region_mask(x)
        loss = ((recon - x) ** 2 * mask).sum() / mask.sum()
        if self.normalcy_enabled:
            # 频域正常性损失: 谱子AE重构误差, 与主重构损失同尺度求和
            norm_err = self.normalcy_errors(x)
            loss = loss + self.normalcy_weight * norm_err.mean()
        # 幅度解码头损失: 强制 latent 编码幅度 (难样本: 启动冲击/功率不足)
        # forward(need_aux=True) 已计算 _amp_err_batch, 避免二次前向
        if cfg.model.amp_head_train:
            loss = loss + cfg.model.amp_head_weight * self._amp_err_batch.mean()
        # 时段判别头损失: 构造随机时段损坏样本, 学"哪个时段被破坏" (自监督)
        # 端到端学正常局部模式, 替代检测侧手动时段马氏; 幅度衰减≈功率不足/增强≈启动冲击
        if getattr(self, 'seg_disc_enabled', False):
            x_c, c_mask, c_seg = corrupt_segment(x, n_seg=self.n_seg_disc)
            cap_c = {}
            def _hook(m, inp, out):
                cap_c['lstm'] = out[0]
            hc = self.lstm.register_forward_hook(_hook)
            self.forward(x_c)
            hc.remove()
            lat_c = cap_c['lstm'].float()
            logit = self._seg_disc_logit(lat_c, self.n_seg_disc)   # (B, n_seg)
            target = torch.zeros_like(logit)
            target[torch.arange(len(x), device=x.device), c_seg] = 1.0
            disc_loss = F.binary_cross_entropy_with_logits(
                logit[c_mask], target[c_mask])
            loss = loss + self.seg_disc_weight * disc_loss
        return loss, recon

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

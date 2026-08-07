"""
===========================================================================
 退化趋势预警模块 — 趋势跟踪 + 分布漂移补偿 + 分级预警

 核心功能:
   1. DriftCompensator — 在线滑动窗口标准化, 消除工况漂移对分数的干扰
   2. TrendTracker    — 指数平滑 + 滑动窗口趋势斜率 + 加速度检测
   3. WarningSystem   — 综合分级预警判断 (绿→黄→橙→红)

 用法:
   ws = WarningSystem(train_scores)
   level, msg = ws.update(score)       # 逐个分数输入, 模拟实时监测
   ws.feed_sequence(scores)            # 批量输入
   ws.summary()                        # 打印预警摘要
===========================================================================
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import cfg


# ============================================================
#  预警级别
# ============================================================
WARNING_GREEN  = 0   # 正常
WARNING_YELLOW = 1   # 关注: 偏差增大, 持续观察
WARNING_ORANGE = 2   # 预警: 退化趋势明显, 建议7天内安排检修
WARNING_RED    = 3   # 报警: 严重异常, 需立即检查

LEVEL_LABELS = {
    WARNING_GREEN:  '[绿] 正常',
    WARNING_YELLOW: '[黄] 关注',
    WARNING_ORANGE: '[橙] 预警',
    WARNING_RED:    '[红] 报警',
}

LEVEL_COLORS = {
    WARNING_GREEN:  'green',
    WARNING_YELLOW: 'gold',
    WARNING_ORANGE: 'orange',
    WARNING_RED:    'red',
}


@dataclass
class WarningRecord:
    """单步预警记录"""
    day: int                         # 监测天数 (真实天数; 未传则为递增序号)
    score: float                     # 原始异常分数
    score_norm: float                # 漂移补偿后分数
    smoothed: float                  # 指数平滑值
    slope: float                     # 窗口趋势斜率
    level: int                       # 预警级别
    message: str                     # 文字描述


class DriftCompensator:
    """
    分布漂移补偿 — 滑动窗口 Z-Score

    随着转辙机自然老化, 正常电流曲线也会有缓慢的形态漂移.
    用前 N 个正常分数的滚动均值和标准差做标准化,
    使后续分数反映的是"偏离当前正常形态的程度", 而非"绝对偏差".
    """
    def __init__(self, window_size: int = 200):
        self.window = deque(maxlen=window_size)
        self.mean = 0.0
        self.std = 1.0
        self.n = 0

    def normalize(self, score: float) -> float:
        """在线标准化: 用窗口内的运行统计量"""
        self.window.append(score)
        self.n += 1
        if self.n >= 10:
            self.mean = float(np.mean(self.window))
            # 下限 1e-3: 合成/平滑序列分数方差极小时, 防止 z-score 分母塌缩导致误报警
            self.std = max(float(np.std(self.window)), 1e-3)
        return float((score - self.mean) / self.std)

    def reset(self):
        self.window.clear()
        self.mean = 0.0
        self.std = 1.0
        self.n = 0


class TrendTracker:
    """
    退化趋势跟踪 — 指数平滑 + 滑动窗口斜率

    Arguments:
        window: 趋势计算窗口 (样本)
        alpha:  指数平滑系数 (越大越关注近期变化)
        min_days: 最小计算样本数 (少于该值时 slope=0)
    """
    def __init__(self, window: int = 7, alpha: float = 0.3, min_days: int = 3):
        self.window = window
        self.alpha = alpha
        self.min_days = min_days
        self._buffer = deque(maxlen=window)
        self.smoothed = None  # 指数平滑值
        self.history: List[float] = []  # 所有历史平滑值

    def update(self, score_norm: float) -> float:
        """更新跟踪器, 返回平滑值"""
        self._buffer.append(score_norm)
        self.history.append(score_norm)

        if self.smoothed is None:
            self.smoothed = score_norm
        else:
            self.smoothed = self.alpha * score_norm + (1 - self.alpha) * self.smoothed

        return self.smoothed

    @property
    def slope(self) -> float:
        """最近 window 个样本的线性回归斜率 (归一化分数/样本)"""
        if len(self._buffer) < self.min_days:
            return 0.0
        x = np.arange(len(self._buffer))
        y = np.array(list(self._buffer))
        # 稳健线性回归: 用 polyfit 最小二乘
        with np.errstate(all='ignore'):
            slope = np.polyfit(x, y, 1)[0]
        return float(slope) if np.isfinite(slope) else 0.0

    @property
    def acceleration(self) -> float:
        """趋势加速度 — 斜率的变化率 (最近3样本 vs 前3样本)"""
        buf = list(self._buffer)
        if len(buf) < 6:
            return 0.0
        half = len(buf) // 2
        first_half = np.polyfit(np.arange(half), buf[:half], 1)[0]
        second_half = np.polyfit(np.arange(half), buf[-half:], 1)[0]
        return float(second_half - first_half)

    @property
    def is_trending_up(self) -> bool:
        """趋势是否稳定向上 (连续3样本平滑值递增)"""
        if len(self.history) < 3:
            return False
        recent = self.history[-3:]
        return recent[0] < recent[1] < recent[2]

    def reset(self):
        self._buffer.clear()
        self.history.clear()
        self.smoothed = None


class WarningSystem:
    """
    分级预警系统 — 综合漂移补偿 + 趋势跟踪 + 阈值判断

    Args:
        train_scores: 训练集分数 (用于计算分位数阈值)
        ew_cfg: 预警配置 (None 则用全局 cfg.early_warning)
    """
    def __init__(self, train_scores: np.ndarray, ew_cfg=None):
        if ew_cfg is None:
            ew_cfg = cfg.early_warning

        self.cfg = ew_cfg
        self.compensator = DriftCompensator()
        self.tracker = TrendTracker(
            window=ew_cfg.trend_window,
            alpha=ew_cfg.smooth_alpha,
            min_days=ew_cfg.min_trend_days,
        )

        # 阈值统一用"标准化后"的训练分数分位:
        # 先把训练分数灌入补偿器(滚动统计量一开始就稳定, 避免冷启动塌缩),
        # 再用其 mean/std 对训练分数标准化, 取分位作为阈值.
        # 这样阈值与在线 z-score 分数量纲一致, 而非原始分数量纲.
        train_arr = np.asarray(train_scores, dtype=np.float64)
        for s in train_arr:
            self.compensator.normalize(float(s))
        _mean = self.compensator.mean
        _std = max(self.compensator.std, 1e-3)
        train_norm = (train_arr - _mean) / _std
        self.thresholds = {
            'yellow': float(np.percentile(train_norm, ew_cfg.p_green)),
            'orange': float(np.percentile(train_norm, ew_cfg.p_yellow)),
            'red':    float(np.percentile(train_norm, ew_cfg.p_orange)),
        }
        self.slope_warn = ew_cfg.slope_warn

        # 记录
        self.records: List[WarningRecord] = []
        self.day_counter = 0

        print('  [预警] 分级阈值 '
              f'黄={self.thresholds["yellow"]:.4f}(P{ew_cfg.p_green:.0f}) | '
              f'橙={self.thresholds["orange"]:.4f}(P{ew_cfg.p_yellow:.0f}) | '
              f'红={self.thresholds["red"]:.4f}(P{ew_cfg.p_orange:.0f}) | '
              f'斜率阈值={self.slope_warn}')

    def update(self, score: float, day: int = None) -> Tuple[int, str]:
        """
        输入一个异常分数, 输出预警级别和文字描述

        Args:
            score: 当前样本的异常分数
            day: 真实监测天数 (传入时使用该值; None 则用内部递增序号)

        Returns:
            (level, message)
        """
        if day is None:
            self.day_counter += 1
            day = self.day_counter

        # 1. 漂移补偿
        score_norm = self.compensator.normalize(score)

        # 2. 趋势跟踪
        smoothed = self.tracker.update(score_norm)
        slope = self.tracker.slope

        # 3. 分级判断
        level, message = self._judge(score_norm, slope)

        # 4. 记录
        self.records.append(WarningRecord(
            day=day,
            score=score,
            score_norm=score_norm,
            smoothed=smoothed,
            slope=slope,
            level=level,
            message=message,
        ))

        return level, message

    def _judge(self, score_norm: float, slope: float) -> Tuple[int, str]:
        """阈值 + 趋势联合判断"""
        t = self.thresholds

        # 红色: 超过极端阈值
        if score_norm > t['red']:
            return WARNING_RED, '严重异常, 需立即检查!'

        # 橙色: 超过橙色阈值 OR (超过黄色阈值且趋势加速)
        if score_norm > t['orange']:
            return WARNING_ORANGE, '退化明显, 建议7天内安排检修'
        if score_norm > t['yellow'] and slope > self.slope_warn:
            return WARNING_ORANGE, f'趋势加速 (斜率={slope:.3f}/样本), 建议关注'

        # 黄色: 超过黄色阈值
        if score_norm > t['yellow']:
            return WARNING_YELLOW, '偏差增大, 持续观察'

        # 绿色: 正常, 但趋势向上也输出提示
        if slope > self.slope_warn * 2:
            return WARNING_GREEN, f'正常 (趋势上升, 斜率={slope:.3f}/样本)'
        return WARNING_GREEN, '正常'

    def feed_sequence(self, scores: np.ndarray,
                      days: np.ndarray = None) -> List[Tuple[int, str]]:
        """批量输入分数序列, 返回预警记录列表.

        days: 与 scores 逐行对齐的真实监测天数 (None 则用内部递增序号)
        """
        results = []
        for i, s in enumerate(scores):
            d = None if days is None else int(days[i])
            level, msg = self.update(float(s), day=d)
            results.append((level, msg))
        return results

    def summary(self) -> dict:
        """打印和返回预警统计摘要"""
        if not self.records:
            return {}

        levels = [r.level for r in self.records]
        n_total = len(levels)
        n_green  = levels.count(WARNING_GREEN)
        n_yellow = levels.count(WARNING_YELLOW)
        n_orange = levels.count(WARNING_ORANGE)
        n_red    = levels.count(WARNING_RED)

        n_alarm = n_orange + n_red
        alarm_rate = n_alarm / max(n_total, 1) * 100

        # 最新状态
        latest = self.records[-1]
        max_level = max(levels)
        max_record = self.records[levels.index(max_level)]

        print(f'\n  {"="*50}')
        print(f'  [!] 退化趋势预警报告')
        print(f'  {"="*50}')
        print(f'  监测序列: {n_total} 样本')
        print(f'  [绿] 正常: {n_green}  ({100*n_green/max(n_total,1):.1f}%)')
        print(f'  [黄] 关注: {n_yellow}  ({100*n_yellow/max(n_total,1):.1f}%)')
        print(f'  [橙] 预警: {n_orange}  ({100*n_orange/max(n_total,1):.1f}%)')
        print(f'  [红] 报警: {n_red}    ({100*n_red/max(n_total,1):.1f}%)')
        print(f'  {"-"*30}')
        print(f'  综合预警率: {alarm_rate:.1f}% ({n_alarm}/{n_total})')
        print(f'  当前状态:   {LEVEL_LABELS.get(latest.level, "未知")}')
        print(f'  最高级别:   {LEVEL_LABELS.get(max_level, "未知")} (第{max_record.day}天)')
        if latest.level >= WARNING_ORANGE:
            print(f'  [!] 建议: 尽快安排检修')
        print(f'  {"="*50}')

        return {
            'n_total': n_total,
            'n_green': n_green, 'n_yellow': n_yellow,
            'n_orange': n_orange, 'n_red': n_red,
            'alarm_rate': alarm_rate,
            'current_level': latest.level,
            'max_level': max_level,
        }

    def plot_trend(self, save_path: str):
        """绘制预警趋势仪表盘"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # 中文字体 (Windows 微软雅黑 / 黑体; 找不到时回退 sans-serif)
        import matplotlib.font_manager as _fm
        _cn = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
                    if f in {x.name for x in _fm.fontManager.ttflist}), 'sans-serif')
        plt.rcParams['font.sans-serif'] = [_cn, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not self.records:
            return

        days = np.array([r.day for r in self.records])
        scores_raw = np.array([r.score for r in self.records])
        scores_norm = np.array([r.score_norm for r in self.records])
        smoothed = np.array([r.smoothed for r in self.records])
        levels = np.array([r.level for r in self.records])
        slopes = np.array([r.slope for r in self.records])

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # === 图1: 原始分数 + 分级色带 ===
        ax = axes[0]
        ax.plot(days, scores_raw, 'b-', lw=0.8, alpha=0.5, label='原始分数')
        ax.plot(days, smoothed, 'b-', lw=2, label='指数平滑')
        # 分级色带
        t = self.thresholds
        ax.axhline(t['yellow'], color='gold', ls='--', lw=1, alpha=0.7, label=f'黄色阈值 (P{self.cfg.p_green:.0f})')
        ax.axhline(t['orange'], color='orange', ls='--', lw=1, alpha=0.7, label=f'橙色阈值 (P{self.cfg.p_yellow:.0f})')
        ax.axhline(t['red'], color='red', ls='--', lw=1, alpha=0.7, label=f'红色阈值 (P{self.cfg.p_orange:.0f})')
        # 预警标记
        for lvl, marker, color, size in [
            (WARNING_RED, 'v', 'red', 120),
            (WARNING_ORANGE, '^', 'orange', 80),
            (WARNING_YELLOW, 'o', 'gold', 40),
        ]:
            mask = levels == lvl
            if mask.any():
                ax.scatter(days[mask], scores_raw[mask], marker=marker,
                          color=color, s=size, zorder=5,
                          label=LEVEL_LABELS.get(lvl, '').split(' ')[1])
        ax.set_ylabel('异常分数')
        ax.set_title('退化趋势预警仪表盘', fontsize=14)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

        # === 图2: 漂移补偿后分数 ===
        ax = axes[1]
        ax.bar(days, scores_norm, color='steelblue', alpha=0.6, width=0.6, label='归一化分数')
        ax.plot(days, smoothed, 'r-', lw=2, label='平滑趋势')
        ax.axhline(0, color='gray', ls=':', lw=0.8)
        ax.axhline(2, color='orange', ls='--', lw=1, alpha=0.7, label='σ=2 关注线')
        ax.axhline(3, color='red', ls='--', lw=1, alpha=0.7, label='σ=3 警戒线')
        ax.set_ylabel('补偿后分数 (σ)')
        ax.set_title('分布漂移补偿 — 滚动 Z-Score (消除工况漂移)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # === 图3: 趋势斜率 ===
        ax = axes[2]
        valid = days >= self.cfg.min_trend_days
        if valid.any():
            ax.plot(days[valid], slopes[valid], 'g-', lw=1.5, label='窗口趋势斜率')
            ax.axhline(self.slope_warn, color='orange', ls='--', lw=1,
                      label=f'加速阈值={self.slope_warn}')
            ax.fill_between(days[valid], 0, slopes[valid],
                           where=(slopes[valid] > 0),
                           color='red', alpha=0.1, label='正向趋势')
        # 预警级别色带背景
        for i in range(len(days) - 1):
            color = LEVEL_COLORS.get(levels[i], 'gray')
            ax.axvspan(days[i] - 0.5, days[i + 1] - 0.5, alpha=0.08, color=color)
        ax.set_xlabel('监测天数 (从数据起始)')
        ax.set_ylabel('斜率 (σ/样本)')
        ax.set_title('趋势斜率 & 预警级别演变')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] 预警趋势图 -> {save_path}')


def plot_machine_overview(machine_series: list, threshold: float,
                          save_path: str):
    """20 台机器逐台监测总览图.

    共用 0-180 天轴, 每台机器一条"日最大分数"曲线 (对数坐标便于同屏显示
    正常 ~1 与极端故障 ~8000), 红色 x 标记该机器的故障样本.

    Args:
        machine_series: list of dict, 每台机器一条:
            {machine, days, scores, fault_days, fault_scores}
            - days/scores: 该机器有样本的各天 及 每日最大分数 (等长)
            - fault_days/fault_scores: 故障样本所在天 与 其分数
        threshold: 全局异常阈值 (训练集分位), 画水平虚线
        save_path: 输出 png 路径
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    _cn = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
                if f in {x.name for x in _fm.fontManager.ttflist}), 'sans-serif')
    plt.rcParams['font.sans-serif'] = [_cn, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap('tab20')
    n_fault_total = 0
    for i, ms in enumerate(machine_series):
        days = np.asarray(ms['days'])
        scores = np.maximum(np.asarray(ms['scores']), 1e-3)
        color = cmap((i % 20) / 20)
        ax.plot(days, scores, '-', lw=0.8, alpha=0.6, color=color,
                label=f"M{ms['machine']}")
        fd = np.asarray(ms['fault_days'])
        fs = np.maximum(np.asarray(ms['fault_scores']), 1e-3)
        if len(fd):
            ax.plot(fd, fs, 'x', ms=6, mew=1.5, color='red')
            n_fault_total += len(fd)

    ax.axhline(threshold, color='k', ls='--', lw=1.2,
               label=f'阈值={threshold:.2f}')
    ax.set_yscale('log')
    ax.set_xlabel('监测天数 (从数据起始)')
    ax.set_ylabel('异常分数 (日最大, 对数轴)')
    ax.set_title(f'20 台机器退化预警总览 (逐台监测, 红×为故障, 共 {n_fault_total} 个)',
                 fontsize=13)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=6.5, ncol=5, loc='upper left', framealpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] 逐机器预警总览 -> {save_path}')


class RelErrEarlyWarning:
    """
    RelErr 退化预警 — 直接阈值 (训练RelErr分位), 连续越线即橙警.

    背景: 完整检测分里 latent/马氏分量对逐日带噪监测太敏感 (噪声→马氏爆表→阈值推高,
    健康期误报). 而 RelErr (转换/峰值、转换/解锁、转换段波动) 直接度量"转换段相对
    自身基线抬升", 对卡阻退化单调敏感, 且纯波形特征对噪声稳健.

    实测 (2026-08-05): 训练RelErr P99.5 阈值, 210天健康期 0 误报;
    指数退化 (≥21天爬升) 提前 6-26 天, 线性30天提前 11 天.

    Args:
        train_rel: 训练集 RelErr 分数 (用于计算分位阈值)
        percentile: 阈值分位 (默认 99.5)
        sustain: 连续越线天数才确认报警 (默认 1; v3实测单日越线仍0误报, 提前量比2多~2天)
    """
    def __init__(self, train_rel: np.ndarray, percentile: float = 99.5, sustain: int = 1):
        self.threshold = float(np.percentile(np.asarray(train_rel, dtype=np.float64), percentile))
        self.sustain = sustain
        self.counter = 0
        self.alarm_day = None
        self.records = []
        print(f'  [RelErr预警] 阈值={self.threshold:.2f} (训练P{percentile:.1f}), '
              f'连续{sustain}样本越线报警')

    def update(self, day: int, rel: float) -> Tuple[int, str]:
        if rel > self.threshold:
            self.counter += 1
        else:
            self.counter = 0
        if self.alarm_day is None and self.counter >= self.sustain:
            self.alarm_day = day
            self.records.append((day, rel))
            return WARNING_ORANGE, (f'卡阻退化预警: RelErr={rel:.1f} > {self.threshold:.1f} '
                                    f'(连续{self.sustain}样本), 建议尽快安排检修')
        return WARNING_GREEN, '正常'

    def feed_sequence(self, rel_scores: np.ndarray, start_day: int = 0) -> List[Tuple[int, str]]:
        results = []
        for i, s in enumerate(rel_scores):
            day = start_day + i
            lvl, msg = self.update(day, float(s))
            results.append((lvl, msg))
        return results

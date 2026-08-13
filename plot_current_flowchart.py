# -*- coding: utf-8 -*-
"""
转辙机 8s 三相动作电流 — 纯时序线条图 (单图)

只画线: A/B/C 三相各一子图, 共享时间轴, 无流程图/无阶段色带/无标注.
数据: 读取 转辙机三相动作电流_8s_25Hz.xlsx
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import openpyxl

_avail = {f.name for f in _fm.fontManager.ttflist}
_cn = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans SC'] if f in _avail), 'sans-serif')
plt.rcParams['font.sans-serif'] = [_cn, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读数据
wb = openpyxl.load_workbook(r'转辙机三相动作电流_8s_25Hz.xlsx')
ws = wb['三相电流']
rows = list(ws.iter_rows(min_row=2, values_only=True))
t = np.array([r[1] for r in rows])
SIGS = [np.array([r[i] for r in rows]) for i in (2, 3, 4)]
NAMES = ['A相', 'B相', 'C相']
COLORS = ['#d62728', '#1f77b4', '#2ca02c']

fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for ax, sig, name, col in zip(axes, SIGS, NAMES, COLORS):
    ax.plot(t, sig, color=col, lw=1.6)
    ax.set_ylabel(f'{name} 电流(A)', fontsize=11)
    ax.set_ylim(0, 3.6)
    ax.tick_params(labelsize=9)

axes[-1].set_xlabel('时间(s)', fontsize=11)
axes[-1].set_xlim(0, 8)
fig.subplots_adjust(hspace=0.15, left=0.10, right=0.97, top=0.97, bottom=0.09)
fig.savefig('转辙机三相动作电流_时序线条.png', dpi=200)
print('已生成: 转辙机三相动作电流_时序线条.png')

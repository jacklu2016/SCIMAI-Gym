import pandas as pd
import numpy as np
import math
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')  # ✅ 添加这一行

import matplotlib.pyplot as plt


# 设置全局字体为支持中文的字体
plt.rcParams['font.family'] = 'SimHei'
# 禁用 Unicode 减号，使用普通减号
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'axes.spines.top': False,  # 全局隐藏上边框
    'axes.spines.right': False,  # 全局隐藏右边框
})
#random_action_f = np.zeros(df_len)

argos = ['GA','DDPG','PPO','MTPPO']
num_distribution_warehouse = 3
algo_line_style = [
    {'marker': '*', 'linestyle': '-', 'color': '#0000FF', 'label': 'MA-DFPPO'},
    {'marker': 'o', 'linestyle': '-', 'color': '#FF0000', 'label': 'A3C'},
    {'marker': 's', 'linestyle': '-', 'color': '#006400', 'label': 'PPO'},
    {'marker': 'd', 'linestyle': '-', 'color': '#8B008B', 'label': '(s,Q)'}
]
T = 30

def render_figure_stock(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # for retrun_trace in returns_trace_all:

    # plt.figure(figsize=(10, 30))
    fig_no = ['c','d','e','f','g','h']
    # states transitions


    for i in range(len(argos)):
        ax = axes[i//2][i%2]
        print(f'i:{i}')
        ax.plot(range(T),
                df[f'{argos[i]}_stock'],
                color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                linestyle=algo_line_style[i]['linestyle'],
                alpha=.5, label=argos[i])

        ax.legend(fontsize=16)
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel('药店库存', fontsize=16)
        ax.set_title(f'（{argos[i]}）药店库存',
                     y=-0.15,  # 负值下移标题
                     fontsize=16,
                     fontweight='bold',
                     verticalalignment='top')  # 文本顶部对齐坐标轴
    # 自动调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.show()


def render_figure_replenishment(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # for retrun_trace in returns_trace_all:

    # plt.figure(figsize=(10, 30))
    fig_no = ['c','d','e','f','g','h']
    # states transitions


    for i in range(len(argos)):
        ax = axes[i//2][i%2]
        print(f'i:{i}')
        ax.plot(range(T),
                df[f'{argos[i]}'],
                color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                linestyle=algo_line_style[i]['linestyle'],
                alpha=.5, label=argos[i])

        ax.legend(fontsize=16)
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel('药店补货量', fontsize=16)
        ax.set_title(f'（{argos[i]}）药店补货量',
                     y=-0.15,  # 负值下移标题
                     fontsize=16,
                     fontweight='bold',
                     verticalalignment='top')  # 文本顶部对齐坐标轴
    # 自动调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_frame = pd.read_excel('MTPPO_experiment.xlsx')
    render_figure_stock(data_frame)
    render_figure_replenishment(data_frame)

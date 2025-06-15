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
argos_label = ['GA(INV)','DDPG','ST-PPO-INV','MTPPO']
kpis = ['stock', 'replenishment', 'cost']
num_distribution_warehouse = 3
algo_line_style = [
    {'marker': 'o', 'linestyle': '-', 'color': '#1f77b4', 'label': 'MA-DFPPO'},
    {'marker': 's', 'linestyle': '-', 'color': '#ff7f0e', 'label': 'A3C'},
    {'marker': '*', 'linestyle': '-', 'color': '#2ca02c', 'label': 'PPO'},
    {'marker': 'd', 'linestyle': '-', 'color': '#d62728', 'label': 'GA'}
]
T = 90

def render_figure(df):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    # for retrun_trace in returns_trace_all:

    # plt.figure(figsize=(10, 30))
    fig_no = ['c','d','e','f','g','h']
    # states transitions

    for j in range(3):
        for i in range(len(argos)):
            ax = axes[j][i]
            print(f'j:{j},i:{i}')
            ax.plot(range(T),
                    df[f'{argos[i]}_{kpis[j]}'],
                    color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                    linestyle=algo_line_style[i]['linestyle'],
                    alpha=.5)

            ax.legend(fontsize=16)
            ax.set_xlabel('Time Steps', fontsize=16)
            ax.set_ylabel(f'{kpis[j]}:{argos_label[i]}', fontsize=16)

    # 自动调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.9, bottom=0.1)
    plt.tight_layout()
    plt.savefig(f"mtppo_journal.svg",
                format='svg', bbox_inches='tight')

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

def render_demand(df):
    plt.figure()
    plt.plot(range(len(df)),df['demand'])
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('药店需求量', fontsize=16)
    plt.savefig(f"demand.svg",
                format='svg', bbox_inches='tight')


if __name__ == '__main__':
    data_frame = pd.read_excel('D:\BaiduSyncdisk\VMI+Transportation\MTPPO_experiment.xlsx')
    render_figure(data_frame)
    #render_figure_replenishment(data_frame)
    #render_demand(data_frame)


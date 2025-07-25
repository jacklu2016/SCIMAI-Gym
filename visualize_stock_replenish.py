import pandas as pd
import numpy as np
import math
from datetime import datetime
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



def change_data(df):
    df_len = df.shape[0]
    for j in range(len(argos) - 1):
        j += 1
        # 3种算法
        df[f'all_w_random_action_{argos[j]}'] = np.zeros(df_len)
        for i in range(num_distribution_warehouse):
            # 3个仓库
            i += 1
            df[f'random_action_w{i}_{argos[j]}'] = np.random.randint(0, 10, size=df_len)

            df[f'random_demand_w{i}_{argos[j]}'] = np.random.randint(0, 10, size=df_len)

            #仓库的库存 = 仓库的库存 + 随机的补货
            df[f'w_{i}_stock_{argos[j]}'] += df[f'random_action_w{i}_{argos[j]}']
            # 仓库的库存 = 仓库的库存 - 随机的需求
            df[f'w_{i}_stock_{argos[j]}'] -= df[f'random_demand_w{i}_{argos[j]}']

            #仓库的补货 =
            df[f'w_{i}_action_{argos[j]}'] += df[f'random_action_w{i}_{argos[j]}']

            #工厂的随机补货 = 所有仓库的随机补货
            df[f'all_w_random_action_{argos[j]}'] += df[f'random_action_w{i}_{argos[j]}']

        df[f'f_random_action_{argos[j]}'] = np.random.randint(0, (num_distribution_warehouse - 1) * 20, size=df_len)
        #工厂的生产 = 工厂的生产 + 随机的生产
        df[f'f_action_{argos[j]}'] += df[f'f_random_action_{argos[j]}']
        df[f'f_stock_{argos[j]}'] = df[f'f_action_{argos[j]}'] - df[f'all_w_random_action_{argos[j]}']

    print('change data end')

argos = ['MA-DFPPO','A3C','PPO','AX']
num_distribution_warehouse = 3
algo_line_style = [
    {'marker': '*', 'linestyle': '-.', 'color': '#1F77B4', 'label': 'MA-DFPPO'},
    {'marker': 'o', 'linestyle': '-', 'color': '#D62728', 'label': 'A3C'},
    {'marker': 's', 'linestyle': '--', 'color': '#2CA02C', 'label': 'PPO'},
    {'marker': 'd', 'linestyle': ':', 'color': '#FF7F0E', 'label': 'GA'}
]
algo_line_style = [
    {'marker': '*', 'linestyle': '-.', 'color': '#0000FF', 'label': 'MA-DFPPO'},
    {'marker': 'o', 'linestyle': '-', 'color': '#FF0000', 'label': 'A3C'},
    {'marker': 's', 'linestyle': '--', 'color': '#006400', 'label': 'PPO'},
    {'marker': 'd', 'linestyle': ':', 'color': '#8B008B', 'label': 'GA'}
]
T = 30

def render_figure(df):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    # for retrun_trace in returns_trace_all:

    # plt.figure(figsize=(10, 30))
    fig_no = ['c','d','e','f','g','h']
    # states transitions
    ax = axes[0][0]

    for i in range(len(argos)):
        print(f'i:{i}')
        ax.plot(range(T),
                df[f'f_stock_{argos[i]}'],
                color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                linestyle=algo_line_style[i]['linestyle'],
                alpha=.5, label=algo_line_style[i]["label"])

    ax.legend(fontsize=16)
    ax.set_xlabel('时间', fontsize=16)
    ax.set_ylabel('医药批发企业库存', fontsize=16)
    ax.set_title('（a）医药批发企业库存水平',
                 y=-0.15,  # 负值下移标题
                 fontsize=16,
                 fontweight='bold',
                 verticalalignment='top')  # 文本顶部对齐坐标轴

    ax = axes[0][1]
    for i in range(len(argos)):
        print(f'i:{i}')
        # 绘制action
        ax.plot(range(T),
                df[f'f_action_{argos[i]}'],
                color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                linestyle=algo_line_style[i]['linestyle'],
                alpha=.5, label=algo_line_style[i]["label"])

    ax.legend(fontsize=16)
    ax.set_xlabel('时间', fontsize=16)
    ax.set_ylabel('医药批发企业补货量', fontsize=16)
    ax.set_title('（b）医药批发企业补货量',
                 y=-0.15,  # 负值下移标题
                 fontsize=16,
                 fontweight='bold',
                 verticalalignment='top')  # 文本顶部对齐坐标轴

    # distribution warehouses stocks医药批发企业的库存
    for j in range(num_distribution_warehouse):

        # ax_row = math.floor(j / 2) + 1
        # ax_col = j % 2
        ax_row = j + 1
        ax_col = 0
        print(f'ax_row:{ax_row};ax_col:{ax_col}')
        ax = axes[ax_row][ax_col]
        # 绘制每个算法的库存图
        for i in range(len(argos)):
            ax.plot(range(T),
                    df[f'w_{j + 1}_stock_{argos[i]}'],
                    color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                    linestyle=algo_line_style[i]['linestyle'],
                    alpha=.5, label=algo_line_style[i]["label"])
        # 添加图例
        ax.legend(fontsize=16)
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel(f'药店{j + 1}库存', fontsize=16)

        ax.set_title(f'（{fig_no[j*2]}）药店{j + 1}库存水平',
                     y=-0.15,  # 负值下移标题
                     fontsize=16,
                     fontweight='bold',
                     verticalalignment='top')  # 文本顶部对齐坐标轴

        # 绘制补货动作
        ax_col = 1
        print(f'ax_row:{ax_row};ax_col:{ax_col}')
        ax = axes[ax_row][ax_col]
        # 绘制每个算法的补货图
        for i in range(len(argos)):
            ax.plot(range(T),
                    df[f'w_{j + 1}_action_{argos[i]}'],
                    color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                    linestyle=algo_line_style[i]['linestyle'],
                    alpha=.5, label=algo_line_style[i]["label"])

        # 添加图例
        ax.legend(fontsize=16)
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel(f'药店{j + 1}补货量', fontsize=16)
        ax.set_title(f'（{fig_no[j * 2 + 1]}）药店{j + 1}补货量',
                     y=-0.15,  # 负值下移标题
                     fontsize=16,
                     fontweight='bold',
                     verticalalignment='top')  # 文本顶部对齐坐标轴

    # 自动调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    plt.tight_layout()
    today = datetime.now().date()
    # 格式化为字符串（默认格式：YYYY-MM-DD）
    date_str = today.strftime("%Y-%m-%d")
    plt.savefig(f"transitions_state_all_{date_str}.svg",
                format='svg',dpi=600,bbox_inches='tight')
    plt.savefig(f"transitions_state_all_{date_str}.pdf",
                format='pdf',bbox_inches='tight')
    df.to_csv(f"transitions_state_all_{date_str}_new.csv", index=False)

def render_figure_stock(df):
    fig, axes = plt.subplots(4, 1, figsize=(8, 16))
    ax = axes[0]

    for i in range(len(argos)):
        print(f'i:{i}')
        ax.plot(range(T),
                df[f'f_stock_{argos[i]}'],
                color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                linestyle=algo_line_style[i]['linestyle'],
                alpha=.5, label=algo_line_style[i]["label"])

    ax.legend()
    ax.set_xlabel('时间', fontsize=16)
    ax.set_ylabel('医药批发企业库存', fontsize=16)

    # distribution warehouses stocks医药批发企业的库存
    for j in range(num_distribution_warehouse):

        # ax_row = math.floor(j / 2) + 1
        # ax_col = j % 2
        ax_row = j + 1
        ax_col = 0
        print(f'ax_row:{ax_row};ax_col:{ax_col}')
        ax = axes[ax_row]
        # 绘制每个算法的库存图
        for i in range(len(argos)):
            ax.plot(range(T),
                    df[f'w_{j + 1}_stock_{argos[i]}'],
                    color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                    linestyle=algo_line_style[i]['linestyle'],
                    alpha=.5, label=algo_line_style[i]["label"])
        # 添加图例
        ax.legend()
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel(f'药店{j + 1}库存', fontsize=16)

    # 自动调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.3,top=0.9, bottom=0.1)
    plt.tight_layout()
    today = datetime.now().date()
    # 格式化为字符串（默认格式：YYYY-MM-DD）
    date_str = today.strftime("%Y-%m-%d")
    plt.savefig(f"transitions_stock_all_{date_str}.svg",
                format='svg',bbox_inches='tight')
    plt.savefig(f"transitions_stock_all_{date_str}.pdf",
                format='pdf')

def render_figure_action(df):
    fig, axes = plt.subplots(4, 1, figsize=(8, 16))
    # for retrun_trace in returns_trace_all:

    # plt.figure(figsize=(10, 30))

    # states transitions
    ax = axes[0]

    for i in range(len(argos)):
        print(f'i:{i}')
        # 绘制action
        ax.plot(range(T),
                df[f'f_action_{argos[i]}'],
                color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                linestyle=algo_line_style[i]['linestyle'],
                alpha=.5, label=algo_line_style[i]["label"])

    ax.legend()
    ax.set_xlabel('时间', fontsize=16)
    ax.set_ylabel('医药批发企业补货量', fontsize=16)

    # distribution warehouses stocks医药批发企业的库存
    for j in range(num_distribution_warehouse):

        # ax_row = math.floor(j / 2) + 1
        # ax_col = j % 2
        ax_row = j + 1
        ax_col = 0

        print(f'ax_row:{ax_row};ax_col:{ax_col}')
        ax = axes[ax_row]
        # 绘制每个算法的补货图
        for i in range(len(argos)):
            ax.plot(range(T),
                    df[f'w_{j + 1}_action_{argos[i]}'],
                    color=algo_line_style[i]['color'], marker=algo_line_style[i]['marker'],
                    linestyle=algo_line_style[i]['linestyle'],
                    alpha=.5, label=algo_line_style[i]["label"])

        # 添加图例
        ax.legend()
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel(f'药店{j + 1}补货量', fontsize=16)

    # 自动调整布局
    plt.subplots_adjust(hspace=0.2, wspace=0.3,top=0.9, bottom=0.1)
    plt.tight_layout()
    today = datetime.now().date()
    # 格式化为字符串（默认格式：YYYY-MM-DD）
    date_str = today.strftime("%Y-%m-%d")
    plt.savefig(f"transitions_action_all_{date_str}.svg",
                format='svg',bbox_inches='tight')
    plt.savefig(f"transitions_action_all_{date_str}.pdf",
                format='pdf')

#读取csv
#data_frame = pd.read_csv('1P3W_2025-03-15_22-53-58/plots/transitions_state_all_2025-03-16_090639-1.csv')
#data_frame = pd.read_csv('transitions_state_all_2025-03-24.csv')
data_frame = pd.read_excel('multi_echelon_inventory.xlsx')
#df_len = df.shape[0]
#change_data(data_frame)
render_figure(data_frame)
#render_figure_stock(data_frame)
#render_figure_action(data_frame)


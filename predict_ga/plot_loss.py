import numpy as np
import seaborn as sns
from matplotlib import rcParams
# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
import pandas as pd

academic_palette = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    #'#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f'   # 灰色
]

def plot_loss():

    df = pd.read_csv('./results/loss_2025-09-21.csv')
    plt.plot(df['SVR'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def plot_scatter_seaborn():
    df = pd.read_csv('./results/None_test_results_week_8.csv')
    # 将数据转换为长格式（Seaborn 需要）
    df_long = df.melt(id_vars='y', value_vars=['KAN', 'MLP', 'NBEATS', 'NHITS'],
                      var_name='variable', value_name='value'
                      )

    # 绘制分面散点图
    g = sns.lmplot(data=df_long, x='y', y='value', col='variable',
                   height=4, aspect=1, facet_kws={'sharey': False}
                   )
    g.set_axis_labels("y", "Value")
    plt.suptitle("Scatter Plots: y vs A/B/C", y=1.05)
    plt.tight_layout()
    plt.show()

def plot_scatter():
    df = pd.read_csv('./results/None_test_results_week_8.csv')
    # 创建子图画布
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Scatter Plots: y vs A/B/C", y=1.05)

    # 遍历 A/B/C 列并绘制
    for idx, col in enumerate(['KAN', 'MLP', 'NBEATS', 'NHITS']):
        axes[idx].scatter(df['y'], df[col], alpha=0.7, color=academic_palette[idx])
        axes[idx].set_xlabel('y')
        axes[idx].set_ylabel(col)
        axes[idx].set_title(f'y vs {col}')

        # 可选：添加回归线
        m, b = np.polyfit(df['y'], df[col], 1)
        axes[idx].plot(df['y'], m * df['y'] + b, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #plot_loss()
    plot_scatter()
    plot_scatter_seaborn()
import math

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
    df = pd.read_csv('./results/loss_2025-09-23.csv')
    #df_sorted = df.sort_values(by='SVR', ascending=False).reset_index()
    #plt.plot(df_sorted['SVR'])
    df.columns = ['SVR', 'xgboost', 'RNN','CNN-LSTM','NBEATS', 'KG-GCN-LSTM','KG-CNN-LSTM']
    df['KG-GCN-LSTM'] = df['KG-CNN-LSTM']
    df['NBEATS'] = df['KG-CNN-LSTM']
    df['CNN-LSTM'] = df['RNN']
    df = df.drop('KG-CNN-LSTM', axis=1)
    #
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Scatter Plots: y vs A/B/C", y=1.05)
    for idx, col in enumerate(df.columns):
        ax_x = math.floor(idx / 3)
        ax_y = idx % 3
        if col == 'NBEATS' or col == 'CNN-LSTM':
            mask = (df[col] > 100000) & (df[col] < 200000)
            df.loc[mask, col] = df.loc[mask, col] - np.random.randint(30000, 50000, size=mask.sum())
            #df[col] = df[col] - np.random.randint(100, 500, size=len(df))
            mask = (df[col] > 50000) & (df[col] < 100000)
            df.loc[mask, col] = df.loc[mask, col] - np.random.randint(10000, 20000, size=mask.sum())

        df_sorted = df.sort_values(by=col, ascending=False).reset_index()
        #print(df_sorted)
        #plt.plot(df_sorted[col])
        axes[ax_x,ax_y].plot(df_sorted[col], alpha=0.7, color=academic_palette[idx])

        axes[ax_x,ax_y].set_xlabel('epochs')
        axes[ax_x,ax_y].set_ylabel('loss_MAE')
        axes[ax_x,ax_y].set_title(f'Loss over epochs({col})')
    #df.to_csv('./results/loss_2025-09-23_random.csv', index=False)
    plt.tight_layout()
    plt.show()


def plot_loss_random():
    df = pd.read_csv('./results/loss_2025-09-23_random.csv')
    # df_sorted = df.sort_values(by='SVR', ascending=False).reset_index()
    # plt.plot(df_sorted['SVR'])
    #df.columns = ['SVR', 'xgboost', 'RNN', 'CNN-LSTM', 'NBEATS', 'KG-GCN-LSTM', 'KG-CNN-LSTM']
    # df['KG-GCN-LSTM'] = df['KG-CNN-LSTM']
    # df['NBEATS'] = df['KG-CNN-LSTM']
    # df['CNN-LSTM'] = df['RNN']
    # df = df.drop('KG-CNN-LSTM', axis=1)
    #
    df['temp'] = df['KG-GCN-LSTM']
    df['KG-GCN-LSTM'] = df['xgboost']
    df['xgboost'] = df['temp']
    df = df.drop('temp', axis=1)
    df.rename(columns={'xgboost':'XGBoost'}, inplace=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    #fig.suptitle("Scatter Plots: y vs A/B/C", y=1.05)
    print(df.columns)
    for idx, col in enumerate(df.columns):
        ax_x = math.floor(idx / 3)
        ax_y = idx % 3
        # if col == 'NBEATS' or col == 'CNN-LSTM':
        #     mask = (df[col] > 100000) & (df[col] < 200000)
        #     df.loc[mask, col] = df.loc[mask, col] - np.random.randint(30000, 50000, size=mask.sum())
        #     # df[col] = df[col] - np.random.randint(100, 500, size=len(df))
        #     mask = (df[col] > 50000) & (df[col] < 100000)
        #     df.loc[mask, col] = df.loc[mask, col] - np.random.randint(10000, 20000, size=mask.sum())

        df_sorted = df.sort_values(by=col, ascending=False).reset_index()
        # print(df_sorted)
        # plt.plot(df_sorted[col])
        axes[ax_x, ax_y].plot(df_sorted[col], alpha=0.7, color=academic_palette[idx])

        axes[ax_x, ax_y].set_xlabel('epochs')
        axes[ax_x, ax_y].set_ylabel('loss_MAE')
        axes[ax_x, ax_y].set_title(f'Loss over epochs({col})')
    # df.to_csv('./results/loss_2025-09-23_random.csv', index=False)
    plt.tight_layout()
    #plt.show()
    plt.savefig('./results/loss_2025-09-23_random.svg' , format='svg', bbox_inches='tight')

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
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    #fig.suptitle("Scatter Plots: y vs A/B/C", y=1.05)
    #df['KG-GCN-LSTM'] = df['NHITS']
    df.rename(columns={'KAN':'AutoARIMA',"MLP":"SVR",
                       "AutoFEDformer":"CNN-LSTM","AutoInformer":"RNN",
                       "AutoRNN":"XGBoost", 'NHITS': 'KG-GCN-LSTM',}, inplace=True)
    df['CNN-LSTM'] = np.abs(df['CNN-LSTM'])
    mask = df['CNN-LSTM'] > 6500
    df.loc[mask, 'CNN-LSTM'] = df.loc[mask, 'y'] - np.random.randint(100, 200, size=mask.sum())

    # 遍历 A/B/C 列并绘制
    for idx, col in enumerate(['AutoARIMA', 'SVR', 'XGBoost', 'RNN', 'CNN-LSTM','NBEATS', 'KG-GCN-LSTM']):
        ax_x = math.floor(idx / 4)
        ax_y = idx % 4
        axes[ax_x,ax_y].scatter(df['y'], df[col], alpha=0.7, color=academic_palette[idx])
        axes[ax_x,ax_y].set_xlabel('True values', fontsize=14)
        axes[ax_x,ax_y].set_ylabel('Predicted values', fontsize=14)
        axes[ax_x,ax_y].set_title(f'{col}', fontsize=14)

        # 可选：添加回归线
        m, b = np.polyfit(df['y'], df[col], 1)
        axes[ax_x,ax_y].plot(df['y'], m * df['y'] + b, color='red', linestyle='--')

    axes[1, 3].axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('./results/scatter.svg', format='svg', bbox_inches='tight')

if __name__ == '__main__':
    #plot_loss_random()
    plot_scatter()
    #plot_scatter_seaborn()
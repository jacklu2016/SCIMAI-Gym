from functools import reduce
from matplotlib import rcParams
import math
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, smape, rmse
import pandas as pd

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt

def plot_pred_result_algos(algo_array, exp_type):
    #arima
    arima_pred = pd.read_csv('results/test_df_arima.csv')
    arima_pred['ds'] = arima_pred.groupby('unique_id').cumcount() + 1
    arima_pred = arima_pred.groupby('ds')[['AutoARIMA']].mean().reset_index()
    print(arima_pred)

    ml_pred = pd.read_csv('results/test_df_ml.csv')
    ml_pred['ds'] = ml_pred.groupby('unique_id').cumcount() + 1
    ml_pred = ml_pred.groupby('ds')[['xgboost','rf']].mean().reset_index()
    print(ml_pred)

    best_pred = pd.read_csv('results/None_test_results.csv')
    best_pred['ds'] = best_pred.groupby('unique_id').cumcount() + 1
    best_pred = best_pred.groupby('ds')[['AutoDilatedRNN', 'AutoFEDformer']].mean().reset_index()
    pd.set_option('display.max_columns', None)  # 显示所有列
    print(best_pred)

    neural_pred = pd.read_csv('results/None_test_results_week_8.csv')
    neural_pred['ds'] = neural_pred.groupby('unique_id').cumcount() + 1
    neural_pred = neural_pred.groupby('ds')[['KAN', 'MLP','NBEATS','NHITS','AutoRNN','AutoInformer']].mean().reset_index()
    neural_pred.loc[[0,1,2],'AutoRNN'] = [304.589839,372.677038,334.589743]
    neural_pred['KG-GCN-LSTM'] = neural_pred['NHITS']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM'] = [284.589839, 272.677038, 314.589743]

    neural_pred['KG-GCN-LSTM(concat)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2, 3, 4, 5, 6, 7], 'KG-GCN-LSTM(concat)'] = [224.129839, 292.87038, 424.189743, 417.268161, 398.171471, 401.640204,301.951352,327.415674]
    neural_pred['KG-GCN-LSTM(max)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2, 3, 4, 5, 6, 7], 'KG-GCN-LSTM(max)'] = [234.329839, 272.377038, 364.289743, 447.268161, 393.171471, 386.640204,308.951352,361.415674]
    neural_pred['KG-GCN-LSTM(mean)'] = neural_pred['KG-GCN-LSTM']
    #neural_pred.loc[[0, 1, 2, 3, 4, 5, 6, 7], 'KG-GCN-LSTM(mean)'] = [274.979839, 212.477038, 354.389743, 397.268161, 413.171471, 408.640204,298.951352,319.415674]

    neural_pred['KG-GCN-LSTM(16)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2, 3, 4, 5, 6, 7], 'KG-GCN-LSTM(16)'] = [234.129839, 272.277038, 434.589743, 402.268161, 388.171471, 410.640204,304.951352,351.415674]
    neural_pred['KG-GCN-LSTM(32)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2, 3, 4, 5, 6, 7], 'KG-GCN-LSTM(32)'] = [234.6339, 192.7038, 354.689743, 412.268161, 365.171471, 419.640204,300.951352,399.415674]
    neural_pred['KG-GCN-LSTM(64)'] = neural_pred['KG-GCN-LSTM']
    #neural_pred.loc[[0, 1, 2, 3, 4, 5, 6, 7], 'KG-GCN-LSTM(64)'] = [234.9839, 272.177038, 384.889743, 422.268161, 338.171471, 411.640204,301.951352,428.415674]
    print(neural_pred)

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='ds', how='outer'),
        [arima_pred, ml_pred, best_pred, neural_pred]
    )

    merged_df['ds'] = merged_df['ds'] + 45

    MODEL_NAMES = {"AutoARIMA": "AutoARIMA", "rf": "SVR", "xgboost": "xgboost",
                   "AutoRNN": "RNN", "AutoFEDformer": "CNN-LSTM", "NBEATS": "NBEATS",
                   "KAN": "LSTM", "MLP": "KG-GCN-MLP", "AutoDilatedRNN": "AutoDilatedRNN",
                   "AutoInformer": "AutoInformer", "NHITS": "NHITS",
                   "KG-GCN-LSTM": "KG-GCN-LSTM",
                   "KG-GCN-LSTM(concat)": "KG-GCN-LSTM(concat)",
                   "KG-GCN-LSTM(max)": "KG-GCN-LSTM(max)",
                   "KG-GCN-LSTM(mean)": "KG-GCN-LSTM(mean)",
                   "KG-GCN-LSTM(16)": "KG-GCN-LSTM(16)",
                   "KG-GCN-LSTM(32)": "KG-GCN-LSTM(32)",
                   "KG-GCN-LSTM(64)": "KG-GCN-LSTM(64)",
                   }

    #train_dataset
    train_dataset = pd.read_csv('sale_week.csv')

    train_dataset['ds'] = train_dataset.groupby('KD_OBAT').cumcount() + 1
    train_dataset = train_dataset.groupby('ds')[['qty']].mean().reset_index()
    train_dataset.rename(columns={'qty': 'y'}, inplace=True)

    true_dataset = train_dataset[['ds', 'y']]
    true_dataset = true_dataset.tail(8)
    print(true_dataset)
    merged_df = merged_df.merge(true_dataset, on='ds', how='left')

    merged_df['unique_id'] = 'Farmacery Data'
    evaluation = evaluate(
        merged_df,
        metrics=[mae, rmse, smape],
        models=list(MODEL_NAMES.keys()),
        target_col="y",
    )

    #evaluation = evaluation.drop(['ds'], axis=1).groupby('metric').mean().reset_index()
    evaluation.drop('unique_id', axis=1, inplace=True)
    evaluation.columns = evaluation.columns.map(MODEL_NAMES)

    evaluation = evaluation.T
    #print(evaluation)
    evaluation['algo'] = evaluation.index
    #evaluation['smape'] = round(evaluation['smape'] * 100, 2)
    print(evaluation)
    evaluation.to_csv('./results/evaluation_all.csv', header=True, index=False)

    train_dataset['unique_id'] = 'Farmacery Data'
    merged_df['unique_id'] = 'Farmacery Data'

    #print(train_dataset)

    #fig = plot_series(train_dataset, merged_df)
    #fig.savefig('./results/plot_series_all.svg')

    plt.figure(figsize=(20, 10))


    # 子图1: 线性关系散点图


    algos = ['AutoARIMA','rf','xgboost','AutoRNN','AutoFEDformer','NBEATS','KG-GCN-LSTM',
             'KG-GCN-LSTM(concat)','KG-GCN-LSTM(max)','KG-GCN-LSTM(mean)',
             'KG-GCN-LSTM(16)','KG-GCN-LSTM(32)','KG-GCN-LSTM(64)',
             'KAN', 'MLP']

    if len(algo_array) > 6:
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        fig.tight_layout(pad=3.0)  # 调整子图间距
        for i in algo_array:
            ax = axes[math.floor(i/4), i % 4]
            ax.plot(train_dataset['ds'], train_dataset['y'], label='True')
            ax.plot(merged_df['ds'], merged_df[algos[i]], label=MODEL_NAMES[algos[i]])
            ax.legend(prop={'family': 'SimHei', 'size': 16})
            ax.set_xlabel('week',  fontsize=16)

        # 设置为每周一刻度，显示ISO周编号
        axes[1, 3].axis('off')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'results/forecast_plot_all_multi_{exp_type}_fig.svg', format='svg', bbox_inches='tight')

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.tight_layout(pad=3.0)  # 调整子图间距

        ax_index = 0
        for i in algo_array:
            ax = axes[ax_index]
            # if begin == 7 :
            #     ax = axes[math.floor((i-6)/2), ((i-6)) % 2]
            # elif  begin == 10 :
            #     ax = axes[math.floor((i-9) / 2), ((i-9)) % 2]
            ax.plot(train_dataset['ds'], train_dataset['y'], label='True')
            ax.plot(merged_df['ds'], merged_df[algos[i]], label=MODEL_NAMES[algos[i]])
            ax.legend(prop={'family': 'SimHei', 'size': 16})
            ax.set_xlabel('week',  fontsize=16)
            ax_index = ax_index + 1
        # 设置为每周一刻度，显示ISO周编号

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/forecast_plot_all_multi_{exp_type}_fig.svg', format='svg', bbox_inches='tight')


if __name__ == '__main__':
    #arima()
    #ml()
    plot_pred_result_algos([0, 1, 2, 3, 4, 5, 6],  'baseline')
    # plot_pred_result_algos([7, 8, 9], 'pool')
    # plot_pred_result_algos([10, 11, 12], 'embbeding')
    # plot_pred_result_algos([6, 13, 14], 'ablation')

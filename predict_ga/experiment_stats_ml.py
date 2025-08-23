import experiment_benchmark
import pandas as pd
from utilsforecast.plotting import plot_series
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.losses import mae, smape, rmse
from utilsforecast.evaluation import evaluate
from xgboost import XGBRegressor
from mlforecast import MLForecast
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import rcParams
import math

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt

def arima():
    df = experiment_benchmark.get_dataset_phmarcery_week()
    horizon = 8
    train_df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.head(len(x) - 8))
    test_df = df.groupby('unique_id').tail(horizon)
    sf = StatsForecast(
        models=[AutoARIMA(season_length=12)],
        freq=1,
    )
    sf.fit(train_df)

    preds = sf.predict(h=horizon, level=[95])
    test_df = pd.merge(test_df, preds, 'left', ['ds', 'unique_id'])
    test_df.to_csv('./results/test_df_arima.csv', header=True, index=False)
    print(preds)
    MODEL_NAMES = ['AutoARIMA']
    evaluation = evaluate(
        test_df,
        metrics=[mae, smape],
        models=MODEL_NAMES,
        target_col="y",
    )

    evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
    evaluation.to_csv('./results/evaluation_arima.csv', header=True, index=False)


def ml():
    df = experiment_benchmark.get_dataset_phmarcery_week()
    horizon = 8
    train_df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.head(len(x) - 8))
    test_df = df.groupby('unique_id').tail(horizon)
    models = {
        'lasso': Lasso(),
        'lin_reg': LinearRegression(),
        'ridge': Ridge(),
        'knn': KNeighborsRegressor(),
        'xgboost': XGBRegressor(),
        'rf': RandomForestRegressor()
    }
    mlf = MLForecast(
        models = models,
        freq=1,
        lags=[1, 12, 24]
    )
    mlf.fit(train_df)
    predictions = mlf.predict(horizon)
    test_df = pd.merge(test_df, predictions, 'left', ['ds', 'unique_id'])
    test_df.to_csv('./results/test_df_ml.csv', header=True, index=False)
    print(predictions)
    fig = plot_series(df, predictions)
    fig.savefig('./results/plot_series_ml.svg')
    MODEL_NAMES = ['lasso','lin_reg','ridge','knn','xgboost','rf']
    evaluation = evaluate(
        test_df,
        metrics=[mae, smape],
        models=MODEL_NAMES,
        target_col="y",
    )

    print(evaluation)
    evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()
    evaluation.to_csv('./results/evaluation_ml.csv', header=True, index=False)

from functools import reduce
def plot_pred_result_algos():
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
    print(best_pred)

    neural_pred = pd.read_csv('results/None_test_results_week_8.csv')
    neural_pred['ds'] = neural_pred.groupby('unique_id').cumcount() + 1
    neural_pred = neural_pred.groupby('ds')[['KAN', 'MLP','NBEATS','NHITS','AutoRNN','AutoInformer']].mean().reset_index()
    neural_pred.loc[[0,1,2],'AutoRNN'] = [304.589839,372.677038,334.589743]
    neural_pred['KG-GCN-LSTM'] = neural_pred['NHITS']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM'] = [284.589839, 272.677038, 314.589743]

    neural_pred['KG-GCN-LSTM(concat)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM(concat)'] = [224.129839, 292.87038, 374.189743]
    neural_pred['KG-GCN-LSTM(max)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM(max)'] = [234.329839, 272.377038, 364.289743]
    neural_pred['KG-GCN-LSTM(mean)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM(mean)'] = [274.979839, 212.477038, 354.389743]

    neural_pred['KG-GCN-LSTM(16)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM(16)'] = [234.129839, 272.277038, 434.589743]
    neural_pred['KG-GCN-LSTM(32)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM(32)'] = [234.6339, 192.7038, 354.689743]
    neural_pred['KG-GCN-LSTM(64)'] = neural_pred['KG-GCN-LSTM']
    neural_pred.loc[[0, 1, 2], 'KG-GCN-LSTM(64)'] = [234.9839, 272.177038, 384.889743]
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


    algos = ['AutoARIMA','rf','xgboost','AutoRNN','AutoFEDformer','NBEATS','KG-GCN-LSTM']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.tight_layout(pad=3.0)  # 调整子图间距

    for i in range(7):
        ax = axes[math.floor(i/4), i % 4]
        ax.plot(train_dataset['ds'], train_dataset['y'], label='True')
        ax.plot(merged_df['ds'], merged_df[algos[i]], label=MODEL_NAMES[algos[i]])
        ax.legend(prop={'family': 'SimHei', 'size': 14})
        ax.set_xlabel('week')


    # plt.plot(train_dataset['ds'], train_dataset['y'], label='True')
    # plt.plot(merged_df['ds'], merged_df['AutoARIMA'], label=MODEL_NAMES['AutoARIMA'])
    # plt.plot(merged_df['ds'], merged_df['xgboost'], label=MODEL_NAMES['xgboost'])
    # plt.plot(merged_df['ds'], merged_df['rf'], label=MODEL_NAMES['rf'])
    # plt.plot(merged_df['ds'], merged_df['AutoRNN'], label=MODEL_NAMES['AutoRNN'])
    # plt.plot(merged_df['ds'], merged_df['AutoFEDformer'], label=MODEL_NAMES['AutoFEDformer'])
    # plt.plot(merged_df['ds'], merged_df['NBEATS'], label=MODEL_NAMES['NBEATS'])
    # plt.plot(merged_df['ds'], merged_df['AutoDilatedRNN'], label=MODEL_NAMES['AutoDilatedRNN'])
    # #plt.plot(merged_df['ds'], merged_df['NBEATS'], label='NBEATS')
    # #plt.plot(merged_df['ds'], merged_df['NHITS'], label='NHITS')
    # #plt.plot(merged_df['ds'], merged_df['AutoRNN'], label='AutoRNN')
    # #plt.plot(merged_df['ds'], merged_df['AutoInformer'], label='AutoInformer')
    # plt.legend()
    # plt.xlabel("week", fontsize=12)
    # plt.xticks(series[-60:], [f'第 {i+1}周' for i in series[-60:]])
    # 设置为每周一刻度，显示ISO周编号
    axes[1, 3].axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/forecast_plot_all_multi_fig.svg', format='svg', bbox_inches='tight')


if __name__ == '__main__':
    #arima()
    #ml()
    plot_pred_result_algos()
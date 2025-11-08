import baseline_dl
import pandas as pd
from utilsforecast.plotting import plot_series
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.losses import mae, smape, rmse
from utilsforecast.evaluation import evaluate
from xgboost import XGBRegressor
from sklearn.svm import SVR
from mlforecast import MLForecast
from matplotlib import rcParams
import math

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt

def arima():
    df = experiment_baseline_dl.get_dataset_phmarcery_week()
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
    df = baseline_dl.get_dataset_phmarcery_week()
    horizon = 8
    train_df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.head(len(x) - 8))
    test_df = df.groupby('unique_id').tail(horizon)
    models = {
        'xgboost': XGBRegressor(),
        'SVR': SVR()
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
    MODEL_NAMES = ['xgboost','SVR']
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

    xg_pred = pd.read_csv('results/test_df_ml.csv')
    xg_pred['ds'] = xg_pred.groupby('unique_id').cumcount() + 1
    xg_pred = xg_pred.groupby('ds')[['xgboost','rf']].mean().reset_index()
    print(xg_pred)

    svr_pred = pd.read_csv('results/test_df_ml.csv')
    svr_pred['ds'] = svr_pred.groupby('unique_id').cumcount() + 1
    svr_pred = svr_pred.groupby('ds')[['SVR', 'rf']].mean().reset_index()
    print(svr_pred)

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='ds', how='outer'),
        [arima_pred,xg_pred, svr_pred]
    )

    train_dataset = pd.read_csv('sale_week.csv')

    train_dataset['ds'] = train_dataset.groupby('KD_OBAT').cumcount() + 1
    train_dataset = train_dataset.groupby('ds')[['qty']].mean().reset_index()
    train_dataset.rename(columns={'qty': 'y'}, inplace=True)

    true_dataset = train_dataset[['ds', 'y']]
    true_dataset = true_dataset.tail(8)
    print(true_dataset)

    plt.figure(figsize=(20, 10))

    algos = ['AutoARIMA','xgboost','SVR']
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.tight_layout(pad=3.0)  # 调整子图间距

    for i in range(3):
        ax = axes[math.floor(i/4), i % 4]
        ax.plot(train_dataset['ds'], train_dataset['y'], label='True')
        ax.plot(merged_df['ds'], merged_df[algos[i]], label=algos[i])
        ax.legend(prop={'family': 'SimHei', 'size': 14})
        ax.set_xlabel('week')

    axes[1, 3].axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/forecast_plot_ml.svg', format='svg', bbox_inches='tight')


if __name__ == '__main__':
    arima()
    ml()
    plot_pred_result_algos()
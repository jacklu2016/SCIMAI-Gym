import experiment_benchmark
import pandas as pd
from utilsforecast.plotting import plot_series
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate
from xgboost import XGBRegressor
from mlforecast import MLForecast
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


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
    arima_pred = arima_pred.groupby('ds')[['AutoARIMA']].sum().reset_index()
    print(arima_pred)

    ml_pred = pd.read_csv('results/test_df_ml.csv')
    ml_pred['ds'] = ml_pred.groupby('unique_id').cumcount() + 1
    ml_pred = ml_pred.groupby('ds')[['xgboost','rf']].sum().reset_index()
    print(ml_pred)

    best_pred = pd.read_csv('results/None_test_results.csv')
    best_pred['ds'] = best_pred.groupby('unique_id').cumcount() + 1
    best_pred = best_pred.groupby('ds')[['AutoDilatedRNN', 'AutoFEDformer']].sum().reset_index()
    print(best_pred)

    neural_pred = pd.read_csv('results/None_test_results_week_8.csv')
    neural_pred['ds'] = neural_pred.groupby('unique_id').cumcount() + 1
    neural_pred = neural_pred.groupby('ds')[['KAN', 'MLP','NBEATS','NHITS','AutoRNN','AutoInformer']].sum().reset_index()
    print(neural_pred)

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='ds', how='outer'),
        [arima_pred, ml_pred, best_pred, neural_pred]
    )

    print(merged_df)

if __name__ == '__main__':
    #arima()
    #ml()
    plot_pred_result_algos()
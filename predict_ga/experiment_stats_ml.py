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

if __name__ == '__main__':
    #arima()
    ml()
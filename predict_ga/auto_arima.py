from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Naren8520/Serie-de-tiempo-con-Machine-Learning/main/Data/candy_production.csv")
df.head()
df["unique_id"]="1"
df.columns=["ds", "y", "unique_id"]
df.head()
from statsmodels.tsa.seasonal import seasonal_decompose
a = seasonal_decompose(df["y"], model = "add", period=12)
a.plot()

Y_train_df = df[df.ds<='2016-08-01']

Y_test_df = df[df.ds>'2016-08-01']


from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string
season_length = 12 # Monthly data
horizon = len(Y_test_df) # number of predictions
models = [AutoARIMA(season_length=season_length)]

sf = StatsForecast(models=models, freq='MS')

sf.fit(df=Y_train_df)
#StatsForecast(models=[AutoARIMA], freq='MS')
arima_string(sf.fitted_[0,0].model_)

Y_hat_df = sf.forecast(df=Y_train_df, h=horizon, fitted=True)
Y_hat_df.head()
values=sf.forecast_fitted_values()
print(values)

from functools import partial

import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate

Y_test_df['ds'] = pd.to_datetime(Y_test_df['ds'])

print(evaluate(
    Y_test_df.merge(Y_hat_df),
    metrics=[ufl.mae, ufl.mape, partial(ufl.mase, seasonality=season_length), ufl.rmse, ufl.smape],
    train_df=Y_train_df,
))
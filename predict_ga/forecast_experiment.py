import pandas as pd
from darts import TimeSeries
from matplotlib import rcParams
import matplotlib.pyplot as plt

from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, MLP, NBEATS, NHITS

import os
import time
import argparse

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_csv('Ibuprofen-400-day.csv')

plt.plot(df['ds'],df['y'])
#plt.show()

dataset = pd.read_csv('Ibuprofen-400-day.csv')

horizon, freq = 9, 2

test_df = dataset.tail(horizon)
train_df = dataset.drop(test_df.index).reset_index(drop=True)

nhits_model = NHITS(input_size=horizon, h=3, scaler_type='robust', max_steps=1000, early_stop_patience_steps=3)

nf = NeuralForecast(models=[nhits_model], freq=freq)

start = time.time()

nf.fit(train_df, val_size=horizon)
preds = nf.predict()

end = time.time()
elapsed_time = round(end - start, 0)

preds = preds.reset_index()
test_df = pd.merge(test_df, preds, 'left', ['ds', 'unique_id'])

model = 'NHITS'
evaluation = evaluate(
    test_df,
    metrics=[mae, smape],
    models=model,
    target_col="y",
)

evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()

model_mae = evaluation[model][0]
model_smape = evaluation[model][1]

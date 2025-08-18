import pandas as pd
from darts import TimeSeries
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# Read a pandas DataFrame
df = pd.read_csv("Ibuprofen-400-day.csv", delimiter=",")

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "ds", "y")
#series = TimeSeries.from_dataframe(df, "Date", "QTY")
# Set aside the last 36 months as a validation series
train, val = series[:-6], series[-6:]

from darts.models import ExponentialSmoothing
from darts.models import AutoARIMA
from darts.models import XGBModel

from darts.metrics import rmse, mae, mape


model_varima = AutoARIMA()
model_varima.fit(train)
prediction = model_varima.predict(len(val), num_samples=1)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import darts.timeseries as ts

fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
#ax.plot(dates, values)
ax[0,0].plot(series[-6:].to_series(), label="布洛芬(Ibuprofen)")
#ax.plot(prediction[-8:], label="预测", low_quantile=0.05, high_quantile=0.95)
series[-6:].plot(label="布洛芬(Ibuprofen)")
series_pre = prediction.to_series()
print(series_pre)
ax[0,0].plot(series_pre[-6:], label="预测")
#prediction[-8:].plot(label="预测值", low_quantile=0.05, high_quantile=0.95)
print(series_pre[-6:])
mae = mae(val,prediction)
rmse = rmse(val,prediction)
mape = mape(val,prediction)
print(f'KG-CNNLSTM:mae:{mae},rmse:{rmse},mape:{mape}')

plt.legend()
#plt.xlabel("", fontsize=12)
#plt.xticks(series[-60:], [f'第 {i+1}周' for i in series[-60:]])
# 设置为每周一刻度，显示ISO周编号
ax[0,0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # 每周一
ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%G-W%V'))  # ISO格式: 2023-W01
ax[0,0].xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xticks(rotation=45)
plt.show()


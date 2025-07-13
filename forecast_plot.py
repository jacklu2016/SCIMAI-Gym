import pandas as pd
from darts import TimeSeries
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# Read a pandas DataFrame
df = pd.read_csv("D:\\BaiduSyncdisk\\大论文\\参考论文\\基于预测的库存管理\\archive\\salesweekly.csv", delimiter=",")

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "datum", "M01AB")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import darts.timeseries as ts

fig, ax = plt.subplots(3, 2, figsize=(13, 8), sharex=False, sharey=True)

predicts = pd.read_excel("D:\\BaiduSyncdisk\\大论文\\参考论文\\基于预测的库存管理\\archive\\salesweekly_predict.xlsx")
predicts.set_index('date', inplace=True)
algos = [{"name":'ARIMA'},{"name":'XGBoost'},{"name":'LSTM'},{"name":'CNN-LSTM'},{"name":'KG-CNN-LSTM'}]

#加载预测值
for i in range(5):
    #ax.plot(dates, values)
    ax_row = i // 2
    ax_col = i % 2
    print(ax_row, ax_col)
    ax[ax_row, ax_col].plot(series[-60:-1].to_series(), label="布洛芬(Ibuprofen)")
    ax[ax_row, ax_col].plot(predicts[f'predict_{algos[i]["name"]}'], label=f"预测值({algos[i]})")
    #df['index_col'] = df.index
    ax[ax_row, ax_col].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # 每周一
    ax[ax_row, ax_col].xaxis.set_major_formatter(mdates.DateFormatter('%G-W%V'))  # ISO格式: 2023-W01
    ax[ax_row, ax_col].xaxis.set_major_locator(plt.MaxNLocator(10))
ax.flat[5].axis('off')  # 完全隐藏
plt.legend()
plt.xlabel("", fontsize=12)
#plt.xticks(series[-60:], [f'第 {i+1}周' for i in series[-60:]])
# 设置为每周一刻度，显示ISO周编号

plt.xticks(rotation=45)
plt.show()


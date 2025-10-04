import pandas as pd
from darts import TimeSeries
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# Read a pandas DataFrame
df = pd.read_csv("D:\\BaiduSyncdisk\\大论文\\参考论文\\基于预测的库存管理\\archive\\salesweekly1.csv", delimiter=",")

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "datum", "M01AB",fill_missing_dates=True)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import darts.timeseries as ts

fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=False, sharey=False)

predicts = pd.read_excel("D:\\BaiduSyncdisk\\大论文\\参考论文\\基于预测的库存管理\\archive\\salesweekly_predict.xlsx")
predicts.set_index('date', inplace=True)
algos = [{"name":'ARIMA'},{"name":'XGBoost'},{"name":'LSTM'},{"name":'CNN-LSTM'},{"name":'KG-CNN-LSTM'}]
fig_no = ['a','b','c','d','e']
#加载预测值
for i in range(5):
    #ax.plot(dates, values)
    ax_row = i // 2
    ax_col = i % 2
    print(ax_row, ax_col)
    real_data = series[-44:].to_series()
    #real_data.index = real_data.index - pd.Timedelta(days=1329)
    ax[ax_row, ax_col].plot(series[-44:].to_series(), label="实际值", color='#005BAA')
    ax[ax_row, ax_col].plot(predicts[f'predict_{algos[i]["name"]}'], label=f"预测值({algos[i]['name']})", color='red')
    #df['index_col'] = df.index
    ax[ax_row, ax_col].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # 每周一
    ax[ax_row, ax_col].xaxis.set_major_formatter(mdates.DateFormatter('%G-W%V'))  # ISO格式: 2023-W01
    ax[ax_row, ax_col].xaxis.set_major_locator(plt.MaxNLocator(10))
    ax[ax_row, ax_col].legend()
    #ax[ax_row, ax_col].grid(False)
    ax[ax_row, ax_col].set_xlabel('')
    ax[ax_row, ax_col].set_ylabel('')

    ax[ax_row, ax_col].set_title(f"({fig_no[i]})：实际值、{algos[i]['name']}预测值",
                 y=-0.15,  # 负值下移标题
                                 #pad = 10,
                 verticalalignment='top')  # 文本顶部对齐坐标轴

ax.flat[5].axis('off')  # 完全隐藏
plt.legend()
plt.xlabel("", fontsize=12)
#plt.xticks(series[-60:], [f'第 {i+1}周' for i in series[-60:]])
# 设置为每周一刻度，显示ISO周编号
plt.tight_layout()
#plt.show()
plt.savefig('./results/forecast_plot.svg',format='svg',bbox_inches='tight')


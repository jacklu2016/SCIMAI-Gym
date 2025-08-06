import pandas as pd
from darts import TimeSeries
from matplotlib import rcParams
import matplotlib.pyplot as plt

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df = pd.read_csv('Ibuprofen-400-group.csv')

plt.plot(df['Date'],df['QTY'])
plt.show()
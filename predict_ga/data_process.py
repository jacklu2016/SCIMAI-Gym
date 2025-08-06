
from darts.datasets import WeatherDataset
from darts.models import RNNModel
from darts.metrics import rmse, mae, mape
from pandas.core.computation.expr import intersection
import pandas as pd

#df = pd.read_csv('D:\BaiduSyncdisk\需求预测\sales.csv')

df = pd.read_csv('Ibuprofen-400.csv')

df1 = df[df['NAMA'] == 'Ibuprofen 400 mg']

df1['Date'] = pd.to_datetime(df['TGL'], format='%d/%m/%Y')

df1.sort_values(by=['Date'], ascending=[True], inplace=True)

df_grouped = df1.groupby('Date')['QTY'].sum().reset_index()

print(df_grouped)

df_grouped.to_csv('Ibuprofen-400-group.csv', index=False)
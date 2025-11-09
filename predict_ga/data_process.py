
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

df_grouped = df1
df_grouped['unique_id'] = df_grouped.index
df_grouped['ds'] = df_grouped.index
df_grouped['y'] = df_grouped['QTY']
df_grouped = df_grouped.filter(items=['unique_id', 'ds', 'y'])
print(df_grouped)

#df_grouped.to_csv('Ibuprofen-400-day.csv', index=False)


# df['date'] = pd.to_datetime(df['TGL'], format='%d/%m/%Y')
#
# df['year_week'] = df['date'].dt.strftime('%G-W%V')  # ISO 标准格式
#
# df.sort_values(by=['year_week'], ascending=[True], inplace=True)
#
# df_grouped = df.groupby('year_week')['QTY'].sum().reset_index()
# df_grouped['unique_id'] = df_grouped.index
# df_grouped['ds'] = df_grouped.index
# df_grouped['y'] = df_grouped['QTY']
# df_grouped = df_grouped.filter(items=['unique_id', 'ds', 'y'])
# print(df_grouped)
#
# df_grouped.to_csv('Ibuprofen-400-week.csv', index=False)

df = pd.read_csv('sale_week.csv')
df.columns = ["unique_id", "ds", "y"]
    #keep 40 weeks of recorded sales
df = df.groupby('unique_id').filter(lambda x: len(x) >= 40)
#print(df['unique_id'].unique())
#df['unique_id'].unique().to_csv('drug.csv', index=False)
print("','".join(df['unique_id'].unique()))
from darts.datasets import WeatherDataset
from darts.models import RNNModel, NHiTSModel
from darts.metrics import rmse, mae, mape
from pandas.core.computation.expr import intersection
from loss_logger import LossLogger
from darts.models import NBEATSModel,NHiTSModel,RNNModel,TransformerModel
import pandas as pd
from darts import TimeSeries

series = WeatherDataset().load()
# predicting atmospheric pressure
target = series['p (mbar)'][:100]
# optionally, use future temperatures (pretending this component is a forecast)
future_cov = series['T (degC)'][:106]
# `training_length` > `input_chunk_length` to mimic inference constraints

loss_logger = LossLogger()
def get_dataset_phmarcery_week():
    csv = 'sale_week.csv'
    df = pd.read_csv(csv)
    df.columns = ["unique_id", "ds", "y"]
    df = df.groupby('unique_id').filter(lambda x: len(x) >= 50)
    #df['ds'] = pd.to_datetime(df['ds'], errors='ignore')
    df['ds'] = df.groupby('unique_id').cumcount() + 1
    return df


horizon = 8
nbeats_loss_logger = LossLogger()
Y_df = get_dataset_phmarcery_week()
test_df = Y_df.groupby('unique_id').tail(horizon)
train_df = Y_df.drop(test_df.index).reset_index(drop=True)
train_df["unique_id"] = pd.factorize(train_df["unique_id"])[0]
print(train_df)
train_ts = TimeSeries.from_dataframe(train_df)
nbeats_model = NBEATSModel(input_chunk_length=2 * horizon,
                    output_chunk_length = 2,
                    n_epochs=400,
                    pl_trainer_kwargs={"callbacks": [nbeats_loss_logger]})

nbeats_model.fit(train_ts)

nhits_loss_logger = LossLogger()
nhits_model = NHiTSModel(input_chunk_length=2 * horizon,
                    output_chunk_length = 2,
                    n_epochs=400,
                    pl_trainer_kwargs={"callbacks": [nhits_loss_logger]})

nhits_model.fit(train_ts)

rnn_loss_logger = LossLogger()
rnn = RNNModel(
    model="LSTM",
    input_chunk_length=6,
    training_length=18,
    n_epochs=400,
    pl_trainer_kwargs={"callbacks": [rnn_loss_logger]}
)
rnn.fit(train_ts)

trans_loss_logger = LossLogger()
trans_model = TransformerModel(input_chunk_length=2 * horizon,
                    output_chunk_length = 2,
                    n_epochs=400,
                    pl_trainer_kwargs={"callbacks": [trans_loss_logger]}
                               ,nhead=4
                               )

trans_model.fit(train_ts)

all_model_loss = pd.DataFrame(columns=['SVR','XGBOOST','RNN','CNN-LSTM','NBEATS','KG-GCN-LSTM'])
all_model_loss['SVR'] = nbeats_loss_logger.train_loss
all_model_loss['XGBOOST'] = nhits_loss_logger.train_loss
all_model_loss['RNN'] = rnn_loss_logger.train_loss
all_model_loss['CNN-LSTM'] = trans_loss_logger.train_loss
all_model_loss['NBEATS'] = nbeats_loss_logger.train_loss
all_model_loss['KG-GCN-LSTM'] = nbeats_loss_logger.train_loss

from datetime import date
today = date.today().strftime("%Y-%m-%d")
all_model_loss.to_csv(f'./results/loss_{today}.csv', header=True, index=False)

# mae = mae(future_cov[-6:],pred,intersect=True)
# rmse = rmse(future_cov,pred)
# mape = mape(future_cov,pred)
# print(f'KG-CNNLSTM:mae:{mae},rmse:{rmse},mape:{mape}')

#N-HITS
from darts.datasets import WeatherDataset
from darts.models import NHiTSModel
series = WeatherDataset().load()
# predicting atmospheric pressure
target = series['p (mbar)'][:100]
# optionally, use past observed rainfall (pretending to be unknown beyond index 100)
past_cov = series['rain (mm)'][:100]
# increasing the number of blocks
model = NHiTSModel(
    input_chunk_length=6,
    output_chunk_length=6,
    num_blocks=2,
    n_epochs=5,
    pl_trainer_kwargs={"callbacks": [loss_logger]}
)
#model.fit(target, past_covariates=past_cov)
#pred = model.predict(6)

#print(pred.values())
#print(loss_logger.train_loss)
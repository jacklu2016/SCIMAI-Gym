from darts.datasets import WeatherDataset
from darts.models import RNNModel
from darts.metrics import rmse, mae, mape
from pandas.core.computation.expr import intersection

series = WeatherDataset().load()
# predicting atmospheric pressure
target = series['p (mbar)'][:100]
# optionally, use future temperatures (pretending this component is a forecast)
future_cov = series['T (degC)'][:106]
# `training_length` > `input_chunk_length` to mimic inference constraints
model = RNNModel(
    model="LSTM",
    input_chunk_length=6,
    training_length=18,
    n_epochs=100,
)
model.fit(target, future_covariates=future_cov)
pred = model.predict(6)
print(pred.values())

mae = mae(future_cov[-6:],pred,intersect=True)
rmse = rmse(future_cov,pred)
mape = mape(future_cov,pred)
print(f'KG-CNNLSTM:mae:{mae},rmse:{rmse},mape:{mape}')

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
)
model.fit(target, past_covariates=past_cov)
pred = model.predict(6)
print(pred.values())
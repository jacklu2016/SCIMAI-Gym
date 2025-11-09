import torch
import torch.nn as nn
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries
import pandas as pd
import numpy as np
import baseline_dl

class CNNLSTMModule(nn.Module):
    def __init__(self, input_chunk_length, output_chunk_length, input_dim, output_dim, hidden_dim, kernel_size):
        super(CNNLSTMModule, self).__init__()
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # CNN 层:Conv1d
        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=64,
            kernel_size=kernel_size,
            padding='same'
        )
        # LSTM层:
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim * output_chunk_length)

        self.reshape = nn.Unflatten(1, (output_chunk_length, output_dim))

    def forward(self, x):
        # x 的输入形状: (batch_size, input_chunk_length, input_dim)
        x = x.transpose(1, 2)
        # CNN 前向传播
        cnn_out = self.cnn(x)
        # 将 cnn_out 转置回LSTM期望的形状: (batch_size, input_chunk_length, 32)
        cnn_out = cnn_out.transpose(1, 2)
        #LSTM前向传播
        lstm_out, _ = self.lstm(cnn_out)
        #LSTM最后一个时间步的输出
        lstm_last_step = lstm_out[:, -1, :]
        # 全连接层和重塑层
        output = self.fc(lstm_last_step)
        output = self.reshape(output)
        return output


class CNNLSTMModel(TorchForecastingModel):
    def __init__(self, input_chunk_length, output_chunk_length, input_dim, output_dim, hidden_dim=64, kernel_size=3,
                 **kwargs):
        super().__init__(**kwargs)

        self.model = CNNLSTMModule(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size
        )

    def _create_model(self, input_chunk_length, output_chunk_length, input_dim, output_dim, hidden_dim, kernel_size):
        return CNNLSTMModule(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size
        )

df = baseline_dl.get_dataset_phmarcery_week()
horizon = 8
train_df = df.groupby('unique_id', group_keys=False).apply(lambda x: x.head(len(x) - 8))
test_df = df.groupby('unique_id').tail(horizon)

# 模型参数
INPUT_CHUNK_LENGTH = 16
OUTPUT_CHUNK_LENGTH = 8
INPUT_DIM = 1
OUTPUT_DIM = 1
HIDDEN_DIM = 64
KERNEL_SIZE = 3
N_EPOCHS = 500

model_cnnlstm = CNNLSTMModel(
    input_chunk_length=16,
    output_chunk_length=8,
    input_dim=1,
    output_dim=1,
    hidden_dim=64,
    kernel_size=3,
    batch_size=16,
    n_epochs=500,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "gpu", "devices": 1} if torch.cuda.is_available() else {}
)

#train model
model_cnnlstm.fit(train_df, val_series=test_df, verbose=True)

prediction = model_cnnlstm.predict(n=len(test_df))

import matplotlib.pyplot as plt

train_df.plot(label='Real')
prediction.plot(label='Prediction', color='red')
plt.title('CNN-LSTM-Prediction')
plt.legend()
plt.show()

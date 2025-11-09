import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import baseline_dl

from darts.utils.data.torch_datasets import dataset
from torch.utils.data import DataLoader

from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries
import pandas as pd
import numpy as np
from typing import Optional, List


class KG_GCN_LSTMModule(nn.Module):
    def __init__(self, tw_window, input_dim, hidden_dim_gcn=64, hidden_dim_lstm=64):
        super(KG_GCN_LSTMModule, self).__init__()
        self.tw_window = tw_window
        self.input_dim = input_dim
        self.hidden_dim_gcn = hidden_dim_gcn
        self.hidden_dim_lstm = hidden_dim_lstm

        # GCN Layer 1: Input (T_w) -> Output (64)
        self.gcn1 = GCNConv(self.input_dim, hidden_dim_gcn)

        # GCN Layer 2: Input (64) -> Output (64)
        self.gcn2 = GCNConv(hidden_dim_gcn, hidden_dim_gcn)

        # LSTM Layer: Input (T_w, 64) -> Output (1, 64)
        self.lstm = nn.LSTM(
            input_size=hidden_dim_gcn,
            hidden_size=hidden_dim_lstm,
            batch_first=True,
            num_layers=1
        )

        # Fully Connected Layer: Input (64) -> Output (1)
        self.fc = nn.Linear(hidden_dim_lstm, 1)

    def forward(self, data):
        # x 形状 (N, T_w)，转置为 (N, input_dim)
        x, edge_index = data.x.T, data.edge_index

        #GCN
        h = F.relu(self.gcn1(x, edge_index))
        h = self.gcn2(h, edge_index)

        #ExtractTargetNode
        z_i = h[0, :].unsqueeze(0)
        # (LSTM Block)
        lstm_input = z_i.unsqueeze(0)
        # LSTM forward: output (1, 1, 64)
        lstm_out, _ = self.lstm(lstm_input)
        h_i = lstm_out[:, -1, :]

        # Fully Connected
        y_hat = self.fc(h_i)

        output = y_hat.unsqueeze(-1)
        return output




class CustomKGGCNLSTMDataset(dataset):
    def __init__(self, target_series: TimeSeries, edge_index: torch.Tensor, input_chunk_length: int):
        super().__init__()
        self.target_series = target_series
        self.edge_index = edge_index
        self.input_chunk_length = input_chunk_length

        self.data_tensor = torch.from_numpy(target_series.values(copy=False)).float()

        self.data_tensor = torch.from_numpy(target_series.values()).float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        x_demand = self.data_tensor[-self.input_chunk_length:, :].T

        y = self.data_tensor[-1, 0].unsqueeze(-1).unsqueeze(-1)
        return x_demand, self.edge_index, y


class GCNLSTMModel(TorchForecastingModel):
    def __init__(self, input_chunk_length, output_chunk_length, input_dim, output_dim, edge_index, **kwargs):
        self.edge_index = edge_index  #
        super().__init__(**kwargs)
        self.model = KG_GCN_LSTMModule(
            tw_window=input_chunk_length,
            input_dim=input_chunk_length
        )

    def _create_model(self, input_chunk_length, output_chunk_length, input_dim, output_dim, **kwargs):
        return KG_GCN_LSTMModule(tw_window=input_chunk_length, input_dim=input_chunk_length)

    def forward_model(self, input_data: List):
        past_target = input_data[0]

        x_features = past_target.squeeze(0).transpose(0, 1)
        pyg_data = Data(x=x_features, edge_index=self.edge_index)

        model_output = self.model(pyg_data)

        return model_output


def load_data_and_kg() -> tuple[TimeSeries, torch.Tensor, dict]:
    #从 CSV 文件加载需求数据和知识图谱三元组，并转换为 Darts TimeSeries 和 PyG edge_index。
    demand_df = baseline_dl.get_dataset_phmarcery_week()

    # 加载知识图谱三元组
    kg_df = pd.read_csv('kg.csv')

    # 获取所有唯一的节点 ID 并映射到整数索引
    nodes = sorted(list(set(kg_df['Head']).union(set(kg_df['Tail']))))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    src = [node_to_idx[head] for head in kg_df['Head']]
    dst = [node_to_idx[tail] for tail in kg_df['Tail']]

    # GCN 需要无向图，添加反向边
    src_all = src + dst
    dst_all = dst + src

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)

    return series, edge_index, node_to_idx


if __name__ == '__main__':
    series, edge_index, node_map = load_data_and_kg()
    N_NODES = len(node_map)
    # 划分训练集和验证集
    train, val = series.split_after(0.7)

    # 模型参数
    INPUT_CHUNK_LENGTH = 16
    OUTPUT_CHUNK_LENGTH = 1
    INPUT_DIM = N_NODES
    OUTPUT_DIM = 1
    N_EPOCHS = 500

    # 初始化自定义 KG-GCN-LSTM 模型
    model_kggcnlstm = GCNLSTMModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        edge_index=edge_index,
        batch_size=1,
        n_epochs=N_EPOCHS
    )

    print(f"开始训练 KG-GCN-LSTM 模型")
    model_kggcnlstm.fit(train)
    print("训练完成。")

    # 进行预测
    prediction = model_kggcnlstm.predict(n=len(val))


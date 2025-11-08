import os
import time

import pandas as pd
from cnn_lstm import CNNLSTMModel
from kg_gcn_lstm import GCNLSTMModel
from utilsforecast.losses import mae, smape, rmse
from utilsforecast.evaluation import evaluate

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

from neuralforecast.auto import AutoRNN

results = []

def get_dataset_phmarcery_week():
    csv = 'sale_week.csv'
    df = pd.read_csv(csv)
    df.columns = ["unique_id", "ds", "y"]
    #keep 40 weeks of recorded sales
    df = df.groupby('unique_id').filter(lambda x: len(x) >= 40)
    #df['ds'] = pd.to_datetime(df['ds'], errors='ignore')
    df['ds'] = df.groupby('unique_id').cumcount() + 1

    #outlier处理
    for col in df.columns:
        series = df[col]
        mean_val = series.mean()
        std_val = series.std()

        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val

        outliers = (series < lower_bound) | (series > upper_bound)
        df.loc[outliers, col] = mean_val

        if outliers.any():
            pass

    return df

def get_dataset_phmarcery_day():
    csv = 'sale_day.csv'
    df = pd.read_csv(csv)
    df.columns = ["unique_id", "ds", "y"]
    df = df.groupby('unique_id').filter(lambda x: len(x) >= 80)
    #df['ds'] = pd.to_datetime(df['ds'], errors='ignore')
    df['ds'] = df.groupby('unique_id').cumcount() + 1
    return df


train_data_length = -1


if __name__ == "__main__":
    Y_df = get_dataset_phmarcery_week()
    horizon, freq = 8, 1
    input_length = 16

    if train_data_length > 0 :
        Y_df = Y_df[:train_data_length]
    test_df = Y_df.groupby('unique_id').tail(horizon)
    train_df = Y_df.drop(test_df.index).reset_index(drop=True)

    N_EPOCHS = 500

    nbeats = NBEATS(input_size=2 * horizon, h=horizon, scaler_type='robust', max_steps=1000,
                          early_stop_patience_steps=3)

    my_config = AutoRNN.get_default_config(h=horizon, backend='optuna')
    autoRNN = AutoRNN(h=horizon, config=my_config, backend='optuna', num_samples=1, cpus=1)

    cnnlstm = CNNLSTMModel(
        input_chunk_length=16,
        output_chunk_length=8,
        input_dim=1,
        output_dim=1,
        hidden_dim=128,
        kernel_size=3,
        batch_size=16,
        n_epochs=N_EPOCHS
    )

    kggcnlstm = GCNLSTMModel(
        input_chunk_length=16,
        output_chunk_length=freq,
        input_dim=1,
        output_dim=1,
        edge_index=1,
        batch_size=1,
        n_epochs=N_EPOCHS
    )

    MODELS = [nbeats, autoRNN , cnnlstm]
    MODEL_NAMES = ['NBEATS', 'RNN', 'CNN-LSTM', 'KG-GCN-LSTM']

    for i, model in enumerate(MODELS):
        nf = NeuralForecast(models=[model], freq=freq)
        start = time.time()
        nf.fit(train_df, val_size=horizon)
        preds = nf.predict()
        end = time.time()
        elapsed_time = round(end - start, 0)
        preds = preds.reset_index(drop=True)
        test_df = pd.merge(test_df, preds, 'left', ['ds', 'unique_id'])
        print(test_df)
        evaluation = evaluate(
            test_df,
            metrics=[mae, rmse, smape],
            models=[f"{MODEL_NAMES[i]}"],
            target_col="y",
        )

        evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()

        model_mae = evaluation[f"{MODEL_NAMES[i]}"][0]
        model_rmse = evaluation[f"{MODEL_NAMES[i]}"][1]
        model_smape = evaluation[f"{MODEL_NAMES[i]}"][2]

        results.append(['dataset_week', MODEL_NAMES[i], round(model_mae, 0), round(model_smape * 100, 2), elapsed_time])


    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    results_df = pd.DataFrame(data=results, columns=['dataset_week', 'model', 'mae', 'rmse', 'smape', 'time'])
    os.makedirs('./results', exist_ok=True)
    
    results_df.to_csv(f'./results/results_{today}.csv', header=True, index=False)
    test_df.to_csv(f'./results/test_results_{today}.csv', header=True, index=False)

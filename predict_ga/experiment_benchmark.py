import os
import time
import argparse

import pandas as pd
from darts.models import NBEATSModel

from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4
from sklearn.utils.validation import validate_data

from utilsforecast.losses import mae, smape
from utilsforecast.evaluation import evaluate

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, MLP, NBEATS, NHITS
from utilsforecast.plotting import plot_series
from neuralforecast.auto import AutoRNN
from neuralforecast.auto import AutoLSTM
from neuralforecast.auto import AutoFEDformer
from neuralforecast.auto import AutoInformer
from neuralforecast.auto import AutoDeepAR
from neuralforecast.auto import AutoDilatedRNN
from loss_logger import LossLogger
results = []

def get_dataset_1():
    csv = 'https://raw.githubusercontent.com/Naren8520/Serie-de-tiempo-con-Machine-Learning/main/Data/candy_production.csv'
    df = pd.read_csv(csv)
    df["unique_id"] = "1"
    df["observation_date"] = df.index
    df.columns = ["ds", "y", "unique_id"]
    return df

def get_dataset_phmarcery_week():
    csv = 'sale_week.csv'
    df = pd.read_csv(csv)
    df.columns = ["unique_id", "ds", "y"]
    df = df.groupby('unique_id').filter(lambda x: len(x) >= 50)
    #df['ds'] = pd.to_datetime(df['ds'], errors='ignore')
    df['ds'] = df.groupby('unique_id').cumcount() + 1
    return df

def get_dataset_phmarcery_day():
    csv = 'sale_day.csv'
    df = pd.read_csv(csv)
    df.columns = ["unique_id", "ds", "y"]
    df = df.groupby('unique_id').filter(lambda x: len(x) >= 80)
    #df['ds'] = pd.to_datetime(df['ds'], errors='ignore')
    df['ds'] = df.groupby('unique_id').cumcount() + 1
    return df

def get_dataset(name):
    Y_df, horizon, freq = pd.DataFrame(), 0, 0
    if name == 'M3-yearly':
        Y_df, *_ = M3.load("./data", "Yearly")
        horizon = 6
        freq = 'Y'
    elif name == 'M3-quarterly':
        Y_df, *_ = M3.load("./data", "Quarterly")
        horizon = 8
        freq = 'Q'
    elif name == 'M3-monthly':
        Y_df, *_ = M3.load("./data", "Monthly")
        horizon = 18
        freq = 'M'
    elif name == 'M4-yearly':
        Y_df, *_ = M4.load("./data", "Yearly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 6
        freq = 1
    elif name == 'M4-quarterly':
        Y_df, *_ = M4.load("./data", "Quarterly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 8
        freq = 1
    elif name == 'M4-monthly':
        Y_df, *_ = M4.load("./data", "Monthly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 18
        freq = 1
    elif name == 'M4-weekly':
        Y_df, *_ = M4.load("./data", "Weekly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 13
        freq = 1
    elif name == 'M4-daily':
        Y_df, *_ = M4.load("./data", "Daily")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 14
        freq = 1
    elif name == 'M4-hourly':
        Y_df, *_ = M4.load("./data", "Hourly")
        Y_df['ds'] = Y_df['ds'].astype(int)
        horizon = 48
        freq = 1

    return Y_df, horizon, freq


train_data_length = -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", type=str)

    args = parser.parse_args()
    dataset = args.dataset

    #Y_df, horizon, freq = get_dataset(dataset)
    Y_df = get_dataset_phmarcery_week()
    horizon, freq = 8, 1

    print(Y_df.head())
    if train_data_length > 0 :
        Y_df = Y_df[:train_data_length]
    test_df = Y_df.groupby('unique_id').tail(horizon)
    train_df = Y_df.drop(test_df.index).reset_index(drop=True)

    kan_loss_logger = LossLogger()
    kan_model = KAN(input_size=2 * horizon, h=horizon, scaler_type='robust', early_stop_patience_steps=3, callbacks=[kan_loss_logger])
    mlp_loss_logger = LossLogger()
    mlp_model = MLP(input_size=2 * horizon, h=horizon, scaler_type='robust', max_steps=1000,
                    early_stop_patience_steps=3, callbacks=[mlp_loss_logger])
    nbeats_loss_logger = LossLogger()
    nbeats_model = NBEATS(input_size=2 * horizon, h=horizon, scaler_type='robust', max_steps=1000,
                          early_stop_patience_steps=3, callbacks=[nbeats_loss_logger])

    nhits_loss_logger = LossLogger()
    nhits_model = NHITS(input_size=2 * horizon, h=horizon, scaler_type='robust', max_steps=1000,
                        early_stop_patience_steps=3, callbacks=[nhits_loss_logger])

    my_config = AutoRNN.get_default_config(h=horizon, backend='optuna')
    autornn_loss_logger = LossLogger()
    autoRNN = AutoRNN(h=horizon, config=my_config, backend='optuna', num_samples=1, cpus=1
                      )

    my_config_LSTM = AutoLSTM.get_default_config(h=horizon, backend='optuna')
    autoLstm_loss_logger = LossLogger()
    autoLSTM = AutoLSTM(h=horizon, config=my_config_LSTM, backend='optuna', num_samples=1, cpus=1
                        )

    my_config_AutoDilatedRNN = AutoDilatedRNN.get_default_config(h=12, backend='optuna')
    def my_config_new(trial):
        config = {**my_config_AutoDilatedRNN(trial)}
        config.update({'max_steps': 1, 'val_check_steps': 1, 'input_size': -1, 'encoder_hidden_size': 8})
        return config

    autoDilatedRNN_loss_logger = LossLogger()
    model_AutoDilatedRNN = AutoDilatedRNN(h=horizon, config=my_config_new, backend='optuna', num_samples=1, cpus=1
                                          )

    my_config_informer = AutoInformer.get_default_config(h=horizon, backend='optuna')
    autoInformer_loss_logger = LossLogger()
    AutoInformer = AutoInformer(h=horizon, config=my_config_informer, backend='optuna', num_samples=1, cpus=1
                                )

    my_config_FEDformer = AutoFEDformer.get_default_config(h=horizon, backend='optuna')
    autofedFormer_loss_logger = LossLogger()
    fedFormer = AutoFEDformer(h=horizon, config=my_config_FEDformer, backend='optuna', num_samples=1, cpus=1
                              )

    #MODELS = [kan_model, mlp_model, nbeats_model, nhits_model,autoRNN, AutoInformer, fedFormer]
    MODELS = [kan_model, mlp_model, nbeats_model, nhits_model,autoRNN, AutoInformer, fedFormer]
    #MODEL_NAMES = ['KAN', 'MLP', 'NBEATS', 'NHITS', 'AutoRNN', 'AutoDilatedRNN', 'AutoInformer', 'AutoFEDformer']
    MODEL_NAMES = ['KAN', 'MLP', 'NBEATS', 'NHITS', 'AutoRNN', 'AutoInformer', 'AutoFEDformer']
    model_loss = {'KAN':kan_loss_logger,'MLP':mlp_loss_logger,
                    'NBEATS':nbeats_loss_logger,
                    'NHITS':nhits_loss_logger,

                    'AutoDilatedRNN':autoDilatedRNN_loss_logger,
                    'AutoInformer':autoInformer_loss_logger,
                    'AutoFEDformer':autofedFormer_loss_logger,}
    #MODELS = [AutoInformer, fedFormer]
    #MODEL_NAMES = ['AutoInformer', 'AutoFEDformer']

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
            metrics=[mae, smape],
            models=[f"{MODEL_NAMES[i]}"],
            target_col="y",
        )

        evaluation = evaluation.drop(['unique_id'], axis=1).groupby('metric').mean().reset_index()

        model_mae = evaluation[f"{MODEL_NAMES[i]}"][0]
        model_smape = evaluation[f"{MODEL_NAMES[i]}"][1]

        #Y_hat_insample = nf.predict_insample(step_size=horizon)
        #plot_series(forecasts_df=Y_hat_insample.drop(columns='cutoff'))

        results.append([dataset, MODEL_NAMES[i], round(model_mae, 0), round(model_smape * 100, 2), elapsed_time])

    print(kan_loss_logger.train_loss)
    # all_model_loss = pd.DataFrame(columns=MODEL_NAMES)
    # for i in MODEL_NAMES:
    #     all_model_loss[i] = model_loss[i].train_loss

    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    results_df = pd.DataFrame(data=results, columns=['dataset', 'model', 'mae', 'smape', 'time'])
    os.makedirs('./results', exist_ok=True)
    
    results_df.to_csv(f'./results/{dataset}_results_KANtuned_{today}.csv', header=True, index=False)
    test_df.to_csv(f'./results/{dataset}_test_results_{today}.csv', header=True, index=False)
    #all_model_loss.to_csv(f'./results/{dataset}_loss_{today}.csv', header=True, index=False)

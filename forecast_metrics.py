import numpy as np


def calculate_metrics(y_true, y_pred):
    """
    计算MAE、RMSE、MAPE
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :return: MAE, RMSE, MAPE
    """
    # 移除NaN值（确保数据对齐）
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # 计算指标
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 百分比形式

    return mae, rmse, mape


# 示例数据
y_true = np.array([40.99,
45.18,
41.87,
30.36,
40.71,
35.51,
46.84,
34.01
])
y_pred = np.array([39.461426,
42.389872,
33.634032,
35.118341,
38.058649,
36.962149,
41.379091,
40.633316


])

mae, rmse, mape = calculate_metrics(y_true, y_pred)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
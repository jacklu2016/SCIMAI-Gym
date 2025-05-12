import numpy as np

# 将以下数字系列转为NumPy数组
data = np.array([
    27, 14, 6, 4, 4, 16,
    25, 40, 48, 53, 48, 40,
    27, 18, 7, 5, 7, 18,
    28, 38, 52, 55, 48, 40,
    29, 18, 8, 4, 8, 14
])

data1 = np.array([
    25, 13, 4, 3, 4, 18,
    26, 42, 50, 51, 49, 39,
    29, 17, 4, 2, 7, 17,
    29, 43, 47, 53, 49, 39,
    25, 15, 3, 1, 5, 13
])

mean = np.mean(data)                # 计算均值
variance_pop = np.std(data)         # 总体方差（默认ddof=0）
variance_sample = np.std(data, ddof=1)  # 样本方差（ddof=1）

mean1 = np.mean(data)                # 计算均值
variance_pop1 = np.var(data)         # 总体方差（默认ddof=0）
variance_sample1 = np.var(data, ddof=1)  # 样本方差（ddof=1）

print("均值:", mean)
print("总体方差:", variance_pop)
print("样本方差:", variance_sample)

print("均值1:", mean1)
print("总体方差1:", variance_pop1)
print("样本方差1:", variance_sample1)

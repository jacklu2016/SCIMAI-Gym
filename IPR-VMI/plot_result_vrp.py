import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 指定目录路径
file_path = r"D:\BaiduSyncdisk\VMI+Transportation\(版本9)实验结果.xlsx"

data = pd.read_excel(file_path, skiprows=9, nrows=4)

labels = data.iloc[:, 0]
values_20 = data.iloc[:, 1]
values_50 = data.iloc[:, 5]
values_80 = data.iloc[:, 9]
values_100 = data.iloc[:, 13]

x = np.arange(len(labels))  # x轴位置
# 颜色
colors = ['red', 'blue', 'purple', 'orange']
academic_palette = [
    '#F08080',  # 蓝色
    '#abd7e5',  # 橙色
    '#e5e7fd',  # 绿色
    #'#d62728',  # 红色
    '#ffd9b7',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f'   # 灰色
]
# 绘图
width = 0.2  # 柱状图宽度
fig, ax = plt.subplots()
min_text_space = 0.3
rects1 = ax.bar(x - 3*width/2, values_20, width, label='20家药店', color=academic_palette[0])
bar = rects1[0]
# 在柱子上方标注
ax.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + min_text_space,f"Min: {values_20[0]:.2f}",ha="center", va="bottom", fontsize=8, color="black")
rects2 = ax.bar(x - width/2, values_50, width, label='50家药店', color=academic_palette[1])
bar = rects2[0]
# 在柱子上方标注
ax.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + min_text_space,f"Min: {values_50[0]:.2f}",ha="center", va="bottom", fontsize=8, color="black")

rects3 = ax.bar(x + width/2, values_80, width, label='80家药店', color=academic_palette[2])
bar = rects3[0]
# 在柱子上方标注
ax.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + min_text_space,f"Min: {values_80[0]:.2f}",ha="center", va="bottom", fontsize=8, color="black")

rects4 = ax.bar(x + 3*width/2, values_100, width, label='100家药店', color=academic_palette[3])
bar = rects4[0]
# 在柱子上方标注
ax.text(bar.get_x() + bar.get_width() / 2,bar.get_height() + min_text_space,f"Min: {values_100[0]:.2f}",ha="center", va="bottom", fontsize=8, color="black")

# 添加标签、标题和图例
ax.set_ylabel('VRP.Dist')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 调整图例位置
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
# 显示柱状图
plt.tight_layout()
plt.savefig('vrp.svg',format='svg',bbox_inches='tight')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 切换到TkAgg后端

from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import numpy as np
import matplotlib.pyplot as plt

# 定义基本参数
D = 1000  # 年需求量 (单位：件)
S = 50  # 每次订货成本 (单位：元)
H = 2  # 单位商品持有成本 (单位：元)
safety_stock_supplier = 300  # 医药批发企业的安全库存
safety_stock_retailer = 100  # 药店的安全库存

# 计算EOQ（经济订货量）
EOQ = np.sqrt(2 * D * S / H)
print('EOQ:{}'.format(EOQ))
# 初始化时间步数
time_steps = 60  # 时间周期

# 初始化库存数据
supplier_initial_stock = 500  # 医药批发企业初始库存
retailer_initial_stock = 200  # 药店初始库存

supplier_stock = [supplier_initial_stock]
retailer_stock = [retailer_initial_stock]
retailer_stock1 = [retailer_initial_stock]

# 模拟库存变化
for t in range(1, time_steps):
    # 药店库存变化：需求消耗库存
    retailer_stock.append(retailer_stock[-1] - 20)  # 假设每周期需求量为20件
    retailer_stock1.append(retailer_stock1[-1] - 15)  # 假设每周期需求量为20件
    supplier_stock.append(supplier_stock[-1])  # 医药批发企业库存增加
    # 药店库存低于安全库存时，进行补货
    if retailer_stock[-1] <= safety_stock_retailer:
        # 医药批发企业的库存减少药店补货量

        supplier_stock[-1] -= EOQ  # 从医药批发企业库存扣除补货量

        retailer_stock[-1] += EOQ  # 药店库存增加补货量
    if retailer_stock1[-1] <= safety_stock_retailer:
        # 医药批发企业的库存减少药店补货量

        supplier_stock[-1] -= EOQ  # 从医药批发企业库存扣除补货量

        retailer_stock1[-1] += EOQ  # 药店库存增加补货量

    # 医药批发企业判断是否补货：如果医药批发企业库存低于安全库存，则补货
    if len(supplier_stock) >=2 and supplier_stock[len(supplier_stock) - 2] < safety_stock_supplier:
        #supplier_stock[-1] += EOQ  # 补货至EOQ
        #print('医药批发企业商补货:{}'.format(supplier_stock[-1]))
        supplier_stock[-1] += EOQ  # 医药批发企业库存增加


# 确保长度一致
if len(supplier_stock) != len(retailer_stock):
    # 如果长度不一致，补齐医药批发企业或药店的库存长度
    max_len = max(len(supplier_stock), len(retailer_stock))
    while len(supplier_stock) < max_len:
        supplier_stock.append(supplier_stock[-1])
    while len(retailer_stock) < max_len:
        retailer_stock.append(retailer_stock[-1])

print(retailer_stock)
print(retailer_stock1)
# 绘制图表
fig, ax = plt.subplots(3, 1, figsize=(8, 5), sharex=True, sharey=True)

# 医药批发企业库存变化图
ax[0].plot(np.arange(time_steps), supplier_stock, label='医药批发企业库存', color='#005BAA')
#ax[0].set_title('医药批发企业库存变化图 (EOQ补货)')
#ax[0].set_xlabel('时间周期')
ax[0].set_ylabel('医药批发企业库存')
ax[0].set_ylim(top=550)
ax[0].set_ylim(bottom=200)  # 设置第一个图的Y轴最小值为200
ax[0].grid(False)
ax[0].legend(loc='lower right')  # 将图例放到右边

# 药店库存变化图
ax[1].plot(np.arange(time_steps), retailer_stock, label='药店库存', color='#89CFF0')
#ax[1].set_title('药店库存变化图 (EOQ补货)')
#ax[1].set_xlabel('时间周期')
ax[1].set_ylabel('药店i库存')

ax[1].set_ylim(bottom=100)
ax[1].grid(False)
ax[1].legend()

# 药店1库存变化图
ax[2].plot(np.arange(time_steps), retailer_stock1, label='药店库存', color='#89CFF0')
#ax[2].set_title('药店库存变化图 (EOQ补货)')
ax[2].set_xlabel('时间')
ax[2].set_ylabel('药店j库存')

ax[2].set_ylim(bottom=100)
ax[2].grid(False)
ax[2].legend()

# 调整子图间距
plt.subplots_adjust(hspace=0.1)  # 调整图表之间的垂直间距

# 显示图表
plt.tight_layout()
plt.show()
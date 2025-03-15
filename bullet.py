import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 定义参数
num_weeks = 20  # 模拟20周
initial_demand = 10  # 客户初始需求
demand_variability = 0.2  # 需求波动率

# 初始化需求和订单量列表
customer_demand = np.zeros(num_weeks)
retailer_orders = np.zeros(num_weeks)
distributor_orders = np.zeros(num_weeks)
manufacturer_orders = np.zeros(num_weeks)
supplier_orders = np.zeros(num_weeks)

# 生成客户需求数据
for week in range(num_weeks):
    customer_demand[week] = initial_demand * (1 + np.random.uniform(-demand_variability, demand_variability))

# 模拟供应链中的各级订单量变化
for week in range(num_weeks):
    if week == 0:
        retailer_orders[week] = customer_demand[week]
    else:
        retailer_orders[week] = customer_demand[week] * (1 + np.random.uniform(0.05, 0.1))  # 零售商需求放大

    distributor_orders[week] = retailer_orders[week] * (1 + np.random.uniform(0.1, 0.2))  # 分销商需求放大
    manufacturer_orders[week] = distributor_orders[week] * (1 + np.random.uniform(0.1, 0.2))  # 制造商需求放大
    supplier_orders[week] = manufacturer_orders[week] * (1 + np.random.uniform(0.1, 0.3))  # 供应商需求放大

# 绘制需求和订单量的变化
plt.figure(figsize=(10, 6))
plt.plot(range(num_weeks), customer_demand, label='Customer Demand', color='blue')
plt.plot(range(num_weeks), retailer_orders, label='Retailer Orders', color='green', linestyle='--')
plt.plot(range(num_weeks), distributor_orders, label='Distributor Orders', color='orange', linestyle='--')
plt.plot(range(num_weeks), manufacturer_orders, label='Manufacturer Orders', color='red', linestyle='--')
plt.plot(range(num_weeks), supplier_orders, label='Supplier Orders', color='purple', linestyle='--')

plt.xlabel('Week')
plt.ylabel('Order Quantity')
plt.title('Bullwhip Effect Simulation')
plt.legend()
plt.grid(True)
plt.show()

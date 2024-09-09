import B4_重做第二题_函数
import B4_重做第三题_函数
from B4_重做第三题_函数 import B4_refrom_Q3
from B4_重做第二题_函数 import B4_reform_Q2

import numpy as np
import matplotlib.pyplot as plt

#重做Q2
# 定义 p2 的初始值、步长和取值范围
p2_initial = 0.1
p2_step = 0.01
p2_range = 0.1  # 取 p2±0.1 范围

# 生成 p2 的取值范围，从 (p2_initial - p2_range) 到 (p2_initial + p2_range)，步长为 0.01
p2_values = np.arange(p2_initial - p2_range, p2_initial + p2_range + p2_step, p2_step)

# 用于存储 p2 值、策略编号、情况编号和最大利润
results = []

# 遍历所有 p2 值，调用 B4_reform_Q2(p2)
for p2 in p2_values:
    strategy_idx2, condition_idx, max_profit_2 = B4_reform_Q2(p2)
    results.append((p2, strategy_idx2, condition_idx, max_profit_2))

# 提取 p2 和对应的最大利润
p2_vals = [result[0] for result in results]
profits = [result[3] for result in results]
strategies = [result[1] for result in results]
conditions = [result[2] for result in results]

# 生成图像，展示 p2 值与最大利润的变化关系
plt.figure(figsize=(10, 6))
plt.plot(p2_vals, profits, marker='o', label='Max Profit')

# 图像美化
plt.title("最大利润随 p2 变化的趋势")
plt.xlabel("p2 值")
plt.ylabel("最大利润")
plt.grid(True)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 输出不同 p2 下的策略编号、情况编号和最大利润
for p2, strategy, condition, profit in results:
    print(f"p2: {p2:.2f}, 策略编号: {strategy}, 情况编号: {condition}, 最大利润: {profit:.2f}")


#重做Q3

# 定义 p3 的初始值、步长和取值范围
p3_initial = 0.1
p3_step = 0.01
p3_range = 0.1  # 取 p3±0.1 的范围

# 生成 p3 的取值范围，从 (p3_initial - p3_range) 到 (p3_initial + p3_range)，步长为 0.01
p3_values = np.arange(p3_initial - p3_range, p3_initial + p3_range + p3_step, p3_step)

# 用于存储 p3 值、策略编号和最大利润
results = []

# 遍历所有 p3 值，调用 B4_refrom_Q3(p3)
for p3 in p3_values:
    strategy_idx3, max_profit_3 = B4_refrom_Q3(p3)
    results.append((p3, strategy_idx3, max_profit_3))

# 提取 p3 和对应的最大利润
p3_vals = [result[0] for result in results]
profits = [result[2] for result in results]
strategies = [result[1] for result in results]

# 生成图像，展示 p3 值与最大利润的变化关系
plt.figure(figsize=(10, 6))
plt.plot(p3_vals, profits, marker='o', label='Max Profit')

# 图像美化
plt.title("最大利润随 p3 变化的趋势")
plt.xlabel("p3 值")
plt.ylabel("最大利润")
plt.grid(True)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 输出不同 p3 下的策略编号和最大利润
for p3, strategy, profit in results:
    print(f"p3: {p3:.2f}, 策略编号: {strategy}, 最大利润: {profit:.2f}")


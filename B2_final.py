import random

import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

plt.rcParams['axes.unicode_minus'] = False  # 禁用 Unicode 负号，使用 ASCII '-'

# 参数设定
np.random.seed(42)  # 设置随机数种子以保证可重复性
simulations = 1000  # 蒙特卡洛模拟次数

# 零配件参数
part1_defect_rate = 0.1  # 零配件1的次品率
part2_defect_rate = 0.1  # 零配件2的次品率
part1_cost = 4  # 零配件1的购买单价
part2_cost = 18  # 零配件2的购买单价
part1_detection_cost = 2  # 零配件1的检测成本
part2_detection_cost = 3  # 零配件2的检测成本

# 成品参数
product_defect_rate = 0.1  # 成品次品率
product_cost = 6  # 成品的装配成本
product_detection_cost = 3  # 成品检测成本
product_market_price = 56  # 成品市场售价
return_loss = 6  # 调换损失
disassembly_cost = 5  # 拆解费用

# 定义每套参数的数据，使用字典来存储每一组参数
data_sets = [
    {
        "part1_defect_rate": 0.1,
        "part2_defect_rate": 0.1,
        "part1_cost": 4,
        "part2_cost": 18,
        "part1_detection_cost": 2,
        "part2_detection_cost": 3,
        "product_defect_rate": 0.1,
        "product_cost": 6,
        "product_detection_cost": 3,
        "product_market_price": 56,
        "return_loss": 6,
        "disassembly_cost": 5
    },
    {
        "part1_defect_rate": 0.2,
        "part2_defect_rate": 0.2,
        "part1_cost": 4,
        "part2_cost": 18,
        "part1_detection_cost": 2,
        "part2_detection_cost": 3,
        "product_defect_rate": 0.2,
        "product_cost": 6,
        "product_detection_cost": 3,
        "product_market_price": 56,
        "return_loss": 6,
        "disassembly_cost": 5
    },
    {
        "part1_defect_rate": 0.1,
        "part2_defect_rate": 0.1,
        "part1_cost": 4,
        "part2_cost": 18,
        "part1_detection_cost": 2,
        "part2_detection_cost": 3,
        "product_defect_rate": 0.1,
        "product_cost": 6,
        "product_detection_cost": 3,
        "product_market_price": 56,
        "return_loss": 30,
        "disassembly_cost": 5
    },
    {
        "part1_defect_rate": 0.2,
        "part2_defect_rate": 0.2,
        "part1_cost": 4,
        "part2_cost": 18,
        "part1_detection_cost": 1,
        "part2_detection_cost": 1,
        "product_defect_rate": 0.2,
        "product_cost": 6,
        "product_detection_cost": 2,
        "product_market_price": 56,
        "return_loss": 30,
        "disassembly_cost": 5
    },
    {
        "part1_defect_rate": 0.1,
        "part2_defect_rate": 0.2,
        "part1_cost": 4,
        "part2_cost": 18,
        "part1_detection_cost": 8,
        "part2_detection_cost": 1,
        "product_defect_rate": 0.1,
        "product_cost": 6,
        "product_detection_cost": 2,
        "product_market_price": 56,
        "return_loss": 10,
        "disassembly_cost": 5
    },
    {
        "part1_defect_rate": 0.05,
        "part2_defect_rate": 0.05,
        "part1_cost": 4,
        "part2_cost": 18,
        "part1_detection_cost": 2,
        "part2_detection_cost": 3,
        "product_defect_rate": 0.05,
        "product_cost": 6,
        "product_detection_cost": 3,
        "product_market_price": 56,
        "return_loss": 10,
        "disassembly_cost": 40
    },
]

# 决策模拟函数
def simulate_decision(detect_part1,detect_part2, detect_product, disassemble):#配件是否检测，成品是否检测，不合格产品是否拆解
    total_cost = 0 #花费
    total_revenue = 0 #收入
    total_return_loss = 0 #返还损失

    #可用零件1和2
    part1_available=0
    part2_available=0
    product_available=0
    #不可用零件1和2
    part1_broken=0
    part2_broken=0
    product_broken=0

    #不考虑拆解零件
    for _ in range(simulations):
        #随机抽取零部件1和2
        part1_defective = np.random.rand() < part1_defect_rate
        part2_defective = np.random.rand() < part2_defect_rate
        #零件1和2是否被丢弃
        part1_exist=True
        part2_exist=True
        # 检测零配件1
        total_cost += part1_cost
        if detect_part1:
            total_cost+=part1_detection_cost
            if part1_defective:
                 # 丢弃次品零配件1
                part1_exist=False
            else:
                part1_available+=1 # 合格零配件1
        else:  # 未检测直接使用零配件1
            if part1_defective:
                part1_broken+=1
            else:
                part1_available+=1

        # 检测零配件2
        total_cost += part2_cost
        if detect_part2:
            if part2_defective:
                part2_exist=False  # 丢弃次品零配件2
            else:
                part2_available+=1 # 合格零配件2
        else: # 未检测直接使用零配件2
            if part2_defective:
                part2_broken+=1
            else:
                part2_available+=1



        #判断成品是否存在
        if part1_exist and part2_exist:#零件1和2都没有被丢弃
            #判断产品好坏
            product_defective= part1_defective or part2_defective or np.random.rand() < product_defect_rate
            #使用零件1
            if  part1_defective:
                part1_broken-=1
            else:
                part1_available-=1
            #使用零件2
            if  part2_defective:
                part2_broken-=1
            else:
                part2_available-=1
            #组装产品
            total_cost += product_cost
            if  product_defective:
                product_broken+=1
            else:
                product_available+=1

            # 检测成品

            if detect_product:#检测产品
                total_cost += product_detection_cost
                if product_defective:
                    if disassemble:
                        total_cost += disassembly_cost
                        # 返还零件1
                        if part1_defective:
                            part1_broken += 1
                        else:
                            part1_available += 1
                        # 返还零件2
                        if part2_defective:
                            part2_broken += 1
                        else:
                            part2_available += 1

                        product_broken -= 1
                    else:
                        product_broken -= 1
                else:
                    total_revenue += product_market_price
                    product_available -= 1
            else:#不检测产品
                if product_defective:#坏产品
                    total_return_loss+=return_loss
                    if disassemble:#拆解
                        total_cost += disassembly_cost
                        # 返还零件1
                        if part1_defective:
                            part1_broken += 1
                        else:
                            part1_available += 1
                        # 返还零件2
                        if part2_defective:
                            part2_broken += 1
                        else:
                            part2_available += 1

                        product_broken -= 1
                    else:#不拆解
                        product_broken -= 1
                else:#好产品
                    total_revenue += product_market_price
                    product_available -= 1



    #考虑拆解出的剩余零件
    while part1_available+part1_broken>0 and part2_available+part2_broken>0:
        #剩余零件
        part1_residue=[]
        for _ in range(part1_available):
            part1_residue.append(False)
        for _ in range(part1_broken):
            part1_residue.append(True)

        part2_residue=[]
        for _ in range(part2_available):
            part2_residue.append(False)
        for _ in range(part2_broken):
            part2_residue.append(True)

        #print(f"part1_residue:{part1_residue} part2_residue:{part2_residue}")
        #零件是否被丢弃
        part1_exist=True
        part2_exist=True
        #选取零件1
        s1=random.choice(part1_residue)
        part1_residue.remove(s1)
        if detect_part1:
            #total_cost+=part1_detection_cost
            if s1:
                part1_broken-=1
                if part1_available+part1_broken==0:
                    break
                part1_exist=False
            else:
                pass
        else:
            pass
        #选取零件2
        s2=random.choice(part2_residue)
        part2_residue.remove(s2)
        if detect_part2:
            #total_cost+=part2_detection_cost
            if s2:
                part2_broken-=1
                if part2_available+part2_broken==0:
                    break
                part2_exist=False
            else:
                pass
        else:
            pass

        #特殊情况
        if not detect_part1 and s1 and disassemble:
            part1_broken-=1
            continue
        if not detect_part2 and s2 and disassemble:
            part2_broken-=1
            continue


        #组装产品
        if part1_exist and part2_exist:
            s3=s1 or s2 or np.random.rand() < product_defect_rate

            if s1:
                part1_broken-=1
            else:
                part1_available-=1

            if s2:
                part2_broken-=1
            else:
                part2_available-=1

            total_cost+=product_cost
            if s3:
                product_broken+=1
            else:
                product_available+=1

            #检测成品
            if detect_product:
                total_cost+=product_detection_cost
                if s3:
                    if disassemble:
                        total_cost+=disassembly_cost


                        if s1:
                            part1_broken+=1
                        else:
                            part1_available+=1


                        if s2:
                            part2_broken+=1
                        else:
                            part2_available+=1

                        product_broken -= 1
                    else:
                       product_broken-=1
                else:
                    total_revenue+=product_market_price
                    product_available -= 1
            else:
                if s3:
                    total_return_loss+=return_loss
                    if disassemble:
                        total_cost+=disassembly_cost


                        if s1:
                            part1_broken+=1
                        else:
                            part1_available-=1


                        if s2:
                            part2_broken+=1
                        else:
                            part2_available-=1
                        product_broken -= 1
                    else:
                        product_broken-=1
                else:
                    total_revenue+=product_market_price
                    product_available -= 1




    # 计算总收益、成本和损失
    net_profit = total_revenue - total_cost - total_return_loss
    return net_profit

def simulate_decision_change_analog(simulate,detect_part1,detect_part2, detect_product, disassemble):#配件是否检测，成品是否检测，不合格产品是否拆解

    total_cost = 0 #花费
    total_revenue = 0 #收入
    total_return_loss = 0 #返还损失

    #可用零件1和2
    part1_available=0
    part2_available=0
    product_available=0
    #不可用零件1和2
    part1_broken=0
    part2_broken=0
    product_broken=0

    #不考虑拆解零件
    for _ in range(simulate):
        #随机抽取零部件1和2
        part1_defective = np.random.rand() < part1_defect_rate
        part2_defective = np.random.rand() < part2_defect_rate
        #零件1和2是否被丢弃
        part1_exist=True
        part2_exist=True
        # 检测零配件1
        total_cost += part1_cost
        if detect_part1:
            total_cost+=part1_detection_cost
            if part1_defective:
                 # 丢弃次品零配件1
                part1_exist=False
            else:
                part1_available+=1 # 合格零配件1
        else:  # 未检测直接使用零配件1
            if part1_defective:
                part1_broken+=1
            else:
                part1_available+=1

        # 检测零配件2
        total_cost += part2_cost
        if detect_part2:
            if part2_defective:
                part2_exist=False  # 丢弃次品零配件2
            else:
                part2_available+=1 # 合格零配件2
        else: # 未检测直接使用零配件2
            if part2_defective:
                part2_broken+=1
            else:
                part2_available+=1



        #判断成品是否存在
        if part1_exist and part2_exist:#零件1和2都没有被丢弃
            #判断产品好坏
            product_defective= part1_defective or part2_defective or np.random.rand() < product_defect_rate
            #使用零件1
            if  part1_defective:
                part1_broken-=1
            else:
                part1_available-=1
            #使用零件2
            if  part2_defective:
                part2_broken-=1
            else:
                part2_available-=1
            #组装产品
            total_cost += product_cost
            if  product_defective:
                product_broken+=1
            else:
                product_available+=1

            # 检测成品

            if detect_product:#检测产品
                total_cost += product_detection_cost
                if product_defective:
                    if disassemble:
                        total_cost += disassembly_cost
                        # 返还零件1
                        if part1_defective:
                            part1_broken += 1
                        else:
                            part1_available += 1
                        # 返还零件2
                        if part2_defective:
                            part2_broken += 1
                        else:
                            part2_available += 1

                        product_broken -= 1
                    else:
                        product_broken -= 1
                else:
                    total_revenue += product_market_price
                    product_available -= 1
            else:#不检测产品
                if product_defective:#坏产品
                    total_return_loss+=return_loss
                    if disassemble:#拆解
                        total_cost += disassembly_cost
                        # 返还零件1
                        if part1_defective:
                            part1_broken += 1
                        else:
                            part1_available += 1
                        # 返还零件2
                        if part2_defective:
                            part2_broken += 1
                        else:
                            part2_available += 1

                        product_broken -= 1
                    else:#不拆解
                        product_broken -= 1
                else:#好产品
                    total_revenue += product_market_price
                    product_available -= 1



    #考虑拆解出的剩余零件
    while part1_available+part1_broken>0 and part2_available+part2_broken>0:
        #剩余零件
        part1_residue=[]
        for _ in range(part1_available):
            part1_residue.append(False)
        for _ in range(part1_broken):
            part1_residue.append(True)

        part2_residue=[]
        for _ in range(part2_available):
            part2_residue.append(False)
        for _ in range(part2_broken):
            part2_residue.append(True)

        #print(f"part1_residue:{part1_residue} part2_residue:{part2_residue}")
        #零件是否被丢弃
        part1_exist=True
        part2_exist=True
        #选取零件1
        s1=random.choice(part1_residue)
        part1_residue.remove(s1)
        if detect_part1:
            #total_cost+=part1_detection_cost
            if s1:
                part1_broken-=1
                if part1_available+part1_broken==0:
                    break
                part1_exist=False
            else:
                pass
        else:
            pass
        #选取零件2
        s2=random.choice(part2_residue)
        part2_residue.remove(s2)
        if detect_part2:
            #total_cost+=part2_detection_cost
            if s2:
                part2_broken-=1
                if part2_available+part2_broken==0:
                    break
                part2_exist=False
            else:
                pass
        else:
            pass

        #特殊情况
        if not detect_part1 and s1 and disassemble:
            part1_broken-=1
            continue
        if not detect_part2 and s2 and disassemble:
            part2_broken-=1
            continue


        #组装产品
        if part1_exist and part2_exist:
            s3=s1 or s2 or np.random.rand() < product_defect_rate

            if s1:
                part1_broken-=1
            else:
                part1_available-=1

            if s2:
                part2_broken-=1
            else:
                part2_available-=1

            total_cost+=product_cost
            if s3:
                product_broken+=1
            else:
                product_available+=1

            #检测成品
            if detect_product:
                total_cost+=product_detection_cost
                if s3:
                    if disassemble:
                        total_cost+=disassembly_cost


                        if s1:
                            part1_broken+=1
                        else:
                            part1_available+=1


                        if s2:
                            part2_broken+=1
                        else:
                            part2_available+=1

                        product_broken -= 1
                    else:
                       product_broken-=1
                else:
                    total_revenue+=product_market_price
                    product_available -= 1
            else:
                if s3:
                    total_return_loss+=return_loss
                    if disassemble:
                        total_cost+=disassembly_cost


                        if s1:
                            part1_broken+=1
                        else:
                            part1_available-=1


                        if s2:
                            part2_broken+=1
                        else:
                            part2_available-=1
                        product_broken -= 1
                    else:
                        product_broken-=1
                else:
                    total_revenue+=product_market_price
                    product_available -= 1




    # 计算总收益、成本和损失
    net_profit = total_revenue - total_cost - total_return_loss
    return net_profit


# 蒙特卡洛模拟不同策略下的收益
# 自定义策略名称的列表
strategy_names = [" "] * 16

# 初始化策略列表
strategies = [[strategy_names[i], False, False, False, False] for i in range(16)]

# 用位运算简化组合生成
for m in range(16):
    # 使用位运算符判断布尔值
    detect_part1 = bool(m & 1)
    detect_part2 = bool(m & 2)
    detect_product = bool(m & 4)
    dismantle = bool(m & 8)

    # 构建策略名称
    strategies[m][0] = f"{'检测' if detect_part1 else '不检测'}配件1 " \
                       f"{'检测' if detect_part2 else '不检测'}配件2 " \
                       f"{'检测' if detect_product else '不检测'}成品 " \
                       f"{'拆解' if dismantle else '不拆解'}"

    # 设置布尔值
    strategies[m][1:] = [detect_part1, detect_part2, detect_product, dismantle]

#画出折线图
def B2_draw():
    # 初始化存储不同数据集的净收益
    all_profits = []

    #1.0

    #遍历所有情况
    for idx, data in enumerate(data_sets):
        profits = []  # 存储当前数据集下各策略的收益
        print(f"处理第 {idx + 1} 套数据:")
        # 提取当前数据集的参数
        part1_defect_rate = data["part1_defect_rate"]
        part2_defect_rate = data["part2_defect_rate"]
        part1_cost = data["part1_cost"]
        part2_cost = data["part2_cost"]
        part1_detection_cost = data["part1_detection_cost"]
        part2_detection_cost = data["part2_detection_cost"]
        product_defect_rate = data["product_defect_rate"]
        product_cost = data["product_cost"]
        product_detection_cost = data["product_detection_cost"]
        product_market_price = data["product_market_price"]
        return_loss = data["return_loss"]
        disassembly_cost = data["disassembly_cost"]

        #遍历所有策略
        for strategy in strategies:
            name, detect_part1,detect_part2, detect_product, disassemble = strategy
            profit = simulate_decision(detect_part1,detect_part2, detect_product, disassemble)
            profits.append(profit)
            print(f"{name} 的模拟净收益为: {profit}")

        # 保存当前数据集下所有策略的收益
        all_profits.append(profits)

    #画图
    plt.figure(figsize=(10, 6))

    # 用于存储最大收益信息
    max_profits_info = []

    # 每个数据集的收益对应一条折线
    for idx, profits in enumerate(all_profits):
        # 绘制折线
        plt.plot(range(len(strategies)), profits, label=f"数据集 {idx + 1}")

        # 找到最大净收益及其对应的策略编号
        max_profit = max(profits)
        max_profit_idx = profits.index(max_profit)
        max_profits_info.append((idx + 1, max_profit_idx, max_profit))

        # 在图中标示最大值点
        plt.annotate(f" ",
                     xy=(max_profit_idx, max_profit),
                     xytext=(max_profit_idx, max_profit + 1),  # 提高文本位置
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     ha='center')

    # 图形美化
    plt.xlabel("策略编号")
    plt.ylabel("净收益")
    plt.title("各数据集下16种策略的净收益折线图")

    # 设置 x 轴的刻度标签为策略编号（0-15），并确保它们显示出来
    plt.xticks(range(len(strategies)), labels=[f"策略 {i}" for i in range(16)], rotation=45)

    # 自动调整 y 轴的范围，以包括负值
    min_profit = min(min(profits) for profits in all_profits)
    max_profit = max(max(profits) for profits in all_profits)
    plt.ylim(min_profit, max_profit + 5)  # 为图例标注提供额外空间

    # 显示图例，添加网格
    plt.legend()
    plt.grid(True)

    # 调整布局，防止标签重叠
    plt.tight_layout()

    # 展示图表
    plt.show()

    # 输出每个数据集的最大净收益策略和净收益
    for dataset_idx, strategy_idx, profit in max_profits_info:
        print(f"数据集 {dataset_idx} 的最大净收益策略是策略 {strategy_idx}，净收益为 {profit:.2f}")

#输出最大收益和最大策略
# 初始化存储不同数据集的净收益
#1.0

def B2_solve():
    all_profits = []

    #1.0 遍历所有情况
    for idx, data in enumerate(data_sets):
        profits = []  # 存储当前数据集下各策略的收益
        dataset_name = data.get("name", f"数据集 {idx + 1}")  # 如果有数据集名称，使用它
        print(f"处理 {dataset_name}:")

        # 提取当前数据集的参数并保存
        dataset_parameters = {
            "part1_defect_rate": data["part1_defect_rate"],
            "part2_defect_rate": data["part2_defect_rate"],
            "part1_cost": data["part1_cost"],
            "part2_cost": data["part2_cost"],
            "part1_detection_cost": data["part1_detection_cost"],
            "part2_detection_cost": data["part2_detection_cost"],
            "product_defect_rate": data["product_defect_rate"],
            "product_cost": data["product_cost"],
            "product_detection_cost": data["product_detection_cost"],
            "product_market_price": data["product_market_price"],
            "return_loss": data["return_loss"],
            "disassembly_cost": data["disassembly_cost"],
        }

        # 遍历所有策略
        for strategy in strategies:
            name, detect_part1, detect_part2, detect_product, disassemble = strategy
            profit = simulate_decision(detect_part1, detect_part2, detect_product, disassemble)
            profits.append(profit)
            print(f"{name} 的模拟净收益为: {profit}")

        # 保存当前数据集下所有策略的收益和对应的参数
        all_profits.append((dataset_name, dataset_parameters, profits))

    #2.0 寻找所有数据集中最大收益的情况
    max_profit_overall = float('-inf')  # 初始化为负无穷大
    best_dataset_name = None  # 用于记录最大收益对应的数据集名称
    best_strategy_name = None  # 用于记录最大收益对应的策略名称
    best_parameters = None  # 用于记录最大收益对应的数据集参数

    # 遍历所有数据集的收益
    for dataset_idx, (dataset_name, dataset_parameters, profits) in enumerate(all_profits):
        for strategy_idx, profit in enumerate(profits):
            if profit > max_profit_overall:
                max_profit_overall = profit
                best_dataset_name = dataset_name
                best_strategy_name = strategies[strategy_idx][0]  # 策略名称
                best_parameters = dataset_parameters  # 保存该数据集的参数

    # 输出结果
    print(f"在所有数据集中，策略 '{best_strategy_name}' 在数据集 '{best_dataset_name}' 中获得了最大净收益 {max_profit_overall:.2f}。")
    print(f"该数据集的参数为：")

    # 输出数据集的具体参数和对应的值
    for param, value in best_parameters.items():
        print(f"{param}: {value}")

#2.0
def B2_solve_average_draw_(simulate):
    all_profits = []

    # 1.0 遍历所有情况
    for idx, data in enumerate(data_sets):
        profits = []  # 存储当前数据集下各策略的收益
        dataset_name = data.get("name", f"数据集 {idx + 1}")  # 如果有数据集名称，使用它
        print(f"处理 {dataset_name}:")

        # 提取当前数据集的参数并保存
        dataset_parameters = {
            "part1_defect_rate": data["part1_defect_rate"],
            "part2_defect_rate": data["part2_defect_rate"],
            "part1_cost": data["part1_cost"],
            "part2_cost": data["part2_cost"],
            "part1_detection_cost": data["part1_detection_cost"],
            "part2_detection_cost": data["part2_detection_cost"],
            "product_defect_rate": data["product_defect_rate"],
            "product_cost": data["product_cost"],
            "product_detection_cost": data["product_detection_cost"],
            "product_market_price": data["product_market_price"],
            "return_loss": data["return_loss"],
            "disassembly_cost": data["disassembly_cost"],
        }

        # 遍历所有策略
        for strategy in strategies:
            name, detect_part1, detect_part2, detect_product, disassemble = strategy
            profit = simulate_decision_change_analog(simulate,detect_part1, detect_part2, detect_product, disassemble)
            profits.append(profit)
            print(f"{name} 的模拟净收益为: {profit}")

        # 保存当前数据集下所有策略的收益和对应的参数
        all_profits.append((dataset_name, dataset_parameters, profits))

    # 2.0 寻找所有数据集中最大收益的情况并计算平均最大利润
    max_profit_overall = float('-inf')  # 初始化为负无穷大
    best_dataset_name = None  # 用于记录最大收益对应的数据集名称
    best_strategy_name = None  # 用于记录最大收益对应的策略名称
    best_parameters = None  # 用于记录最大收益对应的数据集参数
    total_max_profit = 0  # 用于记录所有数据集的最大利润之和
    max_profits = []  # 用于存储每个数据集的最大利润

    # 遍历所有数据集的收益
    for dataset_idx, (dataset_name, dataset_parameters, profits) in enumerate(all_profits):
        max_profit_in_dataset = max(profits)  # 找到当前数据集中的最大利润
        max_profits.append(max_profit_in_dataset)
        total_max_profit += max_profit_in_dataset  # 累加最大利润

        # 检查是否为全局最大利润
        if max_profit_in_dataset > max_profit_overall:
            max_profit_overall = max_profit_in_dataset
            best_dataset_name = dataset_name
            best_strategy_name = strategies[profits.index(max_profit_in_dataset)][0]  # 策略名称
            best_parameters = dataset_parameters  # 保存该数据集的参数

    # 计算平均最大利润
    average_max_profit = total_max_profit / len(all_profits)

    # 输出全局最大利润的结果
    print(
        f"在所有数据集中，策略 '{best_strategy_name}' 在数据集 '{best_dataset_name}' 中获得了最大净收益 {max_profit_overall:.2f}。")
    print(f"该数据集的参数为：")
    for param, value in best_parameters.items():
        print(f"{param}: {value}")

    # 输出所有数据集的平均最大利润
    print(f"所有数据集中最大利润的平均值为: {average_max_profit:.2f}")
    return average_max_profit

#测试
'''
simulate=1000
max_2_profit=B2_solve_average_draw_(simulate)
print(max_2_profit)
'''

#随simulate变化平均最大收益的变化
'''
# 设置 simulate 的取值范围和步长
simulate_values = list(range(10, 1001, 10))  # 从10到1000，步长为10
max_profits = []  # 用于存储每个 simulate 对应的最大收益

# 遍历不同 simulate 的取值
for simulate in simulate_values:
    max_2_profit = B2_solve_average_draw_(simulate)  # 调用函数计算最大收益
    max_profits.append(max_2_profit)  # 将结果存储
    print(f"simulate={simulate}, max_2_profit={max_2_profit}")  # 打印每次结果以供参考

# 绘制最大收益随 simulate 变化的图像
plt.figure(figsize=(10, 6))
plt.plot(simulate_values, max_profits, marker='o', color='b')  # 画出曲线，使用圆点标记
plt.xlabel('Simulate 值')  # x轴标签
plt.ylabel('最大收益')  # y轴标签
plt.title('最大收益随 simulate 变化的关系图')  # 图像标题
plt.grid(True)  # 添加网格
plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图像
'''


import numpy as np
import itertools
import random


import B3_第一步
import B3_第二步
import B3_第三步

# 参数设定
np.random.seed(42)  # 设置随机数种子以保证可重复性
simulations = 10  # 蒙特卡洛模拟次数


# 合成半成品1
def first(x1, x2, x3, h1, dismantle):
    half_product1, all_profit1 = B3_第一步.step1(x1, x2, x3, h1, dismantle)
    return half_product1, all_profit1

# 合成半成品2
def second(x4, x5, x6, h2, dismantle):
    half_product2, all_profit2 = B3_第二步.step2(x4, x5, x6, h2, dismantle)
    return half_product2, all_profit2

# 合成半成品3
def third(x7, x8, h3, dismantle):
    half_product3, all_profit3 = B3_第三步.step3(x7, x8, h3, dismantle)
    return half_product3, all_profit3


#product_defect_rate = 0.1  # 成品次品率
product_cost = 8  # 成品装配成本
product_detect_cost = 6  # 成品检测成本
product_disassembly_cost = 6  # 成品拆解费用
product_price = 200  # 产品市场售价
product_return_loss = 40  # 调换损失

part1_cost = 2
part2_cost = 8
part3_cost = 12
part4_cost = 2
part5_cost = 8
part6_cost = 12
part7_cost = 8
part8_cost = 12


# 决策模拟
# h:是否检测
# disamant:是否拆解
def simulation_product(p,x1, x2, x3, x4, x5, x6, x7, x8, h1, h2, h3, h4, dismantle1, dismantle2, dismantle3, dismantle4):
    product_defect_rate=p
    # 收支
    total_cost = 0
    total_revenue = 0

    # 复用零件库
    half1_available = 0
    half2_available = 0
    half3_available = 0
    half1_broken = 0
    half2_broken = 0
    half3_broken = 0

    # 只拆解，不复用
    for _ in range(simulations):

        # 第一步，选取1，2，3合成半成品1
        half_product1, all_profit1 = first(x1, x2, x3, h1, dismantle1)
        total_revenue += all_profit1

        # 第二步，取4，5，6合成半成品2
        half_product2, all_profit2 = second(x4, x5, x6, h2, dismantle2)
        total_revenue += all_profit2

        # 第三步，取7,8合成半成品3
        half_product3, all_profit3 = third(x7, x8, h3, dismantle3)
        total_revenue += all_profit3

        # 合成成品
        total_cost += product_cost
        product = np.random.rand() > product_defect_rate
        if (half_product1 == 1 and half_product2 == 1 and half_product3 == 1) and product == 1:  # 产品合格
            if h4 == 1:  # 如果检测成品
                total_cost += product_detect_cost
                total_revenue += product_price
            else:
                total_revenue += product_price
        else:  # 产品不合格
            if h4 == 1:  # 如果检测成品
                total_cost += product_detect_cost
                if dismantle4 == 1:  # 如果拆解成品
                    total_cost += product_disassembly_cost
                    # 判断半成品1
                    if half_product1 == 1:  # 如果半成品1是好的
                        half1_available += 1
                    elif half_product1 == 0:  # 如果半成品1是坏的
                        half1_broken += 1
                        if h1 == 1:  # 如果检测
                            half1_broken -= 1  # 直接丢弃
                        else:  # 如果不检测
                            pass  # 计入坏半成品库

                    # 半成品2
                    if half_product2 == 1:  # 如果半成品是好的
                        half2_available += 1
                    elif half_product2 == 0:  # 如果半成品是坏的
                        half2_broken += 1
                        if h2 == 1:  # 如果检测
                            half2_broken += 1
                        else:  # 如果不检测
                            pass
                    # 半成品3
                    if half_product3 == 1:  # 如果半成品是好的
                        half3_available += 1
                    elif half_product3 == 0:  # 如果半成品是坏的
                        half3_broken += 1
                        if h3 == 1:  # 如果检测
                            half3_broken -= 1  # 直接丢弃
                        else:  # 如果不检测
                            pass
                else:
                    pass  # 直接丢弃

            else:  # 不检测
                total_cost += product_return_loss  # 调换损失
                if dismantle4 == 1:  # 如果拆解成品
                    total_cost += product_disassembly_cost
                    # 判断半成品1
                    if half_product1 == 1:  # 如果半成品1是好的
                        half1_available += 1
                    elif half_product1 == 0:  # 如果半成品1是坏的
                        half1_broken += 1
                        if h1 == 1:  # 如果检测
                            half1_broken -= 1  # 直接丢弃
                        else:  # 如果不检测
                            pass  # 计入坏半成品库

                    # 半成品2
                    if half_product2 == 1:  # 如果半成品是好的
                        half2_available += 1
                    elif half_product2 == 0:  # 如果半成品是坏的
                        half2_broken += 1
                        if h2 == 1:  # 如果检测
                            half2_broken += 1
                        else:  # 如果不检测
                            pass
                    # 半成品3
                    if half_product3 == 1:  # 如果半成品是好的
                        half3_available += 1
                    elif half_product3 == 0:  # 如果半成品是坏的
                        half3_broken += 1
                        if h3 == 1:  # 如果检测
                            half3_broken -= 1  # 直接丢弃
                        else:  # 如果不检测
                            pass
                else:
                    pass  # 直接丢弃


    # 处理拆解后的半成品
    while half1_available + half1_broken > 0 and half2_available + half2_broken > 0 and half3_available + half3_broken > 0:

        # 剩余半成品1
        half1_residue = [1] * half1_available + [0] * half1_broken
        half2_residue = [1] * half2_available + [0] * half2_broken
        half3_residue = [1] * half3_available + [0] * half3_broken

        # 处理半成品1
        if half1_residue:
            s1 = random.choice(half1_residue)
            half1_residue.remove(s1)
            if s1 == 1:  # 合格
                half1_available -= 1
            else:  # 不合格，确保 broken 不为负数
                if half1_broken > 0:
                    half1_broken -= 1

        # 处理半成品2
        if half2_residue:
            s2 = random.choice(half2_residue)
            half2_residue.remove(s2)
            if s2 == 1:  # 合格
                half2_available -= 1
            else:  # 不合格，确保 broken 不为负数
                if half2_broken > 0:
                    half2_broken -= 1
                else:
                    print("Error: No broken items available to decrement!")

        # 处理半成品3
        if half3_residue:
            s3 = random.choice(half3_residue)
            half3_residue.remove(s3)
            if s3 == 1:  # 合格
                half3_available -= 1
            else:  # 不合格，确保 broken 不为负数
                if half3_broken > 0:
                    half3_broken -= 1


        # 特殊情况
        if h1 == 0 and s1 == 0 and dismantle4 == 1:
            half1_broken -= 1
            if dismantle1 == 1:
                if x1 == 1:
                    total_cost -= part1_cost
                if x2 == 1:
                    total_cost -= part2_cost
                if x3 == 1:
                    total_cost -= part3_cost
            else:
                pass
            continue
        if h2 == 0 and s2 == 0 and dismantle4 == 1:
            half2_broken -= 1
            if dismantle2 == 1:
                if x4 == 1:
                    total_cost -= part4_cost
                if x5 == 1:
                    total_cost -= part5_cost
                if x6 == 1:
                    total_cost -= part6_cost
            else:
                pass
            continue
        if h3 == 0 and s3 == 0 and dismantle4 == 1:
            half3_broken -= 1
            if dismantle3 == 1:
                if x7 == 1:
                    total_cost -= part1_cost
                if x8 == 1:
                    total_cost -= part2_cost

            else:
                pass
            continue

        # 组装成品
        if s1 == 1 and s2 == 1 and s3 == 1:  # 原料合格
            total_cost += product_cost
            product = np.random.rand() > product_defect_rate
            if h4 == 1:  # 如果检测
                total_cost += product_detect_cost
                if product == 1:  # 产品合格
                    total_revenue += product_price
                else:
                    if dismantle4 == 1:  # 如果拆解成品
                        total_cost += product_disassembly_cost
                        # 判断半成品1
                        if s1 == 1:  # 如果半成品1是好的
                            half1_available += 1
                        elif s1 == 0:  # 如果半成品1是坏的
                            half1_broken += 1
                            if h1 == 1:  # 如果检测
                                half1_broken -= 1  # 直接丢弃
                            else:  # 如果不检测
                                pass  # 计入坏半成品库

                        # 半成品2
                        if s2 == 1:  # 如果半成品是好的
                            half2_available += 1
                        elif s2 == 0:  # 如果半成品是坏的
                            half2_broken += 1
                            if h2 == 1:  # 如果检测
                                half2_broken += 1
                            else:  # 如果不检测
                                pass
                        # 半成品3
                        if s3 == 1:  # 如果半成品是好的
                            half3_available += 1
                        elif s3 == 0:  # 如果半成品是坏的
                            half3_broken += 1
                            if h3 == 1:  # 如果检测
                                half3_broken -= 1  # 直接丢弃
                            else:  # 如果不检测
                                pass
                    else:
                        pass  # 直接丢弃
            else:  # 不检测
                if product == 1:
                    total_revenue += product_price
                else:
                    total_cost += product_return_loss
                    if dismantle4 == 1:  # 如果拆解成品
                        total_cost += product_disassembly_cost
                        # 判断半成品1
                        if s1 == 1:  # 如果半成品1是好的
                            half1_available += 1
                        elif s1 == 0:  # 如果半成品1是坏的
                            half1_broken += 1
                            if h1 == 1:  # 如果检测
                                half1_broken -= 1  # 直接丢弃
                            else:  # 如果不检测
                                pass  # 计入坏半成品库
                        # 半成品2
                        if s2 == 1:  # 如果半成品是好的
                            half2_available += 1
                        elif s2 == 0:  # 如果半成品是坏的
                            half2_broken += 1
                            if h2 == 1:  # 如果检测
                                half2_broken -= 1
                            else:  # 如果不检测
                                pass
                        # 半成品3
                        if s3 == 1:  # 如果半成品是好的
                            half3_available += 1
                        elif s3 == 0:  # 如果半成品是坏的
                            half3_broken += 1
                            if h3 == 1:  # 如果检测
                                half3_broken -= 1  # 直接丢弃
                            else:  # 如果不检测
                                pass
                    else:
                        pass  # 直接丢弃
        else:
            total_cost += product_return_loss
            if dismantle4 == 1:  # 如果拆解成品
                total_cost += product_disassembly_cost
                # 判断半成品1
                if s1 == 1:  # 如果半成品1是好的
                    half1_available += 1
                elif s1 == 0:  # 如果半成品1是坏的
                    half1_broken += 1
                    if h1 == 1:  # 如果检测
                        half1_broken -= 1  # 直接丢弃
                    else:  # 如果不检测
                        pass  # 计入坏半成品库
                # 半成品2
                if s2 == 1:  # 如果半成品是好的
                    half2_available += 1
                elif s2 == 0:  # 如果半成品是坏的
                    half2_broken += 1
                    if h2 == 1:  # 如果检测
                        half2_broken -= 1
                    else:  # 如果不检测
                        pass
                # 半成品3
                if s3 == 1:  # 如果半成品是好的
                    half3_available += 1
                elif s3 == 0:  # 如果半成品是坏的
                    half3_broken += 1
                    if h3 == 1:  # 如果检测
                        half3_broken -= 1  # 直接丢弃
                    else:  # 如果不检测
                        pass
            else:
                pass  # 直接丢弃

    net_profit = total_revenue - total_cost
    return net_profit

# 生成策略
def generate_strategies():
    # 定义策略名和对应的参数
    strategy_names = [
        "半成品1的零件", "半成品2的零件", "半成品3的零件",  # 检测或不检测
        "半成品1", "半成品2", "半成品3",  # 检测或不检测
        "成品", "半成品", "成品"  # 拆解或不拆解
    ]

    # 定义各组的策略长度（检测3项，检测3项，拆解3项）
    group1_len = 3  # 半成品零件检测
    group2_len = 3  # 半成品检测
    group3_len = 3  # 成品/半成品拆解

    # 生成所有可能的组合
    group1 = list(itertools.product([0, 1], repeat=group1_len))  # 检测/不检测
    group2 = list(itertools.product([0, 1], repeat=group2_len))  # 检测/不检测
    group3 = list(itertools.product([0, 1], repeat=group3_len))  # 拆解/不拆解

    # 将三组组合成最终的策略集
    all_combinations = list(itertools.product(group1, group2, group3))

    all_strategies = []
    for g1, g2, g3 in all_combinations:
        strategy = g1 + g2 + g3
        description = " ".join(
            f"{name}: {'检测' if value == 1 else '不检测' if i < group1_len + group2_len else '拆解' if value == 1 else '不拆解'}"
            for i, (name, value) in enumerate(zip(strategy_names, strategy))
        )
        all_strategies.append((description, *strategy))

    return all_strategies


def B4_refrom_Q3(p):
    strategies = generate_strategies()

    # 遍历所有策略
    all_profits = []  # 用于存储所有策略的收益
    best_profit = float('-inf')  # 初始化最大收益为负无穷大
    best_strategy = None  # 用于存储收益最大的策略
    best_strategy_idx = -1  # 用于存储收益最大的策略的编号

    for idx, strategy in enumerate(strategies):  # 添加索引
        name, half_parts_1, half_parts_2, half_parts_3, detect_half1, detect_half2, detect_half3, detect_product, dismantle_half, dismantle_product = strategy

        # 将策略的各部分映射为对应的变量
        x1 = x2 = x3 = half_parts_1
        x4 = x5 = x6 = half_parts_2
        x7 = x8 = half_parts_3
        h1 = detect_half1
        h2 = detect_half2
        h3 = detect_half3
        h4 = detect_product
        dismantle1 = dismantle2 = dismantle3 = dismantle_half
        dismantle4 = dismantle_product

        # 模拟当前策略的收益
        current_profit = simulation_product(p, x1, x2, x3, x4, x5, x6, x7, x8,
                                            h1, h2, h3, h4,
                                            dismantle1, dismantle2, dismantle3, dismantle4)
        all_profits.append(current_profit)

        # 更新最大收益及最佳策略
        if current_profit > best_profit:
            best_profit = current_profit
            best_strategy = strategy
            best_strategy_idx = idx  # 记录当前策略的编号

    # 输出收益最大的策略及其编号
    print(f"收益最大的策略是：{best_strategy[0]} (编号 {best_strategy_idx})，净收益为：{best_profit}")

    return best_strategy_idx, best_profit

#测试
'''
B4_refrom_Q3(0.1)
'''





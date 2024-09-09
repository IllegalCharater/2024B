import numpy as np

total_cost = 0  # 总开销


# 定义状态转移矩阵
def step3_create_transition_matrix(detect_part1, detect_part2, detect_product, dismantle_defective):

    # 定义零件和成品的次品率
    defect_rate_part1 = 0.1  # 零件1的次品率
    defect_rate_part2 = 0.1  # 零件2的次品率

    defect_rate_product = 0.1  # 成品的次品率

    # 状态转移矩阵 P，初始为零矩阵
    P = np.zeros((10, 10))  # 状态数为6，分别是：
    # 0: 零件1、零件2 和 零件3 状态
    # 1: 产品状态判断
    # 2: 检测出的合格状态
    # 3: 未检出的合格状态
    # 4: 检测出的次品状态
    # 5: 未检出的次品状态
    # 6：拆解次品状态
    # 7：次品结束状态
    # 8：成品结束状态
    # 9: 重新开始

    # 零件1、零件2 和 零件3 合成成品的转移概率
    part1_pass = np.random.rand() > defect_rate_part1
    part2_pass = np.random.rand() > defect_rate_part2

    if detect_part1 and detect_part2 :  # 进行检测
        if part1_pass and part2_pass :
            P[0, 1] = 1  # 如果检测的零件为合格品，直接进入产品判断状态
        else:
            P[0, 9] = 1  # 如果检测后零件不合格，重新开始
    else:
        part1_probability=0
        part2_probability=0

        if part1_pass:
            if detect_part1:
                part1_probability=1
            else:
                part1_probability=(1-defect_rate_part1)
        else:
            pass

        if part2_pass:
            if detect_part2:
                part2_probability = 1
            else:
                part2_probability = (1 - defect_rate_part2)
        else:
            pass

        P[0,1]=part1_probability*part2_probability
        if P[0, 1]==0 and (detect_part2 or detect_part1):
            P[0, 9]=1
        else:
            P[0, 5]=1-P[0, 1]

    # 成品检测
    if detect_product:
        product_pass = np.random.rand() > defect_rate_product
        if product_pass:#通过检测
            P[1, 2] = 1  # 成品通过检测，进入检测合格状态
        else:
            P[1, 4] = 1  # 成品为次品
    else:#不检测
        P[1, 3] = (1 - defect_rate_product)  # 成品合格，进入未检出合格状态
        P[1, 5] = 1 - P[1, 3]  # 成品不合格，进入未检出次品状态

    # 检出次品处理
    if dismantle_defective:
        P[4, 6] = 1  # 拆解次品，拆解状态
    else:
        P[4, 9] = 1  # 不拆解次品，重新开始

    # 吸收状态：合格成品或次品
    P[2, 8]= 1 #检出合格
    P[3, 8]= 1 #未检出合格
    P[5, 7]= 1 #未检出次品
    P[7, 7] = 1  # 次品状态结束
    P[8, 8] = 1  # 成品状态结束
    P[9, 9]=1 #重新开始
    return P

# 模拟状态转移并计算总收益
def step3_simulate_markov_chain_with_profit(steps, detect_part1, detect_part2, detect_product, dismantle_defective):
    """
    模拟马尔科夫链的状态转移过程，并计算总收益。
    - detect_part1, detect_part2, detect_part3 表示是否检测零件1、零件2、零件3。
    - detect_product 表示是否检测成品。
    - dismantle_defective 表示是否拆解次品。
    """
    global total_cost
    total_cost = 0  # 总开销

    # 成本和收益设定
    cost_part1 = 8  # 零件1的成本
    cost_part2 = 12  # 零件2的成本


    detect_cost_part1 = 1  # 检测零件1的成本
    detect_cost_part2 = 2  # 检测零件2的成本


    detect_cost_product = 4  # 检测成品成本
    assembly_cost_product = 8  # 装配成品的成本
    #丢弃次品没有成本
    dismantle_cost = 6  # 拆解次品的成本（返还零件时产生的成本）

    state = 0  # 初始状态为零件1、零件2和零件3
    states = [state]  # 记录状态序列
    P = step3_create_transition_matrix(detect_part1, detect_part2, detect_product, dismantle_defective)

    #print("状态转移矩阵：\n", P)

    for _ in range(steps):
        # 打印当前状态和当前成本
        #print(f"step3 当前状态: {state}, 总成本: {total_cost}, 总收益: {total_revenue}")

        # 计算当前步骤的成本和收益
        if state == 0:  # 零件1、零件2检测
            # 加入零件成本
            total_cost += cost_part1 + cost_part2

            # 计算各零件的检测成本
            if detect_part1:
                total_cost += detect_cost_part1
            if detect_part2:
                total_cost += detect_cost_part2
            #print(f"0now_cost:{total_cost}")
        elif state == 1:
            total_cost += assembly_cost_product
            #print(f"1now_cost:{total_cost}")
        elif state == 2:  # 检测出合格成品

            total_cost += detect_cost_product
            #print(f"2now_cost:{total_cost}")
        elif state == 3:  # 未检出的合格
            pass
        elif state == 4:  # 检测出次品

            total_cost += detect_cost_product
            #print(f"4now_cost:{total_cost}")
        elif state == 5:  # 未检出次品
            pass
        elif state == 6: #拆解次品状态
            total_cost += dismantle_cost
            #返还费用
            total_cost-=cost_part1+cost_part2
            #print(f"6now_cost:{total_cost}")
            state=9 #重新开始
        elif state == 7: #次品结束状态
            #print(f"7now_cost:{total_cost}")
            break
        elif state == 8: #成品结束状态
            #print(f"8now_cost:{total_cost}")
            break
        elif state == 9: #重新开始
            if states[-2]==0:
                total_cost -= cost_part1 + cost_part2
            #print(f"now_cost:{total_cost}")
            break

        # 根据当前状态和转移概率进行下一步状态选择
        current_probabilities = P[state]

        if not np.isclose(np.sum(current_probabilities), 1):
            raise ValueError(f"状态 {state} 的转移概率不等于1: {current_probabilities}")

        # 选择下一个状态
        next_state = np.random.choice(range(current_probabilities.shape[0]), p=current_probabilities)
        states.append(next_state)

        state = next_state

    # 计算总收益
    net_profit = - total_cost
    return states, net_profit

# 执行模拟并计算总收益
def step3(detect_part1, detect_part2, detect_product, dismantle_defective):
    #print("step3")
    global total_cost
    total_cost = 0  # 总开销

    steps = 100 #总状态转移次数
    # detect_part1, detect_part2,  detect_product, dismantle_defective 分别表示是否检测 零件1、零件2、成品，是否拆解次品

    half_product3=1
    states=[0]
    all_profit=0
    while not(states[-1]==7 or states[-1]==8):
        #print('states:',states)
        states, profit = step3_simulate_markov_chain_with_profit(steps, detect_part1, detect_part2, detect_product, dismantle_defective)
        all_profit = profit

    if states[-1]==7:
        half_product3=0
    elif states[-1]==8:
        half_product3=1

    #print("状态转移序列:", states)
    #print("成品：","合格" if half_product3 == 1 else "不合格")
    #print("总收益:", all_profit)

    return half_product3,all_profit

#测试3
'''
x7, x8,h3, dismantle = 1, 1, 1, 1  # 检测零件7 检测零件8 检测成品 拆解次品
half_product3, all_profit3 = step3(x7, x8,h3, dismantle)
print("half_product3",half_product3)
print("all_profit3", all_profit3)
'''
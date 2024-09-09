import math

from scipy import stats
from scipy.stats import norm


def calculate_sample_size(p, confidence_level, error_margin):
    """
    根据给定的次品率、信度和允许误差计算最小样本量。

    :param p: 次品率（如0.1代表10%）
    :param confidence_level: 信度（如0.95代表95%）
    :param error_margin: 允许误差
    :return: 计算出的样本量
    """
    # 计算正态分布的临界值
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    # 样本量计算公式
    n = (z_value ** 2) * p * (1 - p) / (error_margin ** 2)
    return math.ceil(n)  # 样本量向上取整

'''
def check_sampling_test(sample_size, defective_items, total_items, confidence_level, threshold_p):
    """
    根据抽样检测结果判断是否接收批次。

    :param sample_size: 抽样样本量
    :param defective_items: 抽样中检测到的次品数
    :param total_items: 样本中的总数量
    :param confidence_level: 信度
    :param threshold_p: 标称次品率
    :return: 决策结果（接收或拒收）
    """

    # 计算样本次品率
    sample_defective_rate = defective_items / total_items
    # 在给定的信度下计算拒收或接收的临界值
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    # 计算总体次品率的置信区间
    lower_bound = sample_defective_rate - z_value * math.sqrt(
        (sample_defective_rate * (1 - sample_defective_rate)) / sample_size)
    upper_bound = sample_defective_rate + z_value * math.sqrt(
        (sample_defective_rate * (1 - sample_defective_rate)) / sample_size)

    print('下边界',lower_bound, '上边界',upper_bound,'次品率',threshold_p)
    # 根据次品率的置信区间判断是否接收或拒收
    if upper_bound < threshold_p:
        return "接收该批次"
    elif lower_bound > threshold_p:
        return "拒收该批次"
    else:
        return "处于置信区间"
'''

def judge_sampling_test(sample_size, defective_items, confidence_level, threshold_p):
    """
    根据抽样检测结果判断是否接收批次。

    :param sample_size: 抽样样本量
    :param defective_items: 抽样中检测到的次品数
    :param confidence_level: 信度
    :param threshold_p: 标称次品率
    :return: 决策结果（接收或拒收）
    """
    '''
    # 计算样本次品率
    sample_defective_rate = defective_items / total_items
    # 在给定的信度下计算拒收或接收的临界值
    z_value = norm.ppf(1 - (1 - confidence_level) / 2)
    # 计算总体次品率的置信区间
    lower_bound = sample_defective_rate - z_value * math.sqrt(
        (sample_defective_rate * (1 - sample_defective_rate)) / sample_size)
    upper_bound = sample_defective_rate + z_value * math.sqrt(
        (sample_defective_rate * (1 - sample_defective_rate)) / sample_size)

    print('下边界',lower_bound, '上边界',upper_bound,'次品率',threshold_p)
    # 根据次品率的置信区间判断是否接收或拒收
    if upper_bound < threshold_p:
        return "接收该批次"
    elif lower_bound > threshold_p:
        return "拒收该批次"
    else:
        return "处于置信区间"
    '''
    result=stats.binomtest(defective_items, sample_size, threshold_p,alternative='greater')
    print('成功概率',result.pvalue)
    lower_bound,high_bound = result.proportion_ci(confidence_level=confidence_level)
    print('lower_bound',lower_bound,'high_bound',high_bound)
    if result.pvalue>=threshold_p:
        return "接受"
    elif result.pvalue<threshold_p:
        return "拒绝"
    else:
        return "无法确定"

# 设置参数
confidence_level_reject = 0.95  # 拒收批次的信度95%
confidence_level_accept = 0.90  # 接收批次的信度90%
p = 0.1  # 标称次品率10%
error_margin = 0.03  # 允许误差2%

# 计算样本量
sample_size_reject = calculate_sample_size(p, confidence_level_reject, error_margin)
sample_size_accept = calculate_sample_size(p, confidence_level_accept, error_margin)

print(f"拒收批次需要的样本量: {sample_size_reject}")
print(f"接收批次需要的样本量: {sample_size_accept}")

# 假设从样本中检测到一定数量的次品
detected_defective_items = 12  # 样本中的次品数
#total_sampled_items = 1000  # 样本总数

# 进行抽样检测决策
result_reject = judge_sampling_test(sample_size_reject, detected_defective_items,
                                    confidence_level_reject, p)
result_accept = judge_sampling_test(sample_size_accept, detected_defective_items,
                                    confidence_level_accept, p)

print(f"拒收批次的决策: {result_reject}")
print(f"接收批次的决策: {result_accept}")
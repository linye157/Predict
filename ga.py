import numpy as np
import torch
import torch.nn as nn


DNA_SIZE = 30  # 编码的位数(本实验使用实数编码)
POP_SIZE = 200  # 初始化种群数量
CROSSOVER_RATE = 0.8  # 交叉率
MUTATION_RATE = 0.005  # 变异率
N_GENERATIONS = 50  # 迭代次数
# 30个特征，每个特征的取值范围不同
lower_bounds = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 735, 1.8, 0.262, 0, 546, 397, 516, 735, 41, 366, 0, 675, 0, 39, 34.3, 1151, 155, 832, 351,
     0])
upper_bounds = np.array(
    [0.192, 0.0373, 1.0609, 0.00501, 0.0228, 0.003, 0.0537, 0.0184, 0.048, 0.0195, 1265, 4, 2.057, 23.25, 842, 749, 634, 1265, 150, 609, 1.6, 779, 127, 114.4, 113, 1276, 1467,
     922, 779, 7078])

model_name='LSTMTransformer'
model_path=f'./model/{model_name}.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义模型
class LSTMTransformerRegressor(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, num_heads, num_targets):
        super(LSTMTransformerRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads)
        self.linear = nn.Linear(hidden_size, num_targets)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Transformer
        transformer_out = self.transformer(lstm_out.permute(1, 0, 2), lstm_out.permute(1, 0, 2))
        # Linear
        out = self.linear(transformer_out.permute(1, 0, 2))
        return out[:, -1, :].unsqueeze(1) # 只使用最后一个时间步的输出

# 加载模型
model = torch.load(model_path)

# 适应度函数
def fitness_func(pop):
    seq_length = 1
    pop_reshaped = pop.reshape(POP_SIZE, seq_length, DNA_SIZE)
    # 转换为 PyTorch 张量
    pop_tensor = torch.tensor(pop_reshaped, dtype=torch.float32).to(device)

    Y_pre = model(pop_tensor)  # 输出: torch.size([200,1,3])
    return Y_pre

# 计算适应度
def get_fitness(pop):
    # x, y = translateDNA(pop)
    # 目标函数
    pred = fitness_func(pop)
    pred = pred.squeeze(1)    # 去掉多余的维度，现在 pred 的形状为 torch.Size([200, 3])
    target1 = pred[:, 0]
    target2 = pred[:, 1]
    target3 = pred[:, 2]
    # 计算适应度
    # 目标值的上下限约束
    low_bounds = [200, 300, 0.5]  # 对应每个目标值的最小值
    up_bounds = [750, 750, 45]  # 对应每个目标值的最大值

    penalty = torch.zeros(pop.shape[0])

    # 计算罚值
    for i in range(pop.shape[0]):
        if target1[i] < low_bounds[0]:
            penalty[i] += low_bounds[0] - target1[i]
        elif target1[i] > up_bounds[0]:
            penalty[i] += target1[i] - up_bounds[0]

        if target2[i] < low_bounds[1]:
            penalty[i] += low_bounds[1] - target2[i]
        elif target2[i] > up_bounds[1]:
            penalty[i] += target2[i] - up_bounds[1]

        if target3[i] < low_bounds[2]:
            penalty[i] += low_bounds[2] - target3[i]
        elif target3[i] > up_bounds[2]:
            penalty[i] += target3[i] - up_bounds[2]

    # 适应度，适应度越大，选择的概率越大
    fitness = -penalty   # 惩罚越大，适应度越低

    fitness = fitness - fitness.min()    # 确保适应度值非负

    return fitness


# # 解码
# def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
#     x_pop = pop[:, 1::2]  # 奇数列表示X
#     y_pop = pop[:, ::2]  # 偶数列表示y
#
#     # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
#     x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
#     y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
#     return x, y

# 交叉变异
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.rand(DNA_SIZE)  # 随机产生交叉的点
            child = father * cross_points + mother * (1 - cross_points)  # 线性插值交叉
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return np.array(new_pop)


def mutation(child, MUTATION_RATE=0.003):
    for i in range(DNA_SIZE):  # 对每个特征单独考虑变异
        if np.random.rand() < MUTATION_RATE:  # 根据变异概率决定是否变异
            mutation_value = np.random.uniform(-0.1, 0.1)  # 添加一个小扰动值
            child[i] += mutation_value  # 特征值加上扰动
            # 限制变异后的特征值在其上下界内
            if child[i] < lower_bounds[i]:
                child[i] = lower_bounds[i]
            if child[i] > upper_bounds[i]:
                child[i] = upper_bounds[i]

# 根据适应度进行选择
def select(pop, fitness):
    # 规范化适应度
    fitness_sum = fitness.sum()

    if fitness_sum == 0:
        fitness = np.ones_like(fitness)  # 如果适应度全为零，设为均匀分布
        fitness_sum = fitness.sum()

    # 选择
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p= (fitness) / (fitness_sum) )
    return pop[idx]

# 输出结果
def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    # x, y = translateDNA(pop)
    best_genotype = pop[max_fitness_index]
    formatted_genotype = [f"{gene:.4f}" for gene in best_genotype]  # 使用列表推导式格式化
    print("最优的基因型：", formatted_genotype)

    formatted_genotype_tensor = torch.tensor(pop[max_fitness_index], dtype=torch.float32).to(device)
    input_tensor = formatted_genotype_tensor.unsqueeze(0).unsqueeze(0)
    Y_PRE = model(input_tensor)
    print("最优化结果为：", Y_PRE)


if __name__ == "__main__":

    # 生成了一个种群pop，并赋予每个个体随机的二进制基因
    # pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)

    pop = np.random.uniform(lower_bounds, upper_bounds, size=(POP_SIZE, DNA_SIZE))

    for _ in range(N_GENERATIONS):  # 迭代N代
        # x, y = translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)

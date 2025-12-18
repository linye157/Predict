import gym
import numpy as np
import pandas as pd
import joblib
from gym import spaces
from stable_baselines3 import PPO
import shap
from mlutils import *
from tqdm import tqdm

# 配置类
class Config:
    # 目标性能阈值
    TARGETS = np.array([130, 270, 26])  # 三个性能指标的最低要求
    # SHAP参数
    SAMPLE_RATIO = 0.1     # 背景数据采样比例
    N_SUMMARY = 100        # SHAP背景数据压缩量
    TOP_K = 5              # 每个样本选择的关键参数数量
    MAX_STEPS = 100        # 最大步数
    BUFFER_RATIO = 0.1     # 动态范围比例
    
    # 文件路径
    DATA_PATH = './data/621useddata2.xlsx'
    SCALER_MODEL_PATH = './models/scaler.pkl'  
    PRE_MODEL_PATH = './models/'
    RANDOM_SEED = 42     # 随机种子
    TEST_SIZE = 0.2      # 测试集比例
    # 模型名称
    MODEL_NAME = 'Random Forest'
    
    # 输出文件
    RESULT_FILE = 'results_PPO.csv'

# 工艺参数头信息
Procedure_header=['化学元素含量', '化学元素含量', '化学元素含量', '化学元素含量', '化学元素含量', '化学元素含量', '化学元素含量', '化学元素含量', '化学元素含量', 
                  '热轧', '热轧', '热轧', '热轧', '热轧', '热轧', '热轧', '热轧', 
                  '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', 
                  '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', 
                  '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧', '冷轧',
                  '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌', '镀锌']

Parameter_header = [
    "碳", "硅", "锰", "磷", "硫", "钛", "铌", "氧", "氮", "材料实际重量", "出口材料实际厚度",
    "出口材料实际宽度", "卷取温度平均值", "出炉温度", "在炉时间", "精轧入口平均温度", "精轧出口平均温度",
    "出口材料实际厚度公差", "出口材料实际宽度公差", "出口材料实际重量", "入口材料1厚度", "入口材料1宽度",
    "入口材料1重量", "S1机架压下率", "S2机架压下率", "S3机架压下率", "S4机架压下率", "S5机架压下率",
    "S1机架入口张力", "S1~S2机架间张力", "S2～S3机架间张力", "S3～S4机架间张力", "S4～S5机架间张力",
    "S5出口张力", "S1机架入口单位张力", "S1~S2机架间单位张力", "S2~S3机架间单位张力",
    "S3~S4机架间单位张力", "S4~S5机架间单位张力", "S5机架工作轧辊粗糙度(底)", "S5机架工作轧辊粗糙度(上)",
    "1#机架轧制力模型设定值", "2#机架轧制力模型设定值", "3#机架轧制力模型设定值", "4#机架轧制力模型设定值",
    "5#机架轧制力模型设定值", "拉矫率平均值", "1#酸槽温度", "2#酸槽温度", "3#酸槽温度", "酸洗工序速度平均值1",
    "上表面镀层重量", "下表面镀层重量", "平整率平均值", "上表面涂油量", "下表面涂油量", "工艺段速度平均值",
    "ES平均温度", "FCS平均温度", "IHS平均温度", "SCS平均温度", "SF平均温度", "RCS平均温度",
    "RTF平均温度", "JPF平均温度"]

# 工具函数
def find_unqualified_samples(y, targets):
    """找到至少两个性能指标不达标的样本"""
    mask = y < targets
    count_below_threshold = np.sum(mask, axis=1)
    indices = np.where(count_below_threshold >= 2)[0]
    return indices

def is_satisfied(y_pred, targets):
    """判断力学性能是否达标"""
    return all(y >= t for y, t in zip(y_pred, targets))

def inverse_normalize(sample, scaler):
    """反标准化函数"""
    inverse_sample = scaler.inverse_transform(sample.reshape(1, -1))
    return inverse_sample[0]

def load_data_and_model():
    """加载数据和模型"""
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    
    # 加载数据
    print("加载数据...")
    df = pd.read_excel(Config.DATA_PATH, header=0, skiprows=[1])
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -3:].values
    
    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
    )
    print(f"数据划分完成。训练集: {train_x.shape[0]}个样本, 测试集: {test_x.shape[0]}个样本")
    
    # 加载或创建数据缩放器
    scaler = None
    if os.path.exists(Config.SCALER_MODEL_PATH):
        print("加载已有缩放器...")
        try:
            scaler = joblib.load(Config.SCALER_MODEL_PATH)
            # 确认缩放器已加载
            print("缩放器加载成功!")
        except Exception as e:
            print(f"加载缩放器失败: {e}")
            scaler = None
    
    if scaler is None:
        print("创建新的缩放器...")
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(Config.SCALER_MODEL_PATH), exist_ok=True)
        scaler = StandardScaler()
        scaler.fit(train_x)
        # 保存缩放器
        try:
            joblib.dump(scaler, Config.SCALER_MODEL_PATH)
            print(f"缩放器已保存到 {Config.SCALER_MODEL_PATH}")
        except Exception as e:
            print(f"保存缩放器失败: {e}")
    
    # 标准化数据
    train_x_scaled = scaler.transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    
    # 加载或训练预测模型
    model_path = os.path.join(Config.PRE_MODEL_PATH, f"{Config.MODEL_NAME}.pkl")
    model = None
    
    if os.path.exists(model_path):
        # print(f"加载已有模型: {Config.MODEL_NAME}...")
        print(f"加载已有模型: DMP_BiLSTM_MHA...")
        try:
            model = joblib.load(model_path)
            print("模型加载成功!")
        except Exception as e:
            print(f"加载模型失败: {e}")
            model = None
    
    if model is None:
        print(f"训练新的{Config.MODEL_NAME}模型...")
        # 创建目录（如果不存在）
        os.makedirs(Config.PRE_MODEL_PATH, exist_ok=True)
        # 创建和训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=Config.RANDOM_SEED)
        model.fit(train_x_scaled, train_y)
        # 保存模型
        try:
            joblib.dump(model, model_path)
            print(f"模型已保存到 {model_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    return train_x_scaled, train_y, test_x_scaled, test_y, model, scaler

def cal_shap_values(x, explainer):
    """计算SHAP值"""
    return explainer.shap_values(x)

def generate_param_bounds(sample, X_data, top_indices, buffer_ratio):
    """根据历史数据和当前样本生成优化参数范围"""
    bounds = {}
    for idx in top_indices:
        # 全局数据范围（考虑工艺限制）
        global_min = X_data[:, idx].min()
        global_max = X_data[:, idx].max()
        
        # 当前值
        current_val = sample[idx]
        
        # 动态范围：当前值±buffer_ratio范围的全局裁剪
        buffer_range = (global_max - global_min) * buffer_ratio
        min_val = max(global_min, current_val - buffer_range)
        max_val = min(global_max, current_val + buffer_range)
        
        bounds[f'x{idx}'] = (min_val, max_val)
    return bounds

def prepare_shap_explainer(model, train_x):
    """准备SHAP解释器"""
    # 随机提取样本
    train_x_sample = train_x[np.random.choice(train_x.shape[0], int(train_x.shape[0] * Config.SAMPLE_RATIO), replace=False)]
    train_x_sample_summary = shap.sample(train_x_sample, Config.N_SUMMARY)
    # 创建SHAP解释器
    explainer = shap.KernelExplainer(model.predict, train_x_sample_summary)
    return explainer, train_x_sample

# 环境类
class ParamOptimizationEnv(gym.Env):
    def __init__(self, sample, top_indices, bounds, model, is_satisfied):
        super(ParamOptimizationEnv, self).__init__()
        self.sample = sample.copy()  # 初始工艺参数
        self.top_indices = top_indices  # 需要优化的参数索引
        self.bounds = bounds  # 优化范围
        self.model = model  # 预测模型
        self.is_satisfied = is_satisfied  # 达标判断函数
        self.current_params = sample.copy()
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(top_indices),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(top_indices),), dtype=np.float32)
        self.max_steps = Config.MAX_STEPS
        self.current_step = 0

    def reset(self):
        self.current_params = self.sample.copy()
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        # 更新需要优化的参数
        for i, idx in enumerate(self.top_indices):
            delta = action[i] * (self.bounds[f'x{idx}'][1] - self.bounds[f'x{idx}'][0]) / 2
            self.current_params[idx] += delta
            self.current_params[idx] = np.clip(self.current_params[idx], self.bounds[f'x{idx}'][0], self.bounds[f'x{idx}'][1])

        # 预测力学性能
        y_pred = self.model.predict(self.current_params.reshape(1, -1))[0]
        satisfied = self.is_satisfied(y_pred, Config.TARGETS)
        
        # 奖励设计
        if satisfied:
            reward = 10  # 达标时高奖励
        else:
            # 未达标时奖励与目标距离负相关
            reward = -sum(abs(y - t) for y, t in zip(y_pred, Config.TARGETS)) / 3
        reward -= 0.1  # 步数惩罚
        
        self.current_step += 1
        done = satisfied or self.current_step >= self.max_steps
        return self._get_obs(), reward, done, {"y_pred": y_pred}

    def _get_obs(self):
        return self.current_params[self.top_indices]

# 优化函数
def optimize_sample(sample, X_data, model, explainer):
    """优化单个样本"""
    # 计算 SHAP 值并选择 top_k 参数
    shap_values = cal_shap_values(sample, explainer)
    top_indices = np.argsort(np.abs(shap_values).mean(1))[::-1][:Config.TOP_K]
    bounds = generate_param_bounds(sample, X_data, top_indices, Config.BUFFER_RATIO)

    # 创建环境
    env = ParamOptimizationEnv(sample, top_indices, bounds, model, is_satisfied)

    # 训练 PPO 模型
    ppo_model = PPO("MlpPolicy", env, verbose=0, batch_size=256, n_epochs=10)
    ppo_model.learn(total_timesteps=100)

    # 测试优化
    obs = env.reset()
    for _ in range(Config.MAX_STEPS):
        action, _ = ppo_model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    if is_satisfied(info['y_pred'], Config.TARGETS):
        status = 'success'
        opt_pred = info['y_pred']
    else:
        status = 'fail'
        opt_pred = info['y_pred']
        
    return env.current_params, top_indices, bounds, status, opt_pred

def optimize_many(test_x, test_y, X_data, model, explainer, scaler):
    """优化多个样本"""
    # 找到不合格样本
    unqualified_samples = find_unqualified_samples(test_y, Config.TARGETS)
    print(f"共找到 {len(unqualified_samples)} 个不合格样本")
    
    results = []
    for idx in tqdm(unqualified_samples):
        ori_sample = test_x[idx]
        ori_pred = test_y[idx]
        optimized_params, top_indices, bounds, status, opt_pred = optimize_sample(ori_sample, X_data, model, explainer)
        
        # 对工艺参数进行反标准化
        if scaler is not None:
            ori_sample_raw = inverse_normalize(ori_sample, scaler)
            optimized_params_raw = inverse_normalize(optimized_params, scaler)
        else:
            ori_sample_raw = ori_sample
            optimized_params_raw = optimized_params
            
        # 记录结果
        if status == 'success':
            param_changes = []
            for p_idx in top_indices:
                ori_val = ori_sample_raw[p_idx]
                opt_val = optimized_params_raw[p_idx]
                change_pct = (opt_val - ori_val) / ori_val * 100 if ori_val != 0 else 0
                param_changes.append({
                    'param_idx': int(p_idx),
                    'param_name': Procedure_header[int(p_idx)] + '过程的' + Parameter_header[int(p_idx)],
                    'ori_val': float(ori_val),
                    'opt_val': float(opt_val),
                    'change_pct': float(change_pct)
                })
                
            # 转换为列表
            ori_pred_list = ori_pred.tolist() if hasattr(ori_pred, 'tolist') else list(ori_pred)
            opt_pred_list = opt_pred.tolist() if hasattr(opt_pred, 'tolist') else list(opt_pred)
            
            results.append({
                'sample_id': int(idx),
                'status': status,
                'original_performance': ori_pred_list,
                'optimized_performance': opt_pred_list,
                'param_changes': param_changes,
                'top_params_index': [int(p_idx) for p_idx in top_indices.tolist()],
                'top_paras': [Parameter_header[int(pidx)] for pidx in top_indices.tolist()]
            })
        else:
            results.append({
                'sample_id': int(idx),
                'status': status,
                'original_performance': ori_pred.tolist() if hasattr(ori_pred, 'tolist') else list(ori_pred),
                'optimized_performance': opt_pred.tolist() if hasattr(opt_pred, 'tolist') else list(opt_pred),
                'param_changes': None,
                'top_params_index': [int(p_idx) for p_idx in top_indices.tolist()],
                'top_paras': [Parameter_header[int(pidx)] for pidx in top_indices.tolist()]
            })
    
    df_results = pd.DataFrame(results)
    success_rate = (df_results['status'] == 'success').mean()
    print(f"\n优化完成！成功率：{success_rate:.1%}")
    
    # 保存到CSV
    df_results.to_csv(Config.RESULT_FILE, index=False, encoding='utf-8-sig')
    
    return df_results

# 展示优化结果
def display_optimization_results(sample_idx, original_sample, optimized_params, original_pred, optimized_pred, top_indices, scaler):
    """展示单个样本的优化结果"""
    print(f"\n样本 {sample_idx} 的优化结果:")
    print(f"优化前的力学性能: {original_pred}")
    print(f"优化后的力学性能: {optimized_pred}")
    print(f"目标阈值: {Config.TARGETS}")
    
    if scaler is not None:
        original_sample_raw = inverse_normalize(original_sample, scaler)
        optimized_params_raw = inverse_normalize(optimized_params, scaler)
    else:
        original_sample_raw = original_sample
        optimized_params_raw = optimized_params
    
    print("\n关键参数的变化:")
    for i, idx in enumerate(top_indices):
        param_name = f"{Procedure_header[idx]}过程的{Parameter_header[idx]}"
        original_value = original_sample_raw[idx]
        optimized_value = optimized_params_raw[idx]
        change = optimized_value - original_value
        change_percent = (change / original_value) * 100 if original_value != 0 else 0
        
        print(f"{i+1}. {param_name}:")
        print(f"   原值: {original_value:.4f} -> 优化值: {optimized_value:.4f}")
        print(f"   变化: {change:.4f} ({change_percent:.2f}%)")

def main():
    """主函数"""
    print("正在加载数据和模型...")
    train_x, train_y, test_x, test_y, model, scaler = load_data_and_model()
    
    print("准备SHAP解释器...")
    explainer, train_x_sample = prepare_shap_explainer(model, train_x)
    
    # 1. 单样本优化演示
    unqualified_samples = find_unqualified_samples(test_y, Config.TARGETS)
    if len(unqualified_samples) > 0:
        print("\n单样本优化演示:")
        sample_idx = unqualified_samples[0]
        sample = test_x[sample_idx]
        print(f"待优化样本索引: {sample_idx}")
        
        print("开始优化单个样本...")
        optimized_params, top_indices, bounds, status, opt_pred = optimize_sample(sample, train_x_sample, model, explainer)
        
        if status == 'success':
            print(f"优化成功！")
            display_optimization_results(sample_idx, sample, optimized_params, test_y[sample_idx], opt_pred, top_indices, scaler)
        else:
            print(f"优化失败!")
            print(f"优化前的力学性能: {test_y[sample_idx]}")
            print(f"优化后的力学性能: {opt_pred}")
            print(f"目标阈值: {Config.TARGETS}")
    
    # 2. 批量优化
    print("\n是否进行批量优化？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        print("\n开始批量优化不合格样本...")
        results_df = optimize_many(test_x, test_y, train_x_sample, model, explainer, scaler)
        
        # 3. 分析结果
        success_count = (results_df['status'] == 'success').sum()
        total_count = len(results_df)
        print(f"\n优化结果分析:")
        print(f"总样本数: {total_count}")
        print(f"成功优化: {success_count} ({success_count/total_count:.1%})")
        print(f"优化结果已保存到: {Config.RESULT_FILE}")
        
        # 4. 展示成功的案例
        success_samples = results_df[results_df['status'] == 'success']
        if len(success_samples) > 0:
            print("\n成功案例展示:")
            sample_count = min(3, len(success_samples))
            for i, (_, row) in enumerate(success_samples.head(sample_count).iterrows()):
                print(f"\n成功案例 {i+1}:")
                print(f"样本ID: {row['sample_id']}")
                print(f"优化前性能: {row['original_performance']}")
                print(f"优化后性能: {row['optimized_performance']}")
                print(f"关键参数: {row['top_paras']}")
                
                if row['param_changes']:
                    print("参数变化:")
                    for change in row['param_changes']:
                        print(f"  {change['param_name']}: {change['ori_val']:.4f} -> {change['opt_val']:.4f} ({change['change_pct']:.2f}%)")
    else:
        print("跳过批量优化。")
    
    print("\n程序执行完毕！")
if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.data import Dataset 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")


pre_model_path = './model/'
train_data_path = './data/train_data.xlsx'
valid_data_path = './data/valid_data.xlsx'
test_data_path = './data/test_data.xlsx'

scaled_data_path = './data/scaled_data.pkl'#标准化的训练集数据
scaler_model_path = './model/scaler.pkl'#标准化的训练集数据

train_data = pd.read_excel(train_data_path, skiprows=0)
# valid_data = pd.read_excel(valid_data_path, skiprows=0)
# test_data = pd.read_excel(test_data_path, skiprows=0)


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
RF= RandomForestRegressor(n_estimators=100, random_state=42)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5) # 设置hue loss使其对异常值具有鲁棒性
model_xgb = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.05,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.8, n_estimators=2200,
                             reg_alpha=0.5, reg_lambda=0.8,
                             subsample=0.5, random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, min_child_samples=20,
                              min_child_weight=0.001, verbosity=-1)

#多输出回归
multi_lasso = MultiOutputRegressor(lasso)
multi_KRR = MultiOutputRegressor(KRR)
multi_RF = MultiOutputRegressor(RF)
multi_GBoost = MultiOutputRegressor(GBoost)
multi_model_xgb = MultiOutputRegressor(model_xgb)
multi_model_lgb = MultiOutputRegressor(model_lgb)


# models = {'Lasso': lasso,'Kernel Ridge': KRR, 'Random Forest': RF,
#           'Gradient Boosting': GBoost, 'XGBoost': model_xgb, 'LightGBM': model_lgb}

models = {'Lasso': multi_lasso,'Kernel Ridge': multi_KRR, 'Random Forest': multi_RF,
            'Gradient Boosting': multi_GBoost, 'XGBoost': multi_model_xgb, 'LightGBM': multi_model_lgb}

def x_y_split(data_dir,scaler):
    data = pd.read_excel(data_dir, skiprows=0)
    data = data.astype('float32')
    x = data.iloc[:, :-3]
    x_scaled = scaler.transform(x)
    y = data.iloc[:, -3:]
    y = y.values
    return x_scaled,y

 #读取训练集数据并标准化保存
def process_and_save_data():
    # 加载数据
    train_data = pd.read_excel(train_data_path)

    # 转为浮点型
    train_data = train_data.astype('float32')

    # 分离特征和目标变量
    train_x = train_data.iloc[:, :-3]

    # 对数据进行标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_x_scaled = scaler.fit_transform(train_x)

    # 保存处理后的数据和scaler
    train_x_scaled_df = pd.DataFrame(train_x_scaled, columns=train_x.columns)
    train_x_scaled_df.to_pickle(scaled_data_path)  # 保存为pickle格式，加载快
    joblib.dump(scaler, scaler_model_path)  # 保存scaler模型

    print("数据预处理完毕并已保存!")


#对数据进行标准化
if not os.path.exists(scaled_data_path) or not os.path.exists(scaler_model_path):
    print('未找到标准化数据或scaler模型，重新处理数据')
    process_and_save_data()
else:
    print('加载标准化数据和scaler模型')
scaler = joblib.load(scaler_model_path)


# valid_data=valid_data.astype('float32')
# valid_data_x=valid_data.iloc[:,:-3]
# valid_data_x_scaled = scaler.transform(valid_data_x)






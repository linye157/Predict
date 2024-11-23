import pandas as pd
import numpy as np

import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 

#生成训练集，测试集
class Train_Loader(Dataset):
    def __init__(
            self, 
            data_dir, 
            seq_length,
    ):
        self.data_dir = data_dir
        self.seq_length = seq_length
        data = pd.read_excel(self.data_dir, skiprows=0)
        #将数据类型转换为float
        data = data.astype('float64')
        #缺失值使用上一行和下一行同一列的平均值填充
        data = data.fillna(data.mean())
        #对目标列(除最后三列)归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        data.iloc[:, :-3] = scaler.fit_transform(data.iloc[:, :-3])
        self.data = data.values
        self.length=len(data)-seq_length+1
        self.scaler = scaler

    def __getitem__(self, i):
        features = self.data[i:i+self.seq_length, :-3]
        targets = self.data[i+self.seq_length-1:i+self.seq_length, -3:]
        return features, targets
        
    def __len__(self):
        return len(self.data)-self.seq_length+1 



class Test_Loader(Dataset):
    def __init__(
            self, 
            data_dir, 
            seq_length,
            scaler
    ):
        self.data_dir = data_dir
        self.seq_length = seq_length
        data = pd.read_excel(self.data_dir, skiprows=0)
        #将数据类型转换为float
        data = data.astype('float64')
        #缺失值使用上一行和下一行同一列的平均值填充
        data = data.fillna(data.mean())
        #对目标列(除最后三列)归一化
        data.iloc[:, :-3] = scaler.transform(data.iloc[:, :-3])
        self.data = data.values
        self.length=len(data)-seq_length+1

    def __getitem__(self, i):
        features = self.data[i:i+self.seq_length, :-3]
        targets = self.data[i+self.seq_length-1:i+self.seq_length, -3:]
        return features, targets
    def __len__(self):
        return len(self.data)-self.seq_length+1 

    


Train_data_dir = './data/train_data.xlsx'
valid_data_dir = './data/valid_data.xlsx'
seq_length = 8
Train_data_Loader = Train_Loader(data_dir=Train_data_dir, seq_length=seq_length)
valid_data_Loader = Test_Loader(data_dir=valid_data_dir, seq_length=seq_length, scaler=Train_data_Loader.scaler)
#查看数据集
# for i in range(10):
#     print(Train_data_Loader[i][0].shape, Train_data_Loader[i][1].shape)
#     print(Train_data_Loader[i][1])
#     print(valid_data_Loader[i][0].shape, valid_data_Loader[i][1].shape)

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
        return out[:, -1, :].unsqueeze(1)
    
# LSTMTransformerRegressor参数设置
feature_size = 30  # 特征数
hidden_size = 128  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
num_heads = 8  # Transformer的头数
num_targets = 3  # 目标值数量

batchsize=512
#创建数据加载器
train_data = DataLoader(Train_data_Loader, batch_size=batchsize, shuffle=True)
val_data = DataLoader(valid_data_Loader, batch_size=batchsize, shuffle=True)
# #查看数据加载器
# for i, (features, targets) in enumerate(train_data):
#     print(i, features.shape, targets.shape)
#     break


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name='LSTMTransformer'

# 创建模型
model = LSTMTransformerRegressor(feature_size, hidden_size, num_layers, num_heads, num_targets)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model_path=f'./model/{model_name}-all+{hidden_size}+{seq_length}.pt'


#计算验证集的损失
def valid(data_loader,model):
    model.eval()
    loss_v=0
    for batch_X, batch_y in data_loader:
        # 将数据移动到GPU上
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # 前向传播
        model=model.to(device)
        batch_X =batch_X.float()
        batch_y =batch_y.float()
        outputs = model(batch_X)
        # 计算损失
        loss_v += criterion(outputs, batch_y).item()
    return loss_v/len(data_loader)

if __name__=='__main__':
    # 训练模型
    num_epochs = 1000
    min_loss = 300
    try:
        model=torch.load(model_path)
        print('load model success')
    except:
        print('load model fail')
        
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_data:
            #训练模式
            model.train()
            # print(batch_X.shape)
            # print(batch_y.shape)
            # 将数据移动到GPU上
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            model=model.to(device)
            batch_X =batch_X.float()
            batch_y =batch_y.float()
            outputs = model(batch_X)

            
            loss = criterion(outputs, batch_y)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

        loss_v=valid(val_data,model)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_v:.4f}')
        if loss_v<min_loss:
            min_loss=loss_v
            torch.save(model,model_path)
            print('save model success')


        

    print('Finished Training')
    print('Min loss: ', min_loss)
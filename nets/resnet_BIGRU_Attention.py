import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.W_omega = nn.Parameter(torch.Tensor(hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.Tensor(attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size, 1))
        
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W_omega)
        nn.init.xavier_uniform_(self.u_omega)
        nn.init.uniform_(self.b_omega)

    def forward(self, inputs):
        v = torch.tanh(torch.matmul(inputs, self.W_omega) + self.b_omega)
        vu = torch.matmul(v, self.u_omega)
        alphas = torch.softmax(vu, dim=1)
        output = torch.sum(inputs * alphas, dim=1)
        return output

class resnet_BiGRU_Attention(nn.Module):
    def __init__(self, n_future, n_class, hidden_size=20, attention_size=64):
        super(resnet_BiGRU_Attention, self).__init__()
        
        # 使用预训练的ResNet模型
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet = models.resnet50(pretrained=False)

        
        # 修改ResNet的第一个卷积层以适应单通道输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 冻结ResNet的参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 修改最后的全连接层
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, hidden_size * 2)  # 假设BiGRU的输入维度为hidden_size * 2
        
        # GRU layers
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = Attention(hidden_size * 2, attention_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, n_class)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 1, n_future, n_future)
        
        # 使用ResNet进行特征提取
        x = self.resnet(x)
        
        # BiGRU
        x, _ = self.gru(x.unsqueeze(1))
        output = x[:, -1, :]  # 取最后一个时间步的输出
        
        # Attention
        attn_output = self.attention(output.unsqueeze(1))
        
        # Fully connected layers
        x = self.relu(self.fc1(attn_output))
        x = self.fc2(x)
        
        return x.unsqueeze(1)

if __name__ == '__main__':
    n_future = 30  # Example value, adjust as needed
    n_class = 3   # Example value, adjust as needed

    model = resnet_BiGRU_Attention(n_future, n_class)
    input_tensor = torch.randn(512, 1, n_future, n_future)  # Example input tensor (batch_size, 1, n_future, n_future)
    output = model(input_tensor)
    print(output.shape)
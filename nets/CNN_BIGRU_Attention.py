import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CNN_BiGRU_Attention(nn.Module):
    def __init__(self, n_future, n_class, hidden_size=20, attention_size=64):
        super(CNN_BiGRU_Attention, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        
        # GRU layers
        self.gru = nn.GRU(16, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = Attention(hidden_size * 2, attention_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, n_class)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x shape: (batch_size, n_future, 1)
        # x = x.permute(0, 2, 1)  # (batch_size, 1, n_future)
        
        # CNN
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape for GRU
        x = x.permute(0, 2, 1)  # (batch_size, new_seq_len, features)
        
        # BiGRU
        _, hidden = self.gru(x)
        output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Attention
        attn_output = self.attention(output.unsqueeze(1))
        
        # Fully connected layers
        x = self.relu(self.fc1(attn_output))
        x = self.fc2(x)
        
        return x.unsqueeze(1)
    

if __name__ == '__main__':
    n_future = 30  # Example value, adjust as needed
    n_class = 3   # Example value, adjust as needed

    model = CNN_BiGRU_Attention(n_future, n_class)
    input_tensor = torch.randn(512, 1,n_future)  # Example input tensor (batch_size, n_future, 1)
    output = model(input_tensor)
    print(output.shape)

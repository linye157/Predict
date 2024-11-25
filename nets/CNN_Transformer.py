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

class CNN_Transformer(nn.Module):
    def __init__(self, n_future, n_class, hidden_size=20, attention_size=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(CNN_Transformer, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=16, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * (n_future // 4), 128)
        self.fc2 = nn.Linear(128, n_class)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x shape: (batch_size, n_future, 1)
        
        # CNN
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Reshape for Transformer
        x = x.permute(2, 0, 1)  # (new_seq_len, batch_size, features)
        
        # Create a target sequence for the decoder
        tgt = torch.zeros_like(x)
        
        # Transformer
        x = self.transformer(x, tgt)
        x = x.permute(1, 0, 2).contiguous().view(x.size(1), -1)  # (batch_size, new_seq_len * features)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x.unsqueeze(1)
    

if __name__ == '__main__':
    n_future = 30  # Example value, adjust as needed
    n_class = 3   # Example value, adjust as needed

    model = CNN_Transformer(n_future, n_class)
    input_tensor = torch.randn(512, 1, n_future)  # Example input tensor (batch_size, 1, n_future)
    output = model(input_tensor)
    print(output.shape)

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

T = config["data"]["T"]


class STGNNLayer(nn.Module):
    def __init__(self, K: int, in_channel: int, out_channel: int):
        super().__init__()
        self.K = K
        self.out_channel = out_channel
        self.weights = nn.Parameter(torch.randn(out_channel, in_channel, T) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channel))

    def forward(self, X: torch.Tensor, S: torch.Tensor):
        """
        X: [B, T, V, F] 
        S: [V, V]  
        """
        B, T, V, F = X.shape
        
        S = S.to(X.device).float()

        # Chuẩn bị tensor để lưu output
        outputs = []

        # t = 1 -> T
        for t in range(T):
            # Lấy dữ liệu tại thời điểm t
            x_t = X[:, t, :, :]          # shape: [B, V, F]
            # Lấy weights cho thời điểm t
            w_t = self.weights[:, :, t]   # shape: [out_channel, in_channel]

            # Reshape
            x_t_reshaped = x_t.reshape(B * V, F)           # [B*V, F]
            
            # [B * V, in_channel] * [in_channel, out_channel]
            h = x_t_reshaped @ w_t.T                       # [B*V, out_channel]
            
            # Đưa về shape gốc
            h = h.reshape(B, V, self.out_channel)          # [B, V, out_channel]

            # Thêm bias
            h = h + self.bias.view(1, 1, -1)

            outputs.append(h)

        # Ghép lại
        out = torch.stack(outputs, dim=1)   # shape: [B, T, V, out_channel]
        return out


if __name__ == "__main__":
    st_gnn_layer = STGNNLayer(K=3, in_channel=5, out_channel=64)
    
    X = torch.randn(64, 24, 202, 5)      # [B, T, V, F]
    S = torch.randint(0, 2, size=(202, 202)).float()   # Ma trận kề giả

    y = st_gnn_layer(X, S)
    
    print("Input shape :", X.shape)
    print("Output shape:", y.shape)      # Mong đợi: [64, 24, 202, 64]
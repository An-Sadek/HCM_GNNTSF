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
        assert T > K, "T phải lớn hơn K"
        
        self.K = K
        self.out_channel = out_channel
        self.weights = nn.Parameter(torch.randn(out_channel, in_channel, K) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channel))

    def forward(self, X: torch.Tensor, S: torch.Tensor):
        pass


if __name__ == "__main__":
    st_gnn_layer = STGNNLayer(K=3, in_channel=5, out_channel=64)
    
    X = torch.randn(64, 24, 202, 5)      # [B, T, V, F]
    S = torch.randint(0, 2, size=(202, 202)).float()   # Ma trận kề giả

    y = st_gnn_layer(X, S)
    
    print("Input shape :", X.shape)
    print("Output shape:", y.shape)      # [64, 24, 202, 64]

    # Định nghĩa
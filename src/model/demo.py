import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class STGNNLayer(nn.Module):
    """
    Một layer của Space-Time Graph Neural Network (ST-GNN)
    Dựa trên FIR space-time filters + pointwise activation.
    """
    def __init__(self, in_features: int, out_features: int, K: int = 4, bias: bool = True):
        super().__init__()
        self.K = K  # số filter taps (độ sâu thời gian)
        # h_{f g k} : learnable weights (out_features, in_features, K)
        self.weights = nn.Parameter(torch.randn(out_features, in_features, K) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, X: torch.Tensor, S_list: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        X: (batch_size, T, N, in_features) - space-time graph signals
        S_list: 
            - torch.Tensor (N, N) nếu đồ thị cố định
            - List[torch.Tensor] độ dài T, mỗi tensor (N, N) nếu đồ thị động theo thời gian
        Trả về: (batch_size, T, N, out_features)
        """
        B, T, N, F_in = X.shape
        F_out = self.weights.shape[0]
        device = X.device

        # Chuẩn bị S_list thành list để dễ xử lý
        if isinstance(S_list, torch.Tensor):
            S_list = [S_list] * T   # fixed graph
        
        assert len(S_list) == T, "S_list phải có độ dài bằng T"

        Y = torch.zeros(B, T, N, F_out, device=device)

        # Duyệt theo từng thời điểm t (causal)
        for t in range(T):
            for k in range(self.K):
                if t - k < 0:
                    continue  # chưa có dữ liệu quá khứ
                
                # x tại thời điểm t-k
                x_past = X[:, t - k, :, :]  # (B, N, F_in)
                
                # Tính product của K-1 đồ thị: prod_{m=1}^k S_{t-k+m}
                S_prod = torch.eye(N, device=device)  # (N, N)
                for m in range(k):
                    s_idx = t - k + m + 1
                    S_current = S_list[s_idx]
                    # S_prod = S_prod @ S_current  (matrix multiplication)
                    S_prod = torch.matmul(S_prod, S_current)
                
                # Áp dụng filter: broadcast batch
                # x_prop = S_prod @ x_past  => (B, N, F_in)
                x_prop = torch.matmul(S_prod, x_past)
                
                # Nhân weights h_k
                h_k = self.weights[:, :, k]  # (F_out, F_in)
                # y_k = x_prop @ h_k.T  => (B, N, F_out)
                y_k = torch.einsum('bif,of->bio', x_prop, h_k)
                
                Y[:, t] += y_k

        # Thêm bias (nếu có)
        if self.bias is not None:
            Y += self.bias.view(1, 1, 1, -1)

        return Y


class STGNN(nn.Module):
    """
    Toàn bộ mô hình ST-GNN (stack nhiều layer)
    """
    def __init__(self, feature_dims: list[int], K: int = 4, activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(len(feature_dims) - 1):
            self.layers.append(STGNNLayer(feature_dims[i], feature_dims[i+1], K=K))

    def forward(self, X: torch.Tensor, S_list: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, S_list)
            X = self.activation(X)
        return X


if __name__ == "__main__":
    # Giả sử: batch=2, T=50 thời điểm, N=10 nodes, input features=6 (position, velocity...)
    batch_size, T, N, F_in = 2, 50, 10, 6
    X = torch.randn(batch_size, T, N, F_in)          # input signals
    
    # Đồ thị động (hoặc fixed)
    S_list = [torch.rand(N, N) for _ in range(T)]   # random adjacency (thay bằng A hoặc normalized Laplacian)
    S_list = [S / (S.sum(dim=1, keepdim=True) + 1e-8) for S in S_list]  # normalize nếu cần
    
    model = STGNN(feature_dims=[6, 64, 32, 2], K=4)  # input 6 → hidden 64 → 32 → output 2
    output = model(X, S_list)                        # (batch, T, N, 2)
    
    print(output.shape)
    print([3]*3)
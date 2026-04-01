import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm

# Set seed cho nhất quán
torch.manual_seed(42)

# lOAD CONFIG
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

STATIC_PATH = config["data"]["static_path"]
DYNAMIC_PATH = config["data"]["dynamic_path"]
PREPROCESS_PATH = config["data"]["preprocess_path"]

SPLIT_RATIO = config["data"]["split_ratio"]
TRAIN_RATIO = SPLIT_RATIO[0]
VAL_RATIO = SPLIT_RATIO[1]
TEST_RATIO = SPLIT_RATIO[2]

BATCH_SIZE = config["train"]["batch_size"]
EPOCH = config["train"]["epoch"]

# === LOAD DATA
# Load đồ thị, không gian đặc trưng tĩnh
with h5py.File(H5_PATH, 'r') as f:
    graph = torch.from_numpy(f["edge_index"][:]) # Cấu trúc đồ thị (2, V) = (2, 202)
    static = torch.from_numpy(f["static"][:]) # Không gian đặc trưng tĩnh (V, F) = (202, 55)

# Load không gian đặc trưng động
class HCM_Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        # Lấy length = total_time
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f["X"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            X = f["X"][idx] # (samples, past, V, 5) = (13961, 24, 202, 5)
            y = f["y"][idx] # (samples, future, V) = (13961, 24, 202)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y
        
# Tạo dataset
dataset = HCM_Dataset(h5_path=H5_PATH)
print("Kích thước của X[0]:")
print(dataset[0][0].shape) # Shape X
print("\nKích thước y[0]:")
print(dataset[0][1].shape) # Shape y

# Chia tập dữ liệu
train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
print("\nChiều dài không gian đặc trưng động:")
print(f"\tTrain: {len(train_dataset)}")
print(f"\tVal:   {len(val_dataset)}")
print(f"\tTest:  {len(test_dataset)}")

# Load DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
# === END
class TemporalGCN(nn.Module):
    def __init__(self, 
                 node_features=5,
                 static_features=55,
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.2,
                 future_steps=24):
        super().__init__()
        
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        
        # === Embedding cho đặc trưng tĩnh (Static Features)
        self.static_embed = nn.Linear(static_features, hidden_dim)
        
        # === GNN layers (Spatial modeling)
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(node_features + hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # === Temporal modeling
        self.temporal = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # === Multi-step decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, future_steps)  # dự báo 24 bước
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, y=None, graph=None, static=None):
        batch_size, past_steps, num_nodes, feat_dim = X.shape
        
        # 1. Static embedding
        static_emb = self.static_embed(static)  # (V, hidden)
        static_emb = static_emb.unsqueeze(0).unsqueeze(0)
        static_emb = static_emb.expand(batch_size, past_steps, -1, -1)
        
        # 2. Combine dynamic + static
        x = torch.cat([X, static_emb], dim=-1)  # (B, past, V, feat+hidden)
        
        # 3. Spatial GCN per timestep
        spatial_out = []
        
        for t in range(past_steps):
            xt = x[:, t]  # (B, V, feat)
            out_batch = []
            
            for b in range(batch_size):
                xb = xt[b]  # (V, feat)
                
                for gcn in self.gcn_layers:
                    xb = F.relu(self.dropout(gcn(xb, graph)))
                
                out_batch.append(xb)
            
            xt_out = torch.stack(out_batch, dim=0)  # (B, V, hidden)
            spatial_out.append(xt_out)
        
        h = torch.stack(spatial_out, dim=1)  # (B, past, V, hidden)
        
        # 4. Temporal modeling per node
        h = h.permute(0, 2, 1, 3)  # (B, V, past, hidden)
        h = h.reshape(batch_size * num_nodes, past_steps, self.hidden_dim)
        
        lstm_out, _ = self.temporal(h)
        temporal_emb = lstm_out[:, -1, :]  # (B*V, hidden)
        
        # 5. Multi-step decoding
        out = self.decoder(temporal_emb)  # (B*V, future_steps)
        out = out.reshape(batch_size, num_nodes, self.future_steps)
        out = out.permute(0, 2, 1)  # (B, future_steps, V)
        
        return out

# === Training setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TemporalGCN(
    node_features=5,
    static_features=55,
    hidden_dim=64,
    num_layers=2,
    future_steps=24
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# graph và static (global)
edge_index = graph.to(device)        # (2, num_edges)
static_feat = static.to(device)      # (202, 55)

print("Bắt đầu training GNN...")

for epoch in tqdm(range(EPOCH), desc="Đang huấn luyện GNN"):  # Bạn có thể tăng số epoch
    model.train()
    train_loss = 0.0
    
    for batch_idx, (X_batch, y_batch) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc="Batch", 
        leave=False
    ):
        X_batch = X_batch.to(device)      # (B, 24, 202, 5)
        y_batch = y_batch.to(device)      # (B, 24, 202)   ← future 24 steps
        
        optimizer.zero_grad()
        
        # Hiện tại mô hình chỉ dự báo 1 bước, ta lấy bước đầu tiên của y để train
        # (Bạn có thể cải tiến sau)
        y_true = y_batch.squeeze(-1)         # (B, 24, 202) - bước tương lai đầu tiên
        
        pred = model(X_batch, graph=edge_index, static=static_feat)
        
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_loader):.6f}")
    
    # Validation (tương tự)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.squeeze(-1) 
            
            pred_val = model(X_val, graph=edge_index, static=static_feat)
            val_loss += criterion(pred_val, y_val).item()
    
    print(f"          | Val Loss:   {val_loss/len(val_loader):.6f}\n")
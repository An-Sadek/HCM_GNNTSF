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


if __name__ == "__main__":
    pass
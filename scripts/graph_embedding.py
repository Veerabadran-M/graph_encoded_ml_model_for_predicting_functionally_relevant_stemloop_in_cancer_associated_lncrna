from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
import pandas as pd
import torch.nn.functional as F

class GINEEncoder(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim, num_layers=3):
        super(GINEEncoder, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(GINEConv(nn.Linear(in_channels, hidden_dim), edge_dim=edge_dim))
            else:
                self.layers.append(GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=edge_dim))

        self.output_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return x

class GraphEmbeddingModel(nn.Module):
    def __init__(self, in_channels, edge_dim, graph_attr_dim, hidden_dim=64, out_dim=64):
        super(GraphEmbeddingModel, self).__init__()
        self.encoder = GINEEncoder(in_channels, edge_dim, hidden_dim)
        total_input = self.encoder.output_dim + graph_attr_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch
        graph_emb = self.encoder(x, edge_index, edge_attr, batch)

        graph_attr = data.graph_attributes
        if len(graph_attr.shape) == 1:
            graph_attr = graph_attr.unsqueeze(0)
        graph_attr = graph_attr.float()

        if graph_emb.size(0) != graph_attr.size(0):
            graph_attr = graph_attr.repeat(graph_emb.size(0), 1)

        concat = torch.cat([graph_emb, graph_attr], dim=1)
        out = self.mlp(concat)
        return out

def train_model(model, dataloader, num_epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        start_epoch = time.time()
        total_loss = 0

        for data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="graph"):
            data = data.to(device)
            optimizer.zero_grad()
            emb = model(data)

            graph_attr = data.graph_attributes
            if len(graph_attr.shape) == 1:
                graph_attr = graph_attr.unsqueeze(0)
            graph_attr = graph_attr.float()
            if graph_attr.size(1) < emb.size(1):
                graph_attr = F.pad(graph_attr, (0, emb.size(1)-graph_attr.size(1)))
            elif graph_attr.size(1) > emb.size(1):
                graph_attr = graph_attr[:, :emb.size(1)]

            loss = loss_fn(emb, graph_attr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} finished. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

cancer_graphs_path = 'cancer graph path'
non_cancer_graphs_path = 'non cancer graph path'

cancer_dataset = torch.load(cancer_graphs_path, weights_only=False)
non_cancer_dataset = torch.load(non_cancer_graphs_path, weights_only=False)

dataset = cancer_dataset + non_cancer_dataset

loader = DataLoader(dataset, batch_size=1)

import torch
torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphEmbeddingModel(in_channels=6, edge_dim=4, graph_attr_dim=8, hidden_dim=64, out_dim=32).to(device)

train_model(model, loader, num_epochs=10)

model.eval()
records = []
for d in dataset:
    d = d.to(device)
    with torch.no_grad():
        vec = model(d).cpu().numpy().flatten()
    records.append({
        'symbol': d.symbol,
        'id': d.id,
        'cancer_association': d.cancer_association,
        'embedded_vector': vec.tolist()
    })

df = pd.DataFrame(records)
df.to_excel("desired output path", index=False)
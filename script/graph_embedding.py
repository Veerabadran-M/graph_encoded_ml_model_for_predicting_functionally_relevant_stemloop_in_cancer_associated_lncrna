from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pandas as pd
import os
from pathlib import Path

# Model
class GINEEncoder(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GINEConv(nn.Linear(in_channels, hidden_dim), edge_dim=edge_dim)
                )
            else:
                self.layers.append(
                    GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=edge_dim)
                )

        self.output_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()

        for conv in self.layers:
            x = torch.relu(conv(x, edge_index, edge_attr))

        return global_mean_pool(x, batch)


class GraphEmbeddingModel(nn.Module):
    def __init__(self, in_channels, edge_dim, graph_attr_dim,
                 hidden_dim=64, out_dim=32):
        super().__init__()

        self.encoder = GINEEncoder(in_channels, edge_dim, hidden_dim)
        total_input = self.encoder.output_dim + graph_attr_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        graph_emb = self.encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        graph_attr = data.graph_attributes.float()
        if len(graph_attr.shape) == 1:
            graph_attr = graph_attr.unsqueeze(0)

        if graph_emb.size(0) != graph_attr.size(0):
            graph_attr = graph_attr.repeat(graph_emb.size(0), 1)

        concat = torch.cat([graph_emb, graph_attr], dim=1)
        return self.mlp(concat)

# Training Function
def train_model(model, dataloader, device,
                num_epochs=10, lr=1e-3):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(num_epochs):
        start = time.time()
        total_loss = 0

        for data in tqdm(dataloader,
                         desc=f"Epoch {epoch+1}/{num_epochs}",
                         unit="graph"):

            data = data.to(device)

            optimizer.zero_grad()
            emb = model(data)

            graph_attr = data.graph_attributes.float()
            if len(graph_attr.shape) == 1:
                graph_attr = graph_attr.unsqueeze(0)

            if graph_attr.size(1) < emb.size(1):
                graph_attr = F.pad(graph_attr,
                                   (0, emb.size(1)-graph_attr.size(1)))
            elif graph_attr.size(1) > emb.size(1):
                graph_attr = graph_attr[:, :emb.size(1)]

            loss = loss_fn(emb, graph_attr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | "
              f"Loss: {total_loss/len(dataloader):.4f} | "
              f"Time: {time.time()-start:.2f}s")

# Main Function
def graph_embedding(graph_paths, output_path='data/results/graph_embedded_vectors.xlsx',
                    in_channels=6,
                    edge_dim=4,
                    graph_attr_dim=8,
                    hidden_dim=64,
                    out_dim=32,
                    num_epochs=10):

    # Allow single path or list
    if isinstance(graph_paths, str):
        graph_paths = [graph_paths]

    # Load graphs
    dataset = []
    for path in graph_paths:
        dataset.extend(torch.load(path, weights_only=False))

    loader = DataLoader(dataset, batch_size=1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GraphEmbeddingModel(
        in_channels,
        edge_dim,
        graph_attr_dim,
        hidden_dim,
        out_dim
    ).to(device)

    # Train
    train_model(model, loader, device, num_epochs=num_epochs)

    # Extract embeddings
    model.eval()
    records = []

    for d in dataset:
        d = d.to(device)

        with torch.no_grad():
            vec = model(d).cpu().numpy().flatten()

        records.append({
            "symbol": d.symbol,
            "id": d.id,
            "cancer_association": d.cancer_association,
            "embedded_vector": vec.tolist()
        })

    df = pd.DataFrame(records)
    os.makedirs("data/results", exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"\nEmbeddings saved to: {output_path}")
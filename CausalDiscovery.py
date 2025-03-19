import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("stainless_steel_energy.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
df.rename(columns={"value": "power_consumption"}, inplace=True)

# Drop irrelevant or empty columns
columns_to_exclude = ['heattreatment_temperatures', 'forming_temperatures', 'dimension', 'input_weight', 'mes_datetime']
df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

# Convert numeric values (handling European format)
numeric_columns = ['weight', 'power_consumption', 'height']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Drop non-numeric categorical variables
df_numeric = df.select_dtypes(include=[np.number])

# Fill missing values with median
df_numeric = df_numeric.fillna(df_numeric.median())

# Normalize data
scaler = StandardScaler()
data = scaler.fit_transform(df_numeric)
data_tensor = torch.tensor(data, dtype=torch.float32)
input_dim = data.shape[1]

# Define Neural Network Model for Causal Discovery
class NotearsMLP(nn.Module):
    def __init__(self, dims):
        super(NotearsMLP, self).__init__()
        self.dims = dims
        self.W = nn.Parameter(torch.randn(dims, dims) * 0.2)  # Increased weight initialization
    
    def forward(self, x):
        return torch.relu(x @ (torch.eye(self.dims) + self.W))  # Apply ReLU to strengthen relationships

# DAG Constraint Function
def dag_constraint(W):
    M = W * W  # Hadamard product
    return torch.trace(torch.matrix_exp(M)) - M.shape[0]

# Train Model
model = NotearsMLP(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)  # Adjusted learning rate decay step
best_loss = float('inf')
patience_counter = 0
max_patience = 20  # Reduced patience to prevent excessive shrinking

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(data_tensor)
    reconstruction_loss = torch.norm(output - data_tensor)
    dag_loss = 0.000005 * dag_constraint(model.W)  # Further reduced DAG constraint weight
    l1_reg = 0.0001 * (1 - epoch / 1000) * torch.norm(model.W, p=1)  # Dynamic L1 regularization
    loss = reconstruction_loss + dag_loss + l1_reg
    
    if torch.isnan(loss):
        print(f"Epoch {epoch}: NaN loss detected! Stopping training.")
        break
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        print(f"Sample W values:\n{model.W.detach().numpy()}")
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Extract adjacency matrix
adj_matrix = model.W.detach().numpy()
adj_matrix = np.nan_to_num(adj_matrix)

# Adaptive thresholding for edge detection
thresh = max(0.0005, np.percentile(np.abs(adj_matrix), 95))

# Visualize adjacency matrix in a pop-up window
plt.figure(figsize=(10, 8))
sns.heatmap(adj_matrix, annot=True, cmap='coolwarm', xticklabels=df_numeric.columns, yticklabels=df_numeric.columns)
plt.title("Causal Adjacency Matrix")
plt.tight_layout()
plt.show()

# Create causal graph
G = nx.DiGraph()
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if abs(adj_matrix[i, j]) > thresh:
            G.add_edge(i, j, weight=adj_matrix[i, j])

# Draw Causal Graph with Edge Weights if edges exist
if G.number_of_edges() > 0:
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10)
    edge_labels = {(i, j): f"{G[i][j]['weight']:.3f}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Causal Graph with Edge Weights")
    plt.show()

# Feature Importance Analysis
importance_scores = np.abs(adj_matrix).sum(axis=0)
importance_df = pd.DataFrame({'Feature': df_numeric.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance in a pop-up window
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance Based on Causal Graph')
plt.tight_layout()
plt.show()

print("Causal analysis complete. Visualizations displayed.")

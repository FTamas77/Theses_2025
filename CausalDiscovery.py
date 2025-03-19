# Import necessary libraries
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
df.rename(columns={"value": "power_consumption"}, inplace=True)  # Rename column for clarity

# Drop irrelevant or empty columns
columns_to_exclude = ['heattreatment_temperatures', 'forming_temperatures', 'dimension', 'input_weight', 'mes_datetime']
df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])

# Convert numeric values (handling European number format with commas)
numeric_columns = ['weight', 'power_consumption', 'height']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Drop non-numeric categorical variables
df_numeric = df.select_dtypes(include=[np.number])

# Fill missing values with median (robust against outliers)
df_numeric = df_numeric.fillna(df_numeric.median())

# Normalize data using StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(df_numeric)
data_tensor = torch.tensor(data, dtype=torch.float32)  # Convert to tensor for PyTorch
input_dim = data.shape[1]  # Number of features (columns)

# Define Neural Network Model for Causal Discovery
class NotearsMLP(nn.Module):
    def __init__(self, dims):
        super(NotearsMLP, self).__init__()
        self.dims = dims
        self.W = nn.Parameter(torch.randn(dims, dims) * 0.3)  # Slightly larger initial weights for better signal

    def forward(self, x):
        return torch.relu(x @ (torch.eye(self.dims) + self.W))  # Apply ReLU for stronger relationships

# DAG Constraint Function
def dag_constraint(W):
    """ 
    Enforces a DAG (Directed Acyclic Graph) constraint by penalizing cycles.
    The idea is to use the trace of matrix exponential.
    """
    M = W * W  # Hadamard product (element-wise multiplication)
    return torch.trace(torch.matrix_exp(M)) - M.shape[0]  # Trace-based constraint

# Initialize Model
model = NotearsMLP(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.005)  # Slightly higher learning rate for faster convergence
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)  # Smoother learning rate decay

best_loss = float('inf')
patience_counter = 0
max_patience = 50  # Increase patience for better convergence

# Training Loop
for epoch in range(1500):  # Increase epochs for better learning
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data_tensor)
    
    # Reconstruction loss (ensuring output is close to input)
    reconstruction_loss = torch.norm(output - data_tensor)
    
    # DAG constraint loss (ensuring a directed acyclic structure)
    dag_loss = 0.00005 * dag_constraint(model.W)  # Increased weight slightly
    
    # L1 Regularization (to encourage sparsity in the adjacency matrix)
    l1_reg = 0.00005 * (1 - epoch / 1500) * torch.norm(model.W, p=1)  # Reduced L1 effect for better signal retention
    
    # Total loss
    loss = reconstruction_loss + dag_loss + l1_reg
    
    # Check for NaN loss
    if torch.isnan(loss):
        print(f"Epoch {epoch}: NaN loss detected! Stopping training.")
        break

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Print progress every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        print(f"Sample W values:\n{model.W.detach().numpy()}")

    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Extract adjacency matrix (causal relationships)
adj_matrix = model.W.detach().numpy()
adj_matrix = np.nan_to_num(adj_matrix)  # Replace NaN values with zeros

# Adjust threshold dynamically to keep significant causal links
thresh = max(0.0002, np.percentile(np.abs(adj_matrix), 90))  # Lower threshold for more edges

# Visualize adjacency matrix (heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(adj_matrix, annot=True, cmap='coolwarm', xticklabels=df_numeric.columns, yticklabels=df_numeric.columns)
plt.title("Causal Adjacency Matrix")
plt.tight_layout()
plt.show()

# Create causal graph from adjacency matrix
G = nx.DiGraph()
column_names = df_numeric.columns.tolist()
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if abs(adj_matrix[i, j]) > thresh:  # Only keep meaningful edges
            G.add_edge(column_names[i], column_names[j], weight=adj_matrix[i, j])

# Draw causal graph if edges exist
if G.number_of_edges() > 0:
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # Position nodes for better visualization
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', 
            edge_color='gray', font_size=10)
    
    # Add edge labels
    edge_labels = {(i, j): f"{G[i][j]['weight']:.3f}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Causal Graph with Edge Weights")
    plt.show()
else:
    print("No significant causal relationships detected.")

# Feature Importance Analysis
importance_scores = np.abs(adj_matrix).sum(axis=0)  # Sum of absolute weights per feature
importance_df = pd.DataFrame({'Feature': df_numeric.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance Based on Causal Graph')
plt.tight_layout()
plt.show()

print("Causal analysis complete. Visualizations displayed.")

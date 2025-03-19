import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx

# Load dataset
df = pd.read_csv("stainless_steel_energy.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
df.rename(columns={"value": "power_consumption"}, inplace=True)

# Print raw data info
print(f"Raw DataFrame shape: {df.shape}")
print(f"Raw DataFrame columns: {df.columns.tolist()}")
print(f"Missing values per column:\n{df.isnull().sum()}")

# Fix numeric values (European format)
numeric_columns = ["input_weight", "weight", "forming_temperatures", "heattreatment_temperatures", "power_consumption"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Instead of dropping rows with NaN, fill them with median values
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())

print(f"DataFrame shape after handling missing values: {df.shape}")

# Prepare numeric data for processing
datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
if datetime_cols:
    print(f"Dropping datetime columns: {datetime_cols}")
    df_numeric = df.drop(columns=datetime_cols)
else:
    df_numeric = df.copy()

# Select only numeric columns for modeling
numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
df_numeric = df_numeric[numeric_cols]
print(f"Using numeric columns: {df_numeric.columns.tolist()}")
print(f"Numeric DataFrame shape: {df_numeric.shape}")

# Initialize StandardScaler and apply if data exists
if df_numeric.shape[0] > 0 and df_numeric.shape[1] > 0:
    scaler = StandardScaler()
    data = scaler.fit_transform(df_numeric)
    print(f"Scaled data shape: {data.shape}")
    
    # Convert to PyTorch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Define Neural Network for Causal Discovery with appropriate dimensions
    input_dim = data.shape[1]
    print(f"Input dimension: {input_dim}")
    
    class NotearsMLP(nn.Module):
        def __init__(self, dims):
            super(NotearsMLP, self).__init__()
            self.dims = dims
            # Create direct weights between variables (square adjacency matrix)
            self.W = nn.Parameter(torch.zeros(dims, dims))
        
        def forward(self, x):
            # Apply learned DAG structure and non-linear transformation
            return x @ (torch.eye(self.dims) + self.W)
    
    # DAG Constraint Function - now using a square matrix
    def dag_constraint(W):
        # Make sure W is a square matrix
        M = W * W  # Hadamard product
        return torch.trace(torch.matrix_exp(M)) - M.shape[0]
    
    # Train Neural Network
    model = NotearsMLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model weight shape: {model.W.shape}")
    
    for epoch in range(1000):
        optimizer.zero_grad()
        # The W matrix directly represents our DAG adjacency matrix
        loss = torch.norm(model(data_tensor) - data_tensor) + 0.1 * dag_constraint(model.W)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    # Extract causal adjacency matrix
    adj_matrix = model.W.detach().numpy()
    
    # Print adjacency matrix information for debugging
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Adjacency matrix contains NaN: {np.isnan(adj_matrix).any()}")
    print(f"Adjacency matrix contains Inf: {np.isinf(adj_matrix).any()}")
    
    # Replace any potential problematic values
    adj_matrix = np.nan_to_num(adj_matrix)
    
    # Create a more explicit graph construction
    G = nx.DiGraph()
    
    # Add nodes to the graph explicitly
    for i in range(adj_matrix.shape[0]):
        G.add_node(i)
    
    # Add edges based on adjacency matrix with threshold
    threshold = 0.1  # Only consider strong enough connections
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if abs(adj_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=adj_matrix[i, j])
    
    # Create node labels safely
    node_labels = {i: f"{str(col)[:10]}" for i, col in enumerate(df_numeric.columns)}
    
    plt.figure(figsize=(10, 8))
    
    try:
        # Use spring_layout to explicitly calculate positions
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)
        
        # Draw labels separately
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        
        plt.title("Neural Network-Based Discovered Causal Graph")
        plt.axis('off')
    except Exception as e:
        print(f"Error during graph drawing: {e}")
        print("Attempting to use a simpler graph visualization method...")
        
        # Fallback to a simpler visualization method
        plt.figure(figsize=(10, 8))
        plt.imshow(adj_matrix, cmap='coolwarm')
        plt.colorbar(label="Causal Strength")
        plt.title("Causal Adjacency Matrix Heatmap")
        plt.xticks(range(len(node_labels)), [node_labels[i] for i in range(len(node_labels))], rotation=90)
        plt.yticks(range(len(node_labels)), [node_labels[i] for i in range(len(node_labels))])
    
    plt.tight_layout()
    plt.show()
    
    # Print graph statistics
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Debug information
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    print(f"Number of nodes in graph: {G.number_of_nodes()}")
    print(f"Number of columns in df_numeric: {len(df_numeric.columns)}")
else:
    print("ERROR: Not enough data for causal discovery.")
    print(f"DataFrame shape: {df.shape}")
    print("First few rows of original data:")
    print(df.head())

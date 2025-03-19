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
        # Initialize with larger weights to prevent convergence to zero
        self.W = nn.Parameter(torch.randn(dims, dims) * 0.8)
                
    def forward(self, x):
        # Use sigmoid to better preserve small relationships
        return torch.sigmoid(x @ (torch.eye(self.dims) + self.W))

# DAG Constraint Function
def dag_constraint(W):
    """ 
    Enforces a DAG (Directed Acyclic Graph) constraint by penalizing cycles.
    The idea is to use the trace of matrix exponential.
    """
    M = W * W  # Hadamard product (element-wise multiplication)
    return torch.trace(torch.matrix_exp(M)) - M.shape[0]  # Trace-based constraint

# Remove domain knowledge constraints section
print("Running causal discovery without domain constraints")

# Initialize Model
model = NotearsMLP(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate for better signal
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.7)

best_loss = float('inf')
patience_counter = 0
max_patience = 30  # Slightly less patience for more aggressive stopping

# Training Loop
for epoch in range(800):  # Fewer epochs but more effective learning
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data_tensor)
    
    # Reconstruction loss (ensuring output is close to input)
    reconstruction_loss = torch.norm(output - data_tensor)
    
    # DAG constraint loss (ensuring a directed acyclic structure) - reduced weight
    dag_loss = 0.00001 * dag_constraint(model.W)
    
    # L1 Regularization (reduced to allow more non-zero connections)
    l1_reg = 0.00001 * torch.norm(model.W, p=1)
    
    # Total loss
    loss = reconstruction_loss + dag_loss + l1_reg
    
    # Check for NaN loss
    if torch.isnan(loss):
        print(f"Epoch {epoch}: NaN loss detected! Stopping training.")
        break

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # Print progress every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        non_zero = torch.sum(torch.abs(model.W) > 0.01).item()
        print(f"Non-zero connections (>0.01): {non_zero}")
        print(f"Sample W values:\n{model.W.detach().numpy()}")

    # Early stopping logic with stricter criteria
    if loss.item() < best_loss * 0.995:  # Must improve by at least 0.5%
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Extract adjacency matrix and apply statistical significance test
adj_matrix = model.W.detach().numpy()
adj_matrix = np.nan_to_num(adj_matrix)

# Calculate bootstrap samples to estimate confidence intervals
n_samples = data.shape[0]
n_bootstrap = 100
bootstrap_matrices = []

print("Calculating statistical significance via bootstrap...")
for _ in range(n_bootstrap):
    # Sample with replacement
    indices = np.random.choice(n_samples, n_samples, replace=True)
    bootstrap_data = data[indices, :]
    bootstrap_tensor = torch.tensor(bootstrap_data, dtype=torch.float32)
    
    # Get output from model
    with torch.no_grad():
        output = model(bootstrap_tensor)
        # Estimate weights by comparing input and output
        bootstrap_W = torch.autograd.functional.jacobian(
            lambda x: model(x).sum(dim=0), 
            bootstrap_tensor.mean(dim=0).unsqueeze(0)
        ).squeeze()
        
    bootstrap_matrices.append(bootstrap_W.numpy())

# Calculate standard deviation across bootstrap samples
bootstrap_std = np.std(np.array(bootstrap_matrices), axis=0)
# Calculate z-scores
z_scores = np.abs(adj_matrix) / (bootstrap_std + 1e-10)

# Apply significance threshold (|z| > 1.96 for p < 0.05)
significant_matrix = np.where(z_scores > 1.96, adj_matrix, 0)

# Use this for further analysis instead of raw adjacency matrix
adj_matrix = significant_matrix

# Apply a lower threshold for more meaningful connections
# Use a dynamic threshold based on standard deviation
thresh = np.std(np.abs(adj_matrix)) * 0.5  # Lower threshold to capture more relationships

# Visualize adjacency matrix (heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(adj_matrix, annot=True, cmap='coolwarm', xticklabels=df_numeric.columns, yticklabels=df_numeric.columns)
plt.title("Causal Adjacency Matrix")
plt.tight_layout()
plt.savefig("causal_adjacency_matrix.png")  # Save figure for later reference
plt.show()

# Create causal graph from adjacency matrix
G = nx.DiGraph()
column_names = df_numeric.columns.tolist()
edge_weights = []  # Store weights for edge width scaling

for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if abs(adj_matrix[i, j]) > thresh:  # Only keep meaningful edges
            G.add_edge(column_names[i], column_names[j], weight=adj_matrix[i, j])
            edge_weights.append(abs(adj_matrix[i, j]))

# Draw causal graph if edges exist
if G.number_of_edges() > 0:
    plt.figure(figsize=(12, 10))
    
    # Use hierarchical layout for clearer causal flow
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G, seed=42)  # Fallback if graphviz not available
    
    # Scale node sizes by out-degree + in-degree (total causal importance)
    node_sizes = [300 + 100 * (G.out_degree(node) + G.in_degree(node)) for node in G]
    
    # Create edge colors based on weight (red for negative, blue for positive)
    edge_colors = ["red" if G[u][v]['weight'] < 0 else "blue" for u, v in G.edges]
    
    # Normalize edge widths
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        edge_widths = [1 + 5 * ((abs(G[u][v]['weight']) - min_weight) / (max_weight - min_weight + 1e-10)) 
                     for u, v in G.edges]
    else:
        edge_widths = [1] * len(G.edges)
    
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, 
            node_color='lightblue', edge_color=edge_colors, 
            font_size=10, width=edge_widths)
    
    # Add edge labels with cleaner formatting
    edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Causal Graph with Edge Weights")
    plt.savefig("causal_graph.png")
    plt.show()
else:
    print("No significant causal relationships detected.")

# Feature Importance Analysis
importance_scores = np.abs(adj_matrix).sum(axis=0)  # Sum of absolute weights per feature
importance_df = pd.DataFrame({'Feature': df_numeric.columns, 'Importance': importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', dodge=False, legend=False, palette='viridis')
plt.title('Feature Importance Based on Causal Graph')
plt.tight_layout()
plt.savefig("feature_importance.png")  # Save figure for later reference
plt.show()

# Add causal effect analysis
print("\nCausal Effect Analysis:")
direct_effects = []

for i, source in enumerate(df_numeric.columns):
    for j, target in enumerate(df_numeric.columns):
        if i != j and abs(adj_matrix[i, j]) > thresh:
            effect = adj_matrix[i, j]
            direct_effects.append((source, target, effect))
            print(f"Direct effect of {source} on {target}: {effect:.4f}")

# Sort and display top indirect effects
print("\nTop Indirect Causal Pathways:")
indirect_effects = []

# Add missing function for indirect effects calculation
def calculate_indirect_effects(G, source, target, max_path_length=3):
    """Calculate indirect effects through all paths from source to target"""
    all_paths = []
    total_effect = 0
    
    try:
        # Find all simple paths from source to target with limited length
        all_paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_path_length))
        
        # Calculate effect through each path
        for path in all_paths:
            path_effect = 1.0
            for i in range(len(path)-1):
                path_effect *= G[path[i]][path[i+1]]['weight']
            total_effect += path_effect
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
        
    return total_effect, all_paths

for source in df_numeric.columns:
    for target in df_numeric.columns:
        if source != target:
            indirect_effect, paths = calculate_indirect_effects(G, source, target)
            if paths and abs(indirect_effect) > 0.1:  # Only show significant effects
                indirect_effects.append((source, target, indirect_effect, paths))

# Sort by absolute effect size
indirect_effects.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 5 indirect effects
for source, target, effect, paths in indirect_effects[:5]:
    print(f"Indirect effect of {source} on {target}: {effect:.4f}")
    print(f"  Pathways: {paths}")

# Create a causal summary table
summary_data = []
for source, target, effect in direct_effects:
    summary_data.append({
        'Source': source, 
        'Target': target, 
        'Effect Type': 'Direct',
        'Strength': effect,
        'Abs Strength': abs(effect)
    })

for source, target, effect, _ in indirect_effects[:5]:
    summary_data.append({
        'Source': source, 
        'Target': target, 
        'Effect Type': 'Indirect',
        'Strength': effect,
        'Abs Strength': abs(effect)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Abs Strength', ascending=False)

print("\nCausal Relationship Summary:")
print(summary_df[['Source', 'Target', 'Effect Type', 'Strength']])

print("\nCausal analysis complete. Visualizations displayed and saved.")

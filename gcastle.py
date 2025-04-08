import numpy as np
from castle.algorithms import GraNDAG, DAG_GNN

# Step 1: Create a random DAG
def create_dag(d=8, edge_prob=0.3, seed=0):
    np.random.seed(seed)
    B = np.tril((np.random.rand(d, d) < edge_prob).astype(int), k=-1)
    return B

# Step 2: Simulate data using a linear SEM
def simulate_linear_sem(B, n=1000, noise_scale=1.0):
    d = B.shape[0]
    W = B * np.random.uniform(0.5, 2.0, size=B.shape)  # Weighted edges
    X = np.zeros((n, d))
    for i in range(d):
        parents = np.where(W[:, i] != 0)[0]
        if len(parents) > 0:
            X[:, i] = X[:, parents] @ W[parents, i] + noise_scale * np.random.randn(n)
        else:
            X[:, i] = noise_scale * np.random.randn(n)
    return X, B

# Step 3: Manual metrics (SHD, precision, recall, F1)
def compute_metrics(true_graph, pred_graph):
    pred_bin = (pred_graph != 0).astype(int)
    true_bin = (true_graph != 0).astype(int)
    np.fill_diagonal(pred_bin, 0)
    np.fill_diagonal(true_bin, 0)

    TP = np.sum((pred_bin == 1) & (true_bin == 1))
    FP = np.sum((pred_bin == 1) & (true_bin == 0))
    FN = np.sum((pred_bin == 0) & (true_bin == 1))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    shd = np.sum(pred_bin != true_bin)

    return {
        "SHD": shd,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# Step 4: Generate synthetic data
B_true = create_dag(d=8, edge_prob=0.3)
X, B = simulate_linear_sem(B_true, n=1000)

# Step 5: Run GraNDAG
grandag = GraNDAG(input_dim=X.shape[1])
grandag.learn(X)
metrics_gran = compute_metrics(B, grandag.causal_matrix)

# Step 6: Run DAG-GNN
daggnn = DAG_GNN()
daggnn.learn(X)
metrics_daggnn = compute_metrics(B, daggnn.causal_matrix)

# Step 7: Output metrics
print("GraNDAG Metrics:")
for k, v in metrics_gran.items():
    print(f"{k}: {v:.4f}")

print("\nDAG-GNN Metrics:")
for k, v in metrics_daggnn.items():
    print(f"{k}: {v:.4f}")

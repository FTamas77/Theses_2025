# Causal Discovery in Stainless Steel Manufacturing: Case Study

## Introduction

This case study explores the application of neural network-based causal discovery techniques to understand energy consumption patterns in stainless steel manufacturing processes. By identifying causal relationships between various manufacturing parameters, we aim to uncover insights that could lead to energy optimization and process improvements.

Causal discovery goes beyond traditional correlation analysis by attempting to establish directional cause-effect relationships between variables. This information is crucial for making effective interventions in complex industrial processes.

## Dataset

The analysis uses a dataset (`stainless_steel_energy.csv`) containing measurements from stainless steel manufacturing operations. The key variables in our analysis include:

- **Power consumption**: Energy usage during manufacturing
- **Weight**: Product weight
- **Height**: Product height/thickness
- **Workplace ID**: Manufacturing station identifier

The dataset required preprocessing to handle European number formats (commas as decimal separators), missing values, and standardization of numeric features.

## Methodology

### NOTEARS Algorithm

This implementation uses a variant of the NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning) algorithm, which enables gradient-based learning of directed acyclic graphs (DAGs). The key innovation of NOTEARS is reformulating the acyclicity constraint as a differentiable function, making it amenable to continuous optimization techniques.

The approach consists of:

1. **Neural network model**: A single-layer neural network represents potential causal relationships
2. **Adjacency matrix**: The weights matrix encodes the strength and direction of causal links
3. **DAG constraint**: A mathematical function ensures the learned graph is acyclic
4. **Statistical significance testing**: Bootstrap sampling to validate discovered relationships

### Loss Function Components

The training process optimizes a composite loss function with three components:
- **Reconstruction loss**: Ensures the model can predict outputs from inputs
- **DAG constraint loss**: Ensures the resulting graph is acyclic (no causal loops)
- **L1 regularization**: Promotes sparsity in the causal graph

## Code Implementation

The implementation follows these primary steps:

### 1. Data Preparation

```python
# Load and preprocess data
df = pd.read_csv("stainless_steel_energy.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
df.rename(columns={"value": "power_consumption"}, inplace=True)

# Convert numeric values (handling European number format with commas)
numeric_columns = ['weight', 'power_consumption', 'height']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Normalize data
scaler = StandardScaler()
data = scaler.fit_transform(df_numeric)
data_tensor = torch.tensor(data, dtype=torch.float32)
```

### 2. Model Definition

```python
class NotearsMLP(nn.Module):
    def __init__(self, dims):
        super(NotearsMLP, self).__init__()
        self.dims = dims
        # Initialize with larger weights to prevent convergence to zero
        self.W = nn.Parameter(torch.randn(dims, dims) * 0.8)
                
    def forward(self, x):
        # Use sigmoid to better preserve small relationships
        return torch.sigmoid(x @ (torch.eye(self.dims) + self.W))
```

### 3. Training Process

The training minimizes the combined loss function:
- Reconstruction loss: `torch.norm(output - data_tensor)`
- DAG constraint: `0.00001 * dag_constraint(model.W)`
- L1 regularization: `0.00001 * torch.norm(model.W, p=1)`

Early stopping is implemented to prevent overfitting, with training halting when improvement plateaus for 30 consecutive epochs.

### 4. Statistical Validation

The bootstrap approach resamples the data 100 times to estimate the variability of the discovered causal relationships:

```python
# Calculate z-scores
z_scores = np.abs(adj_matrix) / (bootstrap_std + 1e-10)

# Apply significance threshold (|z| > 1.96 for p < 0.05)
significant_matrix = np.where(z_scores > 1.96, adj_matrix, 0)
```

### 5. Causal Analysis

After training, the code:
1. Extracts the adjacency matrix representing causal relationships
2. Visualizes these relationships as a heatmap and a directed graph
3. Calculates both direct and indirect causal effects between variables
4. Summarizes the findings with relative importance scores

## Results

### Direct Causal Effects

The analysis discovered several significant direct causal relationships:

1. **Weight → Workplace ID** (1.9969): Product weight strongly influences which workplace is used
2. **Workplace ID → Weight** (2.0568): Reciprocally, workplace setup affects the product weight
3. **Weight → Power Consumption** (1.6222): Heavier products require more energy
4. **Weight → Height** (1.6129): Weight and height are causally linked
5. **Height → Weight** (1.3207): Reciprocal causality between physical dimensions

### Indirect Causal Pathways

The analysis also identified important indirect causal relationships:

1. **Height → Power Consumption** (5.6726): Height indirectly affects energy usage, primarily through weight
2. **Workplace ID → Height** (5.3409): The workplace substantially influences product height through multiple pathways
3. **Workplace ID → Power Consumption** (4.6102): Workplace setup indirectly influences energy consumption

### Key Insights

1. The workplace setup (ID) appears to be a central causal factor, influencing both physical dimensions and energy consumption
2. Product physical characteristics (weight, height) have both direct and indirect effects on energy consumption
3. There are several bidirectional causal relationships, suggesting feedback loops in the manufacturing process

## Conclusion

This causal discovery analysis reveals complex interdependencies in the stainless steel manufacturing process. The identified causal relationships provide actionable insights for energy optimization:

1. **Process optimization opportunities**: By understanding how workplace setup causally influences energy consumption, manufacturers can target specific workstations for efficiency improvements
2. **Product design considerations**: The strong causal link from physical dimensions to energy usage suggests opportunities for energy-efficient product design
3. **Feedback mechanisms**: The bidirectional causal relationships indicate potential process feedback loops that could be leveraged or mitigated

### Limitations and Future Work

1. Additional domain knowledge could further refine the causal model
2. Temporal data would enable more robust causal inferences
3. Expanding the dataset with additional process variables might reveal more complex causal pathways

The neural network-based causal discovery approach demonstrated here provides a powerful tool for understanding complex industrial processes beyond what traditional correlation analysis can reveal.
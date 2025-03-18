import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dowhy import CausalModel
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load Dataset
df = pd.read_csv("stainless_steel_energy.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
df.rename(columns={"value": "power_consumption"}, inplace=True)

# Step 2: Convert Numeric Columns (Handling European Decimal Format)
numeric_columns = ["input_weight", "weight", "forming_temperatures", "heattreatment_temperatures", "power_consumption"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Step 3: Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 4: Define Treatment and Outcome
treatment_var = "workplace_id"
outcome_var = "power_consumption"

# Convert `workplace_id` to categorical integer values
le = LabelEncoder()
df[treatment_var] = le.fit_transform(df[treatment_var])

# Extract treatment, outcome, and features
T = df[treatment_var].astype(int).values
Y = df[outcome_var].values

# Step 5: Select Features (Dropping Irrelevant Columns)
safe_features = ["input_weight", "weight"]
X = df[safe_features].values

# Standardize Features for ML
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Debugging shapes before fitting
print(f"T shape: {T.shape}, dtype: {T.dtype}")
print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")
print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"Expected features: {len(safe_features)}, Actual shape: {X.shape}")

# Check for NaN values
print(f"NaNs in T: {np.isnan(T).sum()}, NaNs in Y: {np.isnan(Y).sum()}, NaNs in X: {np.isnan(X).sum()}")

# Ensure T has the correct shape
if T.shape[0] != Y.shape[0]:
    raise ValueError(f"Mismatch: T has {T.shape[0]} rows, but Y has {Y.shape[0]}")

# Check unique values in treatment
num_treatments = len(np.unique(T))
print(f"Unique treatment values: {num_treatments}")

# Step 6: Apply Causal Forest DML (Handles Multi-Valued Treatments)
dml_estimator = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=100, random_state=42),
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    discrete_treatment=True,
    n_estimators=100,
    random_state=42
)

# Fit Model
dml_estimator.fit(Y, T, X=X)

# Estimate Treatment Effects
treatment_effects = dml_estimator.effect(X)
mean_effect = np.mean(treatment_effects)

# Step 7: Print Results
print(f"\n✅ Estimated Average Treatment Effect (CausalForestDML): {mean_effect:.4f}")
print("\n📊 CAUSAL ANALYSIS RESULTS")
print("="*50)
print(f"Data: {df.shape[0]} observations, {len(safe_features)} features used")
print(f"Treatment Variable: {treatment_var} ({num_treatments} unique categories)")
print(f"Outcome Variable: {outcome_var}")

# Interpret Effect Direction
effect_direction = "decreases" if mean_effect < 0 else "increases"
print(f"\n🔍 FINDING: Different workplaces {effect_direction} power consumption by ≈ {abs(mean_effect):.4f} units.")

print("\n📌 RECOMMENDATIONS:")
print("- Investigate factors behind workplace efficiency differences")
print("- Consider equipment maintenance, age, and operator training")
print("- Collect additional data on forming and heat treatment temperatures")

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['power_consumption'], kde=True)
plt.title('Distribution of Power Consumption')
plt.xlabel('Power Consumption')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df[treatment_var], y=df['power_consumption'])
plt.title('Power Consumption by Workplace')
plt.xlabel('Workplace ID')
plt.ylabel('Power Consumption')
plt.show()

# Ensure only numeric columns are used for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
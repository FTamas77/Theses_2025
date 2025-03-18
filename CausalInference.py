# =========================== #
#    IMPORT LIBRARIES         #
# =========================== #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dowhy import CausalModel
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm

# =========================== #
#   📌 STEP 1: LOAD DATA      #
# =========================== #
df = pd.read_csv("stainless_steel_energy.csv", sep=";", encoding="utf-8", on_bad_lines="skip")
df.rename(columns={"value": "power_consumption"}, inplace=True)

# =========================== #
#   📌 STEP 2: PREPROCESSING  #
# =========================== #
# Convert European decimal format ("," to ".")
numeric_columns = ["input_weight", "weight", "forming_temperatures", "heattreatment_temperatures", "power_consumption"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors='coerce')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert categorical treatment variable
treatment_var = "workplace_id"
outcome_var = "power_consumption"
le = LabelEncoder()
df[treatment_var] = le.fit_transform(df[treatment_var])

# Select features
safe_features = ["input_weight", "weight"]
X = df[safe_features].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define Treatment and Outcome variables
T = df[treatment_var].astype(int).values
Y = df[outcome_var].values

# =========================== #
#   📌 STEP 3: CAUSAL GRAPH   #
# =========================== #
model = CausalModel(
    data=df,
    treatment="workplace_id",
    outcome="power_consumption",
    common_causes=["input_weight", "weight"]  # Add potential confounders
)

# Visualize DAG
model.view_model()

# =========================== #
#   📌 STEP 4: CAUSAL FOREST  #
# =========================== #
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

# ============================== #
#   📌 STEP 5: OLS COMPARISON    #
# ============================== #
# One-hot encode the treatment variable for OLS regression
T_encoded = pd.get_dummies(df[treatment_var], prefix="workplace")

# Add an intercept for OLS
X_ols = np.column_stack((np.ones(len(X)), X, T_encoded))

# Fit OLS model
ols_model = sm.OLS(Y, X_ols).fit()

# Get OLS treatment effect estimates
ols_treatment_effects = ols_model.params[-len(T_encoded.columns) :]  # Last coefficients are for treatment

# ============================== #
# 📌 STEP 6: VALIDATE ASSUMPTIONS #
# ============================== #

# Placebo Test using Random Permutation of Treatment Labels
np.random.shuffle(T)  # Shuffle treatment labels
dml_estimator.fit(Y, T, X=X)  # Refit with shuffled treatments
placebo_effects = dml_estimator.effect(X)
placebo_mean = np.mean(placebo_effects)

# ================================ #
#   📊 STEP 7: RESULTS & INSIGHTS  #
# ================================ #

print("\n✅ Estimated Average Treatment Effect (CausalForestDML): {:.4f}".format(mean_effect))
print("\n📊 CAUSAL ANALYSIS RESULTS")
print("=" * 50)
print(f"Data: {df.shape[0]} observations, {len(safe_features)} features used")
print(f"Treatment Variable: {treatment_var} ({len(np.unique(T))} unique categories)")
print(f"Outcome Variable: {outcome_var}")

effect_direction = "decreases" if mean_effect < 0 else "increases"
print(f"\n🔍 FINDING: Different workplaces {effect_direction} power consumption by ≈ {abs(mean_effect):.4f} units.")

# Placebo Test Results
print("\n🧪 PLACEBO TEST RESULTS")
print(f"✅ True Treatment Effect (CausalForestDML): {mean_effect:.4f}")
print(f"❌ Placebo (Randomized) Treatment Effect: {placebo_mean:.4f}")
print("🔍 If the placebo effect is close to zero, our causal inference is valid.")

# OLS Treatment Effect Estimates
print("\n📊 OLS REGRESSION RESULTS (for comparison):")
print(ols_model.summary())
print(f"\n📌 OLS Estimated Treatment Effects:\n{ols_treatment_effects}")

print("\n📌 RECOMMENDATIONS:")
print("- Investigate factors behind workplace efficiency differences")
print("- Consider equipment maintenance, age, and operator training")
print("- Collect additional data on forming and heat treatment temperatures")

# =========================== #
# 📊 STEP 8: VISUALIZATIONS   #
# =========================== #

# Power Consumption Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['power_consumption'], kde=True)
plt.title('Distribution of Power Consumption')
plt.xlabel('Power Consumption')
plt.ylabel('Frequency')
plt.show()

# Power Consumption by Workplace
plt.figure(figsize=(10, 6))
sns.boxplot(x=df[treatment_var], y=df['power_consumption'])
plt.title('Power Consumption by Workplace')
plt.xlabel('Workplace ID')
plt.ylabel('Power Consumption')
plt.show()

# Correlation Matrix
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Heterogeneous Treatment Effects (HTE) Distribution
plt.figure(figsize=(10, 6))
plt.hist(treatment_effects, bins=30, alpha=0.75)
plt.axvline(mean_effect, color='red', linestyle='dashed', linewidth=2)
plt.title("Distribution of Estimated Treatment Effects")
plt.xlabel("Treatment Effect")
plt.ylabel("Frequency")
plt.show()

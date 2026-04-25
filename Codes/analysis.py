import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
# 'llm_extracted_factors.csv' has been merged with human ratings into 'final_analysis_data.csv'
# columns: human_rating, LLM_Rating, and the 18 factor columns.
df = pd.read_csv('final_analysis_data.csv')

# Ensure factor columns are renamed to match the exact mathematical formula X1LF1 ... X18LF18
factor_mapping = {
    'LF1_Transparency': 'X1LF1', 'LF2_RoadmapCredibility': 'X2LF2',
    'LF3_RiskDisclosure': 'X3LF3', 'LF4_Differentiation': 'X4LF4',
    'LF5_Traction': 'X5LF5', 'LF6_BusinessModelClarity': 'X6LF6',
    'LF7_GoToMarket': 'X7LF7', 'LF8_ExecutionCredibility': 'X8LF8',
    'LF9_TechnicalFeasibility': 'X9LF9', 'LF10_SmartContractSecurity': 'X10LF10',
    'LF11_Scalability': 'X11LF11', 'LF12_IncentiveCompatibility': 'X12LF12',
    'LF13_FinancialSkewness': 'X13LF13', 'LF14_TokenomicsSustainability': 'X14LF14',
    'LF15_UseTestCaseNecessity': 'X15LF15', 'LF16_GovernanceDesign': 'X16LF16',
    'LF17_RegulatoryExposure': 'X17LF17', 'LF18_PartnershipQuality': 'X18LF18'
}
df.rename(columns=factor_mapping, inplace=True)

# Define feature list
features = list(factor_mapping.values())
X = df[features]
y = df['human_rating']
baseline_pred = df['LLM_Rating']

# 2. Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test, base_train, base_test = train_test_split(
    X, y, baseline_pred, test_size=0.2, random_state=42
)

# ==========================================
# MODEL 1: THE BASELINE (Direct LLM Rating)
# ==========================================
baseline_mse = mean_squared_error(y_test, base_test)
baseline_r2 = r2_score(y_test, base_test)

# ==========================================
# MODEL 2: OUR METHOD (OLS REGRESSION)
# ==========================================
# Using the requested formula structure: Y ~ X1LF1 + X2LF2 + ... + X18LF18
formula = 'human_rating ~ ' + ' + '.join(features)
train_df = pd.concat([y_train, X_train], axis=1)

ols_model = smf.ols(formula=formula, data=train_df).fit()
ols_predictions = ols_model.predict(X_test)

ols_mse = mean_squared_error(y_test, ols_predictions)
ols_r2 = r2_score(y_test, ols_predictions)

# Save summary to a text file for academic reporting
with open('OLS_Regression_Summary.txt', 'w') as f:
    f.write(ols_model.summary().as_text())

# ==========================================
# MODEL 3: OUR METHOD + NON-LINEAR (RANDOM FOREST)
# ==========================================
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# ==========================================
# VISUALIZATIONS FOR THE PAPER
# ==========================================
# Plot 1: Random Forest Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Latent Factor Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_features, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("Feature_Importance_RF.png", dpi=300)
plt.close()

# Plot 2: Correlation Heatmap of Latent Factors
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of 18 Latent Factors")
plt.tight_layout()
plt.savefig("Correlation_Heatmap.png", dpi=300)
plt.close()

# ==========================================
# PRINT FINAL ABLATION STUDY TABLE
# ==========================================
print("\n--- ABLATION STUDY RESULTS (TEST SET) ---")
print("-" * 75)
print(f"{'Idea / Variant':<50} | {'Test Set R^2':<10} | {'Test Set MSE':<10}")
print("-" * 75)
print(f"{'Baseline (Direct AI rating, no factors)':<50} | {baseline_r2:10.3f} | {baseline_mse:10.3f}")
print(f"{'Our Method + Ordinal Ratings (18 Factors, OLS)':<50} | {ols_r2:10.3f} | {ols_mse:10.3f}")
print(f"{'Our Method + Ordinal + Random Forest':<50} | {rf_r2:10.3f} | {rf_mse:10.3f}")
print("-" * 75)
print("\nFiles generated: 'Feature_Importance_RF.png', 'Correlation_Heatmap.png', 'OLS_Regression_Summary.txt'")
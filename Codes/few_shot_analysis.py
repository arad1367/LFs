import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf

# 1. Load the original data to get the human ratings
original_df = pd.read_csv('final_analysis_data.csv')
human_ratings_df = original_df[['Paper_Number', 'human_rating']]

# 2. Load our new Few-Shot data
few_shot_df = pd.read_csv('llm_extracted_factors_few_shot.csv')

# Drop paper_100.txt (or any empty/failed papers) to ensure clean data
few_shot_df = few_shot_df[few_shot_df['LF1_Transparency'].notna()]

# 3. Merge them together on Paper_Number
df = pd.merge(human_ratings_df, few_shot_df, on='Paper_Number', how='inner')

# 4. Rename columns for the regression formula
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
features = list(factor_mapping.values())

# 5. Train/Test Split (Use the SAME random_state=42 so the test set is identical!)
X = df[features]
y = df['human_rating']
baseline_pred = df['LLM_Rating'] # This is now the Few-Shot Direct Rating

X_train, X_test, y_train, y_test, base_train, base_test = train_test_split(
    X, y, baseline_pred, test_size=0.2, random_state=42
)

# --- MODEL 1: FEW-SHOT BASELINE ---
fs_baseline_mse = mean_squared_error(y_test, base_test)
fs_baseline_r2 = r2_score(y_test, base_test)

# --- MODEL 2: FEW-SHOT OLS ---
formula = 'human_rating ~ ' + ' + '.join(features)
train_df = pd.concat([y_train, X_train], axis=1)
ols_model = smf.ols(formula=formula, data=train_df).fit()
ols_predictions = ols_model.predict(X_test)
fs_ols_mse = mean_squared_error(y_test, ols_predictions)
fs_ols_r2 = r2_score(y_test, ols_predictions)

# --- MODEL 3: FEW-SHOT RANDOM FOREST ---
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
fs_rf_mse = mean_squared_error(y_test, rf_predictions)
fs_rf_r2 = r2_score(y_test, rf_predictions)

# --- PRINT THE COMPLETE TABLE FOR YOUR DRAFT ---
print("\n--- FINAL ABLATION STUDY RESULTS (INCLUDING FEW-SHOT) ---")
print("-" * 85)
print(f"{'Idea / Variant':<60} | {'Test Set R^2':<10} | {'Test Set MSE':<10}")
print("-" * 85)
print(f"{'Zero-Shot Baseline (from previous run)':<60} |     -3.767 |      1.027")
print(f"{'Zero-Shot OLS (from previous run)':<60} |     -0.141 |      0.246")
print(f"{'Zero-Shot Random Forest (from previous run)':<60} |     -0.137 |      0.245")
print("-" * 85)
print(f"{'Few-Shot Baseline (Direct AI rating)':<60} | {fs_baseline_r2:10.3f} | {fs_baseline_mse:10.3f}")
print(f"{'Few-Shot Method + Ordinal Ratings (18 Factors, OLS)':<60} | {fs_ols_r2:10.3f} | {fs_ols_mse:10.3f}")
print(f"{'Few-Shot Method + Ordinal + Random Forest':<60} | {fs_rf_r2:10.3f} | {fs_rf_mse:10.3f}")
print("-" * 85)
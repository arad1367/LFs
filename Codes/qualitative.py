import pandas as pd

# Load both datasets
original_df = pd.read_csv('final_analysis_data.csv')
few_shot_df = pd.read_csv('llm_extracted_factors_few_shot.csv')

# Merge to get human_rating alongside the LLM's text outputs
df = pd.merge(original_df[['Paper_Number', 'human_rating']], few_shot_df, on='Paper_Number', how='inner')

# Drop any empty or failed extractions to ensure clean text
df = df[df['LLM_Reason'].notna() & (df['LLM_Reason'] != "")]

# Sort by human rating to easily grab the extremes and the median
df_sorted = df.sort_values(by='human_rating')

# Select 3 interesting examples
low_example = df_sorted.iloc[0]                  # Lowest human rating
med_example = df_sorted.iloc[len(df_sorted)//2]  # Median human rating
high_example = df_sorted.iloc[-1]                # Highest human rating

# Function to print the output clearly
def print_example(label, row):
    print(f"--- {label} ---")
    print(f"Paper: {row['Paper_Number']}")
    print(f"Human Rating: {row['human_rating']:.2f}")
    print(f"LLM Rating (Few-Shot): {row['LLM_Rating']:.2f}")
    print(f"LLM Reason: {row['LLM_Reason']}")
    print(f"Brief Evidence: {row['Brief_Evidence']}\n")

print_example("LOW RATED EXAMPLE", low_example)
print_example("MEDIUM RATED EXAMPLE", med_example)
print_example("HIGH RATED EXAMPLE", high_example)
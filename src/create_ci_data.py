import pandas as pd

# Load full dataset
df_full = pd.read_csv('./data/creditcard.csv')

# Use ALL fraud cases (492) and proportional legit cases
fraud = df_full[df_full['Class'] == 1]  # All 492 fraud cases
legit = df_full[df_full['Class'] == 0].sample(n=9508, random_state=42)  # 9508 legit = 10k total

# Combine and shuffle
df_ci = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

# Save for CI use
df_ci.to_csv('./data/creditcard_ci.csv', index=False)

print(f"Created CI dataset: {len(df_ci)} rows, {df_ci['Class'].sum()} fraud cases ({100*df_ci['Class'].mean():.2f}% fraud rate)")
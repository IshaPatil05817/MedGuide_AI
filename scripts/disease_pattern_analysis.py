import pandas as pd

# Load dataset
train = pd.read_csv("datasets/raw/Training.csv")

# Remove empty column
train = train.drop(columns=["Unnamed: 133"], errors="ignore")

# Symptom columns
symptom_columns = [col for col in train.columns if col != "prognosis"]

# Create a unique symptom pattern
train["symptom_pattern"] = (
    train[symptom_columns]
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

# Count unique patterns for each disease
analysis = (
    train.groupby("prognosis")["symptom_pattern"]
    .nunique()
    .sort_values()
)

print("========== DISEASE-WISE UNIQUE PATTERNS ==========")

print(analysis.to_string())

print("\n========== SUMMARY ==========")

print("Number of diseases:", analysis.count())
print("Minimum patterns for a disease:", analysis.min())
print("Maximum patterns for a disease:", analysis.max())
print("Average patterns per disease:", round(analysis.mean(), 2))

print("\n========== DISEASES WITH FEWEST PATTERNS ==========")

print(analysis.head(10).to_string())

print("\n========== DISEASES WITH MOST PATTERNS ==========")

print(analysis.tail(10).to_string())
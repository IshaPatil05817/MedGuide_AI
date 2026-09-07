import pandas as pd

# Load training dataset
train = pd.read_csv("datasets/raw/Training.csv")

# Remove empty column
train = train.drop(columns=["Unnamed: 133"], errors="ignore")

# Separate symptoms and target
symptom_columns = [col for col in train.columns if col != "prognosis"]

print("========== UNIQUE SYMPTOM PATTERNS ==========")

# Count unique symptom combinations
unique_patterns = train[symptom_columns].drop_duplicates()

print("Total training records:", len(train))
print("Unique symptom patterns:", len(unique_patterns))

print("\n========== REPEATED PATTERNS ==========")

pattern_counts = (
    train.groupby(symptom_columns, dropna=False)
    .size()
    .sort_values(ascending=False)
)

print("Most repeated patterns:")
print(pattern_counts.head(20))

print("\n========== CHECK SAME SYMPTOMS → DIFFERENT DISEASES ==========")

disease_per_pattern = (
    train.groupby(symptom_columns)["prognosis"]
    .nunique()
)

ambiguous_patterns = disease_per_pattern[disease_per_pattern > 1]

print("Patterns associated with multiple diseases:",
      len(ambiguous_patterns))

print("\n========== SAME SYMPTOMS WITH DIFFERENT DISEASES ==========")

if len(ambiguous_patterns) > 0:
    for pattern in ambiguous_patterns.head(10).index:
        matching_rows = train[
            (train[symptom_columns] == pattern).all(axis=1)
        ]
        print("\nDiseases:", matching_rows["prognosis"].unique())
else:
    print("No ambiguous symptom patterns found.")

print("\n========== DUPLICATE SUMMARY ==========")

print("Total rows:", len(train))
print("Unique complete records:", train.drop_duplicates().shape[0])
print("Duplicate rows:", train.duplicated().sum())
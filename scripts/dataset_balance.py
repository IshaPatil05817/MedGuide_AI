import pandas as pd

# Load dataset
train = pd.read_csv("datasets/raw/Training.csv")

# Remove empty column
train = train.drop(columns=["Unnamed: 133"], errors="ignore")

# Symptom columns
symptom_columns = [col for col in train.columns if col != "prognosis"]

# Keep only unique symptom-disease combinations
unique_data = train.drop_duplicates(
    subset=symptom_columns + ["prognosis"]
)

print("========== DATASET BALANCE ==========")

print("Original records:", len(train))
print("Unique records:", len(unique_data))

print("\nDisease distribution:")
distribution = unique_data["prognosis"].value_counts().sort_index()

print(distribution.to_string())

print("\n========== BALANCE CHECK ==========")

print("Minimum samples per disease:", distribution.min())
print("Maximum samples per disease:", distribution.max())

print("\nDiseases with only 5 samples:")

print(
    distribution[distribution == 5].to_string()
)

print("\nDiseases with only 6 samples:")

print(
    distribution[distribution == 6].to_string()
)
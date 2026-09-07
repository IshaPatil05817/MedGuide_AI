import pandas as pd
import os

# ==========================================
# 1. Load original dataset
# ==========================================

input_file = "datasets/raw/Training.csv"

df = pd.read_csv(input_file)

print("Original shape:", df.shape)


# ==========================================
# 2. Remove empty columns
# ==========================================

df = df.dropna(axis=1, how="all")

print("After removing empty columns:", df.shape)


# ==========================================
# 3. Remove exact duplicate records
# ==========================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("Rows before duplicate removal:", before)
print("Rows after duplicate removal:", after)
print("Duplicates removed:", before - after)


# ==========================================
# 4. Check disease distribution
# ==========================================

print("\nDisease distribution:")

print(
    df["prognosis"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ==========================================
# 5. Save cleaned dataset
# ==========================================

output_folder = "datasets/cleaned"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "Disease_Symptom_Cleaned.csv"
)

df.to_csv(output_file, index=False)


# ==========================================
# 6. Final information
# ==========================================

print("\n========== FINAL DATASET ==========")

print("Final shape:", df.shape)
print("Number of diseases:", df["prognosis"].nunique())

print("\nSaved successfully:")
print(output_file)
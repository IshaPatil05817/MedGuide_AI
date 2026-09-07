import pandas as pd
import os
import random

# ============================================================
# LOAD EXISTING DATASET 2
# ============================================================

input_file = (
    "datasets/recommendation/Medical_System_Training.csv"
)

df = pd.read_csv(input_file)

print("\n==========================================")
print("EXPANDING MEDICAL SYSTEM DATASET")
print("==========================================")

print(f"\nOriginal records: {len(df)}")


# ============================================================
# EXPAND EACH RECORD
# ============================================================

expanded_data = []

random.seed(42)

for _, row in df.iterrows():

    disease = row["Disease"]
    symptoms = row["Symptoms"].split(",")
    medical_system = row["Medical_System"]

    symptoms = [
        symptom.strip()
        for symptom in symptoms
        if symptom.strip()
    ]

    # Keep original record
    expanded_data.append([
        disease,
        ",".join(symptoms),
        medical_system
    ])

    # Create controlled variations
    for _ in range(8):

        variation = symptoms.copy()

        # Add/remove one symptom where possible
        if len(variation) > 2:
            remove_index = random.randrange(len(variation))
            variation.pop(remove_index)

        # Shuffle symptom order
        random.shuffle(variation)

        expanded_data.append([
            disease,
            ",".join(variation),
            medical_system
        ])


# ============================================================
# CREATE DATAFRAME
# ============================================================

expanded_df = pd.DataFrame(
    expanded_data,
    columns=[
        "Disease",
        "Symptoms",
        "Medical_System"
    ]
)


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

expanded_df = expanded_df.drop_duplicates(
    subset=[
        "Disease",
        "Symptoms",
        "Medical_System"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

output_file = (
    "datasets/recommendation/"
    "Medical_System_Training_Expanded.csv"
)

expanded_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nExpanded records: {len(expanded_df)}"
)

print("\nMedical System Distribution:")

print(
    expanded_df["Medical_System"].value_counts()
)

print("\nDisease count:")

print(
    expanded_df["Disease"].nunique()
)

print("\nSaved at:")

print(output_file)

print("\n==========================================")
print("EXPANSION COMPLETED")
print("==========================================")
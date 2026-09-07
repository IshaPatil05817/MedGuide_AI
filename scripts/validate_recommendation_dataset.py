import pandas as pd

file_path = "datasets/recommendation/Medical_System_Recommendation.csv"

df = pd.read_csv(file_path)

print("========== DATASET 2 VALIDATION ==========")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nUnique diseases:", df["Disease"].nunique())

print("\n========== DUPLICATE DISEASE CHECK ==========")

duplicates = df[df["Disease"].duplicated(keep=False)]

if len(duplicates) == 0:
    print("No duplicate diseases found.")
else:
    print("Duplicate diseases found:")
    print(duplicates["Disease"].to_string(index=False))

print("\n========== MISSING VALUE CHECK ==========")

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])

print("\n========== PRIMARY MEDICAL SYSTEM ==========")

print(df["Primary_Medical_System"].value_counts())

print("\n========== SPECIALIST CHECK ==========")

print("Unique specialists:", df["Specialist"].nunique())

print("\n========== FINAL CHECK ==========")

if (
    len(df) == 41
    and df["Disease"].nunique() == 41
    and missing.sum() == 0
    and len(duplicates) == 0
):
    print("Dataset 2 validation PASSED.")
else:
    print("Dataset 2 validation needs attention.")
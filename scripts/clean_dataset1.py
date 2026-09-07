import pandas as pd

# Load training dataset
train = pd.read_csv("datasets/raw/Training.csv")

# Remove completely empty column
train = train.drop(columns=["Unnamed: 133"], errors="ignore")

print("========== BASIC INFORMATION ==========")
print("Rows:", train.shape[0])
print("Columns:", train.shape[1])

print("\n========== DUPLICATE ANALYSIS ==========")
print("Total rows:", len(train))
print("Duplicate rows:", train.duplicated().sum())
print("Unique rows:", train.drop_duplicates().shape[0])

print("\n========== DISEASE COUNT ==========")
print("Number of diseases:", train["prognosis"].nunique())

print("\n========== UNIQUE RECORDS PER DISEASE ==========")

unique_per_disease = (
    train.groupby("prognosis")
    .apply(lambda x: x.drop_duplicates().shape[0])
)

print(unique_per_disease)

print("\n========== TOTAL RECORDS PER DISEASE ==========")
print(train["prognosis"].value_counts())
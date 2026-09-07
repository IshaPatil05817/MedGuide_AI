import pandas as pd

# Load datasets
train = pd.read_csv("datasets/raw/Training.csv")
test = pd.read_csv("datasets/raw/Testing.csv")

print("========== DATASET SHAPE ==========")
print("Training:", train.shape)
print("Testing :", test.shape)

print("\n========== COLUMN COUNT ==========")
print("Training columns:", len(train.columns))
print("Testing columns :", len(test.columns))

print("\n========== LAST 5 COLUMNS ==========")
print("Training:", train.columns[-5:].tolist())
print("Testing :", test.columns[-5:].tolist())

print("\n========== MISSING VALUES ==========")
print("Training missing values:", train.isnull().sum().sum())
print("Testing missing values :", test.isnull().sum().sum())

print("\n========== DUPLICATES ==========")
print("Training duplicates:", train.duplicated().sum())
print("Testing duplicates :", test.duplicated().sum())

print("\n========== TARGET COLUMN ==========")
print("Training columns containing 'prognosis':")
print([col for col in train.columns if "prognosis" in col.lower()])

print("\nTesting columns containing 'prognosis':")
print([col for col in test.columns if "prognosis" in col.lower()])

print("\n========== DISEASE COUNT ==========")
if "prognosis" in train.columns:
    print("Number of diseases:", train["prognosis"].nunique())
    print("\nDisease distribution:")
    print(train["prognosis"].value_counts())

print("\n========== DATA TYPES ==========")
print(train.dtypes.value_counts())

print("\n========== TESTING DISEASES ==========")

print("Number of testing diseases:")
print(test["prognosis"].nunique())

print("\nTesting disease distribution:")
print(test["prognosis"].value_counts())

print("\n========== TRAINING vs TESTING ==========")

train_diseases = set(train["prognosis"].unique())
test_diseases = set(test["prognosis"].unique())

print("Diseases in training:", len(train_diseases))
print("Diseases in testing:", len(test_diseases))

print("\nDiseases only in training:")
print(sorted(train_diseases - test_diseases))

print("\nDiseases only in testing:")
print(sorted(test_diseases - train_diseases))
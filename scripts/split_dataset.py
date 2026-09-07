import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv(
    "datasets/cleaned/Disease_Symptom_Cleaned.csv"
)

# Separate features and target
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

print("Total records:", len(df))
print("Total features:", X.shape[1])
print("Total diseases:", y.nunique())

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== SPLIT RESULT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining disease classes:", y_train.nunique())
print("Testing disease classes:", y_test.nunique())

print("\nTest set distribution:")
print(y_test.value_counts().sort_index().to_string())
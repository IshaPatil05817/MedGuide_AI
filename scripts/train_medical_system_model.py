import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD EXPANDED DATASET 2
# ============================================================

df = pd.read_csv(
    "datasets/recommendation/Medical_System_Training_Expanded.csv"
)

print("\n==========================================")
print("       MEDICAL SYSTEM MODEL - MODEL 2")
print("==========================================")

print(f"\nTotal records: {len(df)}")


# ============================================================
# 2. CONVERT SYMPTOMS INTO FEATURES
# ============================================================

all_symptoms = set()

for symptoms in df["Symptoms"]:

    for symptom in symptoms.split(","):

        all_symptoms.add(
            symptom.strip()
        )


all_symptoms = sorted(all_symptoms)


# Create feature matrix

X = pd.DataFrame(
    0,
    index=range(len(df)),
    columns=all_symptoms
)


# Set symptom values

for i, symptoms in enumerate(df["Symptoms"]):

    for symptom in symptoms.split(","):

        symptom = symptom.strip()

        if symptom in X.columns:

            X.loc[i, symptom] = 1


# Target variable

y = df["Medical_System"]


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print(
    f"Total symptom features: {len(X.columns)}"
)

print("\nMedical System Distribution:")

print(
    y.value_counts()
)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("TRAIN-TEST SPLIT")
print("==========================================")

print(
    f"Training records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)


# ============================================================
# 5. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


print(
    "\nRandom Forest Model 2 trained successfully."
)


# ============================================================
# 6. PREDICT TEST DATA
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 7. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==========================================")
print("MODEL 2 EVALUATION")
print("==========================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "model/medical_system_model.pkl"
)


print("\n==========================================")
print("MODEL 2 SAVED SUCCESSFULLY")
print("==========================================")

print(
    "\nSaved as:"
)

print(
    "model/medical_system_model.pkl"
)

print("\n==========================================")
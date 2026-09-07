import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load cleaned dataset
# ==========================================

df = pd.read_csv(
    "datasets/cleaned/Disease_Symptom_Cleaned.csv"
)

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# ==========================================
# 2. Separate features and target
# ==========================================

X = df.drop(columns=["prognosis"])
y = df["prognosis"]

print("Features:", X.shape[1])
print("Diseases:", y.nunique())


# ==========================================
# 3. Stratified train-test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 4. Create Random Forest model
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 5. Train model
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 6. Make predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. Evaluate model
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 8. Save model
# ==========================================

model_folder = "model"

import os
os.makedirs(model_folder, exist_ok=True)

model_path = "model/disease_prediction_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully:")
print(model_path)
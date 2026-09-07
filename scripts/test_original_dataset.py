import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load trained model
model = joblib.load("model/disease_prediction_model.pkl")

# Load original testing dataset
test = pd.read_csv("datasets/raw/Testing.csv")

# Remove empty columns if present
test = test.dropna(axis=1, how="all")

# Separate features and target
X_test = test.drop(columns=["prognosis"])
y_test = test["prognosis"]

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("========== ORIGINAL TESTING DATASET ==========")
print("Testing records:", len(test))
print("Actual diseases:", y_test.nunique())
print("Predicted diseases:", len(set(y_pred)))

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\n========== PREDICTIONS ==========")

for actual, predicted in zip(y_test, y_pred):
    status = "✓" if actual == predicted else "✗"
    print(
        f"{status} Actual: {actual} | Predicted: {predicted}"
    )
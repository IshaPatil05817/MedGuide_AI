import pandas as pd
import joblib

# Load model
model = joblib.load("model/disease_prediction_model.pkl")

# Load cleaned dataset
df = pd.read_csv(
    "datasets/cleaned/Disease_Symptom_Cleaned.csv"
)

# Model features
symptom_columns = list(model.feature_names_in_)

# Select first 10 actual records
test_df = df.head(10)

correct = 0

print("\n========== REAL PATTERN TEST ==========\n")

for i, row in test_df.iterrows():

    actual_disease = row["prognosis"]

    # Get symptoms that are 1
    symptoms = [
        symptom
        for symptom in symptom_columns
        if row[symptom] == 1
    ]

    # Create model input
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=symptom_columns
    )

    for symptom in symptoms:
        input_data.loc[0, symptom] = 1

    # Predict
    predicted_disease = model.predict(input_data)[0]

    if predicted_disease == actual_disease:
        correct += 1
        result = "CORRECT"
    else:
        result = "WRONG"

    print(f"Test {i + 1}")
    print(f"Actual Disease:    {actual_disease}")
    print(f"Predicted Disease: {predicted_disease}")
    print(f"Result:            {result}")
    print("-" * 50)

accuracy = (correct / len(test_df)) * 100

print("\n========== FINAL RESULT ==========")
print(f"Correct Predictions: {correct}/{len(test_df)}")
print(f"Accuracy: {accuracy:.2f}%")
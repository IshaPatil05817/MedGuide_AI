import pandas as pd
import joblib


# ============================================================
# LOAD MODELS
# ============================================================

disease_model = joblib.load(
    "model/disease_prediction_model.pkl"
)

medical_system_model = joblib.load(
    "model/medical_system_model.pkl"
)


# ============================================================
# LOAD CLEANED DATASET 1
# ============================================================

df = pd.read_csv(
    "datasets/cleaned/Disease_Symptom_Cleaned.csv"
)


# ============================================================
# SELECT ONE REAL RECORD
# ============================================================

row = df.iloc[0]

actual_disease = row["prognosis"]


# ============================================================
# GET ACTUAL SYMPTOMS FROM THE RECORD
# ============================================================

symptom_columns = list(
    disease_model.feature_names_in_
)

selected_symptoms = []

for symptom in symptom_columns:

    if row[symptom] == 1:
        selected_symptoms.append(symptom)


# ============================================================
# CREATE MODEL 1 INPUT
# ============================================================

disease_input = pd.DataFrame(
    0,
    index=[0],
    columns=symptom_columns
)

for symptom in selected_symptoms:

    disease_input.loc[0, symptom] = 1


# ============================================================
# MODEL 1 - DISEASE PREDICTION
# ============================================================

predicted_disease = disease_model.predict(
    disease_input
)[0]

disease_probability = (
    disease_model.predict_proba(
        disease_input
    )[0]
)

disease_confidence = (
    disease_probability.max() * 100
)


# ============================================================
# MODEL 2 INPUT
# ============================================================

medical_features = list(
    medical_system_model.feature_names_in_
)

medical_input = pd.DataFrame(
    0,
    index=[0],
    columns=medical_features
)

for symptom in selected_symptoms:

    if symptom in medical_features:

        medical_input.loc[
            0,
            symptom
        ] = 1


# ============================================================
# MODEL 2 - MEDICAL SYSTEM
# ============================================================

predicted_system = medical_system_model.predict(
    medical_input
)[0]

system_probability = (
    medical_system_model.predict_proba(
        medical_input
    )[0]
)

system_confidence = (
    system_probability.max() * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("       TWO-MODEL PIPELINE TEST")
print("==========================================")

print("\nActual Disease:")
print(actual_disease)

print("\nSymptoms Used:")
print(", ".join(selected_symptoms))

print("\n------------------------------------------")

print("MODEL 1 - DISEASE PREDICTION")

print("------------------------------------------")

print(
    f"Predicted Disease: {predicted_disease}"
)

print(
    f"Confidence: {disease_confidence:.2f}%"
)

print(
    "\nDisease Prediction:",
    "CORRECT" if predicted_disease == actual_disease else "INCORRECT"
)


print("\n------------------------------------------")

print("MODEL 2 - MEDICAL SYSTEM")

print("------------------------------------------")

print(
    f"Predicted Medical System: {predicted_system}"
)

print(
    f"Confidence: {system_confidence:.2f}%"
)


print("\n==========================================")
print("TWO-MODEL PIPELINE TEST COMPLETED")
print("==========================================")
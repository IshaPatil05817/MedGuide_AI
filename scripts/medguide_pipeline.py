import pandas as pd
import joblib


# ============================================================
# 1. LOAD MODEL 1 - DISEASE PREDICTION
# ============================================================

disease_model = joblib.load(
    "model/disease_prediction_model.pkl"
)

print("Disease prediction model loaded successfully.")


# ============================================================
# 2. LOAD MODEL 2 - MEDICAL SYSTEM RECOMMENDATION
# ============================================================

medical_system_model = joblib.load(
    "model/medical_system_model.pkl"
)

print("Medical system recommendation model loaded successfully.")


# ============================================================
# 3. LOAD RECOMMENDATION DATASET
# ============================================================

recommendation_df = pd.read_csv(
    "datasets/recommendation/Medical_System_Recommendation.csv"
)

print("Recommendation dataset loaded successfully.")


# ============================================================
# 4. GET MODEL 1 SYMPTOMS
# ============================================================

symptom_columns = list(
    disease_model.feature_names_in_
)


# ============================================================
# 5. GET MODEL 2 SYMPTOMS
# ============================================================

medical_system_features = list(
    medical_system_model.feature_names_in_
)


# ============================================================
# 6. DISPLAY MEDGUIDE AI
# ============================================================

print("\n======================================")
print("           MEDGUIDE AI")
print("======================================")


# ============================================================
# 7. DISPLAY AVAILABLE SYMPTOMS
# ============================================================

print("\nAvailable symptoms:")
print(", ".join(symptom_columns))


# ============================================================
# 8. GET USER INPUT
# ============================================================

user_input = input(
    "\nEnter your symptoms separated by comma:\n"
)


# ============================================================
# 9. PROCESS USER SYMPTOMS
# ============================================================

selected_symptoms = [
    symptom.strip()
    for symptom in user_input.split(",")
    if symptom.strip()
]


# ============================================================
# 10. CREATE MODEL 1 INPUT
# ============================================================

input_data = pd.DataFrame(
    0,
    index=[0],
    columns=symptom_columns
)


# ============================================================
# 11. MATCH SYMPTOMS WITH MODEL 1
# ============================================================

valid_symptoms = 0

for symptom in selected_symptoms:

    if symptom in symptom_columns:

        input_data.loc[0, symptom] = 1
        valid_symptoms += 1

    else:

        print(
            f"Warning: '{symptom}' is not a valid symptom."
        )


# ============================================================
# 12. CHECK VALID SYMPTOMS
# ============================================================

if valid_symptoms == 0:

    print("\n======================================")
    print("ERROR")
    print("======================================")

    print(
        "\nNo valid symptoms were entered."
    )

    print(
        "Please enter valid symptoms from the available list."
    )

    print("\n======================================")

    exit()


# ============================================================
# 13. MODEL 1 - DISEASE PREDICTION
# ============================================================

predicted_disease = disease_model.predict(
    input_data
)[0]


# ============================================================
# 14. MODEL 1 - PREDICTION PROBABILITY
# ============================================================

disease_probabilities = disease_model.predict_proba(
    input_data
)[0]


# ============================================================
# 15. DISEASE CONFIDENCE
# ============================================================

disease_confidence = (
    disease_probabilities.max() * 100
)


# ============================================================
# 16. TOP 3 DISEASE PREDICTIONS
# ============================================================

disease_classes = disease_model.classes_

top_indices = (
    disease_probabilities.argsort()[-3:][::-1]
)


print("\n========== TOP 3 DISEASE PREDICTIONS ==========")

for index in top_indices:

    print(
        f"{disease_classes[index]}: "
        f"{disease_probabilities[index] * 100:.2f}%"
    )


print(
    f"\nDisease Prediction Confidence: "
    f"{disease_confidence:.2f}%"
)


# ============================================================
# 17. DISPLAY DISEASE RESULT
# ============================================================

print("\n======================================")
print("          DISEASE PREDICTION")
print("======================================")


print(
    f"\nPossible Condition: "
    f"{predicted_disease}"
)

print(
    f"Disease Confidence: "
    f"{disease_confidence:.2f}%"
)


# ============================================================
# 18. LOW CONFIDENCE HANDLING
# ============================================================

CONFIDENCE_THRESHOLD = 60


if disease_confidence < CONFIDENCE_THRESHOLD:

    print("\n⚠ LOW CONFIDENCE PREDICTION")

    print(
        "\nThe entered symptoms are not sufficient "
        "for a reliable prediction."
    )

    print(
        "\nPlease provide additional symptoms "
        "for a better prediction."
    )

    print(
        "\nYou may also consult a qualified "
        "healthcare professional."
    )

    print("\n======================================")

    print(
        "Medical system recommendation is not "
        "generated because disease confidence is low."
    )

    print("======================================")

    print(
        "\nThis system provides an AI-based preliminary "
        "assessment and is not a medical diagnosis."
    )

    exit()


# ============================================================
# 19. MODEL 2 INPUT
# ============================================================

medical_system_input = pd.DataFrame(
    0,
    index=[0],
    columns=medical_system_features
)


# ============================================================
# 20. MATCH USER SYMPTOMS WITH MODEL 2 FEATURES
# ============================================================

for symptom in selected_symptoms:

    if symptom in medical_system_features:

        medical_system_input.loc[
            0,
            symptom
        ] = 1


# ============================================================
# 21. MODEL 2 - MEDICAL SYSTEM PREDICTION
# ============================================================

predicted_medical_system = (
    medical_system_model.predict(
        medical_system_input
    )[0]
)


# ============================================================
# 22. MODEL 2 - CONFIDENCE
# ============================================================

medical_system_probabilities = (
    medical_system_model.predict_proba(
        medical_system_input
    )[0]
)


medical_system_confidence = (
    medical_system_probabilities.max() * 100
)


# ============================================================
# 23. TOP MEDICAL SYSTEM PREDICTIONS
# ============================================================

medical_system_classes = (
    medical_system_model.classes_
)

medical_top_indices = (
    medical_system_probabilities.argsort()[-3:][::-1]
)


print("\n========== MEDICAL SYSTEM PREDICTIONS ==========")

for index in medical_top_indices:

    print(
        f"{medical_system_classes[index]}: "
        f"{medical_system_probabilities[index] * 100:.2f}%"
    )


# ============================================================
# 24. DISPLAY MODEL 2 RESULT
# ============================================================

print("\n======================================")
print("       MEDICAL SYSTEM RECOMMENDATION")
print("======================================")


print(
    f"\nRecommended Medical System: "
    f"{predicted_medical_system}"
)

print(
    f"Medical System Confidence: "
    f"{medical_system_confidence:.2f}%"
)


# ============================================================
# 25. FIND DISEASE RECOMMENDATION
# ============================================================

recommendation = recommendation_df[
    recommendation_df["Disease"] == predicted_disease
]


# ============================================================
# 26. DISPLAY SPECIALIST + COMPLEMENTARY RECOMMENDATION
# ============================================================

if len(recommendation) > 0:

    row = recommendation.iloc[0]


    print(
        f"\nSpecialist: "
        f"{row['Specialist']}"
    )


    print(
        f"Complementary Recommendation: "
        f"{row['Yoga_Recommendation']}"
    )


    print(
        f"\nRecommendation Reason: "
        f"{row['Recommendation_Reason']}"
    )


else:

    print(
        "\nNo specialist recommendation found "
        "for this condition."
    )


# ============================================================
# 27. MEDICAL DISCLAIMER
# ============================================================

print("\n======================================")

print(
    "Please consult a qualified healthcare "
    "professional for proper diagnosis and treatment."
)

print(
    "This system provides an AI-based preliminary "
    "assessment and is not a medical diagnosis."
)

print("======================================")
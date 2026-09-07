import pandas as pd
import os


# ============================================================
# CREATE MEDICAL SYSTEM TRAINING DATASET
# ============================================================

data = [

    # ---------------- ALLopathy ----------------

    ["Migraine", "headache,nausea,vomiting", "Allopathy"],
    ["Tuberculosis", "cough,fever,weight_loss", "Allopathy"],
    ["Dengue", "high_fever,nausea,pain_behind_the_eyes", "Allopathy"],
    ["Pneumonia", "cough,breathlessness,high_fever", "Allopathy"],
    ["Heart attack", "chest_pain,sweating,fast_heart_rate", "Allopathy"],
    ["Malaria", "high_fever,chills,sweating", "Allopathy"],
    ["Typhoid", "high_fever,abdominal_pain,weakness", "Allopathy"],
    ["Fungal infection", "itching,skin_rash,skin_peeling", "Allopathy"],
    ["Urinary tract infection", "burning_micturition,continuous_feel_of_urine", "Allopathy"],
    ["Hepatitis B", "yellowish_skin,dark_urine,fatigue", "Allopathy"],
    ["Hepatitis C", "yellowish_skin,fatigue,loss_of_appetite", "Allopathy"],
    ["Drug Reaction", "skin_rash,itching,redness_of_eyes", "Allopathy"],
    ["Paralysis", "weakness_of_one_body_side,slurred_speech,loss_of_balance", "Allopathy"],
    ["Chicken pox", "skin_rash,mild_fever,blister", "Allopathy"],


    # ---------------- AYURVEDA ----------------
    # These are framed as complementary/wellness-oriented
    # project categories, not disease-treatment claims.

    ["Indigestion", "indigestion,stomach_pain,acidity", "Ayurveda"],
    ["Constipation", "constipation,abdominal_pain,stomach_pain", "Ayurveda"],
    ["Arthritis", "joint_pain,swelling_joints,movement_stiffness", "Ayurveda"],
    ["Cervical spondylosis", "neck_pain,stiff_neck,movement_stiffness", "Ayurveda"],
    ["Osteoarthristis", "knee_pain,hip_joint_pain,joint_pain", "Ayurveda"],
    ["Hypertension", "headache,dizziness,palpitations", "Ayurveda"],
    ["Obesity", "obesity,weight_gain,increased_appetite", "Ayurveda"],
    ["Anxiety", "anxiety,restlessness,mood_swings", "Ayurveda"],
    ["Fatigue", "fatigue,weakness,lethargy", "Ayurveda"],
    ["Acne", "pus_filled_pimples,blackheads,skin_rash", "Ayurveda"],
    ["Psoriasis", "skin_rash,silver_like_dusting,skin_peeling", "Ayurveda"],
    ["Varicose veins", "swollen_legs,swollen_blood_vessels,painful_walking", "Ayurveda"],


    # ---------------- HOMEOPATHY ----------------
    # These are project-category labels only.
    # They should NOT be interpreted as proof that
    # homeopathy treats these diseases.

    ["Common Cold", "continuous_sneezing,runny_nose,congestion", "Homeopathy"],
    ["Allergy", "continuous_sneezing,watering_from_eyes,runny_nose", "Homeopathy"],
    ["Cough", "cough,throat_irritation,phlegm", "Homeopathy"],
    ["Sinusitis", "sinus_pressure,headache,congestion", "Homeopathy"],
    ["Sore throat", "patches_in_throat,throat_irritation,cough", "Homeopathy"],
    ["Nausea", "nausea,vomiting,stomach_pain", "Homeopathy"],
    ["Vertigo", "dizziness,spinning_movements,loss_of_balance", "Homeopathy"],
    ["Headache", "headache,dizziness,fatigue", "Homeopathy"],
    ["Insomnia", "restlessness,anxiety,fatigue", "Homeopathy"],
    ["Stress", "anxiety,restlessness,lack_of_concentration", "Homeopathy"],
    ["Skin irritation", "itching,skin_rash,redness_of_eyes", "Homeopathy"],
    ["Muscle pain", "muscle_pain,muscle_weakness,fatigue", "Homeopathy"],
]


# ============================================================
# CREATE DATAFRAME
# ============================================================

columns = [
    "Disease",
    "Symptoms",
    "Medical_System"
]

df = pd.DataFrame(data, columns=columns)


# ============================================================
# CREATE DIRECTORY
# ============================================================

output_directory = "datasets/recommendation"

os.makedirs(
    output_directory,
    exist_ok=True
)


# ============================================================
# SAVE DATASET
# ============================================================

output_file = (
    f"{output_directory}/Medical_System_Training.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n==========================================")
print("MEDICAL SYSTEM TRAINING DATASET")
print("==========================================")

print(f"\nTotal records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nMedical System Distribution:")
print(
    df["Medical_System"].value_counts()
)

print("\nDataset Preview:")
print(
    df.head(10).to_string(index=False)
)

print("\n==========================================")
print("Dataset created successfully!")
print("==========================================")

print(f"\nSaved at:")
print(output_file)
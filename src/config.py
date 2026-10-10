"""Central configuration for VitalSign.

All paths are computed relative to the project root, so the project works
wherever the ZIP is extracted (no absolute / Windows-specific paths).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = PROJECT_ROOT / "app" / "static" / "charts"
ML_MODELS_DIR = PROJECT_ROOT / "src" / "ML models"

DATASET_PATH = DATA_DIR / "vital_sign.csv"
APP_TITLE = "Vital Sign Analysis & Prediction System"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------- target ----
# The dataset's own outcome column is "readmitted" with values NO / >30 / <30.
# DECISION (documented in README): we predict whether a patient is READMITTED
# to hospital at all (">30" or "<30" -> 1, "NO" -> 0). This binary target is
# reasonably balanced, which gives meaningful precision / recall / ROC-AUC.
TARGET_RAW = "readmitted"
TARGET = "readmitted_any"
TARGET_LABELS = {0: "Not readmitted", 1: "Readmitted"}

# ------------------------------------------------------------- column roles --
ID_COLS = ["encounter_id", "patient_nbr"]            # identifiers: never features
# Integer columns that are really category codes, not quantities
CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
AGE_COL = "age"                                       # "[50-60)" style band
AGE_NUM = "age_num"                                   # band mid-point (numeric feature)
AGE_MAP = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
           "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
# Columns kept in the cleaned frame for EDA but never used as model inputs
NON_FEATURE_COLS = [AGE_COL, TARGET_RAW, TARGET]

MISSING_DROP_THRESHOLD = 0.90   # drop columns with more than 90 % missing values
# Discharged to hospice / expired: these patients cannot be readmitted
EXCLUDED_DISCHARGE = [11, 13, 14, 19, 20, 21]

DRUG_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
             "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
             "examide", "citoglipton", "insulin", "glyburide-metformin",
             "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
             "metformin-pioglitazone"]

# Human-readable names for the numeric category codes (from the public
# documentation of this dataset; the CSV itself only contains the codes).
CODE_LABELS = {
    "admission_type_id": {
        "1": "Emergency", "2": "Urgent", "3": "Elective", "4": "Newborn",
        "5": "Not available", "6": "Unknown", "7": "Trauma centre", "8": "Not mapped"},
    "discharge_disposition_id": {
        "1": "Discharged to home", "2": "Short-term hospital", "3": "Skilled nursing facility",
        "4": "Intermediate care facility", "5": "Other inpatient institution",
        "6": "Home with home health service", "7": "Left against medical advice",
        "8": "Home with IV services", "9": "Admitted as inpatient here",
        "10": "Neonate to another hospital", "12": "Still a patient / outpatient",
        "15": "Swing bed", "16": "Outpatient - other institution",
        "17": "Outpatient - this institution", "18": "Unknown", "22": "Rehab facility",
        "23": "Long-term care hospital", "24": "Nursing facility (Medicaid)",
        "25": "Not mapped", "27": "Federal health facility", "28": "Psychiatric hospital",
        "29": "Critical access hospital", "30": "Other health institution"},
    "admission_source_id": {
        "1": "Physician referral", "2": "Clinic referral", "3": "HMO referral",
        "4": "Transfer from hospital", "5": "Transfer from skilled nursing facility",
        "6": "Transfer from other facility", "7": "Emergency room", "8": "Court / law enforcement",
        "9": "Not available", "10": "Transfer from critical access hospital",
        "11": "Normal delivery", "13": "Sick baby", "14": "Extramural birth", "17": "Unknown",
        "20": "Not mapped", "22": "Transfer within hospital", "25": "Transfer from ambulatory surgery"},
}

# Friendly labels for form fields / tables
FEATURE_LABELS = {
    "race": "Race", "gender": "Gender", "age_num": "Age band",
    "admission_type_id": "Admission type", "discharge_disposition_id": "Discharge disposition",
    "admission_source_id": "Admission source", "payer_code": "Payer code",
    "medical_specialty": "Medical specialty", "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Lab procedures", "num_procedures": "Other procedures",
    "num_medications": "Medications given", "number_outpatient": "Outpatient visits (prior year)",
    "number_emergency": "Emergency visits (prior year)", "number_inpatient": "Inpatient visits (prior year)",
    "number_diagnoses": "Number of diagnoses", "diag_1": "Primary diagnosis group",
    "diag_2": "Secondary diagnosis group", "diag_3": "Additional diagnosis group",
    "max_glu_serum": "Max glucose serum test", "A1Cresult": "HbA1c test result",
    "change": "Medication change", "diabetesMed": "Diabetes medication prescribed",
}

# Form groups: (title, columns, open-by-default)
FORM_GROUPS = [
    ("Patient", ["race", "gender", "age_num"], True),
    ("Admission", ["admission_type_id", "admission_source_id", "discharge_disposition_id",
                   "payer_code", "medical_specialty", "time_in_hospital"], True),
    ("Hospital utilisation", ["num_lab_procedures", "num_procedures", "num_medications",
                              "number_outpatient", "number_emergency", "number_inpatient",
                              "number_diagnoses"], True),
    ("Diagnoses", ["diag_1", "diag_2", "diag_3"], False),
    ("Lab results", ["max_glu_serum", "A1Cresult"], False),
    ("Diabetes medications", DRUG_COLS + ["change", "diabetesMed"], False),
]

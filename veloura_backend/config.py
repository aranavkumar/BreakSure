"""
Veloura Agentic AI - Configuration & Constants
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 7 Skin Lesion Classes (HAM10000)
CLASS_NAMES = [
    "akiec",  # Actinic keratoses and intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma (malignant)
    "nv",     # Melanocytic nevi (moles)
    "vasc"    # Vascular lesions
]

CLASS_INFO = {
    "mel": {
        "name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "risk_level": "CRITICAL",
        "urgency": "URGENT_BIOPSY_REQUIRED",
        "color": "#e53e3e"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "category": "Malignant Non-Melanoma",
        "risk_level": "HIGH",
        "urgency": "SPECIALIST_REFERRAL",
        "color": "#dd6b20"
    },
    "akiec": {
        "name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "category": "Pre-Malignant / In-Situ",
        "risk_level": "MODERATE",
        "urgency": "CLINICAL_MONITORING",
        "color": "#d69e2e"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "category": "Benign Lesion",
        "risk_level": "LOW",
        "urgency": "ROUTINE_OBSERVATION",
        "color": "#38a169"
    },
    "nv": {
        "name": "Melanocytic Nevus (Common Mole)",
        "category": "Benign Lesion",
        "risk_level": "LOW",
        "urgency": "BENIGN_BASELINE",
        "color": "#319795"
    },
    "df": {
        "name": "Dermatofibroma",
        "category": "Benign Fibrous Nodule",
        "risk_level": "LOW",
        "urgency": "BENIGN_OBSERVATION",
        "color": "#3182ce"
    },
    "vasc": {
        "name": "Vascular Lesion (Hemangioma / Pyogenic)",
        "category": "Benign Vascular Proliferation",
        "risk_level": "LOW",
        "urgency": "ROUTINE_OBSERVATION",
        "color": "#805ad5"
    }
}

# Image Quality Thresholds (from Phase 3)
QUALITY_CONFIG = {
    "min_resolution": 224,
    "blur_threshold": 50.0,
    "min_brightness": 35.0,
    "max_brightness": 220.0,
    "min_contrast": 25.0,
    "max_glare_fraction": 0.15
}

# Safety Engine Rules (from Phase 7)
SAFETY_RULES = {
    "MISSING_PATIENT_CONTEXT": True,
    "EXPERIMENTAL_FEATURES": True,
    "UNVALIDATED_SEGMENTATION": True,
    "MANDATORY_HUMAN_REVIEW": True,
    "AUTOMATIC_ACCEPTANCE_BLOCKED": True,
    "INDEPENDENT_DIAGNOSIS_PROHIBITED": True
}

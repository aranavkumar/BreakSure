"""
Veloura Phase 6 - Reasoning & Context Agent
Combines visual features, patient metadata, and clinical constraints
into a structured clinical reasoning summary.
"""
import time

class ReasoningContextAgent:
    def __init__(self):
        self.agent_name = "Veloura Reasoning & Context Agent"

    def analyze(self, diagnosis_result: dict, lesion_features: dict, patient_context: dict = None) -> dict:
        start_time = time.perf_counter()
        patient_context = patient_context or {}

        # 1. Identify missing information
        missing = []
        if "age" not in patient_context or patient_context.get("age") is None:
            missing.append("patient_age")
        if "sex" not in patient_context or not patient_context.get("sex"):
            missing.append("patient_sex")
        if "anatomical_site" not in patient_context or not patient_context.get("anatomical_site"):
            missing.append("anatomical_location")
        if "history_of_melanoma" not in patient_context:
            missing.append("personal_family_history")
        if "evolution" not in patient_context:
            missing.append("lesion_evolution_or_changes")

        # 2. Risk factor stratification
        risk_factors = []
        age = patient_context.get("age")
        if age and age >= 55:
            risk_factors.append("ADVANCED_AGE_ELEVATED_RISK")

        site = str(patient_context.get("anatomical_site", "")).lower()
        if any(k in site for k in ["back", "trunk", "chest", "shoulder", "scalp"]):
            risk_factors.append("HIGH_RISK_ANATOMICAL_SITE")

        if patient_context.get("history_of_melanoma"):
            risk_factors.append("PRIOR_MELANOMA_HISTORY")

        if patient_context.get("evolution") in ["GROWING", "BLEEDING", "ITCHING", "CHANGING"]:
            risk_factors.append("DYNAMIC_LESION_EVOLUTION")

        # 3. Clinical consistency check between ABCD and CNN prediction
        consistency_notes = []
        primary = diagnosis_result.get("primary_diagnosis", {})
        code = primary.get("code")

        if lesion_features:
            asymmetry = lesion_features.get("asymmetry", {}).get("score", 0)
            color_var = lesion_features.get("color_variation", {}).get("score", 0)
            if code == "mel" and (asymmetry > 0.4 or color_var > 20):
                consistency_notes.append("CNN melanoma prediction correlates with elevated ABCD asymmetry and color variation.")
            elif code == "nv" and asymmetry > 0.5:
                consistency_notes.append("DISCORDANCE: CNN predicts benign nevus, but ABCD shows marked asymmetry. Closer examination warranted.")

        execution_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "agent": self.agent_name,
            "patient_context": {
                "age": patient_context.get("age", "NOT_PROVIDED"),
                "sex": patient_context.get("sex", "NOT_PROVIDED"),
                "anatomical_site": patient_context.get("anatomical_site", "NOT_PROVIDED"),
                "history_of_melanoma": patient_context.get("history_of_melanoma", False),
                "evolution": patient_context.get("evolution", "STABLE")
            },
            "missing_information": missing,
            "clinical_risk_factors": risk_factors,
            "clinical_consistency_notes": consistency_notes,
            "context_completeness_percent": round((1 - (len(missing) / 5)) * 100, 1),
            "execution_ms": execution_ms
        }

"""
Veloura Phase 7 - Safety Agent
Evaluates strict clinical safety rules, regulatory guardrails (FDA/CE MDR),
and determines human dermatologist review requirements.
"""
import time
from veloura_backend.config import SAFETY_RULES

class SafetyAgent:
    def __init__(self, rules=None):
        self.agent_name = "Veloura Clinical Safety Agent"
        self.rules = rules or SAFETY_RULES

    def evaluate(self, image_quality: dict, diagnosis: dict, lesion: dict, reasoning: dict) -> dict:
        start_time = time.perf_counter()

        flags = []
        warnings = []
        requires_immediate_escalation = False

        # Rule 1: Quality check gate
        if image_quality.get("status") == "REJECT":
            flags.append("IMAGE_QUALITY_DEFICIENT")
            warnings.append("Image resolution, blur, or glare failed clinical threshold. Re-imaging required.")
            requires_immediate_escalation = True

        # Rule 2: Missing patient information
        missing = reasoning.get("missing_information", [])
        if missing:
            flags.append("MISSING_PATIENT_CONTEXT")
            warnings.append(f"Incomplete patient clinical context: missing {', '.join(missing)}.")

        # Rule 3: High-risk malignant lesion detected
        primary = diagnosis.get("primary_diagnosis", {})
        if primary.get("code") in ["mel", "bcc"]:
            flags.append("MALIGNANCY_SUSPECTED")
            warnings.append(f"High risk classification: {primary.get('name')} suspected. Urgent in-person biopsy or dermoscopy evaluation required.")
            requires_immediate_escalation = True

        # Rule 4: Low model confidence
        confidence = primary.get("confidence_percent", 0)
        if confidence < 65.0:
            flags.append("LOW_DIAGNOSTIC_CONFIDENCE")
            warnings.append(f"Model diagnostic confidence ({confidence}%) is below clinical safety threshold (65%).")
            requires_immediate_escalation = True

        # Rule 5: Experimental feature evidence
        flags.append("EXPERIMENTAL_FEATURES")
        warnings.append("ABCD measurements are algorithmic estimates and must not replace dermatopathology.")

        # Mandatory regulatory guardrails
        flags.append("MANDATORY_HUMAN_REVIEW")
        flags.append("AUTOMATIC_ACCEPTANCE_BLOCKED")
        flags.append("INDEPENDENT_DIAGNOSIS_PROHIBITED")

        if requires_immediate_escalation:
            safety_decision = "BLOCKED_PENDING_HUMAN_REVIEW"
            triage_tier = "URGENT_CLINICAL_ESCALATION"
        elif image_quality.get("status") == "REVIEW_REQUIRED":
            safety_decision = "HUMAN_REVIEW_REQUIRED"
            triage_tier = "PRIORITY_SPECIALIST_REVIEW"
        else:
            safety_decision = "NO_AUTOMATED_CLINICAL_CLEARANCE"
            triage_tier = "ROUTINE_DERMATOLOGY_TRIAGE"

        execution_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "agent": self.agent_name,
            "safety_decision": safety_decision,
            "triage_tier": triage_tier,
            "safety_gate_passed": safety_decision == "NO_AUTOMATED_CLINICAL_CLEARANCE",
            "regulatory_status": "INVESTIGATIONAL_DEVICE_RESEARCH_ONLY",
            "safety_flags": flags,
            "clinical_warnings": warnings,
            "execution_ms": execution_ms
        }

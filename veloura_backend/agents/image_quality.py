"""
Veloura Phase 3 - Image Quality Agent
Validates resolution, blur, brightness, contrast, and glare before clinical processing.
"""
import cv2
import numpy as np
import time
from veloura_backend.config import QUALITY_CONFIG

class ImageQualityAgent:
    def __init__(self, config=None):
        self.config = config or QUALITY_CONFIG
        self.agent_name = "Veloura Image Quality Agent"

    def analyze(self, image_np: np.ndarray) -> dict:
        start_time = time.perf_counter()
        
        if image_np is None or image_np.size == 0:
            return {
                "agent": self.agent_name,
                "status": "INVALID",
                "quality_score": 0,
                "issues": ["IMAGE_DATA_CORRUPT_OR_EMPTY"],
                "recommendation": "RE_IMAGE",
                "execution_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        height, width = gray.shape

        # 1. Blur measurement using Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Brightness (mean pixel intensity)
        brightness = float(np.mean(gray))

        # 3. Contrast (standard deviation of pixel intensity)
        contrast = float(np.std(gray))

        # 4. Glare fraction (percentage of saturated pixels >= 245)
        glare_fraction = float(np.mean(gray >= 245))

        # 5. Quality scoring & rule checks
        issues = []
        score = 100

        if min(width, height) < self.config["min_resolution"]:
            issues.append("LOW_RESOLUTION")
            score -= 35

        if blur_score < self.config["blur_threshold"]:
            issues.append("EXCESSIVE_BLUR")
            score -= 30

        if brightness < self.config["min_brightness"]:
            issues.append("UNDEREXPOSED")
            score -= 20
        elif brightness > self.config["max_brightness"]:
            issues.append("OVEREXPOSED")
            score -= 20

        if contrast < self.config["min_contrast"]:
            issues.append("LOW_CONTRAST")
            score -= 15

        if glare_fraction > self.config["max_glare_fraction"]:
            issues.append("EXCESSIVE_GLARE")
            score -= 25

        score = max(0, min(100, score))

        if score >= 75 and not issues:
            status = "PASS"
            recommendation = "PROCEED_TO_ANALYSIS"
        elif score >= 50:
            status = "REVIEW_REQUIRED"
            recommendation = "PROCEED_WITH_CAUTION"
        else:
            status = "REJECT"
            recommendation = "RE_IMAGE_RECOMMENDED"

        execution_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "agent": self.agent_name,
            "status": status,
            "quality_score": score,
            "metrics": {
                "resolution": f"{width}x{height}",
                "blur_score": round(blur_score, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "glare_fraction": round(glare_fraction * 100, 2)
            },
            "issues": issues,
            "recommendation": recommendation,
            "execution_ms": execution_ms
        }

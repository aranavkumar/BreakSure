"""
Veloura Phase 5 - Lesion Analysis Agent
Performs lesion segmentation (GrabCut), contour boundary extraction,
and computes experimental ABCD features (Asymmetry, Border, Color, Diameter).
"""
import cv2
import numpy as np
import time
import base64

class LesionAnalysisAgent:
    def __init__(self):
        self.agent_name = "Veloura Lesion Analysis Agent"

    def analyze(self, image_bgr: np.ndarray) -> dict:
        start_time = time.perf_counter()

        if image_bgr is None or image_bgr.size == 0:
            return {
                "agent": self.agent_name,
                "status": "ERROR",
                "error": "IMAGE_EMPTY_OR_UNREADABLE",
                "features": None,
                "execution_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        height, width = image_bgr.shape[:2]

        # Step 1: Candidate GrabCut Segmentation
        mask, segmentation_result = self._segment_lesion(image_bgr)

        # Step 2: Feature extraction (ABCD)
        if mask is not None:
            features = self._extract_abcd_features(image_bgr, mask)
            overlay_base64 = self._generate_overlay_base64(image_bgr, mask)
        else:
            features = None
            overlay_base64 = None

        execution_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "agent": self.agent_name,
            "status": segmentation_result.get("status", "REVIEW_REQUIRED"),
            "segmentation": segmentation_result,
            "features": features,
            "overlay_image_base64": overlay_base64,
            "execution_ms": execution_ms
        }

    def _segment_lesion(self, image_bgr: np.ndarray):
        height, width = image_bgr.shape[:2]
        mask = np.zeros((height, width), np.uint8)

        # Central rectangle search region (70% coverage)
        x = int(width * 0.15)
        y = int(height * 0.15)
        rect_w = int(width * 0.70)
        rect_h = int(height * 0.70)
        rect = (x, y, rect_w, rect_h)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT
            )
            binary_mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return binary_mask, {
                    "status": "LOW_RELIABILITY",
                    "reason": "NO_FOREGROUND_COMPONENT"
                }

            largest_contour = max(contours, key=cv2.contourArea)
            cleaned_mask = np.zeros_like(binary_mask)
            cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            area = cv2.contourArea(largest_contour)
            area_fraction = area / (height * width)

            reasons = []
            if area_fraction < 0.01:
                reasons.append("SEGMENT_TOO_SMALL")
            if area_fraction > 0.65:
                reasons.append("SEGMENT_TOO_LARGE")
            if area < 100:
                reasons.append("INSUFFICIENT_SEGMENT_AREA")

            status = "LOW_RELIABILITY" if reasons else "REVIEW_REQUIRED"

            return cleaned_mask, {
                "status": status,
                "area_pixels": int(area),
                "area_fraction": round(float(area_fraction), 4),
                "issues": reasons
            }
        except Exception as e:
            # Fallback simple Otsu threshold on grayscale
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return binary_mask, {
                "status": "FALLBACK_THRESHOLD",
                "reason": str(e)
            }

    def _extract_abcd_features(self, image_bgr: np.ndarray, mask: np.ndarray) -> dict:
        binary = (mask > 0).astype(np.uint8)

        # 1. Asymmetry (A)
        moments = cv2.moments(binary)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            x1 = max(0, 2 * cx - binary.shape[1])
            x2 = min(binary.shape[1], 2 * cx + 1)
            y1 = max(0, 2 * cy - binary.shape[0])
            y2 = min(binary.shape[0], 2 * cy + 1)

            cropped = binary[y1:y2, x1:x2]
            if cropped.size > 0:
                h_diff = np.mean(np.abs(cropped - np.fliplr(cropped)))
                v_diff = np.mean(np.abs(cropped - np.flipud(cropped)))
                asymmetry_score = float((h_diff + v_diff) / 2)
            else:
                asymmetry_score = 0.5
        else:
            asymmetry_score = 0.5

        # 2. Border Irregularity (B)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter > 0 and area > 0:
                circularity = float(np.clip(4 * np.pi * area / (perimeter ** 2), 0, 1))
                border_irregularity = round(1.0 - circularity, 3)
            else:
                border_irregularity = 0.5
        else:
            border_irregularity = 0.5

        # 3. Color Variation (C)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        selected_pixels = hsv[binary > 0]
        if len(selected_pixels) > 10:
            hue_std = float(np.std(selected_pixels[:, 0]))
            sat_std = float(np.std(selected_pixels[:, 1]))
            val_std = float(np.std(selected_pixels[:, 2]))
            color_variation_score = round((hue_std + sat_std + val_std) / 3.0, 2)
        else:
            color_variation_score = 15.0

        # 4. Diameter / Size (D)
        area_pixels = int(np.sum(binary))
        equivalent_diameter = round(float(2 * np.sqrt(area_pixels / np.pi)), 1)

        return {
            "asymmetry": {
                "score": round(asymmetry_score, 3),
                "interpretation": "HIGH_ASYMMETRY" if asymmetry_score > 0.4 else "SYMMETRICAL"
            },
            "border_irregularity": {
                "score": border_irregularity,
                "interpretation": "IRREGULAR_MARGINS" if border_irregularity > 0.4 else "SMOOTH_CIRCULAR"
            },
            "color_variation": {
                "score": color_variation_score,
                "interpretation": "POLYCHROMATIC_SUSPICIOUS" if color_variation_score > 25.0 else "UNIFORM"
            },
            "diameter": {
                "equivalent_diameter_px": equivalent_diameter,
                "area_pixels": area_pixels
            },
            "feature_status": "EXPERIMENTAL_UNVALIDATED",
            "clinical_warning": "Measurements are technical approximations and must be interpreted by a board-certified dermatologist."
        }

    def _generate_overlay_base64(self, image_bgr: np.ndarray, mask: np.ndarray) -> str:
        overlay = image_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw bright green boundary and translucent fill
        cv2.drawContours(overlay, contours, -1, (0, 255, 128), 2)
        color_mask = np.zeros_like(image_bgr)
        color_mask[mask > 0] = (0, 180, 255) # Amber tint
        combined = cv2.addWeighted(overlay, 0.85, color_mask, 0.15, 0)
        
        _, buffer = cv2.imencode(".png", combined)
        return base64.b64encode(buffer).decode("utf-8")

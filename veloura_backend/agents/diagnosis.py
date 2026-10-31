"""
Veloura Phase 4 & 13 - Diagnosis Agent
Deep Learning classifier for 7 skin lesion types with Test-Time Augmentation (TTA)
and automated checkpoint loading.
"""
import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from veloura_backend.config import CLASS_NAMES, CLASS_INFO, MODEL_DIR

class DiagnosisAgent:
    def __init__(self, model_path: str = None, device: str = None):
        self.agent_name = "Veloura Diagnosis Agent"
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.class_names = CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.model, self.checkpoint_name, self.architecture = self._load_model(model_path)
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, custom_path=None):
        candidate_files = []
        if custom_path:
            candidate_files.append(custom_path)
            
        candidate_files.extend([
            MODEL_DIR / "veloura_efficientnet_b3_90plus.pth",
            MODEL_DIR / "veloura_diagnosis_agent_final.pth",
            MODEL_DIR / "veloura_efficientnet_b0_best.pth",
            MODEL_DIR / "mobilenetv3_phase13_pilot_epoch1.pth"
        ])

        found_checkpoint = None
        for path in candidate_files:
            if os.path.isfile(path):
                found_checkpoint = str(path)
                break

        # Architecture selection
        if found_checkpoint and "b3" in found_checkpoint.lower():
            try:
                model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            except Exception:
                model = models.efficientnet_b3(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            arch = "EfficientNet-B3 (High Resolution)"
        else:
            try:
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            except Exception:
                model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            arch = "EfficientNet-B0"

        if found_checkpoint:
            try:
                ckpt = torch.load(found_checkpoint, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                ckpt_label = os.path.basename(found_checkpoint)
            except Exception as e:
                ckpt_label = f"Pretrained Baseline (failed loading {os.path.basename(found_checkpoint)}: {e})"
        else:
            ckpt_label = "Pretrained ImageNet Weights (drop .pth into backend/models for fine-tuned)"

        model = model.to(self.device)
        return model, ckpt_label, arch

    def predict(self, image_pil: Image.Image, use_tta: bool = True) -> dict:
        start_time = time.perf_counter()

        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # Base tensor
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if use_tta:
                # Test-Time Augmentation (TTA) with flips for higher stability & accuracy
                t_hflip = torch.flip(input_tensor, dims=[3])
                t_vflip = torch.flip(input_tensor, dims=[2])
                batch = torch.cat([input_tensor, t_hflip, t_vflip], dim=0)

                logits = self.model(batch)
                probs_batch = torch.softmax(logits, dim=1)
                probs = torch.mean(probs_batch, dim=0).cpu().numpy()
            else:
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        predicted_code = self.class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Full probability distribution mapped with metadata
        distribution = {}
        for idx, code in enumerate(self.class_names):
            info = CLASS_INFO.get(code, {})
            distribution[code] = {
                "name": info.get("name", code),
                "probability": round(float(probs[idx]) * 100, 2),
                "category": info.get("category", ""),
                "risk_level": info.get("risk_level", "LOW"),
                "color": info.get("color", "#4a5568")
            }

        # Sorted differential diagnosis
        sorted_diff = sorted(
            distribution.items(),
            key=lambda item: item[1]["probability"],
            reverse=True
        )

        top_differential = [
            {
                "code": item[0],
                **item[1]
            }
            for item in sorted_diff[:3]
        ]

        # Calculate prediction entropy (uncertainty metric)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

        execution_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "agent": self.agent_name,
            "architecture": self.architecture,
            "checkpoint": self.checkpoint_name,
            "device": str(self.device),
            "primary_diagnosis": {
                "code": predicted_code,
                "name": CLASS_INFO.get(predicted_code, {}).get("name", predicted_code),
                "category": CLASS_INFO.get(predicted_code, {}).get("category", ""),
                "risk_level": CLASS_INFO.get(predicted_code, {}).get("risk_level", "LOW"),
                "urgency": CLASS_INFO.get(predicted_code, {}).get("urgency", "ROUTINE"),
                "confidence_percent": round(confidence * 100, 2)
            },
            "top_differential": top_differential,
            "all_probabilities": distribution,
            "uncertainty_entropy": round(entropy, 3),
            "test_time_augmentation": use_tta,
            "execution_ms": execution_ms
        }

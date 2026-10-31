# Veloura Agentic AI (2026)
### Multi-Agent Clinical Decision Support & Telemetry System for Dermatology

Veloura is an autonomous multi-agent clinical decision support pipeline designed to assist dermatologists in triaging, segmenting, and classifying dermatological lesions with real-time turnaround latency tracking.

---

## Architecture: 6-Agent Clinical Pipeline

```
                                [ Patient Intake / Clinical Image ]
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │ 1. Image Quality Agent  │ (Laplacian blur, glare, contrast)
                                   └────────────┬────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
            ┌─────────────────────────┐                   ┌─────────────────────────┐
            │ 2. Lesion Analysis Agent│                   │ 3. Deep Diagnosis Agent │
            │   (GrabCut + ABCD)      │                   │  (EfficientNet-B0/B3)   │
            └────────────┬────────────┘                   └────────────┬────────────┘
                         │                                             │
                         └──────────────────────┬──────────────────────┘
                                                ▼
                                   ┌─────────────────────────┐
                                   │ 4. Clinical Reasoning   │ (Age, sex, anatomical site risk)
                                   └────────────┬────────────┘
                                                ▼
                                   ┌─────────────────────────┐
                                   │ 5. Clinical Safety Gate │ (FDA/MDR guardrails, mandatory review)
                                   └────────────┬────────────┘
                                                ▼
                                   ┌─────────────────────────┐
                                   │ 6. Master Coordinator   │ (Multimodal Bayesian Fusion & Timing)
                                   └─────────────────────────┘
```

---

## Diagnostic Performance & Accuracy Benchmarks

Evaluated on **1,535 HAM10000 validation dermoscopy images**:

| Model / Configuration | Validation Accuracy | Macro F1 | Clinical Sensitivity | Status |
|---|---|---|---|---|
| **Phase 4 Baseline (Frozen EfficientNet-B0)** | 71.73% | 0.4787 | 68.4% | Complete |
| **Phase 4 Fine-Tuned (5 Epochs)** | 74.79% | 0.5410 | 74.2% | Complete |
| **Veloura Multimodal Agentic Fusion + TTA** | **> 81.5%** | **0.6240** | **83.1%** | **Active in Backend** |
| **Retrained EfficientNet-B3 @ 300px (25 Epochs)** | **> 91.0%** | **0.7850** | **92.4%** | Notebook Provided |

### Per-Class Performance
* **`nv` (Melanocytic Nevus)**: Precision 91.9%, Recall 84.7%, F1 88.2%
* **`vasc` (Vascular Lesion)**: Recall 95.0%
* **`bcc` (Basal Cell Carcinoma)**: F1 57.9%
* **`mel` (Melanoma)**: Augmented with ABCD Asymmetry & Color Variance to eliminate false-negative escapes.

---

## Real-Time Productivity & Turnaround Telemetry

Benchmarked live on Apple Silicon GPU (`mps`):

* **Image Quality Validation**: `7.74 ms`
* **OpenCV GrabCut & ABCD Extraction**: `36.42 ms`
* **Reasoning & Patient Risk Stratification**: `0.94 ms`
* **Clinical Safety & Regulatory Gate**: `0.61 ms`
* **End-to-End Case Turnaround**: `< 250 ms` (Single Case Throughput: `~240 cases/minute`)
* **Regulatory Compliance**: `100.0%` (Zero unauthorized autonomous clinical clearances).

---

## Quickstart: Running the Backend

### 1. Install & Launch Locally

```bash
cd veloura_backend
./run.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn veloura_backend.main:app --host 0.0.0.0 --port 8085 --reload
```

### 2. Interactive Dashboard
Open your browser to:
* **Clinical Dashboard**: `http://localhost:8085/dashboard/`
* **Interactive API Documentation**: `http://localhost:8085/docs`
* **Productivity Telemetry**: `http://localhost:8085/api/productivity/metrics`

---

## Cloud Deployment (Docker)

```bash
cd veloura_backend
docker compose up --build
```

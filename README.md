# Deepfake & Image Manipulation Detection
## Project 4 — Full System (Parts 1 + 2 + 3 + Bonus)

A multi-layer deepfake and image manipulation detection system combining CNN-based
classification, image forensics, explainability heatmaps, reverse image search,
and eye-blink temporal analysis.

---

## Team

| Part | Member | Responsibility |
|------|--------|----------------|
| Part 1 | Adam   | Dataset loader, ELA preprocessing, MobileNetV2 CNN classifier, shared report structure |
| Part 2 | Amr    | Image splicing detection, AI-generated content detection, JPEG compression analysis |
| Part 3 | Mazen  | Explainability heatmaps, reverse image search, HTML/PDF report generator, eye-blink bonus |

---

## Setup

### Requirements
- Python **3.9 – 3.12** recommended
- Python 3.13 is supported but MediaPipe will not work (OpenCV fallback used automatically)

### 1. Clone / open the project folder

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Install MediaPipe for higher-accuracy blink analysis
```bash
# Python 3.9–3.12 only
pip install mediapipe
```

---

## Quick Start — No Data Needed

Run the full pipeline sanity check with random dummy images:
```bash
python main.py --mode test
```

This verifies all modules (Parts 1, 2, 3) are working without requiring real images.

---

## Running with Real Data

### Step 1 — Add images
```
data/
├── real/    ← put real face images here  (.jpg, .png)
└── fake/    ← put deepfake images here   (.jpg, .png)
```

### Step 2 — Train the CNN model
```bash
python main.py --mode train
```
The best model is saved automatically to `models/deepfake_detector.keras`.

### Step 3 — Analyse a single image
```bash
python main.py --mode predict --image data/fake/your_image.jpg
```
Runs all three parts and saves a full HTML + JSON report to `outputs/reports/`.

---

## Eye-Blink Temporal Analysis (Bonus)

Standalone module for video-based deepfake detection via blink pattern analysis.

```bash
# Analyse a video (auto-detects backend)
python blink_analyzer.py --video path/to/video.mp4

# Force OpenCV backend (works on all Python versions including 3.13)
python blink_analyzer.py --video path/to/video.mp4 --backend opencv

# Save EAR timeline plot to outputs/heatmaps/
python blink_analyzer.py --video path/to/video.mp4 --backend opencv --plot

# Sanity check with dummy frames (no video needed)
python blink_analyzer.py --test
```

### Blink Analysis — Important Notes
- Requires **≥ 5 seconds** of video for reliable scoring (≥ 15s recommended)
- SDFVD dataset clips (~2.7s) are below this threshold — the module returns a
  neutral score and logs a warning rather than producing a false result
- MediaPipe gives better landmark accuracy but requires Python 3.9–3.12
- OpenCV fallback works on all platforms and Python versions

### Blink Metrics Explained

| Metric | Natural Range | Deepfake Pattern |
|--------|--------------|-----------------|
| Blink rate | 12–22 / min | < 5 / min (too few) |
| Blink duration | 100–400 ms | < 80ms or > 450ms |
| L/R asymmetry | < 0.18 | > 0.25 (poor rendering) |
| Regularity (CV) | 0.15–1.0 | < 0.15 (robotic) or > 1.0 (chaotic) |
| Eye openness drift | ~0 slope | Progressive collapse |

---

## Recommended Datasets

### Images
| Dataset | Source |
|---------|--------|
| FaceForensics++ | https://github.com/ondyari/FaceForensics |
| DFDC (Deepfake Detection Challenge) | https://ai.meta.com/datasets/dfdc/ |
| Celeb-DF | https://github.com/yuezunli/celeb-deepfakeforensics |

### Videos (for blink analysis)
| Dataset | Min Clip Length | Notes |
|---------|----------------|-------|
| FaceForensics++ | 10–30s | Best for blink analysis |
| DFDC | 10–30s | Large variety |
| SDFVD | ~2.7s | Too short for blink analysis |

---

## File Structure

```
deepfake_detector/
│
├── data/
│   ├── real/                   ← real face images
│   ├── fake/                   ← deepfake images
│   └── reverse_image_db.json   ← perceptual hash database (auto-generated)
│
├── models/
│   └── deepfake_detector.keras ← trained model (saved after training)
│
├── outputs/
│   ├── reports/                ← JSON + HTML reports
│   ├── heatmaps/               ← gradient heatmaps + EAR timeline plots
│   └── training_history.png    ← accuracy/loss/AUC curves
│
│── Part 1 — Adam ───────────────────────────────────────────────────────────
├── dataset_loader.py           ← loads images, normalises, generates dummy data
├── preprocessor.py             ← ELA transform + face detection + crop pipeline
├── model.py                    ← MobileNetV2 CNN, training, inference, plotting
├── report_stub.py              ← shared report contract (ALL members use this)
│
│── Part 2 — Amr ────────────────────────────────────────────────────────────
├── splicing_detector.py        ← edge/lighting/noise inconsistency detection
├── ai_content_detector.py      ← DCT frequency + co-occurrence + GAN artifact analysis
├── compression_analyzer.py     ← JPEG block boundary + double-compression detection
├── analyzer_part2.py           ← Part 2 pipeline entry point
│
│── Part 3 — Mazen ──────────────────────────────────────────────────────────
├── simple_explainer.py         ← gradient-based heatmap (primary explainer)
├── gradcam_explainer.py        ← Grad-CAM heatmap (alternative explainer)
├── shap_explainer.py           ← SHAP explainability (may conflict with newer TF)
├── reverse_image_search.py     ← perceptual hash database + near-duplicate search
├── report_generator.py         ← HTML + PDF final report renderer
│
│── Bonus — Mazen ───────────────────────────────────────────────────────────
├── blink_analyzer.py           ← eye-blink temporal analysis for video input
│
│── Entry Point ─────────────────────────────────────────────────────────────
├── main.py                     ← runs full pipeline (train / predict / test)
├── requirements.txt            ← all dependencies
└── README.md                   ← this file
```

---

## How the Report System Works

All three parts write into the same shared report dictionary defined in `report_stub.py`.

```
create_report(image_path)
       │
       ├── Part 1 (Adam)   → deepfake_score, deepfake_label, deepfake_confidence
       ├── Part 2 (Amr)    → splicing_score, ai_generated_score, compression_score
       ├── Part 3 (Mazen)  → shap_heatmap_path, reverse_image_match
       └── Bonus  (Mazen)  → blink_score, blink_frames_analyzed, blink_metrics
                                      │
                              compute_final_verdict()
                                      │
                              Weighted score (deepfake 40%, splicing 25%,
                              ai_generated 20%, compression 15%)
                                      │
                              save_report()  +  create_final_report()
                              → outputs/reports/report_<name>_<timestamp>.json
                              → outputs/reports/report_<name>_<timestamp>.html
```

### Final Verdict Thresholds
| Score | Verdict | Risk Level |
|-------|---------|-----------|
| ≥ 0.85 | MANIPULATED | CRITICAL |
| ≥ 0.70 | MANIPULATED | HIGH |
| ≥ 0.40 | SUSPICIOUS  | MEDIUM |
| < 0.40 | AUTHENTIC   | LOW |

---

## Module Import Reference

### Part 2 imports from Part 1
```python
from preprocessor import preprocess_single, preprocess_batch, apply_ela
from report_stub  import create_report, update_report, add_evidence_flag
```

### Part 3 imports from Parts 1 & 2
```python
from model        import load_model, predict_single
from report_stub  import compute_final_verdict, print_report_summary, save_report
```

### Blink module integrates with report system
```python
from blink_analyzer import analyze_blink_from_video, update_report_with_blink
blink_score, metrics = analyze_blink_from_video("clip.mp4", backend="opencv")
report = update_report_with_blink(report, blink_score, metrics)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | MediaPipe ≥ 0.10 | Use `--backend opencv` |
| `AttributeError: function 'free' not found` | MediaPipe + Python 3.13 | Use `--backend opencv` |
| `FileNotFoundError: No saved model` | Model not trained yet | Run `python main.py --mode train` first |
| `blink_score: 0.0` on short clip | Clip < 5s (e.g. SDFVD) | Expected — clip too short to judge |
| `oneDNN custom operations` TF warning | TensorFlow info message | Harmless — set `TF_ENABLE_ONEDNN_OPTS=0` to suppress |
| SHAP `DeepExplainer` fails | TF/Keras version conflict | `simple_explainer.py` is the active fallback |

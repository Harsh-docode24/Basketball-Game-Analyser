# 🏀 Basketball Analyzer — Shot Detection & Form Confidence System

An end-to-end computer vision system that automatically detects the rim, tracks shot trajectories, evaluates shooting biomechanics, and generates structured performance summaries.

The system integrates object detection, pose estimation, geometric evaluation, and an interactive UI to provide deterministic and explainable shot analysis.

---

# 1. Project Overview

The Basketball Analyzer processes shooting videos to automatically determine:

- Shot outcome (MAKE / MISS)
- Form confidence at release
- Aggregate performance across multiple attempts

The system operates fully automatically and requires no manual calibration.

---

# 2. Problem Definition

Traditional basketball training often relies on subjective visual assessment. This project aims to:

- Automate shot outcome detection
- Quantify shooting form biomechanically
- Provide consistent, explainable analytics
- Generate structured coaching summaries

---

# 3. System Architecture

The system follows a modular pipeline:

1. Rim Detection (YOLO Model)
2. Pose Estimation (MediaPipe)
3. Shot Release Detection
4. Ball Tracking
5. Geometric Shot Evaluation
6. Statistical Aggregation
7. Annotated Video Rendering

Each module operates independently and contributes structured output to the final evaluation layer.

---

# 4. Core Components

## 4.1 Rim Detection

- Custom-trained YOLO model (`best.pt`)
- Model parameters defined in `config.yaml`
- Provides bounding box used as geometric reference

The rim detection output acts as the authoritative reference for shot evaluation.

---

## 4.2 Ball Tracking & Trajectory

- Tracks ball center coordinates frame-by-frame
- Computes geometric relationships with rim center
- Draws trajectory overlay on video

---

## 4.3 Shooting Form Evaluation (MediaPipe Pose)

At the release frame, the following biomechanical parameters are analyzed:

- Elbow extension
- Knee bend
- Wrist follow-through

These parameters are aggregated into a:

Form Confidence Score (0–100)

Example:
MAKE | Form Score: 48 / 100

---

# 5. Shot Outcome Logic (Deterministic & Explainable)

Shot evaluation is geometry-based:

MAKE  
Ball center passes within rim bounding area.

MISS  
Ball drops below rim without intersection.

No probabilistic heuristics are used.  
All decisions are deterministic and reproducible.

---

# 6. Multi-Shot Statistical Summary

After processing a full session, the system generates:

- Average Form Confidence
- Joint-level breakdown (Elbow, Knee, Wrist)
- Final shot tally

Example:

Average Form Confidence: 59 / 100  
Breakdown:  
Elbow: 89  
Knee: 1  
Wrist: 94  

Final Score: 25 / 30  

---

# 7. Annotated Video Output

The exported video includes:

- Pose skeleton overlay
- Ball trajectory path
- Rim bounding box
- Shot result overlay (MAKE / MISS)
- Per-shot form score

The annotated output can be downloaded directly from the Streamlit interface.

---

# 8. Technology Stack

- Python 3.9+
- OpenCV (Video Processing)
- MediaPipe Pose (Landmark Detection)
- YOLO (Ultralytics) – Rim Detection
- NumPy (Geometric Computation)
- Streamlit (Interactive UI)
- cvzone (Visualization Utilities)

---

# 9. Project Structure

Basketball-Game-Analyser/
│
├── analyzer.py          # Main Streamlit application  
├── trial.py             # Experimental modules  
├── best.pt              # Custom YOLO rim model  
├── config.yaml          # Model configuration  
├── requirements.txt     # Dependencies  
├── docs/                # Structured documentation modules  
└── README.md  

---

# 10. Installation

## 1. Create Virtual Environment

python -m venv venv

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

---

## 2. Install Dependencies

pip install -r requirements.txt

---

## 3. Run Application

streamlit run analyzer.py

Then:
- Upload a basketball video (.mp4 / .mov)
- Click "Analyze Shot"
- Review annotated video and statistical summary

---

# 11. Known Limitations

- Shot distance measured in pixel space (no real-world calibration)
- Ball detection may fail under heavy occlusion
- Single-player focus
- Net interaction not explicitly modeled

---

# 12. Future Improvements

- Real-world court calibration
- Multi-player tracking
- Shot arc efficiency metrics
- Net-based confirmation logic
- Performance analytics dashboard
- CSV export of shot data

---

# 13. Documentation Approach

This repository follows structured documentation principles:

- Clear hierarchical sections
- Modular system explanation
- Deterministic algorithm breakdown
- Reproducible installation steps

Future updates will include DITA XML–based documentation modules in the `/docs/dita` directory.

---

# 14. Author

Harsh 

---

# Why This Project Stands Out

- Custom-trained YOLO model
- Fully automatic rim detection
- Deterministic shot evaluation logic
- Integrated pose + object tracking
- End-to-end working system
- Structured and explainable output


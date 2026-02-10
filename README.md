🏀Basketball Analyzer — Shot Detector & Form Confidence

An end-to-end computer vision–based basketball shot analysis system that automatically detects the rim, tracks shots, evaluates shooting form, and produces a visual + statistical coaching summary.

The system uses:

A custom-trained YOLO model (best.pt) for rim detection

Ball trajectory tracking

MediaPipe Pose for form evaluation

Streamlit for an interactive UI

🚀 Project Overview

This project analyzes basketball shooting videos to answer:

Was the shot MADE or MISSED?

How good was the shooting form at release?

What is the overall performance across multiple shots?

All analysis is fully automatic — no manual calibration is required.

✨ Key Features
🎯 Automatic Rim Detection (YOLO)

Uses a custom-trained YOLO model (best.pt)

Model is trained using parameters defined in config.yaml

Detects the basketball rim reliably across frames

The detected rim is the single source of truth for shot evaluation

This removes human bias and ensures consistency across videos.

🏀 Ball Tracking & Trajectory

Tracks the basketball after release

Draws a visible trajectory path

Uses the ball center for geometric calculations

🧍 Shooting Form Analysis (MediaPipe Pose)

At the moment of release, the system evaluates:

Elbow extension

Knee bend

Wrist follow-through

These are combined into a Form Confidence Score (0–100).

Displayed directly on the video:

MAKE
Form: 48 / 100

🎯 Shot Outcome Logic (Explainable)

A shot is evaluated after release using geometry:

✅ MAKE

Ball center passes through the detected rim area

❌ MISS

Ball drops below the rim without entering

No heuristics or guessing — decisions are deterministic and explainable.

📊 Multi-Shot Coach Summary

After processing the full video, the app generates:

Average Form Confidence

Average joint breakdown (Elbow, Knee, Wrist)

Final shot tally

Example:

Average Form Confidence: 59 / 100
Average Breakdown - Elbow: 89, Knee: 1, Wrist: 94

Final Tally:
Score: 25 / 30

🎥 Annotated Video Output

The output video includes:

Player pose skeleton

Ball trajectory

Detected rim bounding box

Shot result overlay (MAKE / MISS)

Per-shot form score

The annotated video can be downloaded directly from the app.

🧠 How Shot Calculation Works (High Level)
1. Detect rim using YOLO (best.pt)
2. Detect shot release using pose motion
3. Track ball positions after release
4. Measure distance between ball center and rim center
5. If ball enters rim → MAKE
6. If ball drops below rim → MISS

🛠️ Tech Stack

Python 3.9+

OpenCV — video processing & overlays

MediaPipe Pose — body landmark detection

YOLO (Ultralytics) — rim detection (best.pt)

NumPy — geometry & math

Streamlit — interactive web UI

cvzone — visualization utilities

📂 Project Structure
Basketball-Game-Analyser/
│
├── analyzer.py          # Main Streamlit application
├── trial.py             # Experiments / testing
├── best.pt              # Custom YOLO rim detection model
├── config.yaml          # Training & threshold configuration
├── requirements.txt     # Dependencies
├── .gitignore
└── README.md

⚙️ Installation
1️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run analyzer.py


Then:

Upload a basketball video (.mp4 / .mov)

Click Analyze Shot

Review annotated video and coach summary

⚠️ Known Limitations

Shot distance is pixel-based (no real-world court calibration)

Ball detection may fail under extreme occlusion

Single-player focus

Net interaction not explicitly modeled

🔮 Future Improvements

Real-world court calibration

Multi-player support

Shot arc efficiency metrics

Net-based confirmation

Performance analytics dashboard

CSV export of shot data

👤 Author

Harsh

⭐ Why This Project Stands Out

Uses a custom-trained YOLO model

Fully automatic rim detection

Combines pose estimation + object tracking

Produces coaching-grade feedback

End-to-end, working system — not a demo

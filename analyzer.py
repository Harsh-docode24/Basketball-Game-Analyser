import os
import time
import math
import tempfile
import subprocess
from collections import deque

import cv2
import numpy as np
import streamlit as st
import cvzone
import mediapipe as mp

# --- YOLO import ---
try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False
    YOLO = None

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_LM = mp_pose.PoseLandmark

# ------------------------------
# Streamlit UI setup
# ------------------------------
st.set_page_config(page_title="🏀 Basketball Analyzer — Full Analysis", layout="wide")
st.title("🏀 Basketball Analyzer — Shot Detector & Form Confidence")
st.info("Using 'best.pt' for shot tracking and Mediapipe for form analysis.")

# ------------------------------
# Helper Functions (From Script A)
# ------------------------------
def angle_deg(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    cosang = np.clip(np.dot(v1, v2) / denom, -1, 1)
    return float(np.degrees(np.arccos(cosang)))

def extract_landmarks(results, w, h):
    if not results or not results.pose_landmarks:
        return {}
    return {i: (lm.x * w, lm.y * h) for i, lm in enumerate(results.pose_landmarks.landmark)}

def compute_confidence_at_release(pts):
    need = [int(POSE_LM.RIGHT_SHOULDER), int(POSE_LM.RIGHT_ELBOW), int(POSE_LM.RIGHT_WRIST),
            int(POSE_LM.RIGHT_HIP), int(POSE_LM.RIGHT_KNEE), int(POSE_LM.RIGHT_ANKLE)]
    if any(i not in pts for i in need):
        return 40.0, {"elbow":40, "knee":40, "wrist":40}

    rs = pts[need[0]]; re = pts[need[1]]; rw = pts[need[2]]
    rh = pts[need[3]]; rk = pts[need[4]]; ra = pts[need[5]]

    elbow_ang = angle_deg(rs, re, rw)
    knee_ang = angle_deg(rh, rk, ra)
    elbow_score = max(0, 100 - abs(170 - elbow_ang) * 2)
    knee_score = max(0, 100 - abs(90 - knee_ang) * 1.5)
    wrist_score = 100 if rw[1] < rs[1] else 60
    score = elbow_score * 0.4 + knee_score * 0.35 + wrist_score * 0.25
    
    breakdown = {"elbow": elbow_score, "knee": knee_score, "wrist": wrist_score}
    return float(np.clip(score, 0, 100)), breakdown


# ------------------------------
# Judge shot (HIT or MISS) - (Ball-in-Circle Logic)
# ------------------------------
def judge_shot(ball_path, rim):
    """
    Judges if a shot is made based on the ball's center
    passing inside the rim's 2D circle.
    rim is a tuple (cx, cy, r)
    """
    if not ball_path or rim is None:
        return False, False # (made, missed)

    cx, cy, r = rim
    made, missed = False, False

    # Check the *last 15 points* of the ball path for efficiency
    check_path = ball_path[-15:] 

    for (x, y) in check_path:
        # Calculate distance from ball center to rim center
        d = math.hypot(x - cx, y - cy)
        
        # If ball center enters the radius of the rim
        if d <= r: # Use full radius for max sensitivity
            made = True
            break
            
        # If ball is well below the rim, it's a miss
        if y > cy + (r * 1.5):
            # Only count as a miss if it was never 'made'
            if not made: 
                missed = True
            break
            
    return made, missed

# ------------------------------
# Analyzer function
# ------------------------------
def analyze_and_annotate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        return None, False, "Error reading video."

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    avi_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    if not YOLO_OK:
        st.error("Ultralytics YOLO library not found.")
        return None, False, "YOLO not installed."
        
    yolo = YOLO("best.pt") 
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    class_names = ['Basketball', 'Basketball Hoop']

    frame_idx = 0
    ball_path_deque = deque(maxlen=100)
    hoop_pos_history = deque(maxlen=20)
    current_rim_data = None
    
    made = missed = False
    shot_judged = False
    show_result_time = None
    progress = st.progress(0)

    # --- NEW: Form detection variables ---
    last_elbow_ang = None
    last_wrist_y = None
    last_detected_form_score = None # Holds the most recent form score
    current_shot_form_score_str = "" # The score to display on screen
    all_form_scores = [] # A list to store all breakdown dicts for averaging

    # --- Shot detection state machine ---
    ball_above_hoop = False
    ball_passed_hoop = False
    
    makes = 0
    attempts = 0
    first_shot_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        
        latest_ball_pos = None
        hoop_detections = []
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        pts = extract_landmarks(pose_results, w, h)
        
        res = yolo.predict(source=frame, classes=[0, 1], conf=0.3, verbose=False)
        
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_box, h_box = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                current_class = class_names[cls]
                center = (int(x1 + w_box / 2), int(y1 + h_box / 2))

                if current_class == 'Basketball':
                    if conf > 0.3:
                        ball_path_deque.append(center)
                        latest_ball_pos = center
                        cvzone.cornerRect(frame, (x1, y1, w_box, h_box), rt=3, l=9, colorC=(0, 255, 0))
                
                elif current_class == 'Basketball Hoop':
                    if conf > 0.5:
                        hoop_detections.append((center[0], center[1], w_box, h_box, conf))
                        cvzone.cornerRect(frame, (x1, y1, w_box, h_box), rt=3, l=9, colorC=(255, 0, 255))
        
        if hoop_detections:
            hoop_pos_history.extend(hoop_detections)
        
        if hoop_pos_history:
            best_hoop = sorted(list(hoop_pos_history), key=lambda x: x[4], reverse=True)[0]
            cx, cy, hw, hh, _ = best_hoop
            current_rim_data = (cx, cy, hw, hh)

        # --- NEW: Continuous Release Detection ---
        if pts: # Check if pose landmarks are detected
            try:
                rs = pts[int(POSE_LM.RIGHT_SHOULDER)]
                re = pts[int(POSE_LM.RIGHT_ELBOW)]
                rw = pts[int(POSE_LM.RIGHT_WRIST)]
                elbow_ang = angle_deg(rs, re, rw)
                wrist_y = rw[1]
                wrist_up = last_wrist_y is not None and wrist_y < last_wrist_y - 3
                
                # Check for release motion
                if last_elbow_ang and elbow_ang >= 160 and last_elbow_ang < 160 and wrist_up:
                    # Calculate and store the form score
                    conf_score, breakdown = compute_confidence_at_release(pts)
                    last_detected_form_score = (conf_score, breakdown) # Store tuple
                
                last_elbow_ang = elbow_ang
                last_wrist_y = wrist_y
            except Exception:
                pass # Ignore errors if a landmark is missing

        # --- Shot Detection State Machine ---
        if current_rim_data and latest_ball_pos and not shot_judged:
            ball_x, ball_y = latest_ball_pos
            rim_cx, rim_cy, rim_w, rim_h = current_rim_data
            
            hoop_y_level = rim_cy
            hoop_proximity_x = abs(ball_x - rim_cx) < (rim_w * 1.5)
            
            if not ball_above_hoop and ball_y < (hoop_y_level - rim_h / 2) and hoop_proximity_x:
                ball_above_hoop = True
            
            if ball_above_hoop and ball_y > (hoop_y_level + rim_h / 2):
                ball_passed_hoop = True
                
            if ball_passed_hoop:
                attempts += 1
                shot_judged = True
                show_result_time = time.time()
                
                ball_path = list(ball_path_deque)
                rim_tuple_for_judge = (rim_cx, rim_cy, rim_w / 2) 
                
                made, missed = judge_shot(ball_path, rim_tuple_for_judge)
                
                # --- NEW: Link form score to the shot ---
                if last_detected_form_score:
                    conf_val, breakdown = last_detected_form_score
                    current_shot_form_score_str = f"Form: {conf_val:.0f}/100"
                    all_form_scores.append(last_detected_form_score) # Add for averaging
                    last_detected_form_score = None # Reset for next shot
                else:
                    current_shot_form_score_str = "Form: N/A" # No release detected for this shot

                if made:
                    makes += 1
                    missed = False
                    if attempts == 1:
                        first_shot_result = "MADE"
                elif missed:
                    made = False
                    if attempts == 1:
                        first_shot_result = "MISSED"
                else:
                    made = False
                    missed = False
                    if attempts == 1:
                        first_shot_result = "MISSED"
                
                ball_above_hoop = False
                ball_passed_hoop = False
                ball_path_deque.clear()
                hoop_pos_history.clear()

        # --- Draw ball path (as red dots) ---
        for point in ball_path_deque:
            x, y = map(int, point)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # --- Draw Pose ---
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- Display Score ---
        text = f"{makes} / {attempts}"
        cv2.putText(frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # --- Show "MAKE" / "MISS" & Form Score ---
        if shot_judged and show_result_time:
            elapsed = time.time() - show_result_time
            if elapsed <= 1.5: 
                
                txt = None
                if made:
                    txt, color = "MAKE", (0, 255, 0)
                elif missed:
                    txt, color = "MISS", (0, 0, 255)
                
                if txt: 
                    # Draw "MAKE" or "MISS"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3, 6)
                    tx = (w - tw) // 2
                    ty = (h + th) // 2
                    cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 3, color, 6, cv2.LINE_AA)
                    cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
                    
                    # --- NEW: Draw Form Score Below ---
                    (tw_form, th_form), _ = cv2.getTextSize(current_shot_form_score_str, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4)
                    tx_form = (w - tw_form) // 2
                    ty_form = ty + th_form + 20 # Position below main text
                    cv2.putText(frame, current_shot_form_score_str, (tx_form, ty_form), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, current_shot_form_score_str, (tx_form, ty_form), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

            else:
                # Reset shot state
                shot_judged = False
                made = False
                missed = False
                current_shot_form_score_str = "" # Clear text

        writer.write(frame)
        progress.progress(min(frame_idx / total_frames, 1.0))

    # Clean up
    cap.release()
    writer.release()
    pose.close()
    progress.empty()
    
    # --- NEW: Calculate Average Form Score ---
    if all_form_scores:
        avg_conf = np.mean([score for score, breakdown in all_form_scores])
        avg_elbow = np.mean([breakdown['elbow'] for score, breakdown in all_form_scores])
        avg_knee = np.mean([breakdown['knee'] for score, breakdown in all_form_scores])
        avg_wrist = np.mean([breakdown['wrist'] for score, breakdown in all_form_scores])
    else:
        # Default if no releases were detected
        avg_conf, avg_elbow, avg_knee, avg_wrist = 40.0, 40.0, 40.0, 40.0

    summary = (
        f"--- Average Coach Summary (All Shots) ---\n"
        f"Average Form Confidence: {avg_conf:.0f}/100\n"
        f"Average Breakdown - Elbow: {avg_elbow:.0f}, "
        f"Knee: {avg_knee:.0f}, Wrist: {avg_wrist:.0f}\n"
        f"\n--- Final Tally (All Shots) ---\n"
        f"Score: {makes} / {attempts}"
    )

    mp4_out = avi_path.replace(".avi", "_annotated.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", avi_path, "-vcodec", "libx264", "-crf", "23", mp4_out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        final_out = mp4_out
    except FileNotFoundError:
        st.warning("FFmpeg not found. Returning AVI output.")
        final_out = avi_path

    first_shot_made = (first_shot_result == "MADE") # Still use first shot for balloons
    return final_out, first_shot_made, summary

# ------------------------------
# Streamlit Interface
# ------------------------------
uploaded = st.file_uploader("Upload basketball video (.mp4/.mov)", type=["mp4", "mov"])
if not uploaded:
    st.info("Upload a video (and ensure 'best.pt' is in the same folder) to begin.")
    st.stop()

if not os.path.exists("best.pt"):
    st.error("Error: 'best.pt' model file not found.")
    st.info("Please make sure the 'best.pt' file you uploaded is in the same directory as the Streamlit script.")
    st.stop()
    
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tmp.write(uploaded.read())
tmp.flush()
tmp.close()

if st.button("Analyze Shot"):
    with st.spinner("Analyzing video... This may take a moment. ⏳"):
        out_path, first_shot_made, summary = analyze_and_annotate(tmp.name) 

    if out_path and os.path.exists(out_path):
        st.success("✅ Analysis complete")
        st.video(out_path)
        with open(out_path, "rb") as fh:
            st.download_button("⬇️ Download Annotated Video", fh.read(),
                             file_name=os.path.basename(out_path), mime="video/mp4")
        
        st.subheader("🎙️ Coach Summary")
        st.text(summary)
        if first_shot_made:
            st.balloons()
    else:
        st.error("No output produced or analysis failed.")
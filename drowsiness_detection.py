"""
Driver Anti-Sleep / Drowsiness Detection System
Uses OpenCV + MediaPipe Face Mesh to track eye landmarks and compute
Eye Aspect Ratio (EAR). Triggers an alert when eyes stay closed too long.

Install dependencies first:
    pip install opencv-python mediapipe numpy playsound
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import threading

# ---- Eye landmark indices (MediaPipe Face Mesh) ----
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25       # Below this = eyes considered closed
CONSEC_FRAMES = 20         # Number of consecutive closed frames to trigger alarm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def eye_aspect_ratio(landmarks, eye_points, frame_w, frame_h):
    coords = []
    for idx in eye_points:
        lm = landmarks[idx]
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    coords = np.array(coords)

    # Vertical distances
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    # Horizontal distance
    C = np.linalg.norm(coords[0] - coords[3])

    ear = (A + B) / (2.0 * C)
    return ear

def play_alert():
    try:
        playsound("alarm.wav")  # Place an alarm.wav file in the same folder
    except Exception as e:
        print("Could not play alert sound:", e)

def main():
    cap = cv2.VideoCapture(0)
    closed_frames = 0
    alarm_on = False

    print("Driver Drowsiness Detection Started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1
                    if closed_frames >= CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if not alarm_on:
                            alarm_on = True
                            threading.Thread(target=play_alert, daemon=True).start()
                else:
                    closed_frames = 0
                    alarm_on = False

        cv2.imshow("Driver Anti-Sleep Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

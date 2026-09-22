"""
Driver Anti-Sleep / Drowsiness Detection System (Final Version)
Uses OpenCV + MediaPipe Face Mesh to track eye landmarks and compute
Eye Aspect Ratio (EAR). Triggers a generated alert tone when eyes stay
closed too long. No external audio file needed.

Install dependencies first:
    pip install opencv-python mediapipe numpy sounddevice
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Alerts will be visual only.")

# ---- Eye landmark indices (MediaPipe Face Mesh) ----
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25       # Below this = eyes considered closed
CONSEC_FRAMES = 20         # Consecutive closed frames before alarm triggers
ALARM_COOLDOWN = 3.0       # Seconds between repeated alarm sounds

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

    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])

    ear = (A + B) / (2.0 * C)
    return ear


def generate_beep(frequency=1000, duration=0.6, sample_rate=44100):
    """Generates a simple sine-wave beep tone in memory (no file needed)."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    # Apply a short fade-out to avoid a harsh click at the end
    fade = np.linspace(1, 0, int(sample_rate * 0.05))
    tone[-len(fade):] *= fade
    return tone.astype(np.float32), sample_rate


def play_alert():
    if not AUDIO_AVAILABLE:
        return
    try:
        tone, rate = generate_beep()
        sd.play(tone, rate)
        sd.wait()
    except Exception as e:
        print("Could not play alert sound:", e)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam. Check camera index or permissions.")
        return

    closed_frames = 0
    last_alarm_time = 0

    print("Driver Drowsiness Detection Started... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "No face detected"
        status_color = (0, 165, 255)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                status_text = f"EAR: {avg_ear:.2f}"
                status_color = (0, 255, 0)

                if avg_ear < EAR_THRESHOLD:
                    closed_frames += 1
                    if closed_frames >= CONSEC_FRAMES:
                        status_text = "DROWSINESS ALERT!"
                        status_color = (0, 0, 255)

                        now = time.time()
                        if now - last_alarm_time > ALARM_COOLDOWN:
                            last_alarm_time = now
                            threading.Thread(target=play_alert, daemon=True).start()
                else:
                    closed_frames = 0

        cv2.putText(frame, status_text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Driver Anti-Sleep Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

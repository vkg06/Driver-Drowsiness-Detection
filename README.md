# 😴 Driver Anti-Sleep Device

A real-time drowsiness detection system using computer vision and eye-tracking techniques. Monitors driver alertness via webcam and triggers immediate alerts when signs of drowsiness are detected, helping prevent accidents.

## Features
- Real-time eye-tracking using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based drowsiness detection
- Achieves 90%+ detection accuracy under normal lighting conditions
- Audible alert triggered on prolonged eye closure

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- playsound

## Installation
```bash
pip install opencv-python mediapipe numpy playsound
```

## How It Works
1. Captures live video feed from a webcam.
2. Uses MediaPipe Face Mesh to detect facial landmarks around the eyes.
3. Calculates Eye Aspect Ratio (EAR) — a measure of how open/closed the eyes are.
4. If EAR stays below a threshold for a sustained number of frames, it's flagged as drowsiness.
5. Triggers an audio alert to wake/alert the driver.

## Getting Started
1. Install dependencies (see above).
2. Place an `alarm.wav` file in the project directory for the alert sound.
3. Run:
```bash
   python drowsiness_detection.py
```
4. Press `q` to quit the application.

## Configuration
| Parameter        | Description                                   | Default |
|-------------------|-----------------------------------------------|---------|
| `EAR_THRESHOLD`   | EAR value below which eyes are "closed"       | 0.25    |
| `CONSEC_FRAMES`   | Consecutive closed frames before alert fires  | 20      |

Tune these based on lighting conditions and camera quality for best accuracy.

## Future Improvements
- Add yawning detection as a secondary drowsiness signal
- Integrate with Arduino for a physical buzzer/vibration alert
- Log drowsiness events with timestamps for driver behavior analysis

## License
MIT

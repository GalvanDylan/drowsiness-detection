# 💤 Real-Time Drowsiness Detection System

A real-time computer vision system built to detect signs of drowsiness using facial landmark detection and Eye Aspect Ratio (EAR). This project simulates a practical use case in automotive safety and fatigue monitoring, integrating classic vision techniques with modern facial tracking models.

---

## 🚀 Why This Matters

Driver fatigue is a major contributor to road accidents. This project tackles that problem using **real-time facial analysis**, offering a lightweight, camera-based solution that can detect early signs of drowsiness — such as prolonged eye closure — without any external hardware or wearables.

---

## 💡 Key Highlights

- ✅ **Real-Time Detection** with OpenCV
- 🧠 **Facial Landmark Tracking** using MediaPipe
- 🔄 **Dynamic EAR Thresholding** to adapt to different eye shapes/sizes
- 🔔 **Audio Alert System** that triggers when the user appears drowsy
- 🧪 Designed for experimentation and fast iteration — deployable on any machine with a webcam

---

## 🛠️ Tech Stack

| Tool        | Role                            |
|-------------|---------------------------------|
| **Python**  | Core programming language       |
| **OpenCV**  | Video capture and image processing |
| **MediaPipe** | Facial landmark detection        |
| **NumPy**   | EAR calculations                |
| **Playsound** (optional) | Triggering audio alerts        |

---

## 🧪 How It Works

1. MediaPipe detects facial landmarks (eyes)
2. EAR is calculated by measuring vertical and horizontal eye distances
3. If EAR drops below a set threshold for multiple consecutive frames → alarm is triggered
4. The system runs continuously, monitoring the user’s eye state in real-time

---

## 🖥 Demo

> Coming soon — screen recording or GIF of the system detecting drowsiness and triggering an alarm.

---

## 🧰 Installation

```bash
git clone https://github.com/GalvanDylan/drowsiness-detection.git
cd drowsiness-detection
pip install -r requirements.txt
python main.py

# 🚗 Driver Drowsiness Detection System

A real-time computer vision system that monitors a driver's face via webcam and raises an alarm when drowsiness or yawning is detected — using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) algorithms.

---

## 📁 Project Structure

```
driver_drowsiness_detection/
│
├── main.py               ← Run this to start the system
├── detector.py           ← Core detection logic (EAR / MAR)
├── alert.py              ← Audio + visual alert manager
├── logger.py             ← CSV session logger
├── download_model.py     ← One-time model downloader
├── requirements.txt      ← Python dependencies
│
├── utils/
│   ├── __init__.py
│   └── display.py        ← HUD overlay drawing functions
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat   ← (download separately)
│
├── sounds/
│   ├── alert.wav         ← Loud alarm for drowsiness (optional)
│   └── warning.wav       ← Soft beep for yawning   (optional)
│
└── logs/
    └── session_YYYYMMDD_HHMMSS.csv  ← Auto-generated per session
```

---

## ⚙️ How It Works

```
Webcam Frame
    │
    ▼
Face Detection  (dlib HOG detector or Haar Cascade fallback)
    │
    ▼
Facial Landmark Extraction  (68-point dlib model)
    │
    ├──► Eye Aspect Ratio (EAR)
    │        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    │        EAR < 0.25 for N frames  →  DROWSY ALERT
    │
    └──► Mouth Aspect Ratio (MAR)
             MAR > 0.60 for 15 frames  →  YAWN WARNING
```

---

## 🚀 Setup & Installation

### Step 1 — Clone / Download the project

```bash
cd driver_drowsiness_detection
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` requires CMake. If installation fails:
> - Windows: `pip install cmake` first, then retry
> - Or install a pre-built wheel: `pip install dlib --find-links https://github.com/jloh02/dlib/releases/`

### Step 4 — Download the facial landmark model (one-time)

```bash
python download_model.py
```

This downloads `shape_predictor_68_face_landmarks.dat` (~100 MB) into the `models/` folder.

### Step 5 — Run the system!

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit the program |
| `R` | Reset the blink / yawn counter |

---

## ⚙️ Command-Line Options

```bash
python main.py --help

Options:
  --camera    INT    Camera index (default: 0)
  --threshold FLOAT  EAR threshold (default: 0.25)
  --frames    INT    Consecutive frames to trigger alert (default: 20)
  --no-sound         Disable audio alerts
```

Example with options:
```bash
python main.py --camera 1 --threshold 0.22 --frames 15
```

---

## 📊 HUD Display Explained

| Label   | Meaning |
|---------|---------|
| STATUS  | AWAKE / DROWSY / YAWNING / NO FACE |
| EAR     | Eye Aspect Ratio (< 0.25 = closed) |
| MAR     | Mouth Aspect Ratio (> 0.60 = yawning) |
| FRAMES  | Consecutive frames with closed eyes |
| BLINKS  | Total blinks this session |
| YAWNS   | Total yawns this session |

---

## 🔔 Alert Sounds

Place `.wav` files in the `sounds/` folder:
- `sounds/alert.wav`   — played during drowsiness detection
- `sounds/warning.wav` — played when yawning is detected

If no `.wav` files are found, the system auto-generates a beep tone using `pygame`.

---

## 📄 Session Logs

Every session auto-creates a CSV log in `logs/`:
```
timestamp, elapsed_sec, event, ear, mar, frame_counter, total_blinks, total_yawns
10:05:32,  45.3,        DROWSY, 0.18, 0.22, 22, 14, 2
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `dlib` install fails | Install CMake first: `pip install cmake` |
| Model file not found | Run `python download_model.py` |
| Camera not opening | Try `--camera 1` or `--camera 2` |
| No sound | Install pygame: `pip install pygame` |
| Slow performance | Use `--camera 0`, ensure good lighting |

---

## 📦 Technologies Used

- **Python 3.8+**
- **OpenCV** — Camera capture & image processing
- **dlib** — Face detection & 68-point landmark prediction
- **SciPy** — Euclidean distance for EAR/MAR calculation
- **NumPy** — Array operations
- **Pygame** — Audio alerts

---

## 📌 Key Concepts

**Eye Aspect Ratio (EAR):** Measures eye openness. When a person blinks or falls asleep, EAR drops below the threshold.

**Mouth Aspect Ratio (MAR):** Measures how wide the mouth opens. High MAR over consecutive frames = yawning = early sign of drowsiness.

**Consecutive Frame Counter:** A single low EAR could just be a blink. The alert only fires after N consecutive frames (default 20 ≈ ~0.7 seconds at 30fps).

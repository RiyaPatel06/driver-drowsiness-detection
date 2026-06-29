"""
detector.py
Core detection engine using facial landmarks.
Calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("[WARNING] dlib not found. Using Haar Cascade fallback mode.")


# ─────────────────────────────────────────────
# Facial landmark indices (dlib 68-point model)
# ─────────────────────────────────────────────
LEFT_EYE_IDX  = list(range(36, 42))   # Points 36–41
RIGHT_EYE_IDX = list(range(42, 48))   # Points 42–47
MOUTH_IDX     = list(range(48, 68))   # Points 48–67
JAW_IDX       = list(range(0, 17))


def eye_aspect_ratio(eye_points):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A low EAR means the eye is closed.
    """
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def mouth_aspect_ratio(mouth_points):
    """
    MAR measures mouth opening — high value = yawning.
    """
    A = dist.euclidean(mouth_points[13], mouth_points[19])  # vertical
    B = dist.euclidean(mouth_points[14], mouth_points[18])
    C = dist.euclidean(mouth_points[15], mouth_points[17])
    D = dist.euclidean(mouth_points[12], mouth_points[16])  # horizontal
    mar = (A + B + C) / (2.0 * D + 1e-6)
    return mar


class DrowsinessDetector:
    """
    Detects drowsiness using EAR (eye closure) and MAR (yawning).

    Parameters
    ----------
    ear_threshold  : float  — EAR below this = eye closed (default 0.25)
    mar_threshold  : float  — MAR above this = yawning    (default 0.60)
    consec_frames  : int    — Frames needed to trigger alert (default 20)
    """

    def __init__(self, ear_threshold=0.25, mar_threshold=0.60, consec_frames=20):
        self.EAR_THRESH   = ear_threshold
        self.MAR_THRESH   = mar_threshold
        self.CONSEC_FRAMES = consec_frames

        self.frame_counter = 0   # consecutive frames with closed eyes
        self.yawn_counter  = 0
        self.total_blinks  = 0
        self.total_yawns   = 0

        self._load_models()

    # ── Model loading ──────────────────────────────────────────────────────
    def _load_models(self):
        if DLIB_AVAILABLE:
            self.face_detector  = dlib.get_frontal_face_detector()
            try:
                self.landmark_predictor = dlib.shape_predictor(
                    "models/shape_predictor_68_face_landmarks.dat"
                )
                self.mode = "dlib"
                print("[INFO] dlib landmark model loaded.")
            except RuntimeError:
                print("[WARNING] shape_predictor_68_face_landmarks.dat not found.")
                print("         Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("         Falling back to Haar Cascade mode.")
                self._load_haar()
        else:
            self._load_haar()

    def _load_haar(self):
        self.mode = "haar"
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        print("[INFO] Haar Cascade mode active (basic detection only).")

    # ── Main detect function ───────────────────────────────────────────────
    def detect(self, frame):
        """
        Process one frame and return a result dictionary.

        Returns
        -------
        dict with keys:
            face_detected, ear, mar, is_drowsy, is_yawning,
            frame_counter, total_blinks, total_yawns, landmarks
        """
        result = {
            "face_detected": False,
            "ear":           1.0,
            "mar":           0.0,
            "is_drowsy":     False,
            "is_yawning":    False,
            "frame_counter": self.frame_counter,
            "total_blinks":  self.total_blinks,
            "total_yawns":   self.total_yawns,
            "landmarks":     None,
        }

        if self.mode == "dlib":
            return self._detect_dlib(frame, result)
        else:
            return self._detect_haar(frame, result)

    # ── dlib path ──────────────────────────────────────────────────────────
    def _detect_dlib(self, frame, result):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 0)

        if len(faces) == 0:
            self.frame_counter = 0
            return result

        result["face_detected"] = True
        face = faces[0]                          # Use the first face
        shape = self.landmark_predictor(gray, face)
        coords = np.array([[p.x, p.y] for p in shape.parts()])
        result["landmarks"] = coords

        left_eye  = coords[LEFT_EYE_IDX]
        right_eye = coords[RIGHT_EYE_IDX]
        mouth     = coords[MOUTH_IDX]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        result["ear"] = round(ear, 3)
        result["mar"] = round(mar, 3)

        # ── Drowsiness check ──
        if ear < self.EAR_THRESH:
            self.frame_counter += 1
            if self.frame_counter >= self.CONSEC_FRAMES:
                result["is_drowsy"] = True
        else:
            if self.frame_counter >= 3:
                self.total_blinks += 1
            self.frame_counter = 0

        # ── Yawn check ──
        if mar > self.MAR_THRESH:
            self.yawn_counter += 1
            if self.yawn_counter >= 15:
                result["is_yawning"] = True
        else:
            if self.yawn_counter >= 15:
                self.total_yawns += 1
            self.yawn_counter = 0

        result["frame_counter"] = self.frame_counter
        result["total_blinks"]  = self.total_blinks
        result["total_yawns"]   = self.total_yawns
        return result

    # ── Haar fallback path ─────────────────────────────────────────────────
    def _detect_haar(self, frame, result):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        if len(faces) == 0:
            self.frame_counter = 0
            return result

        result["face_detected"] = True
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, minSize=(20, 20))

        eyes_visible = len(eyes)
        if eyes_visible == 0:
            self.frame_counter += 1
            if self.frame_counter >= self.CONSEC_FRAMES:
                result["is_drowsy"] = True
        else:
            self.frame_counter = 0

        result["frame_counter"] = self.frame_counter
        # EAR approximation: normalize eye count
        result["ear"] = 0.15 if eyes_visible == 0 else 0.30
        return result

    # ── Utilities ─────────────────────────────────────────────────────────
    def reset(self):
        self.frame_counter = 0
        self.yawn_counter  = 0
        self.total_blinks  = 0
        self.total_yawns   = 0

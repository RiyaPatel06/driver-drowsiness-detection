"""
utils/display.py
Draws the HUD (heads-up display) overlay on each video frame.
"""

import cv2
import numpy as np


# ── Colour palette ─────────────────────────────────────────────────────────
GREEN  = (0,   220,   0)
YELLOW = (0,   200, 255)
RED    = (0,     0, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,     0,   0)
CYAN   = (255, 200,   0)


def draw_ui(frame, result: dict, fps: int = 0) -> np.ndarray:
    """
    Overlay all status information on the frame.

    Parameters
    ----------
    frame  : BGR numpy array from OpenCV
    result : dict returned by DrowsinessDetector.detect()
    fps    : current frames-per-second

    Returns
    -------
    Annotated BGR frame
    """
    h, w = frame.shape[:2]

    # ── Draw facial landmarks ────────────────────────────────────────────
    if result.get("landmarks") is not None:
        _draw_landmarks(frame, result["landmarks"])

    # ── Determine status ─────────────────────────────────────────────────
    if result["is_drowsy"]:
        status_text  = "DROWSY!"
        status_color = RED
        _draw_alert_banner(frame, w, h)
    elif result["is_yawning"]:
        status_text  = "YAWNING"
        status_color = YELLOW
    elif not result["face_detected"]:
        status_text  = "NO FACE"
        status_color = YELLOW
    else:
        status_text  = "AWAKE"
        status_color = GREEN

    # ── Top-left stats panel ─────────────────────────────────────────────
    panel_w, panel_h = 230, 160
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    ear = result.get("ear", 0)
    mar = result.get("mar", 0)
    fc  = result.get("frame_counter", 0)
    blinks = result.get("total_blinks", 0)
    yawns  = result.get("total_yawns",  0)

    lines = [
        (f"STATUS : {status_text}", status_color),
        (f"EAR    : {ear:.3f}",     WHITE),
        (f"MAR    : {mar:.3f}",     WHITE),
        (f"FRAMES : {fc}",          YELLOW if fc > 5 else WHITE),
        (f"BLINKS : {blinks}",      CYAN),
        (f"YAWNS  : {yawns}",       CYAN),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (8, 22 + i * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ── FPS (top-right) ─────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps}", (w - 100, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # ── Bottom instructions ──────────────────────────────────────────────
    cv2.putText(frame, "Press Q: Quit  |  R: Reset",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    # ── EAR progress bar ────────────────────────────────────────────────
    _draw_ear_bar(frame, ear, w, h)

    return frame


def _draw_landmarks(frame, landmarks):
    """Draw green dots at each of the 68 facial landmark positions."""
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, GREEN, -1)

    # Highlight eye regions
    for idx_range in [range(36, 42), range(42, 48)]:
        pts = landmarks[list(idx_range)]
        cv2.polylines(frame, [pts], isClosed=True,
                      color=CYAN, thickness=1, lineType=cv2.LINE_AA)

    # Highlight mouth
    mouth_pts = landmarks[48:60]
    cv2.polylines(frame, [mouth_pts], isClosed=True,
                  color=YELLOW, thickness=1, lineType=cv2.LINE_AA)


def _draw_alert_banner(frame, w, h):
    """Red flashing banner across the top when drowsiness detected."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "  !! DROWSINESS ALERT !! WAKE UP !!",
                (10, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, WHITE, 2, cv2.LINE_AA)


def _draw_ear_bar(frame, ear, w, h):
    """Draw a small EAR progress bar at the bottom-right."""
    bar_x, bar_y = w - 130, h - 30
    bar_w, bar_h = 120, 12

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)

    fill = int(min(ear / 0.40, 1.0) * bar_w)
    color = GREEN if ear > 0.25 else (YELLOW if ear > 0.20 else RED)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)

    cv2.putText(frame, "EAR", (bar_x - 35, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1)

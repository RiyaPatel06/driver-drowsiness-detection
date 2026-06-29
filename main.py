"""
Driver Drowsiness Detection System
Main entry point - Run this file to start the system
"""

import cv2
import time
import argparse
from detector import DrowsinessDetector
from alert import AlertSystem
from logger import DrowsinessLogger
from utils.display import draw_ui


def parse_args():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-sound", action="store_true", help="Disable sound alerts")
    parser.add_argument("--threshold", type=float, default=0.25, help="EAR threshold (default: 0.25)")
    parser.add_argument("--frames", type=int, default=20, help="Consecutive frames for alert (default: 20)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("  Driver Drowsiness Detection System")
    print("=" * 50)
    print(f"  Camera Index : {args.camera}")
    print(f"  EAR Threshold: {args.threshold}")
    print(f"  Alert Frames : {args.frames}")
    print(f"  Sound Alert  : {'Disabled' if args.no_sound else 'Enabled'}")
    print("=" * 50)
    print("  Press 'q' to quit | Press 'r' to reset counter")
    print("=" * 50)

    # Initialize components
    detector = DrowsinessDetector(ear_threshold=args.threshold, consec_frames=args.frames)
    alert = AlertSystem(sound_enabled=not args.no_sound)
    logger = DrowsinessLogger()

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check camera index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[INFO] System started. Monitoring driver...")

    fps_counter = 0
    fps_start = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start = time.time()

        # Run drowsiness detection
        result = detector.detect(frame)

        # Handle alerts
        if result["is_drowsy"]:
            alert.trigger()
            logger.log_event("DROWSY", result)
        elif result["is_yawning"]:
            alert.trigger(level="warning")
            logger.log_event("YAWN", result)
        else:
            alert.reset()

        # Draw UI on frame
        frame = draw_ui(frame, result, fps)

        # Display the frame
        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[INFO] Quitting...")
            break
        elif key == ord("r"):
            detector.reset()
            print("[INFO] Counter reset.")

    cap.release()
    cv2.destroyAllWindows()
    logger.save()
    print("[INFO] Session ended. Log saved.")


if __name__ == "__main__":
    main()

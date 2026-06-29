"""
download_model.py
Downloads the dlib 68-point facial landmark model.
Run this ONCE before starting the main system.

Usage:
    python download_model.py
"""

import os
import urllib.request
import bz2
import shutil

MODEL_URL  = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
BZ2_PATH   = "models/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"


def download_with_progress(url, dest):
    print(f"Downloading from:\n  {url}\n")

    def reporthook(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print("\nDownload complete.")


def extract_bz2(src, dst):
    print(f"Extracting {src} ...")
    with bz2.BZ2File(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Saved to: {dst}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH}")
    else:
        download_with_progress(MODEL_URL, BZ2_PATH)
        extract_bz2(BZ2_PATH, MODEL_PATH)
        os.remove(BZ2_PATH)
        print("\n[OK] Model ready! You can now run: python main.py")

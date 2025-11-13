"""
Photo capture and face preprocessing.
- Captures N frames from webcam
- Saves raw frames and cropped/resized face images
- Creates directories only if a face is detected
- Exits after 5 seconds if no face detected
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from tqdm import tqdm
from .config import DATASET_DIR, FACE_CASCADE_PATH, FACE_SIZE, NUM_IMAGES, CAPTURE_DELAY_SEC


def _ensure_user_dirs(user_name: str) -> Tuple[Path, Path]:
    root = DATASET_DIR / user_name
    raw_dir = root / "raw"
    cropped_dir = root / "cropped"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, cropped_dir


def _crop_largest_face(img_bgr, face_size, min_ratio=0.45):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # enlarge bounding box to improve consistency
    expand_w = int(w * min_ratio)
    expand_h = int(h * min_ratio)

    x = max(x - expand_w, 0)
    y = max(y - expand_h, 0)
    w = min(w + 2*expand_w, img_bgr.shape[1] - x)
    h = min(h + 2*expand_h, img_bgr.shape[0] - y)

    face = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(face_rgb, face_size)



def capture_user_images(user_name: str, num_images: int = NUM_IMAGES, delay_sec: float = CAPTURE_DELAY_SEC) -> Tuple[Path | None, Path | None, int]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check permissions or device.")

    print(f"[Camera] Capturing {num_images} images for '{user_name}' ...")
    time.sleep(2)

    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    count = 0
    raw_dir = cropped_dir = None

    while count < num_images:
        ok, frame = cap.read()
        if not ok:
            print("Warn: Failed to read frame; stopping capture.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Eagle Capture", frame)

        # --- keep window on top ---
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, "Eagle Capture")
            if hwnd:
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)
        except Exception as e:
            print("[Warning] Could not set window always-on-top:", e)

        cv2.waitKey(1)

        try:
            # crop face as clean RGB
            cropped_rgb = _crop_largest_face(frame, FACE_SIZE)

            # create folders only once
            if count == 0:
                raw_dir, cropped_dir = _ensure_user_dirs(user_name)

            raw_path = raw_dir / f"img_{count + 1}.jpg"
            cropped_path = cropped_dir / f"img_{count + 1}.jpg"

            # save raw frame
            cv2.imwrite(str(raw_path), frame)

            # save cropped face EXACT RGB (converted properly for OpenCV)
            cv2.imwrite(str(cropped_path), cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))

            count += 1
            print(f"[Capture] Face detected ({count}/{num_images})")
            time.sleep(delay_sec)

        except ValueError as e:
            print("[Warning]", e)
            if count == 0:
                time.sleep(2)
                break
            else:
                time.sleep(1)
                continue

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print(f"[Camera] No valid faces captured for '{user_name}'.")
        return None, None, 0

    print(f"[Camera] Done. Captured {count} valid frames.")
    return raw_dir, cropped_dir, count

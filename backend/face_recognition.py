"""
DeepFace-based live face verification for Eagle Access.
Fully consistent with rebuild pipeline:
- Crop using _crop_largest_face()
- Histogram normalize lighting
- Resize + RGB
- DeepFace.represent(skip) for consistent embeddings
- COSINE similarity match (best for Facenet512)
"""

from __future__ import annotations
import cv2
import numpy as np
import json
import time
from deepface import DeepFace
from pathlib import Path
from .config import EMBED_FILE, FACE_SIZE
from .photo_capture import _crop_largest_face


# ---------------------------------------------------------
# Load stored embeddings
# ---------------------------------------------------------
def load_embeddings():
    if not EMBED_FILE.exists():
        print("⚠️ No embeddings file found.")
        return {}

    try:
        with open(EMBED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        print("⚠️ embeddings.json corrupted.")
        return {}


# ---------------------------------------------------------
# Compute embedding (skip detector — consistent with rebuild)
# ---------------------------------------------------------
def compute_embedding(face_rgb):
    try:
        rep = DeepFace.represent(
            img_path=face_rgb,
            model_name="Facenet512",
            detector_backend="skip",
            enforce_detection=False
        )
        return rep[0]["embedding"]
    except Exception as e:
        print("[Embed ERROR]", e)
        return None


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------------------------------------------------
# Compare embedding with DB using COSINE
# ---------------------------------------------------------
def match_user(embedding, db):
    best_user = None
    best_dist = 9999

    for user, vectors in db.items():
        stored_vec = vectors[0]   # median vector
        dist = 1 - cosine_similarity(embedding, stored_vec)

        if dist < best_dist:
            best_dist = dist
            best_user = user

    # COSINE threshold: lower = better match
    if best_dist <= 0.45:
        return best_user, best_dist

    return "Unknown", best_dist


# ---------------------------------------------------------
# Live verification
# ---------------------------------------------------------
def verify_face_live():
    db = load_embeddings()
    if not db:
        print("❌ No embeddings found.")
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access webcam")

    print("\n📸 Starting verification — hold still...\n")
    time.sleep(1.5)

    ok, frame = cap.read()
    if not ok:
        print("❌ Could not read frame.")
        cap.release()
        return None

    # --- Crop using SAME pipeline as rebuild ---
    try:
        crop = _crop_largest_face(frame, FACE_SIZE)
    except Exception as e:
        print("❌ Face not detected:", e)
        cap.release()
        return None

    # Use pure color crop (no grayscale)
    face_rgb = crop

    # --- Compute embedding ---
    emb = compute_embedding(face_rgb)
    if emb is None:
        cap.release()
        return None

    # --- Match ---
    user, dist = match_user(emb, db)

    # Convert dist → match percentage
    match_percent = max(0.0, 1 - (dist / 1.2))
    match_percent_display = f"{match_percent*100:.1f}%"

    print(f"🎯 MATCH: {user}  dist={dist:.4f}  confidence={match_percent_display}")

    # --- Show SAME SIZE window as registration (640x480) ---
    display = frame.copy()

    # force consistent UI size
    display = cv2.resize(display, (640, 480))

    # draw label
    color = (0, 255, 0) if user != "Unknown" else (0, 0, 255)
    label = f"{user}  ({match_percent_display})"

    cv2.putText(display, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Recognition Result", display)

    # keep open for a moment
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    cap.release()
    return user

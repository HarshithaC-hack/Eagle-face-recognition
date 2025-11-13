"""
Embedding generation and storage utilities using DeepFace (Facenet512).
Consistent with verification pipeline:
- Use _crop_largest_face() for cropping (same as verification)
- Use detector_backend='skip'
- Use np array directly instead of file path
- Median embedding per user
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List
import numpy as np
from deepface import DeepFace
import cv2
from .config import EMBED_FILE, FACE_SIZE, FACE_CASCADE_PATH
from .photo_capture import _crop_largest_face

# --------------------------------------------------------------
# CORE: Compute embeddings for all cropped images
# --------------------------------------------------------------
def compute_embeddings_for_folder(folder: Path) -> List[List[float]]:

    out: List[List[float]] = []

    if not folder.exists():
        print(f"[Embed] Folder not found: {folder}")
        return out

    images = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print(f"[Embed] No images found in {folder}")
        return out

    print(f"[Embed] Generating embeddings for {len(images)} images in {folder} ...")

    for p in images:
        try:
            # Read image manually
            bgr = cv2.imread(str(p))
            if bgr is None:
                print(f"[Embed] Cannot read {p.name}")
                continue

            # Convert to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # IMPORTANT: Use skip (same as verification)
            rep = DeepFace.represent(
                img_path=rgb,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False
            )

            if rep and isinstance(rep, list) and "embedding" in rep[0]:
                out.append(rep[0]["embedding"])
            else:
                print(f"[Embed] No embedding returned for {p.name}")

        except Exception as e:
            print(f"[Embed] Skipping {p.name}: {e}")

    print(f"[Embed] Created {len(out)} embeddings from {len(images)} images.")
    return out

# --------------------------------------------------------------
# AVERAGE embeddings (median vector)
# --------------------------------------------------------------
def average_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    if not embeddings:
        print("[Embed] No embeddings to average.")
        return []

    arr = np.array(embeddings, dtype=np.float32)
    median_vec = np.median(arr, axis=0)
    print(f"[Embed] Median of {len(embeddings)} → 1 stable vector.")
    return [median_vec.tolist()]

# --------------------------------------------------------------
# SAVE embeddings
# --------------------------------------------------------------
def save_user_embeddings(user_name: str, embeddings: List[List[float]]) -> int:
    if not embeddings:
        print(f"[Embed] No embeddings to save for {user_name}.")
        return 0

    if EMBED_FILE.exists():
        try:
            with open(EMBED_FILE, "r", encoding="utf-8") as f:
                db = json.load(f)
        except json.JSONDecodeError:
            db = {}
    else:
        db = {}

    db[user_name] = embeddings
    with open(EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    print(f"[Embed] ✅ Saved {len(embeddings)} vector(s) for '{user_name}' → {EMBED_FILE}")
    return len(embeddings)

# --------------------------------------------------------------
# HIGH-LEVEL: Run embedding generation for a user
# --------------------------------------------------------------
def generate_and_save_embeddings_for_user(user_name: str, cropped_folder: Path) -> None:
    print(f"[Embed] Starting embedding generation for '{user_name}'...")

    embs = compute_embeddings_for_folder(cropped_folder)
    embs = average_embeddings(embs)
    saved = save_user_embeddings(user_name, embs)

    if saved:
        print(f"[Embed] 🎯 Embedding generation complete for '{user_name}'.")
    else:
        print(f"[Embed] ⚠️ No embeddings generated for '{user_name}'.")

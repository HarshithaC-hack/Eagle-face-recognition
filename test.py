"""
Rebuild ALL user data from existing raw images.
- Reads dataset/Custom/<user>/raw
- Crops faces using existing _crop_largest_face()
- Saves cropped images to dataset/Custom/<user>/cropped
- Rebuilds embeddings.json using your embedding_manager
- Rewrites users.json fresh

Run:
    python backend/rebuild_from_raw.py
"""
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import shutil
import json
import cv2
from backend.config import DATASET_DIR, USERS_FILE, EMBED_FILE, FACE_SIZE, FACE_CASCADE_PATH
from backend.embedding_manager import generate_and_save_embeddings_for_user
from backend.user_manager import _write_json
from backend.photo_capture import _crop_largest_face

def rebuild_all():
    print("\n🔄 Rebuilding system from RAW images...\n")

    if not DATASET_DIR.exists():
        print("❌ No dataset directory found.")
        return

    # 1) RESET users.json and embeddings.json
    _write_json(USERS_FILE, {})
    _write_json(EMBED_FILE, {})

    users_db = {}

    # 2) Loop through each user folder
    for user_dir in DATASET_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        user_name = user_dir.name
        raw_dir = user_dir / "raw"

        if not raw_dir.exists():
            print(f"⚠️ No RAW folder for '{user_name}', skipping...")
            continue

        cropped_dir = user_dir / "cropped"

        # Clean cropped folder
        if cropped_dir.exists():
            shutil.rmtree(cropped_dir)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📸 Processing user: {user_name}")
        print(f"RAW folder: {raw_dir}")

        # 3) Crop all faces
        images = sorted([p for p in raw_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not images:
            print(f"❌ No images found in raw for '{user_name}'. Skipping.")
            continue

        count = 0
        for p in images:
            try:
                img = cv2.imread(str(p))
                cropped_rgb = _crop_largest_face(img, FACE_SIZE)
                out_path = cropped_dir / f"img_{count+1}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
                count += 1
            except Exception as e:
                print(f"[Warning] Cannot crop {p.name}: {e}")
                continue

        print(f"✔ Cropped {count} images for {user_name}")

        if count == 0:
            print(f"⚠️ No valid faces for '{user_name}', skipping embedding.")
            continue

        # 4) Save user into users.json
        users_db[user_name] = {"name": user_name}

        # 5) Generate embeddings for this user
        generate_and_save_embeddings_for_user(user_name, cropped_dir)

    # 6) Write users.json
    print("\n📝 Writing users.json...")
    # Convert {name: {name}} -> {id: {name}} format with fake UUIDs
    final_users = {}
    import uuid
    for uname in users_db.keys():
        uid = str(uuid.uuid4())[:8]
        final_users[uid] = {"name": uname}

    _write_json(USERS_FILE, final_users)

    print("\n🎉 REBUILD COMPLETE!")
    print("✔ All cropped images refreshed")
    print("✔ All embeddings rebuilt")
    print("✔ users.json & embeddings.json replaced cleanly\n")


if __name__ == "__main__":
    rebuild_all()

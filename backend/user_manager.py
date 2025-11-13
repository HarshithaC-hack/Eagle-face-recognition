"""
User management utilities.
Handles creation/deletion in users.json, embeddings.json, and dataset folders.
"""

from __future__ import annotations
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from filelock import FileLock  # 🔒 prevents file corruption from concurrent writes

from .config import USERS_FILE, DATASET_DIR, EMBED_FILE


@dataclass
class User:
    user_id: str
    name: str


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------
def _read_json(path: Path, default):
    """Safe read with auto-create if file doesn't exist."""
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[Error] Corrupted JSON file detected: {path}. Resetting it.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        return default


def _write_json(path: Path, data):
    """Thread-safe atomic write using file lock."""
    lock_path = str(path) + ".lock"
    with FileLock(lock_path):
        tmp_path = str(path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        Path(tmp_path).replace(path)  # atomic replace
    print(f"[Write] Updated {path.name} safely ✅")


# ---------------------------------------------------------
# Core Functions
# ---------------------------------------------------------
def list_users() -> Dict[str, Dict[str, str]]:
    """Return the raw users map: {user_id: {name: str}}"""
    return _read_json(USERS_FILE, {})


def add_user_record(name: str, capture_result: Optional[Tuple[Path, Path, int]] = None) -> Optional[str]:
    """
    Create a user entry in users.json **only if capture succeeded**.
    capture_result should be (raw_dir, cropped_dir, count) from capture_user_images().
    Returns the new user_id, or None if no faces captured.
    """
    users = list_users()

    # prevent duplicate names (case-insensitive)
    if any(info["name"].lower() == name.lower() for info in users.values()):
        print(f"[Users] '{name}' already exists, skipping registration.")
        return None

    # ✅ check if capture_result is valid before registering
    if not capture_result or capture_result[2] == 0:
        print(f"[Users] Registration aborted — no valid faces captured for '{name}'.")
        return None

    user_id = str(uuid.uuid4())[:8]
    users[user_id] = {"name": name}
    _write_json(USERS_FILE, users)

    # Create their folders (if not already made by capture)
    user_root = DATASET_DIR / name
    (user_root / "raw").mkdir(parents=True, exist_ok=True)
    (user_root / "cropped").mkdir(parents=True, exist_ok=True)

    print(f"[Users] Registered new user '{name}' (id={user_id})")
    return user_id


def delete_user_record(name_or_id: str) -> bool:
    """
    Deletes a user by name or ID from users.json, embeddings.json, and dataset folder.
    Returns True if successfully deleted, False if not found.
    """
    users = list_users()

    # Find the target user
    target_id = None
    for uid, info in users.items():
        if uid == name_or_id or info["name"].lower() == name_or_id.lower():
            target_id = uid
            break

    if not target_id:
        print(f"[Users] User '{name_or_id}' not found.")
        return False

    # Get the username before removing
    user_name = users[target_id]["name"]

    # 1. Remove from users.json
    del users[target_id]
    _write_json(USERS_FILE, users)
    print(f"[Users] Deleted record for '{user_name}'")

    # 2. Remove from embeddings.json
    if EMBED_FILE.exists():
        embeds = _read_json(EMBED_FILE, {})
        if user_name in embeds:
            del embeds[user_name]
            _write_json(EMBED_FILE, embeds)
            print(f"[Cleanup] Removed embeddings for '{user_name}'")

    # 3. Remove their dataset folder
    user_root = DATASET_DIR / user_name
    if user_root.exists():
        shutil.rmtree(user_root, ignore_errors=True)
        print(f"[Cleanup] Removed dataset folder for '{user_name}'")

    return True

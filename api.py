"""
Phase 2 — Flask backend API for Eagle Access
--------------------------------------------
Exposes REST endpoints to register, delete, list users, and run access verification.
Now includes live registration status tracking for GUI updates.
"""

from flask import Flask, request, jsonify
from backend import user_manager, photo_capture, embedding_manager, face_recognition
import threading
import atexit
import cv2

# Status tracking for registrations
registration_status = {}
status_lock = threading.Lock()

app = Flask(__name__)

# ---------------------------------------------------------------------
# BASIC ROUTES
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Eagle backend is alive 🦅"})


# ---------------------------------------------------------------------
# USER MANAGEMENT
# ---------------------------------------------------------------------
@app.route("/list", methods=["GET"])
def list_users():
    users = user_manager.list_users()
    return jsonify(users)


@app.route("/delete/<string:name>", methods=["DELETE"])
def delete_user(name):
    ok = user_manager.delete_user_record(name)
    if ok:
        return jsonify({"message": f"User '{name}' deleted"})
    return jsonify({"error": "User not found"}), 404


# ---------------------------------------------------------------------
# REGISTER USER (capture + embed)
# ---------------------------------------------------------------------
@app.route("/register", methods=["POST"])
def register_user():
    """
    Registers a new user via webcam capture and embedding.
    You can POST form-data { "name": "Harshi" } from any client.
    """
    name = request.form.get("name")
    if not name:
        return jsonify({"error": "Missing 'name' field"}), 400

    with status_lock:
        users = user_manager.list_users()
        if any(info["name"].lower() == name.lower() for info in users.values()):
            registration_status[name] = {"status": "failed"}
            return jsonify({"error": f"User '{name}' already exists."}), 400


    with status_lock:
        registration_status[name] = {"status": "processing"}

    def process_registration():
        try:
            raw_dir, cropped_dir, n = photo_capture.capture_user_images(name)
            if n == 0:
                print("[Register] No frames captured; aborting.")
                with status_lock:
                    registration_status[name] = {"status": "failed"}
                return

            embedding_manager.generate_and_save_embeddings_for_user(name, cropped_dir)
            print(f"[Register] Completed registration for {name}")

            capture_result = (raw_dir, cropped_dir, n)
            uid = user_manager.add_user_record(name, capture_result)

            with status_lock:
                registration_status[name] = (
                    {"status": "completed", "user_id": uid}
                    if uid else {"status": "failed"}
                )

        except Exception as e:
            print(f"[Register] Failed for {name}: {e}")
            with status_lock:
                registration_status[name] = {"status": "failed"}

    threading.Thread(target=process_registration, daemon=True).start()
    return jsonify({"message": f"User '{name}' registration started"})


# ---------------------------------------------------------------------
# REGISTRATION STATUS CHECK
# ---------------------------------------------------------------------
@app.route("/status/<string:name>", methods=["GET"])
def registration_status_check(name):
    with status_lock:
        status = registration_status.get(name)
    if not status:
        return jsonify({"status": "unknown"})
    return jsonify(status)


# ---------------------------------------------------------------------
# ACCESS VERIFY
# ---------------------------------------------------------------------
@app.route("/access", methods=["POST"])
def access():
    user = face_recognition.verify_face_live()
    return jsonify({"match": user})




# ---------------------------------------------------------------------
# CLEANUP ON EXIT
# ---------------------------------------------------------------------
# @atexit.register
# def release_camera():
#     try:
#         cam = cv2.VideoCapture(0)
#         if cam.isOpened():
#             cam.release()
#             print("[Exit] Webcam released successfully 🎥")
#     except Exception as e:
#         print(f"[Exit] Camera release failed: {e}")


# ---------------------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)


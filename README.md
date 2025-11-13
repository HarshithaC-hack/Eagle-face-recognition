# 🦅 Eagle Face Recognition

### *Real-time Facial Authentication using DeepFace, OpenCV, Flask Backend & Tkinter Frontend*

**Eagle Face Recognition** is a fast, offline, high-accuracy facial authentication system built with:

* 🧠 **DeepFace (Facenet512)** for consistent embeddings
* 🎥 **OpenCV** for face detection & preprocessing
* 🔥 **Flask** for backend APIs
* 🖥️ **Tkinter** for the desktop GUI

The system achieves **85–95% recognition confidence** using stable embeddings and cosine similarity matching.

---

## ✨ Key Features

### 🧑‍💻 Frontend — Tkinter GUI

* Modern dark-themed desktop interface
* Buttons for:

  * Register User
  * Delete User
  * List Users
  * Access (Face Verification)
* Live logs with auto-scroll
* Async status polling
* Real-time recognition results with confidence %

---

### 🔥 Backend — Flask API

| Endpoint                | Description                        |
| ----------------------- | ---------------------------------- |
| `POST /register`        | Capture + embed + save a new user  |
| `DELETE /delete/<name>` | Delete user + dataset + embeddings |
| `GET /list`             | List all registered users          |
| `POST /access`          | Live face verification             |
| `GET /status/<name>`    | Status polling for registration    |

---

### 🎥 Face Recognition Pipeline

* Haar Cascade face detection
* `_crop_largest_face()` for consistent cropping
* RGB normalization
* Facenet512 embeddings (DeepFace, detector skipped)
* Median vector per user for stable identity representation
* Cosine similarity matching
* Confidence conversion for UI display

---

### 📦 Dataset Auto-Management

```
dataset/
└── Custom/
    └── <username>/
         ├── raw/
         └── cropped/
```

Automatically managed:

* ✓ `users.json`
* ✓ `embeddings.json`
* ✓ `access_log.json`

---

## 🏗 Project Structure

```
Eagle-face-recognition/
│
├── backend/
│   ├── config.py              # Global paths & settings
│   ├── photo_capture.py       # Face detection & cropping
│   ├── face_recognition.py    # Embedding + verification
│   ├── embedding_manager.py   # Build embeddings for users
│   ├── user_manager.py        # Add/Delete/List users
│   └── main_console.py        # Optional CLI for debugging
│
├── frontend/
│   └── app_gui.py             # Tkinter GUI
│
├── dataset/
│   └── Custom/                # User folders created automatically
│
├── users.json                 # Registered users
├── embeddings.json            # Stored embeddings
├── access_log.json            # Access events
├── api.py                     # Flask backend entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Start the Backend

```
python backend/api.py
```

You should see:

```
 * Running on http://127.0.0.1:5000
```

### 3️⃣ Start the GUI

```
python app_gui.py
```

Your Eagle Access desktop app launches instantly.

---

## 🧑‍🏫 How Registration Works

1. Enter username in GUI
2. Click **Register User**
3. Backend captures 30 webcam frames
4. Largest face is cropped per frame
5. Embeddings are generated (Facenet512)
6. Median embedding stored
7. JSON + dataset auto-updated
8. Status streamed to GUI

---

## 🔐 How Verification Works

1. Click **Access Eagle**
2. Webcam captures live frame
3. Same cropping pipeline used
4. Embedding computed
5. Cosine similarity applied
6. GUI displays:

   * Recognized user
   * Distance
   * Confidence %
   * Color-coded result window

---

## 🧠 Recognition Logic

### Cosine Distance

```
distance = 1 - cosine_similarity(embed_live, embed_stored)
```

### Acceptance Threshold

```
distance ≤ 0.45
```

### Confidence %

```
confidence = 1 - (distance / 1.2)
```

---

## 🗂 User Management

### Add User

```
POST /register
```

### Delete User

```
DELETE /delete/<name>
```

Also deletes:

* embeddings
* dataset folder
* user record

### List Users

```
GET /list
```

---

## 🦅 Roadmap

* Anti-spoofing (blink detection / depth sensing)
* Web dashboard (HTML/JS)
* Embedding encryption
* Admin login system
* Access log analytics dashboard

---

## 📝 License

MIT License — free to use and modify.

---

## 💛 Author

**Harshitha C**
*"Eagle Access — built with patience, persistence, and passion."*

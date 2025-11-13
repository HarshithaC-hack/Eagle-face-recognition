🦅 Eagle Face Recognition
Real-time Facial Authentication with DeepFace, OpenCV, Flask Backend & Tkinter Frontend

Eagle Face Recognition is a fast, offline, high-accuracy access control system combining:

🧠 DeepFace (Facenet512) for precise face embeddings

🎥 OpenCV for face detection & preprocessing

🔥 Flask backend for registration, deletion, listing & verification APIs

🖥️ Tkinter GUI for a desktop-friendly access control app

This project delivers 85–95% recognition confidence with consistent preprocessing, stable embeddings, and robust cosine similarity matching.

✨ Key Features
🧑‍💻 Frontend (Tkinter GUI)

Modern dark theme UI

Buttons for:

Register User

Delete User

List Users

Access (Face Verification)

Scrollable log window

Automatic status polling

Beautiful JSON → human-friendly messages

Real-time recognition results (with confidence %)

🔥 Backend (Flask API)

/register — capture, crop, embed and save a new user

/delete/<name> — delete user + embeddings + dataset

/list — list all registered users

/access — live face verification

/status/<name> — async registration progress polling

🎥 Face Recognition Pipeline

Haar Cascade face detection

_crop_largest_face() to ensure consistent cropping

RGB normalization

Facenet512 embeddings (via DeepFace)

Median embedding per user (for stability)

Cosine similarity matching

Confidence conversion for UI display

📦 Dataset Auto-Management

dataset/Custom/<user>/raw/

dataset/Custom/<user>/cropped/

Automatic folder creation + cleanup

Auto update of:

users.json

embeddings.json

access_log.json

🏗 Project Structure
Eagle-face-recognition/
│
├── backend/               
│   ├── config.py              # Global paths & settings
│   ├── photo_capture.py       # Face detection & cropping
│   ├── face_recognition.py    # Embedding + verification
│   ├── embedding_manager.py   # Build embeddings for users
│   ├── user_manager.py        # Add/Delete/List users
│   └── main_console.py        # Optional CLI for debugging
├── frontend/
|   |── app_gui.py                 # Tkinter GUI (frontend)
│
├── dataset/
│   └── Custom/                # User folders created automatically
│
├── users.json                 # Registered users
├── embeddings.json            # Stored embeddings
├── access_log.json            # Access events (optional)
├── api.py 
├── requirements.txt
└── README.md

🚀 Getting Started
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start the Backend

From project root:

python backend/api.py


You should see:

 * Running on http://127.0.0.1:5000

3️⃣ Start the GUI

In a second terminal:

python app_gui.py


Your Eagle Access desktop app launches instantly.

🧑‍🏫 How Registration Works

Enter username in the GUI

Click Register User

Backend captures 30 images via webcam

Largest face is cropped from each

Embeddings are generated

Median embedding is stored

JSON + dataset folders update automatically

Status is sent live to the GUI

🔐 How Verification Works

Click Access Eagle

Webcam captures a live frame

Same cropping pipeline is applied

Generate embedding with Facenet512

Compare using cosine distance

Output includes:

Recognized user

Distance

Confidence percentage

Displayed in a 640×480 window (color)

GUI logs the result beautifully

🧠 Recognition Logic
Cosine distance
distance = 1 - cosine_similarity(embed_live, embed_stored)

Good match threshold
distance <= 0.45 → Accepted

Confidence %
confidence = 1 - (distance / 1.2)

🗂 User Management
Add User
POST /register

Delete User
DELETE /delete/<name>


Also deletes:

embeddings

dataset folder

user record

List Users
GET /list

👀 Screenshots (optional)

Add your own later:

Registration in progress

Access Granted

Access Denied

GUI Screenshot

🦅 Roadmap (Future Enhancements)

Anti-spoofing (blink detection / depth check)

Web dashboard instead of Tkinter

Encryption for embeddings

Admin login

Access log viewer UI

📝 License

MIT License — free for personal & professional use.

💛 Author

Harshitha C
“Eagle Access — built with patience, persistence, and passion.”

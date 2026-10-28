
import cv2
import os
import csv
import re
import numpy as np
from datetime import datetime

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
MODEL_FILE = "trainer.yml"

if not hasattr(cv2, "face"):
    raise SystemExit(
        "Install opencv-contrib-python: "
        "pip install opencv-contrib-python"
    )

if not os.path.isdir(DATASET_DIR):
    raise SystemExit("Dataset folder not found. Run register.py first.")

# 1. Load registered face images
faces = []
labels = []
label_to_student = {}
label_number = 0

for folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # Folder format: studentID_studentName
    parts = folder.split("_", 1)
    if len(parts) != 2:
        continue

    student_id, student_name = parts
    if not student_id or not student_name:
        continue

    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                images.append(cv2.resize(image, (200, 200)))

    if not images:
        continue

    label_number += 1
    label_to_student[label_number] = (student_id, student_name)

    for image in images:
        faces.append(image)
        labels.append(label_number)

if not faces:
    raise SystemExit("No face images found. Run register.py first.")

# 2. Train LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels, dtype=np.int32))
recognizer.write(MODEL_FILE)

print(f"Training complete. Students: {len(label_to_student)}")
print("Starting camera. Press Q to exit.")

# 3. Create attendance file if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Student ID", "Name", "Date", "Time"])

# Read today's existing attendance
today = datetime.now().strftime("%Y-%m-%d")
marked_today = set()

with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row.get("Date") == today:
            marked_today.add(row.get("Student ID", ""))

def mark_attendance(student_id, student_name):
    global marked_today

    if student_id in marked_today:
        return "Already Marked"

    now = datetime.now()

    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            student_id,
            student_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ])

    marked_today.add(student_id)
    print(f"Attendance marked: {student_name} ({student_id})")
    return "Attendance Marked"

# 4. Start webcam
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit("Cannot open camera.")

# Lower LBPH confidence distance is better.
# Tune this threshold using your own test images.
THRESHOLD = 65

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < THRESHOLD and label in label_to_student:
            student_id, student_name = label_to_student[label]
            status = mark_attendance(student_id, student_name)
            display_text = f"{student_name}: {status}"
            color = (0, 255, 0)
        else:
            display_text = "Unknown Face"
            color = (0, 0, 255)

        cv2.rectangle(
            frame, (x, y), (x+w, y+h), color, 2
        )
        cv2.putText(
            frame, display_text, (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

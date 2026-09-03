import cv2
import os
from datetime import datetime

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

RECORDS_PATH = os.path.join(
    BASE_DIR,
    "security_records"
)

os.makedirs(RECORDS_PATH, exist_ok=True)

# -------------------------------
# LOAD FACE DETECTOR
# -------------------------------
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    print("ERROR: Haarcascade file not found!")
    print("Expected:", CASCADE_PATH)
    exit()

# -------------------------------
# START CAMERA
# -------------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened!")
    exit()

print("Security camera started.")
print("Press ESC to stop.")

# -------------------------------
# UNKNOWN FACE CAPTURE
# -------------------------------
capture_cooldown = 0

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "UNKNOWN",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Prevent saving hundreds of photos
        if capture_cooldown == 0:

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            filename = f"unknown_{timestamp}.jpg"

            filepath = os.path.join(
                RECORDS_PATH,
                filename
            )

            cv2.imwrite(filepath, frame)

            print("⚠️ UNKNOWN PERSON DETECTED")
            print("Photo saved:", filepath)

            capture_cooldown = 100

    if capture_cooldown > 0:
        capture_cooldown -= 1

    cv2.imshow(
        "AI Smart Lock - Security Camera",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()

print("Security camera stopped.")
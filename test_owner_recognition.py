import cv2
import os
import itertools
import numpy as np
from deepface import DeepFace

# ============================================================
# QUICK OWNER RECOGNITION CHECK
#
# Before doing full threshold calibration with impostor data,
# this just checks: does the system consistently recognize
# Gauri as Gauri across her own photos? Compares every photo
# in owner_photos/Gauri against every other photo in that
# folder and reports the distance for each pair.
# ============================================================

BASE_DIR = r"C:\Users\gauri\python"

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
GENUINE_FOLDER = os.path.join(BASE_DIR, "owner_photos", "Gauri")

MODEL_NAME = "VGG-Face"
DEFAULT_DEEPFACE_THRESHOLD = 0.68  # what DeepFace.verify() uses by default for this model/metric


def load_cascade():

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")

    return face_cascade


def get_face_crop(image_bgr, face_cascade):

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_bgr.shape[1], x + w + pad_x)
    y2 = min(image_bgr.shape[0], y + h + pad_y)

    crop = image_bgr[y1:y2, x1:x2]

    return crop if crop.size > 0 else None


def get_embedding(image_path, face_cascade):

    image = cv2.imread(image_path)

    if image is None:
        print(f"  Could not read: {image_path}")
        return None

    crop = get_face_crop(image, face_cascade)

    if crop is None:
        print(f"  No face detected: {image_path}")
        return None

    try:

        results = DeepFace.represent(
            img_path=crop,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False
        )

        if not results:
            print(f"  No embedding produced: {image_path}")
            return None

        return results[0]["embedding"]

    except Exception as error:

        print(f"  Embedding error on {image_path}: {error}")

        return None


def cosine_distance(a, b):

    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 1.0

    return 1.0 - (np.dot(a, b) / denom)


def main():

    face_cascade = load_cascade()

    if not os.path.exists(GENUINE_FOLDER):

        print(f"Folder does not exist: {GENUINE_FOLDER}")

        return

    files = [
        f for f in os.listdir(GENUINE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(files)} photos in {GENUINE_FOLDER}")
    print()
    print("Extracting embeddings...")

    embeddings = []

    for filename in sorted(files):

        image_path = os.path.join(GENUINE_FOLDER, filename)

        embedding = get_embedding(image_path, face_cascade)

        if embedding is not None:
            embeddings.append((filename, embedding))
            print(f"  OK: {filename}")

    if len(embeddings) < 2:

        print()
        print("Need at least 2 usable photos to compare. Check the")
        print("'No face detected' messages above if some were skipped.")

        return

    print()
    print("================================")
    print("PAIRWISE DISTANCES (same person - Gauri vs Gauri)")
    print("================================")

    distances = []

    for (name_a, emb_a), (name_b, emb_b) in itertools.combinations(embeddings, 2):

        distance = cosine_distance(emb_a, emb_b)

        distances.append(distance)

        verdict = (
            "MATCH"
            if distance < DEFAULT_DEEPFACE_THRESHOLD
            else "MISMATCH <-- problem"
        )

        print(f"{name_a}  vs  {name_b}  ->  {distance:.4f}  [{verdict}]")

    distances = np.array(distances)

    print()
    print("================================")
    print("SUMMARY")
    print("================================")
    print(f"Pairs compared: {len(distances)}")
    print(f"Min distance:   {distances.min():.4f}")
    print(f"Max distance:   {distances.max():.4f}")
    print(f"Mean distance:  {distances.mean():.4f}")

    mismatches = np.sum(distances >= DEFAULT_DEEPFACE_THRESHOLD)

    print()

    if mismatches == 0:

        print(
            f"All {len(distances)} pairs matched under the default "
            f"threshold ({DEFAULT_DEEPFACE_THRESHOLD}). "
            "Basic recognition is working - safe to move on to "
            "impostor testing / full calibration."
        )

    else:

        print(
            f"{mismatches} out of {len(distances)} pairs did NOT match "
            f"under the default threshold ({DEFAULT_DEEPFACE_THRESHOLD})."
        )

        print(
            "This means some of your own photos look 'too different' "
            "to the model - likely due to very different lighting/angle/"
            "resolution between photos. Consider replacing the most "
            "inconsistent photos with ones taken in similar conditions "
            "to your live camera before calibrating further."
        )


if __name__ == "__main__":
    main()
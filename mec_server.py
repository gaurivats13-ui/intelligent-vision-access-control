from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import tempfile
from deepface import DeepFace

app = Flask(__name__)

BASE_DIR = r"C:\Users\gauri\python"

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# All photos in this folder are used as the owner reference set.
# Loading dynamically (instead of a fixed filename list) means you
# can just drop new photos in here anytime and restart the server -
# no code changes needed. Any photo where no face is detected is
# automatically skipped.
OWNER_FOLDER = os.path.join(BASE_DIR, "owner_photos", "Gauri")

MODEL_NAME = "VGG-Face"

# Calibrated via calibrate_threshold.py:
#   genuine max = 0.3751, impostor min = 0.6605 -> midpoint ~0.52
# Nudged up slightly to 0.58 after live testing showed a genuine
# webcam-captured photo landing right at the edge (0.5239) - static
# reference photos and live webcam frames aren't identical in
# compression/lighting, so a bit more tolerance is needed. Still a
# safe ~0.08 margin below the impostor floor (0.6605).
AUTH_THRESHOLD = 0.58

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from {CASCADE_PATH}")


def get_face_crop(image_bgr):
    """
    Detects the largest face with our own Haar cascade and returns
    a padded crop. Using our own cascade (instead of DeepFace's
    detector_backend='opencv') avoids depending on cv2's bundled
    data files, which can be missing/broken on some installs.
    """

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


def get_embedding(image_bgr):

    crop = get_face_crop(image_bgr)

    if crop is None:
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
            return None

        return results[0]["embedding"]

    except Exception as error:

        print("[EMBEDDING ERROR]", error)

        return None


def cosine_distance(a, b):

    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 1.0

    return 1.0 - (np.dot(a, b) / denom)


# --------------------------------------------------------------
# Pre-compute owner embeddings ONCE at startup (not on every
# request) - much faster than re-running DeepFace.verify()
# against each owner photo per incoming request.
# --------------------------------------------------------------

print("Loading owner reference photos and computing embeddings...")

owner_embeddings = []

if not os.path.exists(OWNER_FOLDER):

    raise RuntimeError(f"Owner photos folder not found: {OWNER_FOLDER}")

owner_photo_files = [
    f for f in sorted(os.listdir(OWNER_FOLDER))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for filename in owner_photo_files:

    photo_path = os.path.join(OWNER_FOLDER, filename)

    image = cv2.imread(photo_path)

    if image is None:

        print(f"  [WARNING] Could not read: {filename}")

        continue

    embedding = get_embedding(image)

    if embedding is not None:

        owner_embeddings.append(embedding)

        print(f"  Loaded: {filename}")

    else:

        print(f"  [SKIPPED - no face detected]: {filename}")

if not owner_embeddings:

    raise RuntimeError(
        "No usable owner embeddings loaded. Check OWNER_PHOTOS paths."
    )

print(f"Owner reference set ready: {len(owner_embeddings)} embeddings.")


@app.route("/authenticate", methods=["POST"])
def authenticate():

    if "image" not in request.files:
        return jsonify({
            "authenticated": False,
            "error": "No image received"
        }), 400

    image_file = request.files["image"]

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:

            image_file.save(temp_file)
            temp_path = temp_file.name

        print("\n[REQUEST] Image received")
        print("[AI] Extracting embedding and comparing to owner reference set...")

        live_image = cv2.imread(temp_path)

        if live_image is None:

            return jsonify({
                "authenticated": False,
                "error": "Could not decode received image"
            }), 400

        live_embedding = get_embedding(live_image)

        if live_embedding is None:

            response = {
                "authenticated": False,
                "name": "UNKNOWN",
                "reason": "No face detected in received image",
                "distance": None,
                "threshold": AUTH_THRESHOLD
            }

            print("[RESULT]", response)

            return jsonify(response)

        # Compare against EVERY owner reference embedding, take
        # the best (minimum) distance - more robust than a single
        # reference photo, since lighting/angle varies per photo.
        best_distance = min(
            cosine_distance(live_embedding, owner_emb)
            for owner_emb in owner_embeddings
        )

        # Explicit bool() cast: numpy comparisons return numpy.bool_,
        # which Flask's jsonify cannot serialize - only native Python
        # bool works.
        authenticated = bool(best_distance < AUTH_THRESHOLD)

        response = {
            "authenticated": authenticated,
            "name": "Gauri" if authenticated else "UNKNOWN",
            "distance": round(float(best_distance), 4),
            "threshold": AUTH_THRESHOLD
        }

        print("[RESULT]", response)

        return jsonify(response)

    except Exception as error:

        print("[ERROR]", str(error))

        return jsonify({
            "authenticated": False,
            "error": str(error)
        }), 500

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":

    print("====================================")
    print("      MEC SMART LOCK SERVER")
    print("====================================")
    print("Model:", MODEL_NAME)
    print("Calibrated threshold:", AUTH_THRESHOLD)
    print("Listening on port 5000")

    app.run(host="0.0.0.0", port=5000, debug=False)
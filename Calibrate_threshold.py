import cv2
import os
import itertools
import numpy as np
from deepface import DeepFace

# ============================================================
# THRESHOLD CALIBRATION SCRIPT
#
# Computes:
#   - Genuine distances: every Gauri photo vs every other Gauri photo
#   - Impostor distances: every Gauri photo vs every impostor photo
#
# Then scans candidate thresholds and reports False Acceptance
# Rate (FAR) and False Rejection Rate (FRR) at each, so you can
# pick a threshold based on real data instead of DeepFace's
# generic default (0.68 for VGG-Face/cosine).
#
# Model/metric kept consistent with what you already validated
# through the Flask MEC pipeline: VGG-Face + cosine.
# ============================================================

BASE_DIR = r"C:\Users\gauri\python"

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

GENUINE_FOLDER = os.path.join(BASE_DIR, "owner_photos", "Gauri")
IMPOSTOR_FOLDER = os.path.join(BASE_DIR, "impostor_photos")

MODEL_NAME = "VGG-Face"
DISTANCE_METRIC = "cosine"

# Candidate thresholds to scan (fine-grained around the
# typical useful range for VGG-Face cosine distance)
THRESHOLD_CANDIDATES = [round(t, 2) for t in np.arange(0.20, 0.70, 0.02)]


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


def load_embeddings_from_folder(folder_path, face_cascade, label):

    if not os.path.exists(folder_path):
        print(f"WARNING: folder does not exist: {folder_path}")
        return []

    embeddings = []

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\nProcessing {len(files)} {label} images from {folder_path}")

    for filename in sorted(files):

        image_path = os.path.join(folder_path, filename)

        embedding = get_embedding(image_path, face_cascade)

        if embedding is not None:
            embeddings.append((filename, embedding))
            print(f"  OK: {filename}")

    return embeddings


def main():

    face_cascade = load_cascade()

    genuine_embeddings = load_embeddings_from_folder(
        GENUINE_FOLDER,
        face_cascade,
        "genuine (Gauri)"
    )

    impostor_embeddings = load_embeddings_from_folder(
        IMPOSTOR_FOLDER,
        face_cascade,
        "impostor (unknown people)"
    )

    if len(genuine_embeddings) < 2:

        print()
        print("Need at least 2 usable genuine images to compute genuine pairs.")

        return

    if len(impostor_embeddings) < 1:

        print()
        print(f"No usable impostor images found in {IMPOSTOR_FOLDER}")
        print("Add photos of several different people (not Gauri) there and rerun.")

        return

    # --------------------------------------------------------
    # GENUINE DISTANCES (every Gauri photo vs every other)
    # --------------------------------------------------------

    genuine_distances = []

    for (name_a, emb_a), (name_b, emb_b) in itertools.combinations(genuine_embeddings, 2):

        distance = cosine_distance(emb_a, emb_b)

        genuine_distances.append(distance)

    # --------------------------------------------------------
    # IMPOSTOR DISTANCES (every Gauri photo vs every impostor photo)
    # --------------------------------------------------------

    impostor_distances = []

    for (name_g, emb_g) in genuine_embeddings:

        for (name_i, emb_i) in impostor_embeddings:

            distance = cosine_distance(emb_g, emb_i)

            impostor_distances.append(distance)

    genuine_distances = np.array(genuine_distances)
    impostor_distances = np.array(impostor_distances)

    # --------------------------------------------------------
    # SUMMARY STATS
    # --------------------------------------------------------

    print()
    print("================================")
    print("DISTANCE DISTRIBUTION SUMMARY")
    print("================================")

    print(f"Genuine pairs:  {len(genuine_distances)}")
    print(
        f"  min={genuine_distances.min():.4f}  "
        f"max={genuine_distances.max():.4f}  "
        f"mean={genuine_distances.mean():.4f}  "
        f"std={genuine_distances.std():.4f}"
    )

    print(f"Impostor pairs: {len(impostor_distances)}")
    print(
        f"  min={impostor_distances.min():.4f}  "
        f"max={impostor_distances.max():.4f}  "
        f"mean={impostor_distances.mean():.4f}  "
        f"std={impostor_distances.std():.4f}"
    )

    if genuine_distances.max() < impostor_distances.min():

        print()
        print(
            "Clean separation: genuine max "
            f"({genuine_distances.max():.4f}) is below impostor min "
            f"({impostor_distances.min():.4f})."
        )

        print(
            "Any threshold between these two values works with zero "
            "errors on this dataset."
        )

    else:

        print()
        print(
            "Overlap exists between genuine and impostor distances - "
            "no single threshold will be perfect. See the table below "
            "to choose your trade-off."
        )

    # --------------------------------------------------------
    # THRESHOLD SCAN: FAR / FRR TABLE
    # --------------------------------------------------------

    print()
    print("================================")
    print("THRESHOLD  |  FRR (owner rejected)  |  FAR (stranger accepted)")
    print("================================")

    best_threshold = None
    best_error_sum = float("inf")

    for threshold in THRESHOLD_CANDIDATES:

        frr = np.mean(genuine_distances >= threshold)  # owner wrongly rejected
        far = np.mean(impostor_distances < threshold)   # stranger wrongly accepted

        print(f"  {threshold:.2f}     |  {frr * 100:6.2f}%             |  {far * 100:6.2f}%")

        # Prioritize low FAR (security system - false accepts are worse
        # than false rejects) while still minimizing total errors.
        # Weight FAR more heavily than FRR.
        weighted_error = (far * 3) + frr

        if weighted_error < best_error_sum:
            best_error_sum = weighted_error
            best_threshold = threshold

    print()
    print("================================")
    print(f"SUGGESTED THRESHOLD: {best_threshold}")
    print("================================")
    print(
        "This minimizes a weighted combination of FAR and FRR "
        "(FAR weighted 3x higher, since a false accept on a physical "
        "lock is worse than asking the owner to try again)."
    )
    print(
        "Review the full table above - if you want a stricter security "
        "posture, pick a threshold with 0.00% FAR even if FRR is a bit "
        "higher (owner may occasionally need a second attempt)."
    )


if __name__ == "__main__":
    main()
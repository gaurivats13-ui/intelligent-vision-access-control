import cv2
import numpy as np
import serial
import time
import os
import sqlite3
import shutil
import random
import hashlib
import json
import speech_recognition as sr
import pyttsx3
import difflib

from collections import defaultdict
from datetime import datetime, timedelta
from deepface import DeepFace

# ============================================================
# AI SMART LOCK SYSTEM
# DEEPFACE (FACENET512) VERSION
#
# Replaces LBPH (cv2.face.LBPHFaceRecognizer) with deep
# learning face embeddings. Instead of "training" a statistical
# model on all faces at once, each face gets a numeric
# fingerprint (embedding). Recognition = comparing the live
# face's embedding to stored embeddings via cosine distance.
# Adding a new guest is just appending to a JSON file - no
# retraining, and it does not affect anyone else's recognition.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

COM_PORT = "COM4"
BAUD_RATE = 9600

CAMERA_INDEX = 0
MIC_INDEX = 1

# ---------------- FACE MODEL ----------------

FACE_MODEL = "Facenet512"
DISTANCE_METRIC = "cosine"

# Cosine distance thresholds. Facenet512 gives a much cleaner
# separation than LBPH did: same person is typically well under
# 0.30, different people are typically well above it. Tune these
# up/down slightly after testing with your own camera/lighting.
OWNER_THRESHOLD = 0.30
GUEST_THRESHOLD = 0.30

REQUIRED_MATCHES = 7
GUEST_REQUIRED_MATCHES = 7

# Deep embedding extraction is heavier than LBPH per frame.
# Only run it every N frames to keep the camera feed smooth;
# increase this if the live feed feels laggy on your machine.
PROCESS_EVERY_N_FRAMES = 2

# ---------------- UNKNOWN ----------------

UNKNOWN_REQUIRED_MATCHES = 5
UNKNOWN_PHOTOS = 10
UNKNOWN_PHOTO_DELAY = 0.35

# ---------------- PASSWORD ----------------

PASSWORD_VALID_SECONDS = 60
MAX_PASSWORD_ATTEMPTS = 3
GUEST_LOCKOUT_SECONDS = 20


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\gauri\python"

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

SECURITY_FOLDER = os.path.join(
    BASE_DIR,
    "security_records"
)

GUEST_FOLDER = os.path.join(
    BASE_DIR,
    "temporary_guests"
)

# One-time setup folder for the owner(s). Structure:
#   owner_photos/
#       Gauri/
#           photo1.jpg
#           photo2.jpg
#           ...
# Put 15-20 varied photos of yourself in here (different
# angles/lighting). The system builds embeddings from this
# folder automatically on first run.
OWNER_PHOTOS_FOLDER = os.path.join(
    BASE_DIR,
    "owner_photos"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "smart_lock_database.db"
)

FACE_DATABASE_PATH = os.path.join(
    BASE_DIR,
    "face_database.json"
)


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(SECURITY_FOLDER, exist_ok=True)
os.makedirs(GUEST_FOLDER, exist_ok=True)
os.makedirs(OWNER_PHOTOS_FOLDER, exist_ok=True)


# ============================================================
# TTS
# ============================================================

tts_engine = None


def initialize_tts():

    global tts_engine

    if tts_engine is not None:
        return True

    try:

        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 175)
        tts_engine.setProperty("volume", 1.0)

        print("TTS engine initialized.")

        return True

    except Exception as error:

        print("TTS initialization failed:", error)

        tts_engine = None

        return False


def speak(text):

    global tts_engine

    print()
    print("VOICE:", text)

    try:

        if not initialize_tts():

            print("TTS unavailable. Continuing...")

            return

        tts_engine.say(text)
        tts_engine.runAndWait()

    except Exception as error:

        print("TTS error:", error)

        try:

            if tts_engine is not None:
                tts_engine.stop()

        except Exception:
            pass

        tts_engine = None

        print("TTS disabled. System continues.")


# ============================================================
# DATABASE
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            first_seen TEXT,
            last_seen TEXT,
            visit_count INTEGER DEFAULT 1
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            visitor_name TEXT,
            face_result TEXT,
            authentication_type TEXT,
            access_result TEXT,
            reason TEXT,
            photo_folder TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guest_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_name TEXT,
            temporary_password TEXT,
            created_at TEXT,
            expires_at TEXT,
            attempts INTEGER DEFAULT 0,
            status TEXT,
            guest_folder TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guest_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT,
            last_seen TEXT,
            visit_count INTEGER DEFAULT 1,
            guest_folder TEXT,
            status TEXT DEFAULT 'ACTIVE'
        )
    """)

    connection.commit()
    connection.close()


def save_visitor(visitor_name):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "SELECT id, visit_count FROM visitors WHERE lower(name) = lower(?)",
        (visitor_name,)
    )

    result = cursor.fetchone()

    if result:

        cursor.execute(
            "UPDATE visitors SET last_seen = ?, visit_count = ? WHERE id = ?",
            (now, result[1] + 1, result[0])
        )

    else:

        cursor.execute(
            """
            INSERT INTO visitors (name, first_seen, last_seen, visit_count)
            VALUES (?, ?, ?, ?)
            """,
            (visitor_name, now, now, 1)
        )

    connection.commit()
    connection.close()


def save_access_log(
    visitor_name,
    face_result,
    authentication_type,
    access_result,
    reason,
    photo_folder=""
):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO access_logs
        (timestamp, visitor_name, face_result, authentication_type,
         access_result, reason, photo_folder)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (timestamp, visitor_name, face_result, authentication_type,
         access_result, reason, photo_folder)
    )

    connection.commit()
    connection.close()


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_guest_profile(visitor_name):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, name, password_hash, guest_folder, last_seen,
               visit_count, status
        FROM guest_profiles
        WHERE lower(name) = lower(?)
        """,
        (visitor_name,)
    )

    result = cursor.fetchone()
    connection.close()

    return result


def save_guest_profile(visitor_name, guest_folder):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:

        cursor.execute(
            """
            INSERT INTO guest_profiles
            (name, password_hash, created_at, last_seen,
             visit_count, guest_folder, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (visitor_name, "", now, now, 1, guest_folder, "ACTIVE")
        )

        connection.commit()

    except sqlite3.IntegrityError:

        print("Guest profile already exists.")

    finally:

        connection.close()


def update_guest_visit(visitor_name):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        UPDATE guest_profiles
        SET last_seen = ?, visit_count = visit_count + 1
        WHERE lower(name) = lower(?)
        """,
        (now, visitor_name)
    )

    connection.commit()
    connection.close()


def save_guest_session(
    visitor_name,
    password,
    created_at,
    expires_at,
    guest_folder
):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO guest_sessions
        (visitor_name, temporary_password, created_at, expires_at,
         attempts, status, guest_folder)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (visitor_name, password, created_at, expires_at, 0, "ACTIVE", guest_folder)
    )

    session_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return session_id


def update_guest_session(session_id, attempts, status):

    if session_id is None:
        return

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        "UPDATE guest_sessions SET attempts = ?, status = ? WHERE id = ?",
        (attempts, status, session_id)
    )

    connection.commit()
    connection.close()


# ============================================================
# FACE DATABASE (embeddings, replaces trainer.yml / LBPH)
#
# Structure:
# {
#   "owner": { "Gauri": [[...512 floats...], [...], ...] },
#   "guests": { "Rishika": [[...], ...], "Ramaya": [[...], ...] }
# }
# ============================================================

def load_face_database():

    if not os.path.exists(FACE_DATABASE_PATH):
        return {"owner": {}, "guests": {}}

    try:

        with open(FACE_DATABASE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        data.setdefault("owner", {})
        data.setdefault("guests", {})

        return data

    except Exception as error:

        print("Could not load face_database.json:", error)

        return {"owner": {}, "guests": {}}


def save_face_database(face_database):

    try:

        with open(FACE_DATABASE_PATH, "w", encoding="utf-8") as file:
            json.dump(face_database, file)

    except Exception as error:

        print("Could not save face_database.json:", error)


def get_embedding_from_image(image_bgr, face_cascade=None, already_cropped=False):
    """
    Returns a single embedding (list of floats) for the largest
    face found in image_bgr, or None if no face / error.

    already_cropped=True means image_bgr is already a tight face
    crop (e.g. from the live camera loop where Haar already found
    it) - no detection needed at all.

    When already_cropped=False, detection is done with OUR OWN
    Haar cascade (face_cascade, loaded from CASCADE_PATH) instead
    of DeepFace's internal detector_backend="opencv". DeepFace's
    opencv backend depends on cv2's bundled cascade file
    (cv2.data.haarcascades), which can be missing from some
    opencv-python installs. Using our own cascade avoids that
    dependency entirely, and we always call DeepFace with
    detector_backend="skip" - i.e. only for embedding extraction,
    never for detection.
    """

    if not already_cropped:

        if face_cascade is None:
            print("Embedding extraction error: no face_cascade provided for detection.")
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            return None

        face_x, face_y, face_w, face_h = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        pad_x = int(face_w * 0.15)
        pad_y = int(face_h * 0.15)

        crop_x1 = max(0, face_x - pad_x)
        crop_y1 = max(0, face_y - pad_y)
        crop_x2 = min(image_bgr.shape[1], face_x + face_w + pad_x)
        crop_y2 = min(image_bgr.shape[0], face_y + face_h + pad_y)

        image_bgr = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        if image_bgr.size == 0:
            return None

    try:

        results = DeepFace.represent(
            img_path=image_bgr,
            model_name=FACE_MODEL,
            detector_backend="skip",
            enforce_detection=False,
            align=False
        )

        if not results:
            return None

        return results[0]["embedding"]

    except Exception as error:

        print("Embedding extraction error:", error)

        return None


def build_embeddings_from_folder(folder_path, face_cascade, per_person_subfolders=True):
    """
    Reads images and returns {name: [embedding, embedding, ...]}.

    per_person_subfolders=True expects folder_path/PersonName/*.jpg
    (used for owner_photos).

    per_person_subfolders=False expects folder_path/PersonName_timestamp/*.jpg
    (used for the existing temporary_guests structure, so old guest
    photos can be migrated automatically without re-registering).
    """

    people_embeddings = defaultdict(list)

    if not os.path.exists(folder_path):
        return people_embeddings

    for entry_name in sorted(os.listdir(folder_path)):

        entry_path = os.path.join(folder_path, entry_name)

        if not os.path.isdir(entry_path):
            continue

        if per_person_subfolders:
            person_name = entry_name.strip()
        else:
            # Name_YYYYMMDD_HHMMSS
            parts = entry_name.rsplit("_", 2)

            if len(parts) != 3:
                continue

            person_name = parts[0].strip()

        if not person_name:
            continue

        for filename in os.listdir(entry_path):

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(entry_path, filename)

            image = cv2.imread(image_path)

            if image is None:
                continue

            embedding = get_embedding_from_image(
                image,
                face_cascade=face_cascade,
                already_cropped=False
            )

            if embedding is not None:
                people_embeddings[person_name].append(embedding)

    return people_embeddings


def ensure_owner_database(face_database, face_cascade):
    """
    If no owner embeddings exist yet, build them from
    OWNER_PHOTOS_FOLDER. Run once at startup.
    """

    if face_database.get("owner"):
        return face_database

    print()
    print("No owner embeddings found. Building from owner_photos folder...")

    owner_embeddings = build_embeddings_from_folder(
        OWNER_PHOTOS_FOLDER,
        face_cascade,
        per_person_subfolders=True
    )

    if not owner_embeddings:

        print(
            "WARNING: No owner photos found in:",
            OWNER_PHOTOS_FOLDER
        )

        print(
            "Create a subfolder per owner (e.g. owner_photos/Gauri/) "
            "with 15-20 varied photos, then restart."
        )

        return face_database

    face_database["owner"] = dict(owner_embeddings)

    save_face_database(face_database)

    for name, embeds in owner_embeddings.items():
        print(f"Owner '{name}': {len(embeds)} embeddings built.")

    return face_database


def ensure_guest_database_migrated(face_database, face_cascade):
    """
    If no guest embeddings exist yet but the old temporary_guests
    folder has photos (from the previous LBPH version), migrate
    them automatically so existing guests don't need to re-register.
    """

    if face_database.get("guests"):
        return face_database

    guest_embeddings = build_embeddings_from_folder(
        GUEST_FOLDER,
        face_cascade,
        per_person_subfolders=False
    )

    if not guest_embeddings:
        return face_database

    print()
    print("Migrating existing guest photos into face_database.json...")

    face_database["guests"] = dict(guest_embeddings)

    save_face_database(face_database)

    for name, embeds in guest_embeddings.items():
        print(f"Guest '{name}': {len(embeds)} embeddings migrated.")

    return face_database


def add_guest_encodings(visitor_name, guest_folder, face_database, face_cascade):
    """
    Computes embeddings for all photos in guest_folder and appends
    them to face_database['guests'][visitor_name]. No retraining -
    this never touches any other guest's data.
    """

    new_embeddings = []

    if os.path.exists(guest_folder):

        for filename in os.listdir(guest_folder):

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(guest_folder, filename)

            image = cv2.imread(image_path)

            if image is None:
                continue

            embedding = get_embedding_from_image(
                image,
                face_cascade=face_cascade,
                already_cropped=False
            )

            if embedding is not None:
                new_embeddings.append(embedding)

    if not new_embeddings:

        print("No usable face embeddings extracted for", visitor_name)

        return face_database

    existing = face_database["guests"].get(visitor_name, [])

    face_database["guests"][visitor_name] = existing + new_embeddings

    save_face_database(face_database)

    print(
        f"Guest '{visitor_name}': "
        f"{len(new_embeddings)} new embeddings added "
        f"({len(existing) + len(new_embeddings)} total)."
    )

    return face_database


def cosine_distance(embedding_a, embedding_b):

    vector_a = np.array(embedding_a)
    vector_b = np.array(embedding_b)

    denominator = (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 1.0

    similarity = np.dot(vector_a, vector_b) / denominator

    return 1.0 - similarity


def best_match_in_group(live_embedding, group_dict):
    """
    group_dict: {name: [embedding, embedding, ...]}
    Returns (best_name, best_distance) - the closest person and
    their minimum distance across all their stored embeddings.
    """

    best_name = None
    best_distance = float("inf")

    for name, stored_embeddings in group_dict.items():

        for stored_embedding in stored_embeddings:

            distance = cosine_distance(live_embedding, stored_embedding)

            if distance < best_distance:
                best_distance = distance
                best_name = name

    return best_name, best_distance


def recognize_face(face_crop_bgr, face_database):
    """
    Returns (result_type, name, distance):
      result_type in {"KNOWN", "GUEST", "UNKNOWN"}
    """

    live_embedding = get_embedding_from_image(
        face_crop_bgr,
        face_cascade=None,
        already_cropped=True
    )

    if live_embedding is None:
        return ("UNKNOWN", "Unknown", None)

    # ---------------- OWNER CHECK ----------------

    owner_name, owner_distance = best_match_in_group(
        live_embedding,
        face_database.get("owner", {})
    )

    if owner_name is not None and owner_distance < OWNER_THRESHOLD:
        return ("KNOWN", owner_name, owner_distance)

    # ---------------- GUEST CHECK ----------------

    guest_name, guest_distance = best_match_in_group(
        live_embedding,
        face_database.get("guests", {})
    )

    if guest_name is not None and guest_distance < GUEST_THRESHOLD:
        return ("GUEST", guest_name, guest_distance)

    # ---------------- UNKNOWN ----------------

    return ("UNKNOWN", "Unknown", min(owner_distance, guest_distance)
            if owner_name and guest_name else None)


# ============================================================
# GUEST FOLDER
# ============================================================

def create_guest_folder(visitor_name):

    safe_name = ""

    for character in visitor_name:

        if character.isalnum() or character in (" ", "_", "-"):
            safe_name += character

    safe_name = safe_name.strip()

    if not safe_name:
        safe_name = "Guest"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = f"{safe_name}_{timestamp}"

    folder_path = os.path.join(GUEST_FOLDER, folder_name)

    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def generate_password():
    return str(random.randint(100000, 999999))


def capture_unknown_faces(camera):

    print()
    print("================================")
    print("UNKNOWN PERSON DETECTED")
    print("================================")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    photo_files = []

    for count in range(1, UNKNOWN_PHOTOS + 1):

        ret, frame = camera.read()

        if not ret:

            print("Camera frame error.")

            break

        filename = os.path.join(
            SECURITY_FOLDER,
            f"unknown_{timestamp}_{count}.jpg"
        )

        cv2.imwrite(filename, frame)

        photo_files.append(filename)

        print(f"Unknown face captured {count}/{UNKNOWN_PHOTOS}")

        time.sleep(UNKNOWN_PHOTO_DELAY)

    print()
    print("Security capture complete:", len(photo_files), "photos")

    return photo_files


def copy_photos_to_guest_folder(photo_files, guest_folder):

    copied = []

    for photo in photo_files:

        try:

            destination = os.path.join(guest_folder, os.path.basename(photo))

            shutil.copy2(photo, destination)

            copied.append(destination)

        except Exception as error:

            print("Photo copy error:", error)

    return copied


def send_arduino(arduino, command):

    try:

        if arduino is None:
            return

        arduino.write((command.strip() + "\n").encode())
        arduino.flush()

        print("ARDUINO <-", command)

    except Exception as error:

        print("Arduino communication error:", error)


# ============================================================
# KNOWN NAME MATCHING (voice)
# ============================================================

def get_known_names(face_database):

    known = set()

    for owner_name in face_database.get("owner", {}).keys():
        known.add(owner_name)

    for guest_name in face_database.get("guests", {}).keys():
        known.add(guest_name)

    try:

        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute("SELECT name FROM visitors")

        for (row_name,) in cursor.fetchall():
            if row_name:
                known.add(row_name)

        cursor.execute("SELECT name FROM guest_profiles")

        for (row_name,) in cursor.fetchall():
            if row_name:
                known.add(row_name)

        connection.close()

    except Exception as error:

        print("Could not load known names:", error)

    return list(known)


def pick_best_name_match(candidates, known_names, cutoff=0.72):
    """
    Fuzzy-matches Google's spoken-name candidates against known
    names already in the database. Previously used cutoff=0.4,
    which was far too loose - difflib's ratio() on short strings
    (names are usually 4-8 characters) can land in the 0.4-0.5
    range purely by chance even for names that share almost
    nothing in common (e.g. "Renu"/"Renuka" vs "Apeksha" scored
    0.46 and got accepted as a match).

    Fixes:
      - cutoff raised to 0.72 - only accept a known-name match
        when it's genuinely close, not just the "best available"
        option.
      - length-ratio guard - skip comparing strings whose lengths
        are too different to plausibly be the same name (e.g. a
        4-letter candidate against a 9-letter known name), since
        SequenceMatcher.ratio() can still score those deceptively
        high/low relative to what we want.
      - if nothing clears the cutoff, return None so the caller
        falls back to the name actually spoken, instead of forcing
        a bad match onto an unrelated known name.
    """

    if not known_names:
        return None

    best_name = None
    best_score = 0.0

    for candidate in candidates:

        candidate_clean = candidate.lower().strip()

        if not candidate_clean:
            continue

        for known in known_names:

            known_clean = known.lower().strip()

            if not known_clean:
                continue

            shorter = min(len(candidate_clean), len(known_clean))
            longer = max(len(candidate_clean), len(known_clean))

            # Names whose lengths differ by more than half aren't
            # plausibly the same word - skip to avoid noisy scores.
            if longer > 0 and (shorter / longer) < 0.5:
                continue

            score = difflib.SequenceMatcher(
                None,
                candidate_clean,
                known_clean
            ).ratio()

            if score > best_score:
                best_score = score
                best_name = known

    if best_score >= cutoff:

        print(f"Matched to known name '{best_name}' (similarity {best_score:.2f})")

        return best_name

    print(
        f"No confident match to a known name "
        f"(best similarity {best_score:.2f}, need >= {cutoff}). "
        f"Using the spoken name as-is."
    )

    return None


def ask_visitor_name(face_database):

    recognizer = sr.Recognizer()

    recognizer.energy_threshold = 250
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.3

    print()
    print("================================")
    print("VOICE NAME IDENTIFICATION")
    print("================================")
    print("Microphone index:", MIC_INDEX)

    try:

        mic = sr.Microphone(device_index=MIC_INDEX)

    except Exception as error:

        print("Could not access microphone:", error)

        speak("Microphone is not available.")

        try:

            manual_name = input("Enter visitor name manually: ").strip()

        except (KeyboardInterrupt, EOFError):

            return None

        return manual_name if manual_name else None

    for attempt in range(1, 4):

        print()
        print(f"Name attempt {attempt}/3")

        speak("Please enter your name.")

        try:

            with mic as source:

                print("Microphone opened successfully.")
                print("Calibrating for ambient noise...")

                recognizer.adjust_for_ambient_noise(source, duration=1)

                print("Energy threshold:", recognizer.energy_threshold)

                print()
                print("================================")
                print("SPEAK NOW")
                print("================================")

                audio = recognizer.listen(source, timeout=7, phrase_time_limit=5)

                print()
                print("Audio captured successfully.")

        except KeyboardInterrupt:

            print()
            print("Voice input stopped by user.")

            return None

        except sr.WaitTimeoutError:

            print("No speech detected.")

            if attempt < 3:
                speak("I did not hear you. Please say your name again.")

            continue

        except OSError as error:

            print("Microphone device error:", error)

            if attempt < 3:
                speak("There is a microphone problem. Please try again.")

            continue

        except Exception as error:

            print("Microphone error:", error)

            if attempt < 3:
                speak("Please try again.")

            continue

        except BaseException as error:

            print("UNEXPECTED MIC ERROR:", type(error).__name__, error)

            if attempt < 3:
                speak("Please try again.")

            continue

        print("Sending audio to Google...")

        try:

            result = recognizer.recognize_google(
                audio,
                language="en-IN",
                show_all=True
            )

            name = None

            if result and "alternative" in result:

                candidates = [
                    alt["transcript"].strip()
                    for alt in result["alternative"]
                    if alt.get("transcript")
                ]

                print("Google candidates:", candidates)

                known_names = get_known_names(face_database)

                matched_name = pick_best_name_match(candidates, known_names)

                if matched_name:
                    name = matched_name
                elif candidates:
                    name = candidates[0]
                else:
                    name = None

            print()
            print("Google recognized:")
            print(name)

        except sr.UnknownValueError:

            print()
            print("Google could not understand the speech.")

            if attempt < 3:
                speak("I could not understand your name. Please say it clearly again.")

            continue

        except sr.RequestError as error:

            print()
            print("Google Speech Recognition service error:")
            print(error)

            speak("The speech recognition service is unavailable.")

            break

        except KeyboardInterrupt:

            print()
            print("Stopped by user.")

            return None

        if not name:

            print()
            print("Google returned no usable transcript.")

            if attempt < 3:
                speak("I could not understand your name. Please say it clearly again.")

            continue

        lower_name = name.lower()

        prefixes = ["my name is ", "i am ", "i'm ", "this is ", "my name's "]

        for prefix in prefixes:

            if lower_name.startswith(prefix):
                name = name[len(prefix):].strip()
                break

        if name:

            print()
            print("================================")
            print("VISITOR NAME RECEIVED")
            print("Name:", name)
            print("================================")

            return name

    print()
    print("================================")
    print("VOICE RECOGNITION FAILED")
    print("================================")

    speak("I could not recognize your name. Please enter your name manually.")

    try:

        manual_name = input("Enter visitor name manually: ").strip()

    except (KeyboardInterrupt, EOFError):

        return None

    if manual_name:
        return manual_name

    return None


# ============================================================
# GUEST PASSWORD AUTHENTICATION
# ============================================================

def authenticate_guest(visitor_name, guest_folder, arduino):

    print()
    print("================================")
    print("GUEST AUTHENTICATION")
    print("================================")

    password = generate_password()
    created_time = time.time()

    expires_time = datetime.now() + timedelta(seconds=PASSWORD_VALID_SECONDS)

    print()
    print("Temporary Password:")
    print(password)

    print()
    print(f"Valid for {PASSWORD_VALID_SECONDS} seconds.")
    print("Maximum attempts:", MAX_PASSWORD_ATTEMPTS)

    session_id = save_guest_session(
        visitor_name,
        password,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        expires_time.strftime("%Y-%m-%d %H:%M:%S"),
        guest_folder
    )

    send_arduino(arduino, "PASSWORD")

    time.sleep(0.5)

    speak(f"Hello {visitor_name}. Please enter the temporary password.")

    attempts = 0

    while attempts < MAX_PASSWORD_ATTEMPTS:

        if time.time() - created_time > PASSWORD_VALID_SECONDS:

            print()
            print("================================")
            print("PASSWORD EXPIRED")
            print("ACCESS DENIED")
            print("================================")

            update_guest_session(session_id, attempts, "EXPIRED")

            save_access_log(
                visitor_name,
                "Guest",
                "Face Recognition + Temporary Password",
                "DENIED",
                "Temporary password expired",
                guest_folder
            )

            send_arduino(arduino, "DENIED")

            speak("The password has expired. Access denied.")

            return False

        try:

            entered_password = input("Enter guest password: ").strip()

        except (KeyboardInterrupt, EOFError):

            print()

            send_arduino(arduino, "DENIED")

            return False

        attempts += 1

        if entered_password == password:

            print()
            print("================================")
            print("ACCESS GRANTED")
            print("================================")
            print("Guest:", visitor_name)

            update_guest_session(session_id, attempts, "GRANTED")

            update_guest_visit(visitor_name)

            save_visitor(visitor_name)

            save_access_log(
                visitor_name,
                "Guest",
                "Face Recognition + Temporary Password",
                "GRANTED",
                "Correct temporary password",
                guest_folder
            )

            send_arduino(arduino, f"WELCOME:{visitor_name}")

            time.sleep(1)

            send_arduino(arduino, "UNLOCK")

            speak(f"Welcome {visitor_name}. Authentication successful.")

            return True

        remaining = MAX_PASSWORD_ATTEMPTS - attempts

        print()
        print("WRONG PASSWORD")

        if remaining > 0:

            print("Attempts remaining:", remaining)

            speak("Incorrect password. Please try again.")

        else:

            print()
            print("================================")
            print("ACCESS DENIED")
            print("SYSTEM LOCKED")
            print("================================")

            update_guest_session(session_id, attempts, "DENIED")

            save_access_log(
                visitor_name,
                "Guest",
                "Face Recognition + Temporary Password",
                "DENIED",
                "Maximum password attempts exceeded",
                guest_folder
            )

            send_arduino(arduino, "DENIED")

            speak("Access denied. The system is locked.")

            print(f"Lockout: {GUEST_LOCKOUT_SECONDS} seconds")

            time.sleep(GUEST_LOCKOUT_SECONDS)

            return False

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("================================")
    print("AI SMART LOCK SYSTEM (DEEPFACE)")
    print("================================")

    initialize_database()

    print("Database:", DATABASE_PATH)
    print("Security records:", SECURITY_FOLDER)
    print("Temporary guests:", GUEST_FOLDER)

    print()
    print("Connecting to Arduino...")

    arduino = None

    try:

        arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)

        time.sleep(2)

        print("Arduino connected!")

    except Exception as error:

        print("Arduino connection failed:")
        print(error)

        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():

        print("Haar Cascade could not load.")

        arduino.close()

        return

    print("Haar Cascade loaded!")

    # --------------------------------------------------------
    # FACE DATABASE (owner + guest embeddings)
    # --------------------------------------------------------

    print()
    print("Loading face database...")

    face_database = load_face_database()

    face_database = ensure_owner_database(face_database, face_cascade)

    face_database = ensure_guest_database_migrated(face_database, face_cascade)

    if not face_database.get("owner"):

        print()
        print("FATAL: No owner face data available.")
        print(f"Add photos to: {OWNER_PHOTOS_FOLDER}\\<YourName>\\")

        arduino.close()

        return

    print(
        "Owner(s):",
        list(face_database.get("owner", {}).keys())
    )

    print(
        "Guest(s):",
        list(face_database.get("guests", {}).keys())
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():

        print("Camera could not start.")

        arduino.close()

        return

    print("Camera started!")

    print()
    print("================================")
    print("SYSTEM READY")
    print("================================")
    print("Face model:", FACE_MODEL)
    print("Owner threshold:", OWNER_THRESHOLD)
    print("Guest threshold:", GUEST_THRESHOLD)
    print("Required owner matches:", REQUIRED_MATCHES)
    print("Required guest matches:", GUEST_REQUIRED_MATCHES)
    print("Unknown required matches:", UNKNOWN_REQUIRED_MATCHES)
    print("Microphone:", MIC_INDEX)
    print()
    print("Show a face to the camera.")
    print("Press ESC to stop.")
    print("================================")

    known_count = 0
    unknown_count = 0
    guest_count_matches = 0

    known_streak_name = None
    guest_streak_name = None

    unknown_triggered = False

    frame_counter = 0

    # Cache the last recognition result so frames we skip
    # (PROCESS_EVERY_N_FRAMES) still draw a label/box.
    last_result_type = "UNKNOWN"
    last_result_name = "Unknown"
    last_result_distance = None

    try:

        while True:

            ret, frame = camera.read()

            if not ret:

                print("Camera frame error.")

                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) == 0:

                known_count = 0
                unknown_count = 0
                guest_count_matches = 0

                known_streak_name = None
                guest_streak_name = None

                unknown_triggered = False

            frame_counter += 1

            run_recognition = (frame_counter % PROCESS_EVERY_N_FRAMES == 0)

            for (x, y, w, h) in faces:

                # Pad the crop slightly - deep models perform
                # better with a bit of context around the face
                # than a razor-tight Haar box.
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)

                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(frame.shape[1], x + w + pad_x)
                crop_y2 = min(frame.shape[0], y + h + pad_y)

                face_color_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if face_color_crop.size == 0:
                    continue

                if run_recognition:

                    result_type, result_name, result_distance = recognize_face(
                        face_color_crop,
                        face_database
                    )

                    last_result_type = result_type
                    last_result_name = result_name
                    last_result_distance = result_distance

                else:

                    result_type = last_result_type
                    result_name = last_result_name
                    result_distance = last_result_distance

                # ---------------- STREAK LOGIC ----------------

                if result_type == "KNOWN":

                    if known_streak_name == result_name:
                        known_count += 1
                    else:
                        known_streak_name = result_name
                        known_count = 1

                    guest_count_matches = 0
                    guest_streak_name = None
                    unknown_count = 0
                    unknown_triggered = False

                elif result_type == "GUEST":

                    if guest_streak_name == result_name:
                        guest_count_matches += 1
                    else:
                        guest_streak_name = result_name
                        guest_count_matches = 1

                    known_count = 0
                    known_streak_name = None
                    unknown_count = 0
                    unknown_triggered = False

                else:

                    unknown_count += 1

                    known_count = 0
                    guest_count_matches = 0

                    known_streak_name = None
                    guest_streak_name = None

                # ---------------- PRINT ----------------

                distance_text = (
                    f"{result_distance:.3f}"
                    if result_distance is not None
                    else "N/A"
                )

                print(
                    f"Distance: {distance_text} | "
                    f"Result: {result_type} | "
                    f"Name: {result_name} | "
                    f"Owner Count: {known_count} | "
                    f"Guest Count: {guest_count_matches} | "
                    f"Unknown Count: {unknown_count}"
                )

                # ---------------- CAMERA LABEL ----------------

                if result_type == "KNOWN":

                    label = f"{result_name} {distance_text}"
                    box_color = (0, 255, 0)

                elif result_type == "GUEST":

                    label = f"Guest: {result_name} {distance_text}"
                    box_color = (255, 255, 0)

                else:

                    label = f"Unknown {distance_text}"
                    box_color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

                # =================================================
                # OWNER VERIFIED
                # =================================================

                if result_type == "KNOWN" and known_count >= REQUIRED_MATCHES:

                    print()
                    print("================================")
                    print("KNOWN FACE VERIFIED")
                    print("================================")
                    print("Name:", result_name)
                    print("Distance:", distance_text)

                    save_visitor(result_name)

                    save_access_log(
                        result_name,
                        "Known",
                        "Face Recognition",
                        "GRANTED",
                        "Known face verified"
                    )

                    send_arduino(arduino, f"WELCOME:{result_name}")

                    time.sleep(1)

                    send_arduino(arduino, "UNLOCK")

                    speak(f"Hello {result_name}. Authentication successful.")

                    print("Access granted.")

                    return

                # =================================================
                # GUEST VERIFIED
                # =================================================

                if (
                    result_type == "GUEST"
                    and guest_count_matches >= GUEST_REQUIRED_MATCHES
                ):

                    verified_guest_name = result_name

                    print()
                    print("================================")
                    print("GUEST FACE VERIFIED")
                    print("================================")
                    print("Guest:", verified_guest_name)

                    profile = get_guest_profile(verified_guest_name)

                    if profile:

                        guest_folder = profile[3]

                    else:

                        guest_folder = create_guest_folder(verified_guest_name)

                        save_guest_profile(verified_guest_name, guest_folder)

                    save_access_log(
                        verified_guest_name,
                        "Guest",
                        "Face Recognition",
                        "PENDING",
                        "Guest face verified",
                        guest_folder
                    )

                    camera.release()
                    cv2.destroyAllWindows()

                    authenticate_guest(verified_guest_name, guest_folder, arduino)

                    return

                # =================================================
                # UNKNOWN VERIFIED
                # =================================================

                if (
                    result_type == "UNKNOWN"
                    and unknown_count >= UNKNOWN_REQUIRED_MATCHES
                    and not unknown_triggered
                ):

                    unknown_triggered = True

                    print()
                    print("================================")
                    print("UNKNOWN FACE CONFIRMED")
                    print("================================")

                    photo_files = capture_unknown_faces(camera)

                    camera.release()
                    cv2.destroyAllWindows()

                    send_arduino(arduino, "ASK_NAME")

                    visitor_name = ask_visitor_name(face_database)

                    if not visitor_name:

                        print("Visitor name not received.")

                        save_access_log(
                            "Unknown",
                            "Unknown",
                            "Voice Recognition",
                            "DENIED",
                            "Visitor name not received",
                            SECURITY_FOLDER
                        )

                        send_arduino(arduino, "DENIED")

                        return

                    visitor_name = visitor_name.strip()

                    existing_guest = get_guest_profile(visitor_name)

                    if existing_guest:

                        print()
                        print("Guest profile already exists.")
                        print("The current face was not recognized as that guest.")

                        speak(
                            f"A guest account for {visitor_name} already exists, "
                            "but this face was not recognized. Access denied."
                        )

                        send_arduino(arduino, "DENIED")

                        save_access_log(
                            visitor_name,
                            "Unknown",
                            "Voice Identification",
                            "DENIED",
                            "Existing guest name but face mismatch",
                            SECURITY_FOLDER
                        )

                        return

                    # ---------------- NEW GUEST ----------------

                    guest_folder = create_guest_folder(visitor_name)

                    print()
                    print("Guest folder created:")
                    print(guest_folder)

                    copied_files = copy_photos_to_guest_folder(
                        photo_files,
                        guest_folder
                    )

                    print("Photos copied:", len(copied_files))

                    save_visitor(visitor_name)

                    print()
                    print("Adding guest to face database...")

                    face_database = add_guest_encodings(
                        visitor_name,
                        guest_folder,
                        face_database,
                        face_cascade
                    )

                    save_guest_profile(visitor_name, guest_folder)

                    save_access_log(
                        visitor_name,
                        "Unknown",
                        "Voice Identification",
                        "PENDING",
                        "New guest registered",
                        guest_folder
                    )

                    authenticate_guest(visitor_name, guest_folder, arduino)

                    return

            cv2.imshow("AI Smart Lock", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:

                print()
                print("System stopped by user.")

                break

    except KeyboardInterrupt:

        print()
        print("System stopped by user (Ctrl+C).")

    finally:

        try:
            camera.release()
        except Exception:
            pass

        cv2.destroyAllWindows()

        try:
            if arduino is not None:
                arduino.close()
        except Exception:
            pass

        print()
        print("================================")
        print("AI SMART LOCK STOPPED")
        print("================================")


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Program terminated by user.")
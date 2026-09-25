from flask import Flask, request, jsonify, send_file
import manual_auth
import cv2
import numpy as np
import os
import tempfile
import time
from datetime import datetime
from deepface import DeepFace
import guest_auth

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"C:\Users\gauri\python"
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
OWNER_FOLDER = os.path.join(BASE_DIR, "owner_photos", "Gauri")
GUEST_FOLDER = os.path.join(BASE_DIR, "guest_database")
DASHBOARD_IMAGE = os.path.join(BASE_DIR, "dashboard_latest.jpg")

OWNER_NAME = "Gauri"

# ============================================================
# MODEL CONFIG
# ============================================================
MODEL_NAME = "VGG-Face"
AUTH_THRESHOLD = 0.58
DETECTOR_BACKEND = "skip"
ENFORCE_DETECTION = False
ALIGN = False

# ============================================================
# STATE
# ============================================================
dashboard_state = {
    "person": "None",
    "role": "UNKNOWN",
    "distance": None,
    "threshold": AUTH_THRESHOLD,
    "authenticated": False,
    "access": "DENIED",
    "timestamp": None,
    "guest_password": None,
    "guest_password_valid_seconds": None,
    "guest_password_attempts": None,
}
activity_log = []

owner_embeddings_cache = []
guest_embeddings_cache = {}

# ============================================================
# HAAR CASCADE
# ============================================================
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("[ERROR] Haar cascade could not be loaded.")
else:
    print("[MEC] Haar cascade loaded successfully.")


def get_face_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.03, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return image[y:y + h, x:x + w]


def get_embedding(image):
    try:
        results = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION,
            align=ALIGN,
        )
        if not results:
            return None
        return np.array(results[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        return None


def cosine_distance(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(v1, v2) / denom)


# ============================================================
# LOAD FACE DATABASE
# FIX: reference photos now go through the SAME Haar crop as the
# live image. If no face is found in a reference photo, the full
# image is used (so already-cropped photos still work).
# ============================================================
def embedding_from_reference(image):
    crop = get_face_crop(image)
    if crop is not None:
        return get_embedding(crop), True
    return get_embedding(image), False


def load_embeddings_from_folder(folder):
    embeddings = []
    if not os.path.exists(folder):
        print(f"[WARNING] Folder not found: {folder}")
        return embeddings

    cropped = 0
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            image = cv2.imread(path)
            if image is None:
                continue
            emb, was_cropped = embedding_from_reference(image)
            if emb is not None:
                embeddings.append(emb)
                cropped += int(was_cropped)
        except Exception as e:
            print(f"[WARNING] Failed to process {filename}: {e}")

    print(f"[MEC]   {folder}: {len(embeddings)} embedding(s), {cropped} face-cropped")
    return embeddings


def build_owner_cache():
    global owner_embeddings_cache
    print("[MEC] Building owner embedding cache...")
    owner_embeddings_cache = load_embeddings_from_folder(OWNER_FOLDER)
    print(f"[MEC] Owner cache ready: {len(owner_embeddings_cache)} embedding(s).")


def build_guest_cache():
    global guest_embeddings_cache
    print("[MEC] Building guest embedding cache...")
    guest_embeddings_cache = {}

    if not os.path.exists(GUEST_FOLDER):
        print("[MEC] Guest cache ready: 0 guest(s).")
        return

    for guest_name in os.listdir(GUEST_FOLDER):
        guest_path = os.path.join(GUEST_FOLDER, guest_name)
        if not os.path.isdir(guest_path):
            continue
        # A guest folder named like the owner would poison matching.
        if guest_name.strip().lower() == OWNER_NAME.lower():
            print(f"[WARNING] Ignoring guest folder '{guest_name}' (same as owner).")
            continue
        guest_embeddings_cache[guest_name] = load_embeddings_from_folder(guest_path)

    total = sum(len(v) for v in guest_embeddings_cache.values())
    print(f"[MEC] Guest cache ready: {len(guest_embeddings_cache)} guest(s), {total} embedding(s).")


def add_guest_embedding_to_cache(guest_name, embedding):
    if embedding is None:
        return
    guest_embeddings_cache.setdefault(guest_name, []).append(embedding)
    print(f"[MEC] Cache updated: '{guest_name}' now has "
          f"{len(guest_embeddings_cache[guest_name])} embedding(s).")


def warm_up_model():
    print(f"[MEC] Warming up DeepFace model ({MODEL_NAME})...")
    try:
        get_embedding(np.zeros((100, 100, 3), dtype=np.uint8))
        print("[MEC] Model warm-up complete.")
    except Exception as e:
        print(f"[WARNING] Model warm-up failed: {e}")


# ============================================================
# MATCHING (closest distance only, NO threshold here)
# ============================================================
def best_owner_match(embedding):
    if not owner_embeddings_cache:
        return None, None
    best = min(cosine_distance(embedding, e) for e in owner_embeddings_cache)
    return OWNER_NAME, best


def best_guest_match(embedding):
    best_name, best_distance = None, float("inf")
    for name, embs in guest_embeddings_cache.items():
        for e in embs:
            d = cosine_distance(embedding, e)
            if d < best_distance:
                best_distance, best_name = d, name
    if best_name is None:
        return None, None
    return best_name, best_distance


# ============================================================
# DASHBOARD HELPERS
# ============================================================
def clear_dashboard_password():
    dashboard_state["guest_password"] = None
    dashboard_state["guest_password_valid_seconds"] = None
    dashboard_state["guest_password_attempts"] = None


def update_dashboard(person, role, distance, authenticated, access):
    dashboard_state.update({
        "person": person,
        "role": role,
        "distance": distance,
        "threshold": AUTH_THRESHOLD,
        "authenticated": authenticated,
        "access": access,
        "timestamp": datetime.now().isoformat(),
    })
    activity_log.insert(0, {
        "person": person,
        "role": role,
        "distance": distance,
        "authenticated": authenticated,
        "access": access,
        "timestamp": dashboard_state["timestamp"],
    })
    if len(activity_log) > 100:
        del activity_log[100:]


def unknown_response(distance=None, access="DENIED", error=None):
    body = {
        "authenticated": False,
        "name": "UNKNOWN",
        "role": "UNKNOWN",
        "distance": distance,
        "threshold": AUTH_THRESHOLD,
        "access": access,
    }
    if error:
        body["error"] = error
    return jsonify(body)


# ============================================================
# FACE AUTHENTICATION
# ============================================================
@app.route("/authenticate", methods=["POST"])
def authenticate():
    temp_path = None
    try:
        if "image" not in request.files:
            return jsonify({"authenticated": False, "error": "No image provided"}), 400

        uploaded = request.files["image"]
        if uploaded.filename == "":
            return jsonify({"authenticated": False, "error": "Empty filename"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            uploaded.save(tmp.name)
            temp_path = tmp.name

        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({"authenticated": False, "error": "Could not read image"}), 400

        try:
            cv2.imwrite(DASHBOARD_IMAGE, image)
        except Exception as e:
            print(f"[WARNING] Dashboard image save failed: {e}")

        # ---- face crop ----
        face_crop = get_face_crop(image)
        if face_crop is None:
            print("[AUTH] No face detected in captured image.")
            update_dashboard("UNKNOWN", "UNKNOWN", None, False, "DENIED")
            return unknown_response(error="No face detected")

        # ---- embedding ----
        embedding = get_embedding(face_crop)
        if embedding is None:
            update_dashboard("UNKNOWN", "UNKNOWN", None, False, "DENIED")
            return unknown_response(error="Embedding generation failed")

        # ---- closest owner / closest guest ----
        owner_name, owner_distance = best_owner_match(embedding)
        guest_name, guest_distance = best_guest_match(embedding)

        # DEBUG: this line tells you exactly why a role was chosen
        print(
            f"[DEBUG] owner={owner_name} "
            f"{'None' if owner_distance is None else f'{owner_distance:.4f}'} | "
            f"guest={guest_name} "
            f"{'None' if guest_distance is None else f'{guest_distance:.4f}'} | "
            f"threshold={AUTH_THRESHOLD}"
        )

        candidates = []
        if owner_name is not None and owner_distance is not None:
            candidates.append((owner_name, "OWNER", owner_distance))
        if guest_name is not None and guest_distance is not None:
            candidates.append((guest_name, "GUEST", guest_distance))

        if not candidates:
            update_dashboard("UNKNOWN", "UNKNOWN", None, False, "DENIED")
            return unknown_response(error="No reference embeddings loaded")

        # ---- global minimum, THEN threshold once ----
        closest_name, closest_role, closest_distance = min(candidates, key=lambda c: c[2])
        distance = round(float(closest_distance), 4)
        authenticated = distance <= AUTH_THRESHOLD

        if authenticated:
            update_dashboard(closest_name, closest_role, distance, True, "GRANTED")
            print(f"[AUTH] {closest_role}: {closest_name} | distance={distance:.4f}")
            return jsonify({
                "authenticated": True,
                "name": closest_name,
                "role": closest_role,
                "distance": distance,
                "threshold": AUTH_THRESHOLD,
                "access": "GRANTED",
            })

        update_dashboard("UNKNOWN", "UNKNOWN", distance, False, "PASSWORD_REQUIRED")
        print(f"[AUTH] UNKNOWN | closest={closest_name} ({closest_role}) | "
              f"distance={distance:.4f}")
        return unknown_response(distance=distance, access="PASSWORD_REQUIRED")

    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return jsonify({"authenticated": False, "error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# GUEST PASSWORD GENERATION
# ============================================================
@app.route("/guest/generate_password", methods=["POST"])
def generate_guest_password():
    try:
        data = request.get_json(silent=True) or {}
        name = str(data.get("name", "")).strip()
        if not name:
            return jsonify({"success": False, "error": "Guest name is required"}), 400

        print(f"[MEC] Requesting password for '{name}'...")
        password = guest_auth.generate_password(name)
        valid_seconds = guest_auth.PASSWORD_VALID_SECONDS
        max_attempts = guest_auth.MAX_ATTEMPTS

        dashboard_state["guest_password"] = password
        dashboard_state["guest_password_valid_seconds"] = valid_seconds
        dashboard_state["guest_password_attempts"] = max_attempts

        print(f"[GUEST] Generated password for '{name}': {password}")
        return jsonify({
            "success": True,
            "name": name,
            "password": password,
            "valid_seconds": valid_seconds,
            "max_attempts": max_attempts,
        })
    except Exception as e:
        print(f"[ERROR] Guest password generation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# GUEST PASSWORD VERIFICATION
# ============================================================
@app.route("/guest/verify_password", methods=["POST"])
def verify_guest_password():
    try:
        data = request.get_json(silent=True) or {}
        name = str(data.get("name", "")).strip()
        password = str(data.get("password", "")).strip()

        if not name:
            return jsonify({"success": False, "verified": False, "reason": "invalid_name"}), 400
        if not password:
            return jsonify({"success": False, "verified": False, "reason": "empty_password"}), 400

        ok, reason = guest_auth.verify_password(name, password)

        if ok:
            print(f"[GUEST] Password verified for '{name}'")
            clear_dashboard_password()
            update_dashboard(name, "GUEST", None, True, "GRANTED")
            return jsonify({"success": True, "verified": True, "reason": reason, "name": name})

        print(f"[GUEST] Password verification failed for '{name}': {reason}")
        return jsonify({"success": True, "verified": False, "reason": reason, "name": name})

    except Exception as e:
        print(f"[ERROR] Guest password verification failed: {e}")
        return jsonify({"success": False, "verified": False,
                        "reason": "server_error", "error": str(e)}), 500


# ============================================================
# GUEST REGISTRATION
# FIX: owner name is reserved; filenames never collide.
# ============================================================
@app.route("/guest/register", methods=["POST"])
def register_guest():
    try:
        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "error": "Guest name is required"}), 400

        if name.lower() == OWNER_NAME.lower():
            print(f"[GUEST] Registration blocked: '{name}' is a reserved owner name.")
            return jsonify({"success": False, "error": "Reserved name"}), 400

        # keep folder names filesystem-safe
        if any(c in name for c in '\\/:*?"<>|'):
            return jsonify({"success": False, "error": "Invalid characters in name"}), 400

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400

        guest_path = os.path.join(GUEST_FOLDER, name)
        os.makedirs(guest_path, exist_ok=True)

        filename = f"{int(time.time() * 1000)}.jpg"
        save_path = os.path.join(guest_path, filename)
        request.files["image"].save(save_path)
        print(f"[GUEST] Registered image for '{name}': {save_path}")

        try:
            img = cv2.imread(save_path)
            if img is not None:
                emb, _ = embedding_from_reference(img)
                add_guest_embedding_to_cache(name, emb)
        except Exception as e:
            print(f"[WARNING] Failed to update guest cache for '{name}': {e}")

        return jsonify({"success": True, "name": name, "filename": filename})

    except Exception as e:
        print(f"[ERROR] Guest registration failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# MANUAL AUTHENTICATION
# ============================================================
@app.route("/manual/verify", methods=["POST"])
def manual_verify():
    try:
        data = request.get_json(silent=True) or {}
        name = str(data.get("name", "")).strip()
        password = str(data.get("password", "")).strip()

        if not name or not password:
            return jsonify({"success": False, "authenticated": False,
                            "error": "Name and password required"}), 400

        if manual_auth.verify_credentials(name, password):
            update_dashboard(name, "MANUAL", None, True, "GRANTED")
            return jsonify({"success": True, "authenticated": True, "name": name})

        update_dashboard(name, "MANUAL", None, False, "DENIED")
        return jsonify({"success": True, "authenticated": False, "name": name})

    except Exception as e:
        print(f"[ERROR] Manual authentication failed: {e}")
        return jsonify({"success": False, "authenticated": False, "error": str(e)}), 500


# ============================================================
# DASHBOARD APIs
# ============================================================
@app.route("/dashboard/state")
def dashboard_state_api():
    return jsonify(dashboard_state)


@app.route("/dashboard/activity")
def dashboard_activity():
    return jsonify(activity_log)


@app.route("/dashboard/latest_image")
def dashboard_latest_image():
    if not os.path.exists(DASHBOARD_IMAGE):
        return jsonify({"error": "No image available"}), 404
    return send_file(DASHBOARD_IMAGE, mimetype="image/jpeg")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Lock MEC Dashboard</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Arial,sans-serif;background:#0f172a;color:#e5e7eb}
.container{width:94%;max-width:1400px;margin:auto;padding:25px 0 40px}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:25px}
.header h1{margin:0;font-size:28px}.header p{margin:6px 0 0;color:#94a3b8}
.status{display:flex;align-items:center;gap:9px;background:#111827;border:1px solid #334155;padding:10px 16px;border-radius:10px}
.dot{width:11px;height:11px;background:#22c55e;border-radius:50%;box-shadow:0 0 10px #22c55e}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:20px}
.card,.panel{background:#111827;border:1px solid #263244;border-radius:14px;padding:20px}
.panel{margin-bottom:20px}.panel h2{margin:0 0 16px;font-size:19px}
.ct{color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}
.big{font-size:26px;font-weight:bold}.sub{color:#94a3b8;margin-top:6px;font-size:13px}
.main{display:grid;grid-template-columns:1.35fr 1fr;gap:20px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:15px}
.metric{background:#0f172a;border:1px solid #263244;border-radius:10px;padding:15px}
.ml{color:#94a3b8;font-size:12px;margin-bottom:7px}.mv{font-size:19px;font-weight:bold}
.granted{color:#22c55e}.pending{color:#f59e0b}.denied{color:#ef4444}
.cam{width:100%;height:390px;background:#020617;border-radius:12px;overflow:hidden;border:1px solid #263244;display:flex;align-items:center;justify-content:center}
.cam img{width:100%;height:100%;object-fit:contain}
.pw{font-size:30px;font-weight:bold;letter-spacing:5px;margin:10px 0}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:#94a3b8;font-size:12px;text-transform:uppercase;padding:12px;border-bottom:1px solid #263244}
td{padding:12px;border-bottom:1px solid #1e293b;font-size:14px}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,1fr)}.main{grid-template-columns:1fr}}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="container">

<div class="header">
  <div><h1>&#128272; Smart Lock MEC Dashboard</h1>
  <p>Intelligent Vision-Based Multi-Factor Physical Access Control</p></div>
  <div class="status"><div class="dot"></div><span>MEC SERVER ONLINE</span></div>
</div>

<div class="grid">
  <div class="card"><div class="ct">Current Person</div><div class="big" id="person">None</div><div class="sub" id="role">Role: UNKNOWN</div></div>
  <div class="card"><div class="ct">Authentication</div><div class="big" id="auth">--</div><div class="sub" id="ts">No activity yet</div></div>
  <div class="card"><div class="ct">Face Distance</div><div class="big" id="dist">--</div><div class="sub" id="thr">Threshold: 0.58</div></div>
  <div class="card"><div class="ct">Access</div><div class="big" id="access">--</div><div class="sub">Door authorization status</div></div>
</div>

<div class="main">
  <div>
    <div class="panel"><h2>Guest Password</h2>
      <div class="pw" id="pw">------</div>
      <div class="sub">Valid for: <span id="pwv">--</span> seconds</div>
      <div class="sub">Maximum attempts: <span id="pwa">--</span></div>
    </div>
    <div class="panel"><h2>Recent Authentication Activity</h2>
      <div style="overflow-x:auto"><table>
        <thead><tr><th>Time</th><th>Person</th><th>Role</th><th>Distance</th><th>Access</th></tr></thead>
        <tbody id="act"><tr><td colspan="5" style="text-align:center;color:#64748b">Waiting for activity...</td></tr></tbody>
      </table></div>
    </div>
  </div>
  <div>
    <div class="panel"><h2>Latest Camera Image</h2>
      <div class="cam"><img id="cam" src="/dashboard/latest_image" alt="Latest frame"
        onerror="this.style.display='none'" onload="this.style.display='block'"></div>
    </div>
    <div class="panel"><h2>System Information</h2>
      <div class="metrics">
        <div class="metric"><div class="ml">Model</div><div class="mv" style="font-size:16px">VGG-Face</div></div>
        <div class="metric"><div class="ml">Detector</div><div class="mv" style="font-size:16px">Haar Cascade</div></div>
        <div class="metric"><div class="ml">Matching</div><div class="mv" style="font-size:16px">Cosine Distance</div></div>
        <div class="metric"><div class="ml">Refresh</div><div class="mv" style="font-size:16px">1 second</div></div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
const $ = id => document.getElementById(id);
const esc = v => String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const fmtT = t => t ? new Date(t).toLocaleString() : "No activity yet";
const fmtD = d => (d !== null && d !== undefined) ? Number(d).toFixed(4) : "--";
const cls = a => a === "GRANTED" ? "granted" : a === "PASSWORD_REQUIRED" ? "pending" : "denied";

async function updateState(){
  try{
    const d = await (await fetch("/dashboard/state")).json();
    $("person").textContent = d.person || "None";
    $("role").textContent = "Role: " + (d.role || "UNKNOWN");
    $("auth").textContent = d.authenticated ? "SUCCESS" : "FAILED";
    $("ts").textContent = fmtT(d.timestamp);
    $("dist").textContent = fmtD(d.distance);
    $("thr").textContent = "Threshold: " + Number(d.threshold || 0.58).toFixed(2);
    $("access").textContent = d.access || "--";
    $("access").className = "big " + cls(d.access);
    $("pw").textContent = d.guest_password || "------";
    $("pwv").textContent = d.guest_password_valid_seconds ?? "--";
    $("pwa").textContent = d.guest_password_attempts ?? "--";
  }catch(e){console.error(e)}
}

async function updateActivity(){
  try{
    const data = await (await fetch("/dashboard/activity")).json();
    if(!data.length){
      $("act").innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b">No activity yet.</td></tr>';
      return;
    }
    $("act").innerHTML = data.slice(0,20).map(i => `<tr>
      <td>${fmtT(i.timestamp)}</td>
      <td><strong>${esc(i.person || "UNKNOWN")}</strong></td>
      <td>${esc(i.role || "UNKNOWN")}</td>
      <td>${fmtD(i.distance)}</td>
      <td class="${cls(i.access)}">${esc(i.access || "--")}</td></tr>`).join("");
  }catch(e){console.error(e)}
}

updateState(); updateActivity();
setInterval(updateState, 1000);
setInterval(updateActivity, 1000);
setInterval(() => { $("cam").src = "/dashboard/latest_image?t=" + Date.now(); }, 1500);
</script></body></html>"""


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


# ============================================================
# START
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SMART LOCK MEC SERVER")
    print("=" * 60)
    print(f"MODEL: {MODEL_NAME} | THRESHOLD: {AUTH_THRESHOLD}")
    print(f"OWNER FOLDER: {OWNER_FOLDER}")
    print(f"GUEST FOLDER: {GUEST_FOLDER}")
    print("SERVER: http://0.0.0.0:5000")
    print("=" * 60)

    warm_up_model()
    build_owner_cache()
    build_guest_cache()

    app.run(host="0.0.0.0", port=5000, debug=False)
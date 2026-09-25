"""
pi_client.py
------------
Runs on the Raspberry Pi. 
1. Listens for AUTH_REQUEST from Arduino over USB serial, captures webcam photo,
   sends to MEC server for face authentication, and relays AUTH_SUCCESS/FAILED.
2. Updates Firebase Realtime Database with entry logs for the Flutter mobile app.
3. Polls Firebase to listen for remote emergency unlock requests from the app.
"""

import serial
import requests
import cv2
import time
import sys
import json

# ----------------------- CONFIG -----------------------

SERIAL_PORT = "/dev/ttyACM0"   # Arduino Uno serial port
BAUD_RATE = 9600               # Matches Arduino Serial.begin()

MEC_SERVER_URL = "http://51.0.0.227:5000/authenticate"

# Firebase Realtime Database
FIREBASE_URL = "https://smart-lock-hub-default-rtdb.firebaseio.com"

CAMERA_INDEX = 0               # Default webcam
CAPTURE_IMAGE_PATH = "/tmp/capture.jpg"

REQUEST_TIMEOUT = 10           # Seconds to wait for MEC server
SERIAL_READ_TIMEOUT = 1        # Non-blocking read timeout
FIREBASE_POLL_INTERVAL = 2.0   # Seconds between checking app unlock status

# -----------------------------------------------------


def connect_serial():
    """Open serial connection to Arduino, retrying if it fails."""
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_READ_TIMEOUT)
            time.sleep(2)  # Allow Arduino to reset after serial connect
            print(f"[SERIAL] Connected to Arduino on {SERIAL_PORT} @ {BAUD_RATE} baud")
            return ser
        except serial.SerialException as e:
            print(f"[SERIAL] Could not open {SERIAL_PORT}: {e}")
            print("[SERIAL] Retrying in 3s... (check cable / port name)")
            time.sleep(3)


def capture_photo():
    """Capture a single frame from the webcam and save to disk."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[CAMERA] ERROR: Could not open webcam.")
        return False

    # Force the camera mode that we proved works with v4l2-ctl
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(
        f"[CAMERA] Configured: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS)} FPS"
    )

    # Warm up camera
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[CAMERA] ERROR: Failed to capture frame.")
        return False

    # Debug information
    print(
        f"[CAMERA] Frame stats: "
        f"shape={frame.shape}, "
        f"min={frame.min()}, "
        f"max={frame.max()}, "
        f"mean={frame.mean():.2f}"
    )

    # Don't send a completely black image to MEC
    if frame.max() == 0:
        print("[CAMERA] ERROR: Captured frame is completely black.")
        return False

    if not cv2.imwrite(CAPTURE_IMAGE_PATH, frame):
        print("[CAMERA] ERROR: Failed to save image.")
        return False

    print(f"[CAMERA] Photo captured -> {CAPTURE_IMAGE_PATH}")
    return True

def send_to_mec_server():
    """POST the captured photo to the MEC server and return auth result & user."""
    try:
        with open(CAPTURE_IMAGE_PATH, "rb") as f:
            files = {"image": f}
            response = requests.post(MEC_SERVER_URL, files=files, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            print(f"[MEC] Server returned status {response.status_code}: {response.text}")
            return False, "Unknown"

        data = response.json()
        print(f"[MEC] Response: {data}")
        is_authenticated = bool(data.get("authenticated", False))
        user_name = data.get("user", "Authorized User")
        return is_authenticated, user_name

    except requests.exceptions.Timeout:
        print("[MEC] ERROR: Request timed out.")
        return False, "Unknown"
    except requests.exceptions.ConnectionError as e:
        print(f"[MEC] ERROR: Could not connect to MEC server: {e}")
        return False, "Unknown"
    except Exception as e:
        print(f"[MEC] ERROR: Unexpected error: {e}")
        return False, "Unknown"


def log_entry_to_firebase(visitor_name):
    """Update latest entry card on the Flutter mobile app."""
    try:
        current_time = time.strftime("%I:%M %p")
        payload = {
            "latest_entry": {
                "name": visitor_name,
                "timestamp": current_time
            }
        }
        requests.patch(
            f"{FIREBASE_URL}/.json",
            headers={"Connection": "close"},
            json=payload,
            timeout=3
        )
        print(f"[FIREBASE] Logged access for: {visitor_name} at {current_time}")
    except Exception as e:
        print(f"[FIREBASE] Failed to log entry: {e}")


def check_app_unlock_status():
    """Check if the mobile app triggered an EMERGENCY UNLOCK."""
    try:
        res = requests.get(
            f"{FIREBASE_URL}/door_status.json",
            headers={"Connection": "close"},
            timeout=2
        )
        if res.status_code == 200 and res.text != "null":
            return res.json()  # Returns "LOCKED" or "UNLOCKED"
    except Exception as e:
        pass
    return "LOCKED"


def send_result_to_arduino(ser, authenticated):
    """Send AUTH_SUCCESS or AUTH_FAILED back to Arduino over serial."""
    message = "AUTH_SUCCESS\n" if authenticated else "AUTH_FAILED\n"
    ser.write(message.encode("utf-8"))
    print(f"[SERIAL] Sent to Arduino: {message.strip()}")


def main():
    ser = connect_serial()
    print("[SYSTEM] Ready. Waiting for Arduino requests or Mobile App unlock...")

    last_firebase_poll = 0
    app_unlocked_active = False

    try:
        while True:
            current_time = time.time()

            # 1. Periodically check if the Flutter app pressed EMERGENCY UNLOCK
            if current_time - last_firebase_poll >= FIREBASE_POLL_INTERVAL:
                last_firebase_poll = current_time
                door_status = check_app_unlock_status()

                if door_status == "UNLOCKED" and not app_unlocked_active:
                    print("[APP] Emergency Unlock detected from mobile app! Triggering Arduino...")
                    send_result_to_arduino(ser, True)
                    log_entry_to_firebase("Remote App Unlock")
                    app_unlocked_active = True
                elif door_status == "LOCKED":
                    app_unlocked_active = False

            # 2. Check for Serial requests from Arduino (Keypad/Sensor/Button trigger)
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                print(f"[SERIAL] Received: {line}")

                if line == "AUTH_REQUEST":
                    print("[SYSTEM] AUTH_REQUEST received. Capturing image...")

                    if not capture_photo():
                        send_result_to_arduino(ser, False)
                        continue

                    authenticated, user_name = send_to_mec_server()
                    send_result_to_arduino(ser, authenticated)

                    if authenticated:
                        print(f"[SYSTEM] Authenticated: {user_name}. Opening lock.")
                        log_entry_to_firebase(user_name)
                    else:
                        print("[SYSTEM] Authentication failed. Access denied.")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down (Ctrl+C).")
    except serial.SerialException as e:
        print(f"[SERIAL] Connection lost: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
"""
pi_client.py
------------
Runs on the Raspberry Pi. Listens for AUTH_REQUEST from Arduino over
USB serial, captures a webcam photo, sends it to the MEC (Flask/DeepFace)
server on the Windows laptop for authentication, and relays the result
(AUTH_SUCCESS / AUTH_FAILED) back to Arduino for servo/OLED/buzzer control.

Pipeline:
  Arduino --AUTH_REQUEST--> Pi --webcam--> MEC server --DeepFace-->
  Pi --AUTH_SUCCESS/AUTH_FAILED--> Arduino
"""

import serial
import requests
import cv2
import time
import sys

# ----------------------- CONFIG (edit these) -----------------------

SERIAL_PORT = "/dev/ttyACM0"   # Try /dev/ttyACM0 first (Arduino Uno on
                                # Raspberry Pi OS). If not found, check
                                # with: ls /dev/tty*  or  dmesg | grep tty
BAUD_RATE = 9600                # MUST match Serial.begin() value in
                                # the Arduino sketch exactly

MEC_SERVER_URL = "http://51.0.0.227:5000/authenticate"

CAMERA_INDEX = 0                 # 0 = default webcam, change if using
                                  # a USB webcam that enumerates differently
CAPTURE_IMAGE_PATH = "/tmp/capture.jpg"

REQUEST_TIMEOUT = 10              # seconds to wait for MEC server response
SERIAL_READ_TIMEOUT = 1           # seconds, non-blocking-ish serial reads

# ---------------------------------------------------------------------


def connect_serial():
    """Open serial connection to Arduino, retrying if it fails."""
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_READ_TIMEOUT)
            time.sleep(2)  # allow Arduino to reset after serial connect
            print(f"[SERIAL] Connected to Arduino on {SERIAL_PORT} @ {BAUD_RATE} baud")
            return ser
        except serial.SerialException as e:
            print(f"[SERIAL] Could not open {SERIAL_PORT}: {e}")
            print("[SERIAL] Retrying in 3s... (check cable / port name)")
            time.sleep(3)


def capture_photo():
    """Capture a single frame from the webcam and save to disk."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[CAMERA] ERROR: Could not open webcam.")
        return False

    # Warm up the camera - first few frames are often dark/unfocused
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[CAMERA] ERROR: Failed to capture frame.")
        return False

    cv2.imwrite(CAPTURE_IMAGE_PATH, frame)
    print(f"[CAMERA] Photo captured -> {CAPTURE_IMAGE_PATH}")
    return True


def send_to_mec_server():
    """POST the captured photo to the MEC server and return auth result."""
    try:
        with open(CAPTURE_IMAGE_PATH, "rb") as f:
            files = {"image": f}
            response = requests.post(MEC_SERVER_URL, files=files, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            print(f"[MEC] Server returned status {response.status_code}: {response.text}")
            return False

        data = response.json()
        print(f"[MEC] Response: {data}")
        return bool(data.get("authenticated", False))

    except requests.exceptions.Timeout:
        print("[MEC] ERROR: Request timed out.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"[MEC] ERROR: Could not connect to MEC server: {e}")
        return False
    except Exception as e:
        print(f"[MEC] ERROR: Unexpected error: {e}")
        return False


def send_result_to_arduino(ser, authenticated):
    """Send AUTH_SUCCESS or AUTH_FAILED back to Arduino over serial."""
    message = "AUTH_SUCCESS\n" if authenticated else "AUTH_FAILED\n"
    ser.write(message.encode("utf-8"))
    print(f"[SERIAL] Sent to Arduino: {message.strip()}")


def main():
    ser = connect_serial()
    print("[SYSTEM] Ready. Waiting for AUTH_REQUEST from Arduino...")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not line:
                    continue

                print(f"[SERIAL] Received: {line}")

                if line == "AUTH_REQUEST":
                    print("[SYSTEM] AUTH_REQUEST received. Starting authentication flow...")

                    if not capture_photo():
                        send_result_to_arduino(ser, False)
                        continue

                    authenticated = send_to_mec_server()
                    send_result_to_arduino(ser, authenticated)

                    if authenticated:
                        print("[SYSTEM] Authenticated as Gauri. Servo should unlock.")
                    else:
                        print("[SYSTEM] Authentication failed. Access denied.")

            time.sleep(0.1)  # small delay to avoid busy-waiting the CPU

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
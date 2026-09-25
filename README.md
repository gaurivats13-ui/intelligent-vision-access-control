# Intelligent Vision Access Control

A smart physical access control system that combines face recognition, risk-aware authentication, Raspberry Pi edge processing, MEC-based AI authentication, and Arduino-controlled locking.

The basic idea is simple: instead of relying on only a password or only face recognition, the system uses multiple authentication steps depending on who is trying to access the system.

## How it works

Camera → Raspberry Pi → MEC Server → AI Authentication → Decision → Raspberry Pi → Arduino → Lock

The Raspberry Pi acts as the edge client. It captures the person's face and communicates with the MEC server, where the heavier face-recognition processing is performed.

Once authentication is completed, the Raspberry Pi communicates with the Arduino, which handles the physical lock mechanism.

## Authentication

- Owner — recognized through face authentication and allowed to access the system.
- Guest — recognized as a registered guest and required to complete additional password authentication.
- Unknown person — not matched with the registered database and treated as an unrecognized user.

## Main Components

### Raspberry Pi

The Raspberry Pi works as the edge-side controller.

It handles:
- Camera input
- Face detection
- Communication with the MEC server
- Communication with Arduino
- Receiving authentication results

The Pi avoids running the heavy AI authentication model locally and instead offloads that processing to the MEC server.

### MEC Server

The MEC server performs the computationally heavier authentication tasks.

It handles:
- Face embedding generation
- Face matching
- Owner/guest identification
- Authentication decisions
- Guest password generation and validation
- Authentication state

### Arduino

The Arduino handles the physical side of the system.

It is responsible for:
- Receiving commands from the Raspberry Pi
- Controlling the servo-based lock
- Handling the physical keypad
- Providing buzzer and OLED feedback

## Technologies Used

- Python
- Raspberry Pi
- OpenCV
- DeepFace
- TensorFlow
- Flask
- Arduino
- Servo motor
- Webcam
- Physical keypad
- OLED display

## Project Structure

`	ext
.
├── mec_server.py
├── pi_client.py
├── guest_auth.py
├── delete_guest.py
├── list_registered_users.py
├── Calibrate_threshold.py
├── security_capture.py
├── unknown_capture.py
├── test_owner_recognition.py
├── face_arduino_guest_v3_deepface.py
├── haarcascade_frontalface_default.xml
├── templates/
├── README.md
└── .gitignore

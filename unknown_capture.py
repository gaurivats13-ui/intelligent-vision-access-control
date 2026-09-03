import cv2
import serial
import time
import os
from datetime import datetime

import speech_recognition as sr
import pyttsx3


# =====================================================
# SETTINGS
# =====================================================

COM_PORT = "COM5"
BAUD_RATE = 9600

THRESHOLD = 75
REQUIRED_MATCHES = 15

UNKNOWN_REQUIRED_MATCHES = 5
MAX_UNKNOWN_CAPTURES = 10
CAPTURE_INTERVAL = 0.5

CASCADE_PATH = r"C:\Users\gauri\python\haarcascade_frontalface_default.xml"

TRAINER_PATH = r"C:\Users\gauri\python\trainer\trainer.yml"

SECURITY_FOLDER = r"C:\Users\gauri\python\security_records"

os.makedirs(SECURITY_FOLDER, exist_ok=True)


# =====================================================
# VOICE
# =====================================================

def speak(text):

    print("VOICE:", text)

    engine = pyttsx3.init()

    engine.setProperty("rate", 165)

    engine.say(text)
    engine.runAndWait()

    engine.stop()


def ask_visitor_name():

    recognizer_voice = sr.Recognizer()

    speak("Please tell me your name.")

    print("Listening for visitor name...")

    with sr.Microphone() as source:

        recognizer_voice.adjust_for_ambient_noise(
            source,
            duration=0.5
        )

        try:

            audio = recognizer_voice.listen(
                source,
                timeout=5,
                phrase_time_limit=5
            )

            name = recognizer_voice.recognize_google(
                audio
            )

            name = name.strip()

            print("Visitor name:", name)

            speak(
                f"Hello {name}, please enter password."
            )

            return name

        except sr.WaitTimeoutError:

            print("No voice detected.")

            speak(
                "No name detected. Please try again."
            )

            return None

        except sr.UnknownValueError:

            print("Could not understand the visitor.")

            speak(
                "Sorry, I could not understand."
            )

            return None

        except sr.RequestError as error:

            print(
                "Speech recognition error:",
                error
            )

            speak(
                "Speech recognition is unavailable."
            )

            return None


# =====================================================
# UNKNOWN FACE CAPTURE
# =====================================================

def capture_unknown_faces():

    print()
    print("================================")
    print("UNKNOWN FACE DETECTED")
    print("Starting security capture...")
    print("================================")

    capture_count = 0

    last_capture_time = 0

    while capture_count < MAX_UNKNOWN_CAPTURES:

        ret, frame = camera.read()

        if not ret:

            print("Camera error during security capture.")

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:

            current_time = time.time()

            if (
                current_time - last_capture_time
                >= CAPTURE_INTERVAL
            ):

                face_image = frame[
                    y:y+h,
                    x:x+w
                ]

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                filename = os.path.join(
                    SECURITY_FOLDER,
                    f"unknown_{timestamp}.jpg"
                )

                cv2.imwrite(
                    filename,
                    face_image
                )

                capture_count += 1

                last_capture_time = current_time

                print(
                    f"Unknown face captured "
                    f"{capture_count}/{MAX_UNKNOWN_CAPTURES}"
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"SECURITY CAPTURE: "
            f"{capture_count}/{MAX_UNKNOWN_CAPTURES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "AI Smart Lock",
            frame
        )

        if cv2.waitKey(1) & 0xFF == 27:

            print("Security capture stopped.")

            break

    print()
    print("================================")
    print("UNKNOWN FACE CAPTURE COMPLETE")
    print(f"{capture_count} images saved.")
    print("Location:")
    print(SECURITY_FOLDER)
    print("================================")

    return capture_count


# =====================================================
# ARDUINO
# =====================================================

try:

    arduino = serial.Serial(
        COM_PORT,
        BAUD_RATE
    )

    time.sleep(2)

    print("Arduino connected!")

except Exception as error:

    print("Arduino connection failed:")
    print(error)

    exit()


# =====================================================
# CAMERA
# =====================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Camera could not be opened!")

    arduino.close()

    exit()


# =====================================================
# HAARCASCADE
# =====================================================

face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_cascade.empty():

    print("Haarcascade not found!")

    camera.release()
    arduino.close()

    exit()


# =====================================================
# LBPH
# =====================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    TRAINER_PATH
)


names = {
    1: "Gauri",
    2: "Ramaya"
}


# =====================================================
# VARIABLES
# =====================================================

last_valid_name = None

match_count = 0

unknown_match_count = 0

unlocked = False

unknown_handled = False


# =====================================================
# START
# =====================================================

print("--------------------------------")
print("AI SMART LOCK STARTED")
print("Threshold:", THRESHOLD)
print("Required matches:", REQUIRED_MATCHES)
print("--------------------------------")


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("Camera error!")

        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )


    # =================================================
    # NO FACE
    # =================================================

    if len(faces) == 0:

        last_valid_name = None

        match_count = 0

        unknown_match_count = 0


    # =================================================
    # FACE FOUND
    # =================================================

    for (x, y, w, h) in faces:

        face = gray[
            y:y+h,
            x:x+w
        ]


        person_id, distance = recognizer.predict(
            face
        )


        distance = float(distance)


        # =================================================
        # KNOWN OR UNKNOWN
        # =================================================

        if (
            person_id in names
            and distance < THRESHOLD
        ):

            current_name = names[person_id]

        else:

            current_name = "Unknown"


        print(
            "ID:", person_id,
            "| Distance:", round(distance, 2),
            "| Result:", current_name,
            "| Known Count:", match_count,
            "| Unknown Count:", unknown_match_count
        )


        # =================================================
        # UNKNOWN
        # =================================================

        if current_name == "Unknown":

            last_valid_name = None

            match_count = 0

            unknown_match_count += 1


            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )


            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            # -----------------------------------------
            # UNKNOWN CONFIRMED
            # -----------------------------------------

            if (
                unknown_match_count
                >= UNKNOWN_REQUIRED_MATCHES
                and not unknown_handled
            ):

                unknown_handled = True


                # Capture security photos
                capture_unknown_faces()


                # -------------------------------------
                # CLOSE CAMERA BEFORE VOICE
                # -------------------------------------

                camera.release()

                cv2.destroyAllWindows()


                # -------------------------------------
                # ASK VISITOR NAME
                # -------------------------------------

                visitor_name = ask_visitor_name()


                if visitor_name:

                    print()
                    print("================================")
                    print("UNKNOWN VISITOR")
                    print(
                        "NAME:",
                        visitor_name
                    )
                    print("================================")

                else:

                    print(
                        "Visitor name was not received."
                    )


                # -------------------------------------
                # STOP FOR NOW
                # -------------------------------------

                print()
                print(
                    "Guest password authentication "
                    "will be added next."
                )

                arduino.close()

                exit()


        # =================================================
        # KNOWN
        # =================================================

        else:

            unknown_match_count = 0


            if current_name == last_valid_name:

                match_count += 1

            else:

                last_valid_name = current_name

                match_count = 1


            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                current_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # =================================================
        # KNOWN USER → UNLOCK
        # =================================================

        if (
            current_name in ("Gauri", "Ramaya")
            and last_valid_name == current_name
            and match_count >= REQUIRED_MATCHES
            and not unlocked
        ):

            print()
            print("================================")
            print("AUTHENTICATION SUCCESSFUL")
            print("USER:", current_name)
            print("MATCH COUNT:", match_count)
            print("================================")


            arduino.write(
                b"UNLOCK\n"
            )

            arduino.flush()

            unlocked = True


            camera.release()

            cv2.destroyAllWindows()


            print("UNLOCK SENT TO ARDUINO")
            print("Camera closed.")
            print("Door unlocked.")
            print("Arduino will lock after 5 seconds.")

            break


    # =================================================
    # UNLOCKED → STOP CAMERA
    # =================================================

    if unlocked:

        break


    cv2.imshow(
        "AI Smart Lock",
        frame
    )


    if cv2.waitKey(1) & 0xFF == 27:

        print("Emergency exit.")

        break


# =====================================================
# CLEANUP
# =====================================================

camera.release()

cv2.destroyAllWindows()

arduino.close()

print("System stopped.")
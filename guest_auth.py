import random
import time


# ==============================
# SETTINGS
# ==============================

OTP_VALIDITY = 60       # seconds
MAX_ATTEMPTS = 3


# ==============================
# GENERATE TEMPORARY PASSWORD
# ==============================

password = str(random.randint(100000, 999999))

created_time = time.time()

print()
print("================================")
print("   GUEST AUTHENTICATION SYSTEM")
print("================================")

print()
print("Temporary Password Generated:")
print(password)

print()
print("Password is valid for 60 seconds.")
print("Maximum attempts:", MAX_ATTEMPTS)
print()


# ==============================
# PASSWORD VERIFICATION
# ==============================

attempts = 0

while attempts < MAX_ATTEMPTS:

    # Check expiry
    if time.time() - created_time > OTP_VALIDITY:

        print()
        print("================================")
        print("PASSWORD EXPIRED")
        print("ACCESS DENIED")
        print("================================")

        break


    entered_password = input(
        "Enter temporary password: "
    ).strip()


    # ==============================
    # CORRECT PASSWORD
    # ==============================

    if entered_password == password:

        print()
        print("================================")
        print("       ACCESS GRANTED")
        print("================================")
        print("Guest authentication successful.")
        print()

        break


    # ==============================
    # WRONG PASSWORD
    # ==============================

    else:

        attempts += 1

        remaining = MAX_ATTEMPTS - attempts

        print()
        print("WRONG PASSWORD")

        if remaining > 0:

            print(
                "Attempts remaining:",
                remaining
            )

        else:

            print()
            print("================================")
            print("       ACCESS DENIED")
            print("       SYSTEM LOCKED")
            print("================================")

            # 20-second lockout
            print("Lockout: 20 seconds")

            time.sleep(20)

            print("Lockout finished.")
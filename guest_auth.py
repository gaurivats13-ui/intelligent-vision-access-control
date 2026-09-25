import secrets
import time
import hashlib


PASSWORD_LENGTH = 6
PASSWORD_VALID_SECONDS = 60
MAX_ATTEMPTS = 3

# Allowed digits for generated passwords (1, 2 and 3 are excluded)
PASSWORD_DIGITS = "4567890"


active_passwords = {}


def hash_password(password):

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def generate_password(name):

    name = str(name).strip()

    if not name:
        raise ValueError(
            "Guest name cannot be empty."
        )

    password = "".join(
        secrets.choice(PASSWORD_DIGITS)
        for _ in range(PASSWORD_LENGTH)
    )

    active_passwords[name.lower()] = {
        "password_hash": hash_password(password),
        "created_at": time.time(),
        "attempts": 0
    }

    return password


def verify_password(name, password):

    name = str(name).strip()
    password = str(password).strip()

    if not name:
        return False, "invalid_name"

    if not password:
        return False, "empty_password"

    key = name.lower()

    if key not in active_passwords:
        return False, "no_active_password"

    record = active_passwords[key]

    elapsed = time.time() - record["created_at"]

    if elapsed > PASSWORD_VALID_SECONDS:

        del active_passwords[key]

        return False, "expired"

    if record["attempts"] >= MAX_ATTEMPTS:

        del active_passwords[key]

        return False, "max_attempts"

    record["attempts"] += 1

    entered_hash = hash_password(password)

    if entered_hash == record["password_hash"]:

        del active_passwords[key]

        return True, "success"

    if record["attempts"] >= MAX_ATTEMPTS:

        del active_passwords[key]

        return False, "max_attempts"

    return False, "wrong"


def get_password_info(name):

    name = str(name).strip()
    key = name.lower()

    if key not in active_passwords:
        return None

    record = active_passwords[key]

    elapsed = time.time() - record["created_at"]

    remaining = PASSWORD_VALID_SECONDS - elapsed

    if remaining <= 0:

        del active_passwords[key]

        return None

    return {
        "valid": True,
        "remaining_seconds": int(remaining),
        "attempts_used": record["attempts"],
        "max_attempts": MAX_ATTEMPTS
    }


def clear_password(name):

    name = str(name).strip()
    key = name.lower()

    if key in active_passwords:

        del active_passwords[key]

        return True

    return False


if __name__ == "__main__":

    test_name = "TestGuest"

    print("========================================")
    print("GUEST AUTH TEST")
    print("========================================")

    password = generate_password(test_name)

    print(f"Guest: {test_name}")
    print(f"Generated password: {password}")
    print(
        f"Valid for: {PASSWORD_VALID_SECONDS} seconds"
    )
    print(
        f"Maximum attempts: {MAX_ATTEMPTS}"
    )

    print()
    print("Testing correct password...")

    success, reason = verify_password(
        test_name,
        password
    )

    print(f"Success: {success}")
    print(f"Reason: {reason}")
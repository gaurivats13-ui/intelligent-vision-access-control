import os
import sys
import json
import sqlite3
import shutil

BASE_DIR = r"C:\Users\gauri\python"

FACE_DATABASE_PATH = os.path.join(
    BASE_DIR, "face_database.json"
)

DATABASE_PATH = os.path.join(
    BASE_DIR, "smart_lock_database.db"
)

GUEST_FOLDER = os.path.join(
    BASE_DIR, "temporary_guests"
)


def delete_guest(guest_name):

    print(f"\nDeleting guest: {guest_name}")

    # ==========================================
    # 1. Delete from face_database.json
    # ==========================================

    if os.path.exists(FACE_DATABASE_PATH):

        with open(FACE_DATABASE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.setdefault("owner", {})
        data.setdefault("guests", {})

        # Case-insensitive guest name search
        matching_name = None

        for name in data["guests"]:
            if name.lower() == guest_name.lower():
                matching_name = name
                break

        if matching_name:

            del data["guests"][matching_name]

            with open(FACE_DATABASE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)

            print("✓ Removed from face_database.json")

        else:
            print("• Guest not found in face_database.json")

    # ==========================================
    # 2. Delete guest folders
    # ==========================================

    deleted_folders = 0

    if os.path.exists(GUEST_FOLDER):

        for folder_name in os.listdir(GUEST_FOLDER):

            if folder_name.lower().startswith(
                guest_name.lower() + "_"
            ):

                folder_path = os.path.join(
                    GUEST_FOLDER,
                    folder_name
                )

                if os.path.isdir(folder_path):

                    shutil.rmtree(folder_path)

                    deleted_folders += 1

                    print(
                        f"✓ Deleted folder: {folder_name}"
                    )

    if deleted_folders == 0:
        print("• No guest photo folder found")

    # ==========================================
    # 3. Delete guest profile from SQLite
    # ==========================================

    if os.path.exists(DATABASE_PATH):

        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute(
            "DELETE FROM guest_profiles WHERE lower(name) = lower(?)",
            (guest_name,)
        )

        print(
            f"✓ Deleted guest_profiles records: {cursor.rowcount}"
        )

        cursor.execute(
            "DELETE FROM guest_sessions WHERE lower(visitor_name) = lower(?)",
            (guest_name,)
        )

        print(
            f"✓ Deleted guest_sessions records: {cursor.rowcount}"
        )

        connection.commit()
        connection.close()

    print("\n================================")
    print("GUEST DELETION COMPLETE")
    print("================================")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("\nUsage:")
        print("py -3.12 delete_guest.py GuestName")
        sys.exit(1)

    delete_guest(sys.argv[1])
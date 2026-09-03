import sqlite3
import os

BASE_DIR = r"C:\Users\gauri\python"
DATABASE_PATH = os.path.join(BASE_DIR, "smart_lock_database.db")

if not os.path.exists(DATABASE_PATH):
    print("Database not found at:", DATABASE_PATH)
    exit()

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

# ============================================================
# OWNER (hardcoded in script, not in DB)
# ============================================================
print("================================")
print("OWNER (from script's `names` dict)")
print("================================")
print("Gauri (ID: 1)")
print()

# ============================================================
# VISITORS TABLE
# ============================================================
print("================================")
print("VISITORS TABLE (everyone ever seen)")
print("================================")

cursor.execute("""
    SELECT name, first_seen, last_seen, visit_count
    FROM visitors
    ORDER BY last_seen DESC
""")

rows = cursor.fetchall()

if not rows:
    print("No visitors recorded yet.")
else:
    for name, first_seen, last_seen, visit_count in rows:
        print(f"- {name}")
        print(f"    First seen : {first_seen}")
        print(f"    Last seen  : {last_seen}")
        print(f"    Visits     : {visit_count}")
        print()

# ============================================================
# GUEST PROFILES TABLE (face-trained guests with a folder)
# ============================================================
print("================================")
print("GUEST PROFILES (face-recognized guests)")
print("================================")

cursor.execute("""
    SELECT name, created_at, last_seen, visit_count, guest_folder, status
    FROM guest_profiles
    ORDER BY last_seen DESC
""")

rows = cursor.fetchall()

if not rows:
    print("No guest profiles found.")
else:
    for name, created_at, last_seen, visit_count, guest_folder, status in rows:
        print(f"- {name}  [{status}]")
        print(f"    Registered : {created_at}")
        print(f"    Last seen  : {last_seen}")
        print(f"    Visits     : {visit_count}")
        print(f"    Folder     : {guest_folder}")
        print()

# ============================================================
# GUEST NAMES JSON (used for face-model label mapping)
# ============================================================
GUEST_NAMES_PATH = os.path.join(BASE_DIR, "guest_names.json")

print("================================")
print("GUEST FACE-MODEL LABELS (guest_names.json)")
print("================================")

if os.path.exists(GUEST_NAMES_PATH):
    import json
    with open(GUEST_NAMES_PATH, "r", encoding="utf-8") as f:
        guest_names = json.load(f)

    if guest_names:
        for guest_id, guest_name in guest_names.items():
            print(f"- ID {guest_id}: {guest_name}")
    else:
        print("File exists but is empty.")
else:
    print("guest_names.json not found.")

connection.close()

print()
print("================================")
print("DONE")
print("================================")
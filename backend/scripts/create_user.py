"""Create a user (or set a password on an existing passwordless one):  uv run python scripts/create_user.py"""
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import User, UserRole
from app.utils.database import SessionLocal, init_db
from app.utils.security import hash_password

init_db()
email = input("Email: ").strip().lower()
name = input("Full name: ").strip() or email
role = input("Role (STUDENT / TEACHER / ADMIN) [STUDENT]: ").strip().upper() or "STUDENT"
if role not in UserRole.__members__:
    sys.exit("Role must be STUDENT, TEACHER or ADMIN.")
password = getpass.getpass("Password (at least 8 characters, typing is hidden): ")
if len(password) < 8:
    sys.exit("Password is too short.")

db = SessionLocal()
user = db.query(User).filter(User.email == email).first()
if user:
    if user.password_hash:
        sys.exit("That account already has a password.")
    user.password_hash = hash_password(password)
    print(f"Password set for existing {user.role.value} account {email}.")
else:
    user = User(email=email, full_name=name, role=UserRole[role], password_hash=hash_password(password))
    db.add(user)
    print(f"Created {role} account {email}.")
db.commit()
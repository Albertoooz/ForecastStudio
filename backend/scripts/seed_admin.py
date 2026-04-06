"""
Seed local admin user: email=admin, password=admin.

Run from repo root:
  cd backend && uv run python scripts/seed_admin.py
Or with docker (after migrate):
  docker compose run --rm backend python scripts/seed_admin.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure backend/app is importable when run as script
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from passlib.context import CryptContext

from app.config import get_settings
from app.db.models import Tenant, User
from app.db.session import async_session_factory

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Use valid email so frontend type="email" and backend EmailStr (register) accept it
ADMIN_EMAIL = "admin@local.dev"
ADMIN_PASSWORD = "admin"
TENANT_NAME = "Local"


async def seed_admin():
    settings = get_settings()
    if settings.environment not in ("local", "dev"):
        print(f"Skipping seed: only run in local/dev (current: {settings.environment})")
        return

    async with async_session_factory() as session:
        from sqlalchemy import select

        # Check if admin already exists
        r = await session.execute(select(User).where(User.email == ADMIN_EMAIL))
        if r.scalar_one_or_none():
            print(f"User {ADMIN_EMAIL!r} already exists. No change.")
            return

        # Create tenant
        tenant = Tenant(name=TENANT_NAME, plan="free")
        session.add(tenant)
        await session.flush()

        # Create admin user
        user = User(
            tenant_id=tenant.id,
            email=ADMIN_EMAIL,
            hashed_password=pwd_context.hash(ADMIN_PASSWORD),
            full_name="Admin",
            role="admin",
            is_active=True,
        )
        session.add(user)
        await session.commit()
        print(f"Created admin user: email={ADMIN_EMAIL!r}, password={ADMIN_PASSWORD!r}")
        print("Log in at http://localhost:3000/login with these credentials.")


if __name__ == "__main__":
    asyncio.run(seed_admin())

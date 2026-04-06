"""
Auth API — JWT-based authentication.

MVP: email + password → JWT.  Phase 2: Azure AD B2C / SSO.
"""

from datetime import datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db
from app.db.models import User, Tenant

router = APIRouter()
settings = get_settings()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── Schemas ──────────────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""
    tenant_name: str = "My Organization"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    tenant_id: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str
    tenant_id: UUID


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_token(user_id: str, tenant_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "exp": expire,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency — decode JWT → load user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user + tenant."""
    # Check duplicate email
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create tenant
    tenant = Tenant(name=body.tenant_name)
    db.add(tenant)
    await db.flush()

    # Create user
    user = User(
        tenant_id=tenant.id,
        email=body.email,
        hashed_password=pwd_context.hash(body.password),
        full_name=body.full_name,
        role="admin",
    )
    db.add(user)
    await db.flush()

    token = _create_token(str(user.id), str(tenant.id))
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        tenant_id=str(tenant.id),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
):
    """Login with email + password → JWT."""
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = _create_token(str(user.id), str(user.tenant_id))
    return TokenResponse(
        access_token=token,
        user_id=str(user.id),
        tenant_id=str(user.tenant_id),
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    """Get currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
    )

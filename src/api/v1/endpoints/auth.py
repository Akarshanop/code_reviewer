from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, EmailStr

from src.core.logger import get_logger
from src.core.security import create_access_token
from src.services.auth_service import auth_service

logger = get_logger(__name__)
router = APIRouter()


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class WhitelistCreateRequest(BaseModel):
    email: EmailStr
    role: str
    added_by: Optional[str] = None


class WhitelistUpdateRequest(BaseModel):
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    status: Optional[str] = None


@router.post("/signup")
async def signup(payload: SignUpRequest):
    user = await auth_service.signup(payload.email, payload.password)
    token = create_access_token(
        {
            "sub": user.get("email"),
            "user_id": user.get("id"),
            "role": user.get("role", "viewer"),
        }
    )
    return {
        "message": "Signup successful",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.get("id"),
            "email": user.get("email"),
            "role": user.get("role", "viewer"),
        },
    }


@router.post("/signin")
async def signin(payload: SignInRequest):
    user = await auth_service.signin(payload.email, payload.password)
    token = create_access_token(
        {
            "sub": user.get("email"),
            "user_id": user.get("id"),
            "role": user.get("role", "viewer"),
        }
    )
    return {
        "message": "Signin successful",
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.get("id"),
            "email": user.get("email"),
            "role": user.get("role", "viewer"),
        },
    }


@router.get("/whitelist/status")
async def whitelist_status(email: EmailStr = Query(...)):
    whitelisted, role = await auth_service.get_whitelist_status(str(email))
    return {"whitelisted": whitelisted, "role": role}


@router.get("/whitelist")
async def get_whitelisted_users():
    rows = await auth_service.list_whitelisted_users()
    return {"users": rows}


@router.post("/whitelist")
async def create_whitelisted_user(payload: WhitelistCreateRequest):
    created = await auth_service.add_whitelisted_user(
        email=str(payload.email),
        role=payload.role,
        added_by=payload.added_by,
    )
    return {"user": created}


@router.patch("/whitelist/{row_id}")
async def patch_whitelisted_user(row_id: str, payload: WhitelistUpdateRequest):
    updated = await auth_service.update_whitelisted_user(
        row_id=row_id,
        updates=payload.model_dump(exclude_none=True),
    )
    return {"user": updated}


@router.post("/whitelist/{row_id}/revoke")
async def revoke_whitelisted_user(row_id: str):
    updated = await auth_service.revoke_whitelisted_user(row_id)
    return {"user": updated}

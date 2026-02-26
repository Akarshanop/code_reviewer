from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, status
from postgrest.exceptions import APIError

from src.core.logger import get_logger
from src.core.security import get_password_hash, verify_password
from src.database.supabase_session import supabase_client

logger = get_logger(__name__)


class AuthService:
    """Auth service backed by Supabase tables and local JWT issuance."""

    async def _get_client(self):
        await supabase_client.initialize()
        client = supabase_client.client
        if not client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Supabase is not configured",
            )
        return client

    @staticmethod
    def _as_single_row(response: Any) -> Optional[Dict[str, Any]]:
        """Normalize Supabase response to a single dict row."""
        data = getattr(response, "data", None)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            first = data[0]
            return first if isinstance(first, dict) else None
        return None

    @staticmethod
    def _as_rows(response: Any) -> list[Dict[str, Any]]:
        """Normalize Supabase response to a list of dict rows."""
        data = getattr(response, "data", None)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []

    async def get_whitelist_status(self, email: str) -> Tuple[bool, Optional[str]]:
        """Return (whitelisted, role) for an email."""
        normalized_email = email.strip().lower()
        client = await self._get_client()

        try:
            rpc_response = client.rpc(
                "get_whitelist_status",
                {"check_email": normalized_email},
            ).execute()
            rpc_row = self._as_single_row(rpc_response)
            if rpc_row is not None:
                return bool(rpc_row.get("whitelisted")), rpc_row.get("role")
        except Exception as exc:
            logger.debug("Whitelist RPC unavailable, using table lookup", error=str(exc))

        response = (
            client.table("whitelisted_users")
            .select("email, role, status")
            .eq("email", normalized_email)
            .limit(1)
            .execute()
        )
        row = self._as_single_row(response)
        if not row:
            return False, None
        if row.get("status") != "active":
            return False, None
        return True, row.get("role")

    async def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find user row by email."""
        normalized_email = email.strip().lower()
        client = await self._get_client()
        response = (
            client.table("user_profiles")
            .select("id, email, role, hashed_password")
            .eq("email", normalized_email)
            .limit(1)
            .execute()
        )
        return self._as_single_row(response)

    async def signup(self, email: str, plain_password: str) -> Dict[str, Any]:
        """Create a whitelisted user with hashed password."""
        normalized_email = email.strip().lower()
        client = await self._get_client()

        whitelisted, role = await self.get_whitelist_status(normalized_email)

        if not whitelisted:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not whitelisted",
            )

        existing = await self.find_user_by_email(normalized_email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists",
            )

        auth_user_id: Optional[str] = None
        try:
            auth_resp = client.auth.sign_up(
                {"email": normalized_email, "password": plain_password}
            )
            auth_user = getattr(auth_resp, "user", None)
            auth_user_id = getattr(auth_user, "id", None)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create auth user: {str(exc)}",
            ) from exc

        if not auth_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create auth user: missing user id",
            )

        try:
            hashed_password = get_password_hash(plain_password)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process password",
            ) from exc

        payload = {
            "id": auth_user_id,
            "email": normalized_email,
            "role": role or "viewer",
            "hashed_password": hashed_password,
        }
        try:
            client.table("user_profiles").insert(payload).execute()
        except APIError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create user profile: {exc.message}",
            ) from exc

        created = await self.find_user_by_email(normalized_email)
        if not created:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user profile",
            )
        return {
            "id": created.get("id"),
            "email": created.get("email"),
            "role": created.get("role"),
        }

    async def signin(self, email: str, plain_password: str) -> Dict[str, Any]:
        """Validate credentials and return user row."""
        user = await self.find_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        hashed_password = user.get("hashed_password", "")
        if not hashed_password or not verify_password(plain_password, hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        return user

    async def list_whitelisted_users(self) -> list[Dict[str, Any]]:
        """Return all whitelist rows ordered by most-recent first."""
        client = await self._get_client()
        response = (
            client.table("whitelisted_users")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return self._as_rows(response)

    async def _find_whitelisted_user_by_id(self, row_id: str) -> Optional[Dict[str, Any]]:
        client = await self._get_client()
        response = (
            client.table("whitelisted_users")
            .select("*")
            .eq("id", row_id)
            .limit(1)
            .execute()
        )
        return self._as_single_row(response)

    async def add_whitelisted_user(
        self,
        email: str,
        role: str,
        added_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new whitelist row."""
        client = await self._get_client()
        payload: Dict[str, Any] = {
            "email": email.strip().lower(),
            "role": role,
            "status": "active",
        }
        if added_by:
            payload["added_by"] = added_by

        response = client.table("whitelisted_users").insert(payload).execute()
        created = self._as_single_row(response)
        if not created:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add whitelisted user",
            )
        return created

    async def update_whitelisted_user(
        self,
        row_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update a whitelist row by id."""
        client = await self._get_client()
        allowed = {"email", "role", "status"}
        safe_updates = {k: v for k, v in updates.items() if k in allowed}
        if "email" in safe_updates and isinstance(safe_updates["email"], str):
            safe_updates["email"] = safe_updates["email"].strip().lower()
        if not safe_updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid update fields provided",
            )

        client.table("whitelisted_users").update(safe_updates).eq("id", row_id).execute()
        updated = await self._find_whitelisted_user_by_id(row_id)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Whitelisted user not found",
            )
        return updated

    async def revoke_whitelisted_user(self, row_id: str) -> Dict[str, Any]:
        """Mark a whitelist row as revoked."""
        return await self.update_whitelisted_user(row_id, {"status": "revoked"})


auth_service = AuthService()

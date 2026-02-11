import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from config.settings import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Redis client for token blacklisting
redis_client: Optional[redis.Redis] = None

def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client if configured"""
    global redis_client
    if redis_client is None and settings.REDIS_URL:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return redis_client

class TokenBlacklist:
    """Manage blacklisted JWT tokens"""
    
    @staticmethod
    async def add_token(token: str, expires_in: int):
        """Add token to blacklist"""
        redis_cli = get_redis_client()
        if redis_cli:
            key = f"blacklist:{token}"
            await redis_cli.setex(key, expires_in, "1")
    
    @staticmethod
    async def is_blacklisted(token: str) -> bool:
        """Check if token is blacklisted"""
        redis_cli = get_redis_client()
        if redis_cli:
            key = f"blacklist:{token}"
            return await redis_cli.exists(key) == 1
        return False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate JWT token"""
    try:
        # Check blacklist first
        if TokenBlacklist.is_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def revoke_token(token: str):
    """Revoke a JWT token"""
    try:
        payload = decode_access_token(token)
        if payload and "exp" in payload:
            expires_in = payload["exp"] - int(datetime.utcnow().timestamp())
            if expires_in > 0:
                await TokenBlacklist.add_token(token, expires_in)
    except Exception:
        pass

def create_api_key(name: str, user_id: str) -> Dict[str, str]:
    """Create API key for service authentication"""
    raw_key = secrets.token_urlsafe(32)
    key_id = hashlib.sha256(f"{name}:{user_id}".encode()).hexdigest()[:16]
    
    # Store key hash in database (simulated)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    return {
        "key_id": key_id,
        "api_key": raw_key,
        "key_hash": key_hash,
        "name": name,
        "created_at": datetime.utcnow().isoformat()
    }

def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify API key against stored hash"""
    computed_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return hmac.compare_digest(computed_hash, stored_hash)

class RBAC:
    """Role-Based Access Control"""
    
    ROLES = {
        "admin": ["*"],
        "architect": ["read:all", "write:blueprints", "read:codebase"],
        "developer": ["read:codebase", "write:analysis", "read:blueprints"],
        "qa": ["read:blueprints", "write:tests", "read:codebase"],
        "viewer": ["read:codebase"],
    }
    
    @staticmethod
    def has_permission(role: str, permission: str) -> bool:
        """Check if role has permission"""
        if role not in RBAC.ROLES:
            return False
        
        role_permissions = RBAC.ROLES[role]
        return "*" in role_permissions or permission in role_permissions
    
    @staticmethod
    def get_role_permissions(role: str) -> list:
        """Get all permissions for a role"""
        return RBAC.ROLES.get(role, [])

class JWTBearer(HTTPBearer):
    """JWT Bearer authentication scheme"""
    
    def __init__(self, auto_error: bool = True, required_roles: list = None):
        super().__init__(auto_error=auto_error)
        self.required_roles = required_roles or []
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if credentials:
            if credentials.scheme != "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            
            token_data = decode_access_token(credentials.credentials)
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or expired token"
                )
            
            # Check role permissions
            user_role = token_data.get("role", "viewer")
            if self.required_roles and user_role not in self.required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return token_data
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization credentials"
            )

auth_scheme = JWTBearer()

def require_role(*roles: str):
    """Decorator to require specific roles"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authorization header"
                )
            
            token = auth_header.replace("Bearer ", "")
            token_data = decode_access_token(token)
            
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            user_role = token_data.get("role", "viewer")
            if user_role not in roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
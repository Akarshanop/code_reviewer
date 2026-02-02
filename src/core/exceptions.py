from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: str

class BaseAPIException(HTTPException):
    """Base exception for API errors"""
    
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        self.error = error
        self.message = message
        self.details = details
        super().__init__(
            status_code=status_code,
            detail=self.to_dict(),
            headers=headers
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error": self.error,
            "message": self.message,
            "details": self.details or {}
        }

class ValidationException(BaseAPIException):
    """Validation error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="validation_error",
            message=message,
            details=details
        )

class AuthenticationException(BaseAPIException):
    """Authentication error"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error="authentication_error",
            message=message
        )

class AuthorizationException(BaseAPIException):
    """Authorization error"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error="authorization_error",
            message=message
        )

class ResourceNotFoundException(BaseAPIException):
    """Resource not found"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error="resource_not_found",
            message=f"{resource_type} with ID '{resource_id}' not found"
        )

class ConflictException(BaseAPIException):
    """Resource conflict"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error="conflict_error",
            message=message,
            details=details
        )

class RateLimitException(BaseAPIException):
    """Rate limit exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error="rate_limit_exceeded",
            message=message,
            headers={"Retry-After": str(retry_after)}
        )

class InternalServerException(BaseAPIException):
    """Internal server error"""
    
    def __init__(self, message: str = "Internal server error"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="internal_server_error",
            message=message
        )

class ServiceUnavailableException(BaseAPIException):
    """Service unavailable"""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="service_unavailable",
            message=message
        )

class CodeAnalysisException(BaseAPIException):
    """Code analysis error"""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        details = {"file_path": file_path} if file_path else None
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="code_analysis_error",
            message=message,
            details=details
        )

class LLMServiceException(BaseAPIException):
    """LLM service error"""
    
    def __init__(self, message: str, model: Optional[str] = None):
        details = {"model": model} if model else None
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error="llm_service_error",
            message=message,
            details=details
        )

class GraphDatabaseException(BaseAPIException):
    """Graph database error"""
    
    def __init__(self, message: str, query: Optional[str] = None):
        details = {"query": query} if query else None
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="graph_database_error",
            message=message,
            details=details
        )


class AgentException(BaseAPIException):
    """AI Agent processing error"""
    
    def __init__(self, message: str, agent_type: Optional[str] = None):
        details = {"agent_type": agent_type} if agent_type else None
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="agent_error",
            message=message,
            details=details
        )


class IngestionException(BaseAPIException):
    """Data ingestion error"""
    
    def __init__(self, message: str, repository: Optional[str] = None):
        details = {"repository": repository} if repository else None
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="ingestion_error",
            message=message,
            details=details
        )


class VectorSearchException(BaseAPIException):
    """Vector search error"""
    
    def __init__(self, message: str, collection: Optional[str] = None):
        details = {"collection": collection} if collection else None
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="vector_search_error",
            message=message,
            details=details
        )


def create_error_response(
    error: str,
    message: str,
    status_code: int,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    from datetime import datetime
    
    return {
        "error": error,
        "code": error.upper().replace(" ", "_"),
        "message": message,
        "details": details or {},
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat()
    }
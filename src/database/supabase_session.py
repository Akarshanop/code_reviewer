import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
import json

from supabase import create_client, Client
from postgrest import APIResponse
import httpx

from config.settings import settings
from src.core.exceptions import ServiceUnavailableException
from src.core.logger import get_logger

logger = get_logger(__name__)

class SupabaseClient:
    """Enterprise Supabase client with connection pooling and advanced features"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._connection_pool = []
        self.max_pool_size = 10
        self._metrics = {
            "queries_executed": 0,
            "errors": 0,
            "connection_count": 0,
            "last_health_check": None
        }
        
    async def initialize(self):
        """Initialize Supabase client with connection pooling"""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    try:
                        # Skip if Supabase is not configured
                        if not settings.SUPABASE_URL or "your-supabase" in settings.SUPABASE_URL.lower():
                            logger.warning("Supabase not configured - skipping initialization")
                            self._initialized = True  # Mark as initialized but without client
                            return
                        
                        # Create client with minimal options (newer SDK versions have limited options)
                        self.client = create_client(
                            supabase_url=settings.SUPABASE_URL,
                            supabase_key=settings.SUPABASE_KEY
                        )
                        
                        # Initialize connection pool
                        await self._initialize_connection_pool()
                        
                        # Verify connectivity with timeout (non-blocking)
                        try:
                            await asyncio.wait_for(self._verify_connectivity(), timeout=5.0)
                        except asyncio.TimeoutError:
                            logger.warning("Supabase connectivity check timed out (non-critical) - proceeding without verification")
                        
                        self._initialized = True
                        logger.info(
                            "Supabase client initialized successfully",
                            url=settings.SUPABASE_URL[:50] + "...",
                            pool_size=self.max_pool_size
                        )
                        
                    except Exception as e:
                        logger.warning(f"Supabase initialization failed (non-critical): {str(e)}")
                        self._initialized = True  # Continue without Supabase
    
    async def _initialize_connection_pool(self):
        """Initialize connection pool for concurrent operations"""
        # For Supabase, we simulate connection pooling by managing multiple clients
        # In production, you'd want to use proper connection pooling
        self._connection_pool = []
        
        for i in range(min(5, self.max_pool_size)):  # Start with 5 connections
            try:
                client = create_client(
                    supabase_url=settings.SUPABASE_URL,
                    supabase_key=settings.SUPABASE_KEY
                )
                self._connection_pool.append({
                    "client": client,
                    "in_use": False,
                    "last_used": datetime.utcnow(),
                    "id": f"conn_{i}"
                })
                self._metrics["connection_count"] += 1
            except Exception as e:
                logger.warning(f"Failed to create connection {i}", error=str(e))
        
        logger.debug(f"Connection pool initialized with {len(self._connection_pool)} connections")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    async def _verify_connectivity(self):
        """Verify Supabase connectivity with retry logic"""
        try:
            # Test with a simple query
            response = await self.execute_query("SELECT 1 as test")
            if response and len(response) > 0 and response[0].get("test") == 1:
                logger.info("Supabase connectivity verified")
                self._metrics["last_health_check"] = datetime.utcnow().isoformat()
                return True
            else:
                raise ConnectionError("Connectivity test failed")
        except Exception as e:
            logger.error("Supabase connectivity check failed", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with context management"""
        if not self._initialized:
            await self.initialize()
        
        connection = None
        try:
            # Find available connection
            for conn in self._connection_pool:
                if not conn["in_use"]:
                    connection = conn
                    conn["in_use"] = True
                    conn["last_used"] = datetime.utcnow()
                    break
            
            # If no connections available, create a new one (up to max)
            if not connection and len(self._connection_pool) < self.max_pool_size:
                client = create_client(
                    supabase_url=settings.SUPABASE_URL,
                    supabase_key=settings.SUPABASE_KEY
                )
                connection = {
                    "client": client,
                    "in_use": True,
                    "last_used": datetime.utcnow(),
                    "id": f"conn_{len(self._connection_pool)}"
                }
                self._connection_pool.append(connection)
                self._metrics["connection_count"] += 1
                logger.debug("Created new Supabase connection")
            
            if not connection:
                raise ServiceUnavailableException("No database connections available")
            
            yield connection["client"]
            
        finally:
            if connection:
                connection["in_use"] = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query with parameters"""
        start_time = datetime.utcnow()
        
        async with self.get_connection() as client:
            try:
                # For complex queries, use raw SQL execution
                # Supabase PostgREST doesn't support raw SQL directly,
                # so we use RPC calls for stored procedures or direct table access
                
                # This is a simplified example - in production you'd use proper RPC calls
                # or the query builder based on your table structure
                
                if query.strip().upper().startswith("SELECT"):
                    # For SELECT queries, parse and use table methods
                    # This is a placeholder - implement based on your schema
                    table_name = self._extract_table_name(query)
                    
                    if table_name:
                        response = client.table(table_name).select("*").limit(10).execute()
                        result = self._format_response(response)
                    else:
                        # Fallback to direct RPC if available
                        result = await self._execute_rpc(query, params or {})
                else:
                    # For other queries, use RPC
                    result = await self._execute_rpc(query, params or {})
                
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._metrics["queries_executed"] += 1
                
                logger.debug(
                    "Supabase query executed",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    duration_ms=round(duration, 2),
                    result_count=len(result) if result else 0
                )
                
                return result or []
                
            except Exception as e:
                self._metrics["errors"] += 1
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.error(
                    "Supabase query failed",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    error=str(e),
                    duration_ms=round(duration, 2)
                )
                raise
    
    async def _execute_rpc(self, function_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a stored procedure/RPC function"""
        try:
            response = self.client.rpc(function_name, params).execute()
            return self._format_response(response)
        except Exception as e:
            logger.error(f"RPC execution failed: {function_name}", error=str(e))
            raise
    
    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract table name from SQL query (simplified)"""
        # Simple table name extraction - in production use proper SQL parsing
        query_lower = query.lower()
        from_index = query_lower.find("from ")
        
        if from_index != -1:
            rest = query_lower[from_index + 5:].strip()
            # Take first word after FROM
            table_name = rest.split()[0].strip(";")
            # Remove schema prefix if present
            if "." in table_name:
                table_name = table_name.split(".")[-1]
            return table_name
        
        return None
    
    def _format_response(self, response: APIResponse) -> List[Dict[str, Any]]:
        """Format Supabase API response to consistent format"""
        if hasattr(response, 'data'):
            return response.data or []
        elif isinstance(response, list):
            return response
        elif isinstance(response, dict) and 'data' in response:
            return response['data']
        else:
            return []
    
    async def insert_audit_log(self, log_data: Dict[str, Any]) -> str:
        """Insert audit log entry with structured data"""
        try:
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": "code-intelligence-platform",
                "level": log_data.get("level", "info"),
                "action": log_data.get("action", "unknown"),
                "user_id": log_data.get("user_id"),
                "repository": log_data.get("repository"),
                "file_path": log_data.get("file_path"),
                "details": json.dumps(log_data.get("details", {})),
                "duration_ms": log_data.get("duration_ms"),
                "status": log_data.get("status", "success"),
                "error_message": log_data.get("error_message"),
                "metadata": json.dumps(log_data.get("metadata", {}))
            }
            
            async with self.get_connection() as client:
                response = client.table("audit_logs").insert(audit_entry).execute()
                
                if response.data and len(response.data) > 0:
                    log_id = response.data[0].get("id")
                    logger.debug("Audit log inserted", log_id=log_id)
                    return log_id
            
            return None
            
        except Exception as e:
            logger.error("Failed to insert audit log", error=str(e))
            return None
    
    async def store_analysis_result(self, result_data: Dict[str, Any]) -> str:
        """Store code analysis results"""
        try:
            analysis_entry = {
                "created_at": datetime.utcnow().isoformat(),
                "analysis_type": result_data.get("analysis_type"),
                "repository": result_data.get("repository"),
                "branch": result_data.get("branch", "main"),
                "commit_hash": result_data.get("commit_hash"),
                "trigger": result_data.get("trigger", "manual"),
                "status": result_data.get("status", "completed"),
                "summary": json.dumps(result_data.get("summary", {})),
                "findings_count": result_data.get("findings_count", 0),
                "duration_seconds": result_data.get("duration_seconds"),
                "agent_model": result_data.get("agent_model"),
                "raw_results": json.dumps(result_data.get("raw_results", {})),
                "metadata": json.dumps(result_data.get("metadata", {}))
            }
            
            async with self.get_connection() as client:
                response = client.table("analysis_results").insert(analysis_entry).execute()
                
                if response.data and len(response.data) > 0:
                    result_id = response.data[0].get("id")
                    
                    # Store detailed findings if available
                    findings = result_data.get("findings", [])
                    if findings:
                        findings_entries = []
                        for i, finding in enumerate(findings):
                            findings_entries.append({
                                "analysis_id": result_id,
                                "finding_index": i,
                                "type": finding.get("type"),
                                "severity": finding.get("severity"),
                                "description": finding.get("description"),
                                "file_path": finding.get("file_path"),
                                "line_number": finding.get("line_number"),
                                "suggestion": finding.get("suggestion"),
                                "confidence": finding.get("confidence"),
                                "metadata": json.dumps(finding.get("metadata", {}))
                            })
                        
                        client.table("analysis_findings").insert(findings_entries).execute()
                    
                    logger.info("Analysis result stored", result_id=result_id, findings_count=len(findings))
                    return result_id
            
            return None
            
        except Exception as e:
            logger.error("Failed to store analysis result", error=str(e))
            return None
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a given ID"""
        try:
            async with self.get_connection() as client:
                response = client.table("conversations") \
                    .select("*") \
                    .eq("conversation_id", conversation_id) \
                    .order("created_at", desc=True) \
                    .limit(100) \
                    .execute()
                
                return response.data or []
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {conversation_id}", error=str(e))
            return []
    
    async def store_conversation_message(self, message_data: Dict[str, Any]) -> str:
        """Store a conversation message"""
        try:
            message_entry = {
                "conversation_id": message_data.get("conversation_id"),
                "message_id": message_data.get("message_id"),
                "created_at": datetime.utcnow().isoformat(),
                "role": message_data.get("role"),
                "content": message_data.get("content"),
                "agent_type": message_data.get("agent_type"),
                "model": message_data.get("model"),
                "tokens_used": message_data.get("tokens_used"),
                "context_files": json.dumps(message_data.get("context_files", [])),
                "metadata": json.dumps(message_data.get("metadata", {}))
            }
            
            async with self.get_connection() as client:
                response = client.table("conversation_messages").insert(message_entry).execute()
                
                if response.data and len(response.data) > 0:
                    return response.data[0].get("id")
            
            return None
            
        except Exception as e:
            logger.error("Failed to store conversation message", error=str(e))
            return None
    
    async def get_system_metrics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get system metrics for monitoring"""
        try:
            async with self.get_connection() as client:
                # Get recent audit logs count
                time_threshold = (datetime.utcnow() - timedelta(hours=time_range_hours)).isoformat()
                
                audit_response = client.table("audit_logs") \
                    .select("status", count="exact") \
                    .gte("timestamp", time_threshold) \
                    .execute()
                
                analysis_response = client.table("analysis_results") \
                    .select("analysis_type, status", count="exact") \
                    .gte("created_at", time_threshold) \
                    .execute()
                
                # Combine metrics
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "time_range_hours": time_range_hours,
                    "audit_logs": {
                        "total": audit_response.count or 0,
                        "by_status": {}
                    },
                    "analysis_results": {
                        "total": analysis_response.count or 0,
                        "by_type": {},
                        "by_status": {}
                    },
                    "connection_pool": {
                        "total": len(self._connection_pool),
                        "in_use": sum(1 for conn in self._connection_pool if conn["in_use"]),
                        "available": sum(1 for conn in self._connection_pool if not conn["in_use"])
                    },
                    "client_metrics": self._metrics
                }
                
                # Process audit logs by status
                if audit_response.data:
                    for entry in audit_response.data:
                        status = entry.get("status", "unknown")
                        metrics["audit_logs"]["by_status"][status] = \
                            metrics["audit_logs"]["by_status"].get(status, 0) + 1
                
                # Process analysis results
                if analysis_response.data:
                    for entry in analysis_response.data:
                        analysis_type = entry.get("analysis_type", "unknown")
                        status = entry.get("status", "unknown")
                        
                        metrics["analysis_results"]["by_type"][analysis_type] = \
                            metrics["analysis_results"]["by_type"].get(analysis_type, 0) + 1
                        
                        metrics["analysis_results"]["by_status"][status] = \
                            metrics["analysis_results"]["by_status"].get(status, 0) + 1
                
                return metrics
                
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check Supabase connectivity and health"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # If client is None (Supabase not configured), return False but don't error
            if self.client is None:
                return False
            
            # Try a simple query
            async with self.get_connection() as client:
                if client is None:
                    return False
                # Just check we can connect
                response = client.table("audit_logs").select("id").limit(1).execute()
                self._metrics["last_health_check"] = datetime.utcnow().isoformat()
                return True
        except Exception as e:
            logger.debug(f"Supabase health check failed: {e}")
            return False
    
    async def get_audit_logs(self, limit: int = 100, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get audit logs with optional filters"""
        try:
            async with self.get_connection() as client:
                query = client.table("audit_logs").select("*").order("timestamp", desc=True).limit(limit)
                
                if filters:
                    if filters.get("action"):
                        query = query.eq("action", filters["action"])
                    if filters.get("user_id"):
                        query = query.eq("user_id", filters["user_id"])
                    if filters.get("status"):
                        query = query.eq("status", filters["status"])
                    if filters.get("start_date"):
                        query = query.gte("timestamp", filters["start_date"])
                    if filters.get("end_date"):
                        query = query.lte("timestamp", filters["end_date"])
                
                response = query.execute()
                
                logs = []
                for row in (response.data or []):
                    logs.append({
                        "id": row.get("id"),
                        "timestamp": row.get("timestamp"),
                        "action": row.get("action"),
                        "user_id": row.get("user_id"),
                        "repository": row.get("repository"),
                        "status": row.get("status"),
                        "level": row.get("level"),
                        "details": json.loads(row.get("details", "{}")) if row.get("details") else {},
                        "error_message": row.get("error_message"),
                        "duration_ms": row.get("duration_ms")
                    })
                
                return logs
                
        except Exception as e:
            logger.error("Failed to get audit logs", error=str(e))
            return []
    
    async def get_chat_history(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            async with self.get_connection() as client:
                response = client.table("chat_messages") \
                    .select("*") \
                    .eq("session_id", session_id) \
                    .order("created_at", desc=False) \
                    .limit(limit) \
                    .execute()
                
                messages = []
                for row in (response.data or []):
                    messages.append({
                        "id": row.get("id"),
                        "role": row.get("role"),
                        "content": row.get("content"),
                        "timestamp": row.get("created_at"),
                        "metadata": json.loads(row.get("metadata", "{}")) if row.get("metadata") else {}
                    })
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get chat history: {session_id}", error=str(e))
            return []
    
    async def get_all_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all chat sessions with their first message as title"""
        try:
            async with self.get_connection() as client:
                # Get distinct session IDs with their first message
                response = client.table("chat_messages") \
                    .select("session_id, content, created_at, role") \
                    .order("created_at", desc=True) \
                    .execute()
                
                # Group by session_id
                sessions_map = {}
                for row in (response.data or []):
                    session_id = row.get("session_id")
                    if session_id not in sessions_map:
                        sessions_map[session_id] = {
                            "session_id": session_id,
                            "created_at": row.get("created_at"),
                            "title": "",
                            "last_message_at": row.get("created_at"),
                            "message_count": 0
                        }
                    
                    sessions_map[session_id]["message_count"] += 1
                    
                    # Use first user message as title
                    if row.get("role") == "user" and not sessions_map[session_id]["title"]:
                        content = row.get("content", "")
                        sessions_map[session_id]["title"] = content[:100] + ("..." if len(content) > 100 else "")
                
                # Convert to list and sort by last message
                sessions = list(sessions_map.values())
                sessions.sort(key=lambda x: x.get("last_message_at", ""), reverse=True)
                
                return sessions[:limit]
                
        except Exception as e:
            logger.error("Failed to get chat sessions", error=str(e))
            return []
    
    async def delete_chat_session(self, session_id: str) -> bool:
        """Delete all messages in a chat session"""
        try:
            async with self.get_connection() as client:
                response = client.table("chat_messages") \
                    .delete() \
                    .eq("session_id", session_id) \
                    .execute()
                
                logger.info(f"Deleted chat session: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete chat session: {session_id}", error=str(e))
            return False
    
    async def store_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """Store a chat message"""
        try:
            message_entry = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": datetime.utcnow().isoformat(),
                "metadata": json.dumps(metadata or {})
            }
            
            async with self.get_connection() as client:
                response = client.table("chat_messages").insert(message_entry).execute()
                
                if response.data and len(response.data) > 0:
                    return response.data[0].get("id")
            
            return None
            
        except Exception as e:
            logger.error("Failed to store chat message", error=str(e))
            return None
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
            
            async with self.get_connection() as client:
                # Delete old audit logs
                audit_deleted = client.table("audit_logs") \
                    .delete() \
                    .lt("timestamp", cutoff_date) \
                    .execute()
                
                # Delete old analysis results (keep only aggregated data)
                analysis_deleted = client.table("analysis_results") \
                    .delete() \
                    .lt("created_at", cutoff_date) \
                    .execute()
                
                logger.info(
                    "Data cleanup completed",
                    cutoff_date=cutoff_date,
                    audit_logs_deleted=audit_deleted.count if hasattr(audit_deleted, 'count') else 0,
                    analysis_results_deleted=analysis_deleted.count if hasattr(analysis_deleted, 'count') else 0
                )
                
                return {
                    "audit_logs_deleted": audit_deleted.count if hasattr(audit_deleted, 'count') else 0,
                    "analysis_results_deleted": analysis_deleted.count if hasattr(analysis_deleted, 'count') else 0
                }
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            return {}
    
    async def close(self):
        """Close all connections and clean up"""
        self._connection_pool.clear()
        self._initialized = False
        self._metrics["connection_count"] = 0
        logger.info("Supabase client closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Singleton instance
supabase_client = SupabaseClient()

# FastAPI dependency
async def get_supabase():
    """Dependency for FastAPI to get Supabase client"""
    client = supabase_client
    await client.initialize()
    try:
        yield client
    finally:
        await client.close()
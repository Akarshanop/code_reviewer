import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.types import EventDict, Processor
from config.settings import settings

class ContextLogger:

    def __init__(self):
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure structured logging for enterprise environment"""
        
        # Common processors
        timestamper = TimeStamper(fmt="iso")
        
        def add_service_context(_, __, event_dict: EventDict) -> EventDict:
            event_dict["service"] = "code-intelligence-platform"
            event_dict["environment"] = settings.ENVIRONMENT
            return event_dict
        
        def drop_color_message(_, __, event_dict: EventDict) -> EventDict:
            # Remove the 'color_message' key if it exists
            event_dict.pop('color_message', None)
            return event_dict
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                timestamper,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                add_service_context,
                drop_color_message,
                JSONRenderer(indent=2) if settings.DEBUG else JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
            wrapper_class=structlog.stdlib.BoundLogger,
        )
        
        # Configure standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, settings.LOG_LEVEL.upper()),
        )
        
        # Suppress noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("neo4j").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger instance"""
        return structlog.get_logger(name)
    
    def bind_context(self, **kwargs) -> structlog.BoundLogger:
        """Create a logger with bound context"""
        logger = structlog.get_logger()
        return logger.bind(**kwargs)

# Singleton logger instance
_logger = ContextLogger()

def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get logger instance"""
    return _logger.get_logger(name)

def log_with_context(name: str, **context) -> structlog.BoundLogger:
    """Get logger with bound context"""
    return get_logger(name).bind(**context)

class RequestContextFilter(logging.Filter):
    """Add request context to log records"""
    
    def __init__(self, request_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self.request_id = request_id
        self.start_time = datetime.utcnow()
    
    def filter(self, record):
        record.request_id = self.request_id or "global"
        record.duration_ms = int((datetime.utcnow() - self.start_time).total_seconds() * 1000)
        return True

# Export commonly used functions
setup_logging = _logger._configure_logging
get_context_logger = _logger.bind_context
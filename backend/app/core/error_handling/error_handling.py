"""
Global Error Handling and Monitoring System

Provides comprehensive error tracking, user-friendly error messages,
and integration with Sentry for production monitoring.
"""

import os
import sys
import logging
import traceback
import streamlit as st
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime, timezone
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Central error handling and monitoring system"""
    
    def __init__(self):
        self.sentry_enabled = False
        self.error_count = 0
        self.init_sentry()
    
    def init_sentry(self):
        """Initialize Sentry error monitoring if configured"""
        try:
            sentry_dsn = os.getenv('SENTRY_DSN')
            if sentry_dsn:
                import sentry_sdk
                from sentry_sdk.integrations.logging import LoggingIntegration
                
                sentry_logging = LoggingIntegration(
                    level=logging.INFO,        # Capture info and above as breadcrumbs
                    event_level=logging.ERROR  # Send errors as events
                )
                
                sentry_sdk.init(
                    dsn=sentry_dsn,
                    integrations=[sentry_logging],
                    traces_sample_rate=0.1,  # Capture 10% of transactions for performance monitoring
                    environment=os.getenv('ENVIRONMENT', 'production'),
                    release=os.getenv('APP_VERSION', '1.0.0')
                )
                
                self.sentry_enabled = True
                logger.info("Sentry error monitoring initialized")
            else:
                logger.info("Sentry not configured (SENTRY_DSN not set)")
                
        except ImportError:
            logger.warning("Sentry SDK not installed - error monitoring disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
    
    def capture_exception(self, error: Exception, context: Optional[Dict[str, Any]] = None, 
                         user_context: Optional[Dict[str, Any]] = None):
        """Capture exception with context for monitoring"""
        self.error_count += 1
        
        # Log error locally
        logger.error(f"Exception captured: {type(error).__name__}: {error}", exc_info=True)
        
        # Send to Sentry if enabled
        if self.sentry_enabled:
            try:
                import sentry_sdk
                
                with sentry_sdk.configure_scope() as scope:
                    if context:
                        for key, value in context.items():
                            scope.set_extra(key, value)
                    
                    if user_context:
                        scope.set_user(user_context)
                    
                    # Add Streamlit session info if available
                    if hasattr(st, 'session_state') and st.session_state:
                        scope.set_extra("streamlit_session", {
                            "session_id": getattr(st.session_state, 'session_id', 'unknown'),
                            "authenticated": getattr(st.session_state, 'authenticated', False),
                            "current_page": st.session_state.get('current_page', 'unknown')
                        })
                
                sentry_sdk.capture_exception(error)
                
            except Exception as sentry_error:
                logger.error(f"Failed to send error to Sentry: {sentry_error}")
    
    def show_user_error(self, title: str, message: str, error: Optional[Exception] = None,
                       show_details: bool = False, error_id: Optional[str] = None):
        """Display user-friendly error message in Streamlit"""
        
        # Generate error ID if not provided
        if not error_id:
            error_id = f"ERR_{int(datetime.now().timestamp())}"
        
        st.error(f"**{title}**")
        st.write(message)
        
        if error and show_details:
            with st.expander("🔍 Technical Details", expanded=False):
                st.code(f"{type(error).__name__}: {error}")
                if hasattr(error, '__traceback__'):
                    st.code(traceback.format_exc())
        
        # Show error ID for support
        st.caption(f"Error ID: `{error_id}` • If this persists, please contact support with this ID.")
        
        # Log with error ID
        logger.error(f"User error displayed [ID: {error_id}]: {title} - {message}")
        
        return error_id
    
    def get_user_friendly_message(self, error: Exception) -> tuple[str, str]:
        """Convert technical error to user-friendly title and message"""
        error_type = type(error).__name__
        error_str = str(error).lower()
        
        # Database errors
        if 'database' in error_str or 'sqlite' in error_str:
            return (
                "Database Connection Issue",
                "We're having trouble accessing the database. This might be temporary - please try again in a moment."
            )
        
        # Network/API errors
        if 'connection' in error_str or 'timeout' in error_str or 'network' in error_str:
            return (
                "Connection Problem",
                "We're having trouble connecting to our services. Please check your internet connection and try again."
            )
        
        # File system errors
        if 'permission' in error_str or 'file not found' in error_str:
            return (
                "File Access Issue",
                "We couldn't access some required files. Please make sure the application has proper permissions."
            )
        
        # Memory errors
        if 'memory' in error_str or error_type == 'MemoryError':
            return (
                "Resource Limitation",
                "The system is running low on memory. Try reducing the size of your request or try again later."
            )
        
        # GPU/CUDA errors
        if 'cuda' in error_str or 'gpu' in error_str:
            return (
                "GPU Processing Issue",
                "There's a problem with GPU processing. The system will try to use CPU instead."
            )
        
        # Training errors
        if 'training' in error_str or 'model' in error_str:
            return (
                "Training Issue",
                "There was a problem during model training. Please check your configuration and try again."
            )
        
        # Authentication errors
        if 'auth' in error_str or 'token' in error_str or 'permission' in error_str:
            return (
                "Authentication Problem",
                "There's an issue with your authentication. Please try logging in again."
            )
        
        # Generic errors
        return (
            "Unexpected Error",
            "Something went wrong that we didn't expect. Our team has been notified and will investigate."
        )


def streamlit_error_boundary(func: Callable) -> Callable:
    """Decorator to wrap Streamlit pages with error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            error_handler = get_error_handler()
            
            # Get user context if available
            user_context = None
            if hasattr(st.session_state, 'current_user') and st.session_state.current_user:
                user_context = {
                    "user_id": getattr(st.session_state.current_user, 'id', None),
                    "email": getattr(st.session_state.current_user, 'email', None),
                    "role": getattr(st.session_state.current_user, 'role', None)
                }
            
            # Capture the error
            context = {
                "function": func.__name__,
                "module": func.__module__,
                "args": str(args)[:500],  # Limit length
                "kwargs": str(kwargs)[:500]
            }
            
            error_handler.capture_exception(error, context=context, user_context=user_context)
            
            # Show user-friendly error
            title, message = error_handler.get_user_friendly_message(error)
            error_id = error_handler.show_user_error(
                title=title,
                message=message,
                error=error,
                show_details=os.getenv('ENVIRONMENT', 'production') != 'production'
            )
            
            # Add recovery options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Try Again", type="primary"):
                    st.rerun()
            
            with col2:
                if st.button("🏠 Go Home"):
                    st.switch_page("app.py")
            
            with col3:
                if st.button("📞 Contact Support"):
                    st.switch_page("pages/support.py")
            
            return None
    
    return wrapper


def safe_execute(func: Callable, fallback_value=None, error_message: str = "Operation failed") -> Any:
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as error:
        error_handler = get_error_handler()
        error_handler.capture_exception(error, context={"operation": func.__name__})
        
        # Show user notification
        st.warning(f"⚠️ {error_message}")
        
        return fallback_value


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def setup_global_error_handling():
    """Setup global error handling for the application"""
    error_handler = get_error_handler()
    
    # Override default exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error_handler.capture_exception(exc_value, context={
            "type": exc_type.__name__,
            "traceback": ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        })
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    logger.info("Global error handling configured") 
"""
Authentication Middleware

Provides decorators and utilities for protecting Streamlit pages
and checking user permissions in a clean, reusable way.
"""

import streamlit as st
import functools
import logging
from typing import Callable, List, Optional, Any
from .models import User, UserRole
from .auth_manager import AuthManager

logger = logging.getLogger(__name__)


def require_auth(redirect_page: str = "pages/login.py"):
    """
    Decorator to require authentication for a Streamlit page
    
    Args:
        redirect_page: Page to redirect to if not authenticated
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated', False):
                st.warning("⚠️ Please log in to access this page.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔐 Login", type="primary"):
                        st.switch_page(redirect_page)
                with col2:
                    if st.button("📝 Register"):
                        st.switch_page("pages/register.py")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(required_roles: List[UserRole], redirect_page: str = "pages/login.py"):
    """
    Decorator to require specific roles for a Streamlit page
    
    Args:
        required_roles: List of roles that can access the page
        redirect_page: Page to redirect to if access denied
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First check authentication
            if not st.session_state.get('authenticated', False):
                st.warning("⚠️ Please log in to access this page.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔐 Login", type="primary"):
                        st.switch_page(redirect_page)
                with col2:
                    if st.button("📝 Register"):
                        st.switch_page("pages/register.py")
                return None
            
            # Check role permissions
            current_user = st.session_state.get('current_user')
            if not current_user or current_user.role not in required_roles:
                st.error("❌ You don't have permission to access this page.")
                
                role_names = [role.value.title() for role in required_roles]
                st.info(f"This page requires one of the following roles: {', '.join(role_names)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏠 Back to Dashboard"):
                        st.switch_page("app.py")
                with col2:
                    if st.button("👤 View Profile"):
                        st.switch_page("pages/profile.py")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(permission: str, redirect_page: str = "pages/login.py"):
    """
    Decorator to require specific permission for a Streamlit page
    
    Args:
        permission: Permission string required to access the page
        redirect_page: Page to redirect to if access denied
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First check authentication
            if not st.session_state.get('authenticated', False):
                st.warning("⚠️ Please log in to access this page.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔐 Login", type="primary"):
                        st.switch_page(redirect_page)
                with col2:
                    if st.button("📝 Register"):
                        st.switch_page("pages/register.py")
                return None
            
            # Check specific permission
            current_user = st.session_state.get('current_user')
            auth_manager = st.session_state.get('auth_manager')
            
            if not current_user or not auth_manager:
                st.error("❌ Authentication error. Please log in again.")
                return None
                
            if not auth_manager.check_permission(current_user, permission):
                st.error(f"❌ You don't have permission to {permission.replace('_', ' ')}.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏠 Back to Dashboard"):
                        st.switch_page("app.py")
                with col2:
                    if st.button("👤 View Profile"):
                        st.switch_page("pages/profile.py")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_current_user() -> Optional[User]:
    """Get the currently authenticated user"""
    return st.session_state.get('current_user')


def is_authenticated() -> bool:
    """Check if user is currently authenticated"""
    return st.session_state.get('authenticated', False)


def has_role(role: UserRole) -> bool:
    """Check if current user has a specific role"""
    current_user = get_current_user()
    return current_user and current_user.role == role


def has_permission(permission: str) -> bool:
    """Check if current user has a specific permission"""
    current_user = get_current_user()
    auth_manager = st.session_state.get('auth_manager')
    
    if not current_user or not auth_manager:
        return False
        
    return auth_manager.check_permission(current_user, permission)


def logout_user():
    """Logout the current user"""
    try:
        # Get current JWT ID for session invalidation
        if 'access_token' in st.session_state and 'auth_manager' in st.session_state:
            # In production, we'd decode the token to get JWT ID and invalidate session
            pass
        
        # Clear authentication state
        auth_keys = ['authenticated', 'current_user', 'access_token', 'refresh_token']
        for key in auth_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Logged out successfully!")
        logger.info("User logged out")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Logout failed: {str(e)}")
        logger.error(f"Logout error: {e}")
        return False


def init_auth_manager():
    """Initialize the auth manager in session state"""
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()


def render_auth_sidebar():
    """Render authentication status in sidebar"""
    init_auth_manager()
    
    st.sidebar.markdown("---")
    
    if is_authenticated():
        current_user = get_current_user()
        
        # User info
        st.sidebar.markdown(f"**👤 {current_user.username}**")
        st.sidebar.markdown(f"*{current_user.role.value.title()}*")
        
        # User actions
        if st.sidebar.button("👤 Profile", use_container_width=True):
            st.switch_page("pages/profile.py")
            
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
    else:
        # Login/Register buttons
        st.sidebar.markdown("**🔐 Authentication**")
        
        if st.sidebar.button("🔐 Login", use_container_width=True):
            st.switch_page("pages/login.py")
            
        if st.sidebar.button("📝 Register", use_container_width=True):
            st.switch_page("pages/register.py")


def render_role_based_navigation():
    """Render navigation based on user role"""
    if not is_authenticated():
        return
    
    current_user = get_current_user()
    
    if current_user.role == UserRole.CREATOR:
        st.sidebar.markdown("### 🎨 Creator Tools")
        
        creator_pages = [
            ("🎭 Character Studio", [
                ("📁 Character Upload", "pages/character_upload.py"),
                ("🗨️ Conversational Builder", "pages/character_builder.py"),
                ("📋 Character Management", "pages/character_management.py"),
            ]),
            ("🌍 World & Data", [
                ("🌍 World Management", "pages/world_management.py"),
                ("🎨 Dataset Studio", "pages/dataset_studio.py"),
            ]),
            ("📊 Training & Testing", [
                ("⚙️ Training Config", "pages/training_config.py"),
                ("📊 Training Dashboard", "pages/training_dashboard.py"),
                ("💬 Character Chat", "pages/character_chat.py"),
                ("🧪 Model Testing", "pages/model_testing.py"),
                ("⚔️ Model Comparison", "pages/model_comparison.py"),
                ("🔧 Model Management", "pages/model_management.py"),
            ])
        ]
        
        for section_name, pages in creator_pages:
            with st.sidebar.expander(section_name):
                for page_name, page_path in pages:
                    if st.button(page_name, key=f"nav_{page_path}"):
                        st.switch_page(page_path)
    
    elif current_user.role == UserRole.PLAYER:
        st.sidebar.markdown("### 🎮 Player Features")
        
        player_pages = [
            ("🌍 Discover Worlds", "pages/world_discovery.py"),
            ("🎮 Play Characters", "pages/character_selection.py"),
            ("💬 Chat History", "pages/conversation_history.py"),
            ("⭐ My Reviews", "pages/my_reviews.py"),
        ]
        
        for page_name, page_path in player_pages:
            if st.sidebar.button(page_name, key=f"nav_{page_path}"):
                # For now, show coming soon message since these pages don't exist yet
                st.info(f"🚧 {page_name} coming soon!")
    
    elif current_user.role == UserRole.ADMIN:
        st.sidebar.markdown("### ⚙️ Admin Panel")
        
        admin_pages = [
            ("👥 User Management", "pages/admin_users.py"),
            ("📊 Platform Analytics", "pages/admin_analytics.py"),
            ("🛡️ Content Moderation", "pages/admin_moderation.py"),
            ("⚙️ System Settings", "pages/admin_settings.py"),
        ]
        
        for page_name, page_path in admin_pages:
            if st.sidebar.button(page_name, key=f"nav_{page_path}"):
                # For now, show coming soon message since these pages don't exist yet
                st.info(f"🚧 {page_name} coming soon!")


def check_token_validity():
    """Check if the current access token is still valid"""
    if not is_authenticated():
        return False
    
    access_token = st.session_state.get('access_token')
    auth_manager = st.session_state.get('auth_manager')
    
    if not access_token or not auth_manager:
        return False
    
    try:
        result = auth_manager.verify_token(access_token)
        if not result.valid:
            # Token is invalid, try to refresh
            refresh_token = st.session_state.get('refresh_token')
            if refresh_token:
                refresh_result = auth_manager.refresh_token(refresh_token)
                if refresh_result.success:
                    # Update tokens in session state
                    st.session_state.access_token = refresh_result.access_token
                    st.session_state.refresh_token = refresh_result.refresh_token
                    return True
                else:
                    # Refresh failed, logout user
                    logout_user()
                    st.warning("⚠️ Your session has expired. Please log in again.")
                    return False
        
        return result.valid
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return False 
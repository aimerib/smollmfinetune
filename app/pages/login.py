"""
🔐 Login Page

User authentication page with clean, modern design and comprehensive
error handling. Integrates with AuthManager for secure login.
"""

import streamlit as st
import logging
from typing import Optional

# Import auth components
try:
    from utils.auth import AuthManager
except ImportError:
    from app.utils.auth import AuthManager

logger = logging.getLogger(__name__)


def page_login():
    """Render the login page"""
    
    # Page configuration
    st.set_page_config(
        page_title="Login - Character Platform",
        page_icon="🔐",
        layout="centered"
    )
    
    # Initialize auth manager
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    # Check if already logged in
    if st.session_state.get('authenticated', False):
        st.success("✅ You are already logged in!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Go to Dashboard", type="primary"):
                st.switch_page("app.py")
        with col2:
            if st.button("🚪 Logout"):
                logout_user()
                st.rerun()
        return
    
    # Main login interface
    render_login_form()


def render_login_form():
    """Render the main login form"""
    
    # Header with branding
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>🎭 Character Platform</h1>
            <h2>🔐 Sign In</h2>
            <p style="color: #64748b;">Welcome back! Sign in to access your characters and worlds.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Login form
    with st.container():
        # Add some spacing and styling
        st.markdown("""
            <style>
            .login-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 2rem;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                background: #ffffff;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            # Email input
            email = st.text_input(
                "📧 Email Address",
                placeholder="Enter your email address",
                help="The email address you used when registering"
            )
            
            # Password input
            password = st.text_input(
                "🔑 Password",
                type="password",
                placeholder="Enter your password",
                help="Your account password"
            )
            
            # Remember me checkbox (future feature)
            col1, col2 = st.columns([1, 1])
            with col1:
                remember_me = st.checkbox("🔄 Remember me")
            
            # Login button
            submitted = st.form_submit_button(
                "🔐 Sign In",
                type="primary",
                use_container_width=True
            )
            
            # Handle form submission
            if submitted:
                handle_login(email, password, remember_me)
    
    # Additional options
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Create Account", use_container_width=True):
            st.switch_page("pages/register.py")
    
    with col2:
        if st.button("🔄 Forgot Password", use_container_width=True):
            st.info("🚧 Password reset coming soon!")
    
    # Demo accounts section (for development)
    with st.expander("🧪 Demo Accounts (Development)"):
        st.info("""
        **Demo Creator Account:**
        - Email: `creator@demo.com`
        - Password: `demo123456`
        
        **Demo Player Account:**
        - Email: `player@demo.com`
        - Password: `demo123456`
        
        *Note: These accounts are created automatically for testing purposes.*
        """)
        
        if st.button("🎭 Create Demo Accounts"):
            create_demo_accounts()


def handle_login(email: str, password: str, remember_me: bool = False):
    """Handle login form submission"""
    
    # Validate input
    if not email or not password:
        st.error("❌ Please enter both email and password.")
        return
    
    if not email.count("@") == 1:
        st.error("❌ Please enter a valid email address.")
        return
    
    # Show loading spinner
    with st.spinner("🔐 Signing in..."):
        try:
            # Attempt authentication
            auth_manager = st.session_state.auth_manager
            result = auth_manager.authenticate_user(email, password)
            
            if result.success:
                # Store authentication state
                st.session_state.authenticated = True
                st.session_state.current_user = result.user
                st.session_state.access_token = result.access_token
                st.session_state.refresh_token = result.refresh_token
                
                # Log successful login
                logger.info(f"User logged in successfully: {email}")
                
                # Show success message
                st.success(f"✅ Welcome back, {result.user.username}!")
                
                # Add role-specific welcome message
                role_messages = {
                    "creator": "🎨 Ready to create amazing characters!",
                    "player": "🎮 Time to explore new worlds!",
                    "admin": "⚙️ Admin dashboard awaits you!"
                }
                
                role_msg = role_messages.get(result.user.role.value, "🎭 Welcome to the platform!")
                st.info(role_msg)
                
                # Redirect after successful login
                st.balloons()
                st.rerun()
                
            else:
                # Show error message
                st.error(f"❌ {result.error_message}")
                logger.warning(f"Login failed for {email}: {result.error_message}")
                
        except Exception as e:
            st.error(f"❌ Login failed: {str(e)}")
            logger.error(f"Login error: {e}")


def logout_user():
    """Handle user logout"""
    try:
        # Get current JWT ID for session invalidation
        if 'access_token' in st.session_state:
            # In a production app, we'd decode the token to get JWT ID
            # For now, we'll just clear the session state
            pass
        
        # Clear authentication state
        for key in ['authenticated', 'current_user', 'access_token', 'refresh_token']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Logged out successfully!")
        logger.info("User logged out")
        
    except Exception as e:
        st.error(f"❌ Logout failed: {str(e)}")
        logger.error(f"Logout error: {e}")


def create_demo_accounts():
    """Create demo accounts for testing"""
    try:
        auth_manager = st.session_state.auth_manager
        
        # Create creator demo account
        creator_result = auth_manager.register_user(
            email="creator@demo.com",
            username="demo_creator",
            password="demo123456",
            role=auth_manager.permissions.get("CREATOR", "creator")
        )
        
        # Create player demo account
        player_result = auth_manager.register_user(
            email="player@demo.com",
            username="demo_player",
            password="demo123456",
            role=auth_manager.permissions.get("PLAYER", "player")
        )
        
        success_messages = []
        if creator_result.success:
            success_messages.append("✅ Creator demo account created")
        if player_result.success:
            success_messages.append("✅ Player demo account created")
        
        if success_messages:
            for msg in success_messages:
                st.success(msg)
        else:
            st.info("ℹ️ Demo accounts may already exist")
            
    except Exception as e:
        st.error(f"❌ Failed to create demo accounts: {str(e)}")
        logger.error(f"Demo account creation error: {e}")


# Run the page
if __name__ == "__main__":
    page_login() 
"""
📝 Registration Page

User registration page with comprehensive form validation,
role selection, and integration with AuthManager.
"""

import streamlit as st
import re
import logging
from typing import Optional

# Import auth components
try:
    from utils.auth import AuthManager, UserRole
except ImportError:
    from app.utils.auth import AuthManager, UserRole

logger = logging.getLogger(__name__)


def page_register():
    """Render the registration page"""
    
    # Page configuration
    st.set_page_config(
        page_title="Register - Character Platform",
        page_icon="📝",
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
    
    # Main registration interface
    render_registration_form()


def render_registration_form():
    """Render the main registration form"""
    
    # Header with branding
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>🎭 Character Platform</h1>
            <h2>📝 Create Account</h2>
            <p style="color: #64748b;">Join the platform to create characters or explore worlds!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Registration form
    with st.container():
        # Add styling
        st.markdown("""
            <style>
            .register-container {
                max-width: 500px;
                margin: 0 auto;
                padding: 2rem;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                background: #ffffff;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        with st.form("register_form", clear_on_submit=False):
            # Email input
            email = st.text_input(
                "📧 Email Address",
                placeholder="Enter your email address",
                help="We'll use this for login and important notifications"
            )
            
            # Username input
            username = st.text_input(
                "👤 Username",
                placeholder="Choose a unique username",
                help="This will be your display name on the platform"
            )
            
            # Password inputs
            password = st.text_input(
                "🔑 Password",
                type="password",
                placeholder="Create a strong password",
                help="Must be at least 8 characters long"
            )
            
            confirm_password = st.text_input(
                "🔑 Confirm Password",
                type="password",
                placeholder="Re-enter your password",
                help="Must match the password above"
            )
            
            # Role selection
            st.subheader("🎭 Account Type")
            
            role_option = st.radio(
                "Choose your account type:",
                ["Creator", "Player"],
                help="You can change this later in your profile settings"
            )
            
            # Show role descriptions
            if role_option == "Creator":
                st.info("""
                **🎨 Creator Account** - Perfect for:
                - Building and training AI characters
                - Creating immersive worlds and lore
                - Managing datasets and training pipelines
                - Exporting characters for deployment
                """)
            else:
                st.info("""
                **🎮 Player Account** - Perfect for:
                - Discovering and exploring worlds
                - Interacting with AI characters
                - Saving conversation history
                - Rating and reviewing characters
                """)
            
            # Terms and conditions
            agree_terms = st.checkbox(
                "✅ I agree to the Terms of Service and Privacy Policy",
                help="Required to create an account"
            )
            
            # Registration button
            submitted = st.form_submit_button(
                "📝 Create Account",
                type="primary",
                use_container_width=True
            )
            
            # Handle form submission
            if submitted:
                handle_registration(
                    email, username, password, confirm_password, 
                    role_option, agree_terms
                )
    
    # Additional options
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔐 Already have an account?", use_container_width=True):
            st.switch_page("pages/login.py")
    
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("app.py")


def handle_registration(
    email: str, 
    username: str, 
    password: str, 
    confirm_password: str,
    role_option: str,
    agree_terms: bool
):
    """Handle registration form submission"""
    
    # Validate input
    validation_errors = validate_registration_input(
        email, username, password, confirm_password, agree_terms
    )
    
    if validation_errors:
        for error in validation_errors:
            st.error(f"❌ {error}")
        return
    
    # Convert role option to UserRole
    role = UserRole.CREATOR if role_option == "Creator" else UserRole.PLAYER
    
    # Show loading spinner
    with st.spinner("📝 Creating your account..."):
        try:
            # Attempt registration
            auth_manager = st.session_state.auth_manager
            result = auth_manager.register_user(
                email=email,
                username=username,
                password=password,
                role=role
            )
            
            if result.success:
                # Show success message
                st.success("✅ Account created successfully!")
                
                # Add role-specific welcome message
                role_messages = {
                    UserRole.CREATOR: "🎨 Welcome to the creator community! You can now build amazing characters.",
                    UserRole.PLAYER: "🎮 Welcome to the platform! Start exploring worlds and characters."
                }
                
                role_msg = role_messages.get(role, "🎭 Welcome to the platform!")
                st.info(role_msg)
                
                # Auto-login after registration
                st.info("🔐 Signing you in...")
                
                auth_result = auth_manager.authenticate_user(email, password)
                if auth_result.success:
                    # Store authentication state
                    st.session_state.authenticated = True
                    st.session_state.current_user = auth_result.user
                    st.session_state.access_token = auth_result.access_token
                    st.session_state.refresh_token = auth_result.refresh_token
                    
                    logger.info(f"User registered and logged in: {email}")
                    
                    # Show success and redirect
                    st.balloons()
                    st.rerun()
                else:
                    # Registration succeeded but auto-login failed
                    st.warning("⚠️ Account created but auto-login failed. Please sign in manually.")
                    if st.button("🔐 Go to Login"):
                        st.switch_page("pages/login.py")
                
            else:
                # Show registration error
                st.error(f"❌ {result.error_message}")
                logger.warning(f"Registration failed for {email}: {result.error_message}")
                
        except Exception as e:
            st.error(f"❌ Registration failed: {str(e)}")
            logger.error(f"Registration error: {e}")


def validate_registration_input(
    email: str, 
    username: str, 
    password: str, 
    confirm_password: str,
    agree_terms: bool
) -> list[str]:
    """Validate registration form input"""
    
    errors = []
    
    # Check required fields
    if not email:
        errors.append("Email address is required")
    elif not is_valid_email(email):
        errors.append("Please enter a valid email address")
    
    if not username:
        errors.append("Username is required")
    elif len(username) < 3:
        errors.append("Username must be at least 3 characters long")
    elif not is_valid_username(username):
        errors.append("Username can only contain letters, numbers, and underscores")
    
    if not password:
        errors.append("Password is required")
    elif len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    elif not is_strong_password(password):
        errors.append("Password must contain at least one letter and one number")
    
    if not confirm_password:
        errors.append("Please confirm your password")
    elif password != confirm_password:
        errors.append("Passwords don't match")
    
    if not agree_terms:
        errors.append("You must agree to the Terms of Service")
    
    return errors


def is_valid_email(email: str) -> bool:
    """Check if email is valid format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def is_valid_username(username: str) -> bool:
    """Check if username contains only valid characters"""
    pattern = r'^[a-zA-Z0-9_]+$'
    return re.match(pattern, username) is not None


def is_strong_password(password: str) -> bool:
    """Check if password meets strength requirements"""
    has_letter = any(c.isalpha() for c in password)
    has_number = any(c.isdigit() for c in password)
    return has_letter and has_number


def logout_user():
    """Handle user logout"""
    try:
        # Clear authentication state
        for key in ['authenticated', 'current_user', 'access_token', 'refresh_token']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Logged out successfully!")
        logger.info("User logged out")
        
    except Exception as e:
        st.error(f"❌ Logout failed: {str(e)}")
        logger.error(f"Logout error: {e}")


# Run the page
if __name__ == "__main__":
    page_register() 
"""
👤 User Profile Page

User profile management page with account information,
settings, and role-based features.
"""

import streamlit as st
import logging
from datetime import datetime
from typing import Optional

# Import auth components
try:
    from utils.auth import AuthManager, UserRole
except ImportError:
    from app.utils.auth import AuthManager, UserRole

logger = logging.getLogger(__name__)


def page_profile():
    """Render the user profile page"""
    
    # Check authentication
    if not st.session_state.get('authenticated', False):
        st.warning("⚠️ Please log in to access your profile.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Login", type="primary"):
                st.switch_page("pages/login.py")
        with col2:
            if st.button("📝 Register"):
                st.switch_page("pages/register.py")
        return
    
    # Get current user
    current_user = st.session_state.get('current_user')
    if not current_user:
        st.error("❌ User information not found. Please log in again.")
        return
    
    # Page header
    st.markdown('<h2 class="gradient-text">👤 User Profile</h2>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Account Info", 
        "⚙️ Settings", 
        "📊 Activity", 
        "🔒 Security"
    ])
    
    with tab1:
        render_account_info(current_user)
    
    with tab2:
        render_account_settings(current_user)
    
    with tab3:
        render_activity_overview(current_user)
    
    with tab4:
        render_security_settings(current_user)


def render_account_info(user):
    """Render account information section"""
    
    st.subheader("📋 Account Information")
    
    # User info display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # User avatar placeholder
        st.markdown("""
            <div style="
                width: 120px; 
                height: 120px; 
                border-radius: 50%; 
                background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto;
                font-size: 48px;
                color: white;
            ">
                🎭
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<h3 style='text-align: center; margin-top: 1rem;'>{user.username}</h3>", 
                   unsafe_allow_html=True)
    
    with col2:
        # Account details
        st.markdown("**Account Details:**")
        
        # Create info table
        info_data = {
            "Username": user.username,
            "Email": user.email,
            "Role": user.role.value.title(),
            "Status": "Active" if user.is_active else "Inactive",
            "Email Verified": "✅ Yes" if user.email_verified else "❌ No",
            "Member Since": user.created_at.strftime("%B %d, %Y") if user.created_at else "Unknown",
            "Last Login": user.last_login.strftime("%B %d, %Y at %I:%M %p") if user.last_login else "Never"
        }
        
        for key, value in info_data.items():
            st.write(f"**{key}:** {value}")
    
    # Role-specific information
    st.markdown("---")
    render_role_specific_info(user)


def render_role_specific_info(user):
    """Render role-specific account information"""
    
    if user.role == UserRole.CREATOR:
        st.subheader("🎨 Creator Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Characters Created",
                value="0",  # TODO: Get from database
                help="Total number of characters you've created"
            )
        
        with col2:
            st.metric(
                label="Training Runs",
                value="0",  # TODO: Get from database
                help="Total number of training sessions completed"
            )
        
        with col3:
            st.metric(
                label="Worlds Managed",
                value="0",  # TODO: Get from database
                help="Number of worlds you're managing"
            )
        
        st.info("""
        **Creator Features Available:**
        - 🎭 Character Creation & Management
        - 🌍 World Building & Lore Management
        - 🎨 Dataset Studio & Training Pipeline
        - 📊 Advanced Analytics & Metrics
        - 📦 Character Export & Deployment
        """)
    
    elif user.role == UserRole.PLAYER:
        st.subheader("🎮 Player Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Conversations",
                value="0",  # TODO: Get from database
                help="Total conversations with characters"
            )
        
        with col2:
            st.metric(
                label="Worlds Explored",
                value="0",  # TODO: Get from database
                help="Number of worlds you've explored"
            )
        
        with col3:
            st.metric(
                label="Characters Met",
                value="0",  # TODO: Get from database
                help="Unique characters you've interacted with"
            )
        
        st.info("""
        **Player Features Available:**
        - 🌍 World Discovery & Exploration
        - 🎮 Character Interaction & Chat
        - 💾 Conversation History & Bookmarks
        - ⭐ Character Rating & Reviews
        """)
    
    elif user.role == UserRole.ADMIN:
        st.subheader("⚙️ Admin Dashboard")
        
        st.info("""
        **Admin Features Available:**
        - 👥 User Management & Role Assignment
        - 📊 Platform Analytics & Monitoring
        - 🛡️ Content Moderation & Safety
        - ⚙️ System Configuration & Maintenance
        """)
        
        if st.button("🛡️ Go to Admin Panel"):
            st.info("🚧 Admin panel coming soon!")


def render_account_settings(user):
    """Render account settings section"""
    
    st.subheader("⚙️ Account Settings")
    
    # Username update
    with st.expander("✏️ Update Username"):
        new_username = st.text_input(
            "New Username",
            value=user.username,
            help="Choose a new username for your account"
        )
        
        if st.button("Update Username"):
            if new_username and new_username != user.username:
                st.info("🚧 Username update coming soon!")
            else:
                st.warning("Please enter a different username")
    
    # Email update
    with st.expander("📧 Update Email"):
        new_email = st.text_input(
            "New Email Address",
            value=user.email,
            help="Update your email address"
        )
        
        if st.button("Update Email"):
            if new_email and new_email != user.email:
                st.info("🚧 Email update coming soon!")
            else:
                st.warning("Please enter a different email address")
    
    # Role change request
    with st.expander("🎭 Change Account Type"):
        current_role_display = user.role.value.title()
        
        st.write(f"**Current Role:** {current_role_display}")
        
        new_role = st.selectbox(
            "Request Role Change",
            ["Creator", "Player"],
            index=0 if user.role == UserRole.CREATOR else 1
        )
        
        if st.button("Request Role Change"):
            if new_role.lower() != user.role.value:
                st.info("🚧 Role change requests coming soon!")
            else:
                st.info("You already have this role")
    
    # Notification preferences
    with st.expander("🔔 Notification Preferences"):
        email_notifications = st.checkbox(
            "📧 Email Notifications",
            value=True,
            help="Receive email notifications for important updates"
        )
        
        training_notifications = st.checkbox(
            "🎯 Training Completion Alerts",
            value=True,
            help="Get notified when character training completes"
        )
        
        new_features = st.checkbox(
            "✨ New Features Announcements",
            value=True,
            help="Stay updated on new platform features"
        )
        
        if st.button("Save Notification Preferences"):
            st.success("✅ Notification preferences saved!")


def render_activity_overview(user):
    """Render activity overview section"""
    
    st.subheader("📊 Activity Overview")
    
    # Recent activity placeholder
    st.info("🚧 Activity tracking coming soon!")
    
    # Placeholder metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Days Active", "0")
        st.metric("Average Session Length", "0 min")
    
    with col2:
        st.metric("Last Active", "Today")
        st.metric("Longest Session", "0 min")
    
    # Activity chart placeholder
    st.subheader("📈 Activity Chart")
    st.info("Activity visualization will be available soon!")


def render_security_settings(user):
    """Render security settings section"""
    
    st.subheader("🔒 Security Settings")
    
    # Password change
    with st.expander("🔑 Change Password"):
        with st.form("change_password_form"):
            current_password = st.text_input(
                "Current Password",
                type="password",
                help="Enter your current password"
            )
            
            new_password = st.text_input(
                "New Password",
                type="password",
                help="Must be at least 8 characters long"
            )
            
            confirm_new_password = st.text_input(
                "Confirm New Password",
                type="password",
                help="Re-enter your new password"
            )
            
            if st.form_submit_button("Change Password"):
                if not all([current_password, new_password, confirm_new_password]):
                    st.error("❌ Please fill in all password fields")
                elif new_password != confirm_new_password:
                    st.error("❌ New passwords don't match")
                elif len(new_password) < 8:
                    st.error("❌ New password must be at least 8 characters long")
                else:
                    st.info("🚧 Password change coming soon!")
    
    # Active sessions
    with st.expander("📱 Active Sessions"):
        st.write("**Current Session:**")
        
        session_info = {
            "Device": "Web Browser",
            "Location": "Unknown",  # Would be populated from IP geolocation
            "Started": user.last_login.strftime("%B %d, %Y at %I:%M %p") if user.last_login else "Unknown",
            "Status": "Active"
        }
        
        for key, value in session_info.items():
            st.write(f"**{key}:** {value}")
        
        if st.button("🚪 Sign Out All Devices"):
            st.info("🚧 Session management coming soon!")
    
    # Account deletion
    with st.expander("⚠️ Delete Account", expanded=False):
        st.warning("""
        **Warning:** Account deletion is permanent and cannot be undone.
        
        This will:
        - Delete your account and all associated data
        - Remove all characters and worlds you've created
        - Cancel any ongoing training sessions
        - Permanently delete conversation history
        """)
        
        if st.checkbox("I understand the consequences"):
            delete_confirmation = st.text_input(
                "Type 'DELETE' to confirm:",
                help="Type DELETE in capital letters to confirm account deletion"
            )
            
            if st.button("🗑️ Delete Account", type="secondary"):
                if delete_confirmation == "DELETE":
                    st.error("🚧 Account deletion coming soon!")
                else:
                    st.error("❌ Please type 'DELETE' to confirm")


# Run the page
if __name__ == "__main__":
    page_profile() 
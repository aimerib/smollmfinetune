"""
Authentication module for Character Creation Platform

Provides user authentication, authorization, and session management
for the multi-user platform functionality.
"""

from .auth_manager import AuthManager, UserRole, AuthenticationError, PermissionError
from .models import User, UserSession, SocialAccount

__all__ = [
    'AuthManager',
    'UserRole', 
    'AuthenticationError',
    'PermissionError',
    'User',
    'UserSession',
    'SocialAccount'
] 
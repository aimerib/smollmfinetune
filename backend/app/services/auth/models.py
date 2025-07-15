"""
Data models for authentication system

Defines the database schema for users, sessions, and social accounts
using a lightweight approach that can integrate with R3-1 SQLAlchemy models later.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(Enum):
    """User roles for permission management"""
    ADMIN = "admin"
    CREATOR = "creator"
    PLAYER = "player"


@dataclass
class User:
    """User model representing a platform user"""
    id: Optional[int] = None
    email: str = ""
    username: str = ""
    hashed_password: str = ""
    role: UserRole = UserRole.PLAYER
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    email_verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for database storage"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'hashed_password': self.hashed_password,
            'role': self.role.value if self.role else UserRole.PLAYER.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'email_verified': self.email_verified
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create user from dictionary (database row)"""
        return cls(
            id=data.get('id'),
            email=data.get('email', ''),
            username=data.get('username', ''),
            hashed_password=data.get('hashed_password', ''),
            role=UserRole(data.get('role', UserRole.PLAYER.value)),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            is_active=data.get('is_active', True),
            email_verified=data.get('email_verified', False)
        )


@dataclass
class UserSession:
    """User session model for JWT token management"""
    id: Optional[int] = None
    user_id: int = 0
    jwt_jti: str = ""  # JWT ID for token revocation
    access_token: str = ""
    refresh_token: str = ""
    expires_at: Optional[datetime] = None
    refresh_expires_at: Optional[datetime] = None
    device_info: str = ""
    created_at: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for database storage"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'jwt_jti': self.jwt_jti,
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'refresh_expires_at': self.refresh_expires_at.isoformat() if self.refresh_expires_at else None,
            'device_info': self.device_info,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Create session from dictionary (database row)"""
        return cls(
            id=data.get('id'),
            user_id=data.get('user_id', 0),
            jwt_jti=data.get('jwt_jti', ''),
            access_token=data.get('access_token', ''),
            refresh_token=data.get('refresh_token', ''),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            refresh_expires_at=datetime.fromisoformat(data['refresh_expires_at']) if data.get('refresh_expires_at') else None,
            device_info=data.get('device_info', ''),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            is_active=data.get('is_active', True)
        )


@dataclass
class SocialAccount:
    """Social account model for OAuth integration"""
    id: Optional[int] = None
    user_id: int = 0
    provider: str = ""  # 'google', 'discord', etc.
    provider_id: str = ""
    provider_email: str = ""
    access_token: str = ""
    refresh_token: str = ""
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert social account to dictionary for database storage"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'provider': self.provider,
            'provider_id': self.provider_id,
            'provider_email': self.provider_email,
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialAccount':
        """Create social account from dictionary (database row)"""
        return cls(
            id=data.get('id'),
            user_id=data.get('user_id', 0),
            provider=data.get('provider', ''),
            provider_id=data.get('provider_id', ''),
            provider_email=data.get('provider_email', ''),
            access_token=data.get('access_token', ''),
            refresh_token=data.get('refresh_token', ''),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        )


# Result classes for auth operations
@dataclass
class AuthResult:
    """Result of authentication operation"""
    success: bool = False
    user: Optional[User] = None
    user_id: Optional[int] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    error_message: str = ""


@dataclass
class RegistrationResult:
    """Result of user registration operation"""
    success: bool = False
    user: Optional[User] = None
    user_id: Optional[int] = None
    error_message: str = ""


@dataclass
class TokenResult:
    """Result of token verification operation"""
    valid: bool = False
    user: Optional[User] = None
    jwt_jti: Optional[str] = None
    error_message: str = ""


@dataclass
class RefreshResult:
    """Result of token refresh operation"""
    success: bool = False
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    error_message: str = "" 
"""
Authentication Manager for Character Creation Platform

Handles user registration, login, token management, and role-based permissions
using SQLite database and JWT tokens for a Streamlit-native approach.
"""

import sqlite3
import hashlib
import secrets
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from .models import (
    User, UserSession, SocialAccount, UserRole,
    AuthResult, RegistrationResult, TokenResult, RefreshResult
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class PermissionError(Exception):
    """Raised when user lacks required permissions"""
    pass


class AuthManager:
    """
    Authentication manager for the Character Creation Platform
    
    Provides comprehensive user management including:
    - User registration and authentication
    - JWT token generation and validation
    - Role-based access control
    - Session management
    - Password hashing and verification
    """
    
    def __init__(self, db_path: str = "platform.db", jwt_secret: Optional[str] = None):
        """
        Initialize AuthManager
        
        Args:
            db_path: Path to SQLite database file
            jwt_secret: Secret key for JWT token signing (auto-generated if None)
        """
        self.db_path = db_path
        self.jwt_secret = jwt_secret or self._generate_jwt_secret()
        self.access_token_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=30)
        
        # Initialize database
        self._init_database()
        
        # Permission mappings for role-based access control
        self.permissions = {
            UserRole.ADMIN: [
                "admin_panel", "manage_users", "delete_users", "change_user_roles",
                "create_character", "train_model", "manage_worlds", "delete_characters",
                "play_character", "discover_worlds", "view_analytics"
            ],
            UserRole.CREATOR: [
                "create_character", "train_model", "manage_worlds", "export_character",
                "view_training_logs", "manage_datasets", "character_analytics"
            ],
            UserRole.PLAYER: [
                "play_character", "discover_worlds", "save_conversations", "rate_characters"
            ]
        }
    
    def _generate_jwt_secret(self) -> str:
        """Generate a secure JWT secret key"""
        return secrets.token_urlsafe(32)
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'player',
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    email_verified BOOLEAN DEFAULT 0
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    jwt_jti TEXT UNIQUE NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    refresh_expires_at TEXT NOT NULL,
                    device_info TEXT,
                    created_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Social accounts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS social_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    provider_email TEXT,
                    access_token TEXT,
                    refresh_token TEXT,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(provider, provider_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_jti ON user_sessions(jwt_jti)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def _get_table_names(self) -> List[str]:
        """Get list of table names in database (for testing)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor.fetchall()]
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def _generate_jwt_token(self, user: User, token_type: str = "access") -> str:
        """Generate JWT token for user"""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        if token_type == "access":
            exp = now + self.access_token_expiry
        else:  # refresh
            exp = now + self.refresh_token_expiry
        
        payload = {
            'user_id': user.id,
            'email': user.email,
            'role': user.role.value,
            'type': token_type,
            'iat': int(now.timestamp()),
            'exp': int(exp.timestamp()),
            'jti': secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def _decode_jwt_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            # Ensure token is string, not bytes
            if isinstance(token, bytes):
                token = token.decode('utf-8')
                
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # PyJWT automatically handles expiration checking, but we can be explicit
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            if datetime.fromtimestamp(payload['exp'], timezone.utc) < now:
                raise jwt.ExpiredSignatureError("Token expired")
            
            return payload
        except jwt.ExpiredSignatureError as e:
            logger.warning(f"JWT token expired: {e}")
            raise AuthenticationError(f"Token expired: {e}")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise AuthenticationError(f"Invalid token: {e}")
    
    def register_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.PLAYER
    ) -> RegistrationResult:
        """
        Register a new user
        
        Args:
            email: User's email address
            username: Unique username
            password: Plain text password (will be hashed)
            role: User role (default: PLAYER)
            
        Returns:
            RegistrationResult with success status and user info
        """
        try:
            # Validate input
            if not email or not username or not password:
                return RegistrationResult(
                    success=False,
                    error_message="Email, username, and password are required"
                )
            
            if len(password) < 8:
                return RegistrationResult(
                    success=False,
                    error_message="Password must be at least 8 characters long"
                )
            
            # Check for existing users
            if self.get_user_by_email(email):
                return RegistrationResult(
                    success=False,
                    error_message="Email already exists"
                )
            
            if self.get_user_by_username(username):
                return RegistrationResult(
                    success=False,
                    error_message="Username already exists"
                )
            
            # Hash password
            hashed_password = self._hash_password(password)
            
            # Create user
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            user = User(
                email=email,
                username=username,
                hashed_password=hashed_password,
                role=role,
                created_at=now,
                is_active=True,
                email_verified=False
            )
            
            # Insert into database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (email, username, hashed_password, role, created_at, is_active, email_verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.email, user.username, user.hashed_password,
                    user.role.value, user.created_at.isoformat(),
                    user.is_active, user.email_verified
                ))
                user.id = cursor.lastrowid
                conn.commit()
            
            logger.info(f"User registered successfully: {email}")
            
            return RegistrationResult(
                success=True,
                user=user,
                user_id=user.id
            )
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return RegistrationResult(
                success=False,
                error_message=f"Registration failed: {str(e)}"
            )
    
    def authenticate_user(self, email: str, password: str) -> AuthResult:
        """
        Authenticate user with email and password
        
        Args:
            email: User's email address
            password: Plain text password
            
        Returns:
            AuthResult with tokens and user info if successful
        """
        try:
            # Get user from database
            user = self.get_user_by_email(email)
            if not user:
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Check if user is active
            if not user.is_active:
                return AuthResult(
                    success=False,
                    error_message="Account is disabled"
                )
            
            # Verify password
            if not self._verify_password(password, user.hashed_password):
                return AuthResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Generate tokens
            access_token = self._generate_jwt_token(user, "access")
            refresh_token = self._generate_jwt_token(user, "refresh")
            
            # Get JWT ID from access token
            access_payload = self._decode_jwt_token(access_token)
            jwt_jti = access_payload['jti']
            
            # Store session
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            session = UserSession(
                user_id=user.id,
                jwt_jti=jwt_jti,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=now + self.access_token_expiry,
                refresh_expires_at=now + self.refresh_token_expiry,
                device_info="",  # Could be populated from request headers
                created_at=now,
                is_active=True
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store session
                cursor.execute("""
                    INSERT INTO user_sessions 
                    (user_id, jwt_jti, access_token, refresh_token, expires_at, refresh_expires_at, device_info, created_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.user_id, session.jwt_jti, session.access_token, session.refresh_token,
                    session.expires_at.isoformat(), session.refresh_expires_at.isoformat(),
                    session.device_info, session.created_at.isoformat(), session.is_active
                ))
                
                # Update last login
                cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                             (now.isoformat(), user.id))
                
                conn.commit()
            
            user.last_login = now
            
            logger.info(f"User authenticated successfully: {email}")
            
            return AuthResult(
                success=True,
                user=user,
                user_id=user.id,
                access_token=access_token,
                refresh_token=refresh_token
            )
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthResult(
                success=False,
                error_message=f"Authentication failed: {str(e)}"
            )
    
    def verify_token(self, token: str) -> TokenResult:
        """
        Verify JWT access token
        
        Args:
            token: JWT access token
            
        Returns:
            TokenResult with user info if valid
        """
        try:
            # Decode token
            payload = self._decode_jwt_token(token)
            
            # Check token type
            if payload.get('type') != 'access':
                return TokenResult(
                    valid=False,
                    error_message="Invalid token type"
                )
            
            # Check if session is still active
            jwt_jti = payload['jti']
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT is_active FROM user_sessions 
                    WHERE jwt_jti = ? AND is_active = 1
                """, (jwt_jti,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    return TokenResult(
                        valid=False,
                        error_message="Session not found or inactive"
                    )
            
            # Get user
            user = self.get_user_by_id(payload['user_id'])
            if not user or not user.is_active:
                return TokenResult(
                    valid=False,
                    error_message="User not found or inactive"
                )
            
            return TokenResult(
                valid=True,
                user=user,
                jwt_jti=jwt_jti
            )
            
        except AuthenticationError as e:
            return TokenResult(
                valid=False,
                error_message=str(e)
            )
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return TokenResult(
                valid=False,
                error_message=f"Token verification failed: {str(e)}"
            )
    
    def refresh_token(self, refresh_token: str) -> RefreshResult:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: JWT refresh token
            
        Returns:
            RefreshResult with new tokens if successful
        """
        try:
            # Decode refresh token
            payload = self._decode_jwt_token(refresh_token)
            
            # Check token type
            if payload.get('type') != 'refresh':
                return RefreshResult(
                    success=False,
                    error_message="Invalid token type"
                )
            
            # Get user
            user = self.get_user_by_id(payload['user_id'])
            if not user or not user.is_active:
                return RefreshResult(
                    success=False,
                    error_message="User not found or inactive"
                )
            
            # Check if refresh token session exists and is active
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM user_sessions 
                    WHERE refresh_token = ? AND is_active = 1
                """, (refresh_token,))
                
                session_row = cursor.fetchone()
                if not session_row:
                    return RefreshResult(
                        success=False,
                        error_message="Invalid refresh token"
                    )
                
                session_id = session_row[0]
            
            # Generate new tokens
            new_access_token = self._generate_jwt_token(user, "access")
            new_refresh_token = self._generate_jwt_token(user, "refresh")
            
            # Get new JWT ID
            new_payload = self._decode_jwt_token(new_access_token)
            new_jwt_jti = new_payload['jti']
            
            # Update session with new tokens
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_sessions 
                    SET jwt_jti = ?, access_token = ?, refresh_token = ?,
                        expires_at = ?, refresh_expires_at = ?
                    WHERE id = ?
                """, (
                    new_jwt_jti, new_access_token, new_refresh_token,
                    (now + self.access_token_expiry).isoformat(),
                    (now + self.refresh_token_expiry).isoformat(),
                    session_id
                ))
                conn.commit()
            
            logger.info(f"Token refreshed for user: {user.email}")
            
            return RefreshResult(
                success=True,
                access_token=new_access_token,
                refresh_token=new_refresh_token
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return RefreshResult(
                success=False,
                error_message=f"Token refresh failed: {str(e)}"
            )
    
    def logout_user(self, jwt_jti: str) -> bool:
        """
        Logout user by invalidating session
        
        Args:
            jwt_jti: JWT ID from access token
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_sessions 
                    SET is_active = 0 
                    WHERE jwt_jti = ?
                """, (jwt_jti,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def check_permission(self, user: User, permission: str) -> bool:
        """
        Check if user has required permission
        
        Args:
            user: User object
            permission: Permission string to check
            
        Returns:
            True if user has permission
        """
        user_permissions = self.permissions.get(user.role, [])
        return permission in user_permissions
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = cursor.fetchone()
                
                if row:
                    return User.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                row = cursor.fetchone()
                
                if row:
                    return User.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                
                if row:
                    return User.from_dict(dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    def update_user_role(self, user_id: int, new_role: UserRole) -> bool:
        """Update user role (admin functionality)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users 
                    SET role = ? 
                    WHERE id = ?
                """, (new_role.value, user_id))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update user role: {e}")
            return False
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List all users (admin functionality)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM users 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                rows = cursor.fetchall()
                return [User.from_dict(dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return [] 
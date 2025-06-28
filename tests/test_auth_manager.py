"""
Unit tests for AuthManager authentication system (R3-0.5)

Tests core authentication functionality including:
- User registration and login
- Password hashing and verification
- JWT token generation and validation
- Role-based permissions
- Session management
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# These imports will be created during implementation
from app.utils.auth.auth_manager import AuthManager, UserRole, AuthenticationError, PermissionError
from app.utils.auth.models import User, UserSession


class TestAuthManager(unittest.TestCase):
    """Test suite for AuthManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Initialize AuthManager with test database
        self.auth_manager = AuthManager(db_path=self.db_path)
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary database
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_init_creates_tables(self):
        """Test that AuthManager initialization creates required database tables"""
        # Check that tables exist in database
        tables = self.auth_manager._get_table_names()
        self.assertIn('users', tables)
        self.assertIn('user_sessions', tables)
        self.assertIn('social_accounts', tables)
    
    def test_register_user_success(self):
        """Test successful user registration"""
        result = self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="secure_password123",
            role=UserRole.CREATOR
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.user_id)
        self.assertEqual(result.user.email, "test@example.com")
        self.assertEqual(result.user.username, "testuser")
        self.assertEqual(result.user.role, UserRole.CREATOR)
        
    def test_register_user_duplicate_email(self):
        """Test registration with duplicate email fails"""
        # Register first user
        self.auth_manager.register_user(
            email="test@example.com",
            username="testuser1",
            password="password123"
        )
        
        # Try to register with same email
        result = self.auth_manager.register_user(
            email="test@example.com",
            username="testuser2",
            password="password456"
        )
        
        self.assertFalse(result.success)
        self.assertIn("email already exists", result.error_message.lower())
    
    def test_register_user_duplicate_username(self):
        """Test registration with duplicate username fails"""
        # Register first user
        self.auth_manager.register_user(
            email="test1@example.com",
            username="testuser",
            password="password123"
        )
        
        # Try to register with same username
        result = self.auth_manager.register_user(
            email="test2@example.com",
            username="testuser",
            password="password456"
        )
        
        self.assertFalse(result.success)
        self.assertIn("username already exists", result.error_message.lower())
    
    def test_authenticate_user_success(self):
        """Test successful user authentication"""
        # Register user first
        self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        # Authenticate user
        result = self.auth_manager.authenticate_user(
            email="test@example.com",
            password="password123"
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.access_token)
        self.assertIsNotNone(result.refresh_token)
        self.assertEqual(result.user.email, "test@example.com")
    
    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password fails"""
        # Register user first
        self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        # Try to authenticate with wrong password
        result = self.auth_manager.authenticate_user(
            email="test@example.com",
            password="wrong_password"
        )
        
        self.assertFalse(result.success)
        self.assertIn("invalid credentials", result.error_message.lower())
    
    def test_authenticate_user_nonexistent(self):
        """Test authentication with non-existent user fails"""
        result = self.auth_manager.authenticate_user(
            email="nonexistent@example.com",
            password="password123"
        )
        
        self.assertFalse(result.success)
        self.assertIn("invalid credentials", result.error_message.lower())
    
    def test_verify_token_valid(self):
        """Test token verification with valid token"""
        # Register and authenticate user
        self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        auth_result = self.auth_manager.authenticate_user(
            email="test@example.com",
            password="password123"
        )
        
        # Verify token
        token_result = self.auth_manager.verify_token(auth_result.access_token)
        
        self.assertTrue(token_result.valid)
        self.assertEqual(token_result.user.email, "test@example.com")
    
    def test_verify_token_invalid(self):
        """Test token verification with invalid token"""
        result = self.auth_manager.verify_token("invalid_token")
        
        self.assertFalse(result.valid)
        self.assertIsNone(result.user)
    
    def test_verify_token_expired(self):
        """Test token verification with expired token"""
        # Create AuthManager with very short token expiry for testing
        test_auth_manager = AuthManager(db_path=self.db_path + "_expiry_test")
        test_auth_manager.access_token_expiry = timedelta(seconds=1)
        
        # Register and authenticate user
        test_auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        auth_result = test_auth_manager.authenticate_user(
            email="test@example.com",
            password="password123"
        )
        
        # Wait for token to expire
        import time
        time.sleep(2)
        
        # Verify expired token
        result = test_auth_manager.verify_token(auth_result.access_token)
        
        self.assertFalse(result.valid)
        self.assertIn("expired", result.error_message.lower())
    
    def test_refresh_token_valid(self):
        """Test token refresh with valid refresh token"""
        # Register and authenticate user
        self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        auth_result = self.auth_manager.authenticate_user(
            email="test@example.com",
            password="password123"
        )
        
        # Refresh token
        refresh_result = self.auth_manager.refresh_token(auth_result.refresh_token)
        
        self.assertTrue(refresh_result.success)
        self.assertIsNotNone(refresh_result.access_token)
        self.assertNotEqual(refresh_result.access_token, auth_result.access_token)
    
    def test_refresh_token_invalid(self):
        """Test token refresh with invalid refresh token"""
        result = self.auth_manager.refresh_token("invalid_refresh_token")
        
        self.assertFalse(result.success)
        self.assertIn("invalid", result.error_message.lower())
    
    def test_check_permission_allowed(self):
        """Test permission check for allowed role"""
        # Register creator user
        register_result = self.auth_manager.register_user(
            email="creator@example.com",
            username="creator",
            password="password123",
            role=UserRole.CREATOR
        )
        
        # Check creator permissions
        self.assertTrue(
            self.auth_manager.check_permission(register_result.user, "create_character")
        )
        self.assertTrue(
            self.auth_manager.check_permission(register_result.user, "train_model")
        )
    
    def test_check_permission_denied(self):
        """Test permission check for denied role"""
        # Register player user
        register_result = self.auth_manager.register_user(
            email="player@example.com",
            username="player",
            password="password123",
            role=UserRole.PLAYER
        )
        
        # Check player permissions (should not have creator permissions)
        self.assertFalse(
            self.auth_manager.check_permission(register_result.user, "create_character")
        )
        self.assertFalse(
            self.auth_manager.check_permission(register_result.user, "train_model")
        )
        self.assertTrue(
            self.auth_manager.check_permission(register_result.user, "play_character")
        )
    
    def test_get_user_by_id(self):
        """Test getting user by ID"""
        # Register user
        register_result = self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        # Get user by ID
        user = self.auth_manager.get_user_by_id(register_result.user_id)
        
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.username, "testuser")
    
    def test_get_user_by_email(self):
        """Test getting user by email"""
        # Register user
        self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123"
        )
        
        # Get user by email
        user = self.auth_manager.get_user_by_email("test@example.com")
        
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "test@example.com")
        self.assertEqual(user.username, "testuser")
    
    def test_update_user_role(self):
        """Test updating user role (admin functionality)"""
        # Register user as creator
        register_result = self.auth_manager.register_user(
            email="test@example.com",
            username="testuser",
            password="password123",
            role=UserRole.CREATOR
        )
        
        # Update to admin
        success = self.auth_manager.update_user_role(
            register_result.user_id,
            UserRole.ADMIN
        )
        
        self.assertTrue(success)
        
        # Verify role updated
        user = self.auth_manager.get_user_by_id(register_result.user_id)
        self.assertEqual(user.role, UserRole.ADMIN)


if __name__ == '__main__':
    unittest.main() 
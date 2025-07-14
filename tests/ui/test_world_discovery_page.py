"""
UI tests for World Discovery page

Tests the world discovery interface basic rendering and functionality.
"""

import pytest
from unittest.mock import Mock, patch
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.app.services.auth.models import User, UserRole


class TestWorldDiscoveryPage:
    """Test the World Discovery page UI - simplified to avoid framework limitations"""
    
    def test_discovery_page_renders_without_errors_unauthenticated(self):
        """Test that discovery page renders without errors when not authenticated"""
        mock_session_state = {
            'authenticated': False,
            'world_discovery_manager': Mock()
        }
        
        with patch('streamlit.session_state', mock_session_state):
            app = AppTest.from_file("app/pages/world_discovery.py")
            app.run()
            
            # Just verify it runs without throwing exceptions
            # The page should have some content
            assert len(app.markdown) > 0
            assert len(app.button) > 0  # Should have login button

    def test_discovery_page_renders_without_errors_authenticated(self):
        """Test that discovery page renders without errors when authenticated"""
        mock_user = Mock()
        mock_user.id = 1
        mock_user.username = "player"
        mock_user.role = UserRole.PLAYER
        
        mock_discovery_manager = Mock()
        mock_discovery_manager.get_published_worlds.return_value = []
        mock_discovery_manager.get_user_sessions.return_value = []
        
        session_state = {
            'authenticated': True,
            'user': mock_user,
            'world_discovery_manager': mock_discovery_manager
        }
        
        with patch('streamlit.session_state', session_state):
            app = AppTest.from_file("app/pages/world_discovery.py")
            app.run()
            
            # Verify it renders and the discovery manager is called
            assert len(app.markdown) > 0
            assert mock_discovery_manager.get_published_worlds.called

    def test_discovery_page_has_search_elements(self):
        """Test that discovery page contains expected search UI elements"""
        mock_user = Mock()
        mock_user.id = 1
        mock_user.role = UserRole.PLAYER
        
        mock_discovery_manager = Mock()
        mock_discovery_manager.get_published_worlds.return_value = []
        mock_discovery_manager.get_user_sessions.return_value = []
        
        session_state = {
            'authenticated': True,
            'user': mock_user,
            'world_discovery_manager': mock_discovery_manager
        }
        
        with patch('streamlit.session_state', session_state):
            app = AppTest.from_file("app/pages/world_discovery.py")
            app.run()
            
            # Verify search elements are present
            assert len(app.text_input) > 0  # Search box
            assert len(app.multiselect) > 0  # Tag filter
            assert len(app.selectbox) > 0  # Sort options

    def test_discovery_page_creator_access(self):
        """Test that creators can access the discovery interface"""
        mock_user = Mock()
        mock_user.id = 1
        mock_user.role = UserRole.CREATOR
        
        mock_discovery_manager = Mock()
        mock_discovery_manager.get_published_worlds.return_value = []
        mock_discovery_manager.get_user_sessions.return_value = []
        
        mock_world_manager = Mock()
        mock_world_manager.list_worlds.return_value = ["My World"]
        
        session_state = {
            'authenticated': True,
            'user': mock_user,
            'world_discovery_manager': mock_discovery_manager,
            'world_manager': mock_world_manager
        }
        
        with patch('streamlit.session_state', session_state):
            app = AppTest.from_file("app/pages/world_discovery.py")
            app.run()
            
            # Should render without errors and call the discovery manager
            assert len(app.markdown) > 0
            assert mock_discovery_manager.get_published_worlds.called

    def test_discovery_page_basic_structure(self):
        """Test that discovery page has the expected basic structure"""
        mock_user = Mock()
        mock_user.id = 1
        mock_user.role = UserRole.PLAYER
        
        mock_discovery_manager = Mock()
        mock_discovery_manager.get_published_worlds.return_value = []
        mock_discovery_manager.get_user_sessions.return_value = []
        
        session_state = {
            'authenticated': True,
            'user': mock_user,
            'world_discovery_manager': mock_discovery_manager
        }
        
        with patch('streamlit.session_state', session_state):
            app = AppTest.from_file("app/pages/world_discovery.py")
            app.run()
            
            # Verify basic page structure
            assert len(app.markdown) > 0  # Should have markdown content
            # Verify discovery manager interaction
            assert mock_discovery_manager.get_published_worlds.called
            assert mock_discovery_manager.get_user_sessions.called 
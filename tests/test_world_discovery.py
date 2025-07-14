"""
Unit tests for WorldDiscoveryManager

Tests world discovery, publishing, ratings, and session management functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from backend.app.services.world.world_discovery import WorldDiscoveryManager, PublishResult
from backend.app.services.auth.models import User, UserRole
from backend.app.services.world.world import WorldManager, WorldLore


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_worlds_dir():
    """Create temporary worlds directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir





@pytest.fixture
def world_manager(temp_worlds_dir):
    """Create WorldManager with test directory"""
    return WorldManager(worlds_root=temp_worlds_dir)


@pytest.fixture
def discovery_manager(temp_db, temp_worlds_dir):
    """Create WorldDiscoveryManager with test database and worlds directory"""
    return WorldDiscoveryManager(db_path=temp_db, worlds_root=temp_worlds_dir)


@pytest.fixture
def sample_user(discovery_manager):
    """Create a sample user for testing"""
    # Create a user directly in the database since WorldDiscoveryManager 
    # creates its own simple users table
    import sqlite3
    
    with sqlite3.connect(discovery_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO users (id, username, email, created_at)
            VALUES (1, 'creator', 'creator@test.com', '2024-01-01T00:00:00Z')
        """)
        conn.commit()
    
    # Return a simple User object for testing
    return User(
        id=1,
        username="creator",
        email="creator@test.com",
        role=UserRole.CREATOR,
        is_active=True
    )


@pytest.fixture
def sample_world(world_manager):
    """Create a sample world for testing"""
    world_manager.create_world("Fantasy Realm")
    lore = world_manager.load_world("Fantasy Realm")
    
    # Add some sample content
    lore.facts["setting"] = "Medieval fantasy world"
    lore.facts["magic"] = "Magic exists but is rare"
    world_manager.save_world_lore("Fantasy Realm", lore)
    
    return "Fantasy Realm", lore


class TestWorldDiscoveryManager:
    """Test WorldDiscoveryManager functionality"""
    
    def test_init_creates_discovery_tables(self, discovery_manager):
        """Test that WorldDiscoveryManager creates required database tables"""
        tables = discovery_manager._get_table_names()
        
        required_tables = [
            'published_worlds', 'world_stats', 'world_ratings', 
            'play_sessions', 'session_characters'
        ]
        
        for table in required_tables:
            assert table in tables, f"Table {table} should be created"
    
    def test_publish_world_success(self, discovery_manager, sample_user, sample_world):
        """Test successful world publishing"""
        world_name, lore = sample_world
        
        result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="An epic fantasy world with magic and adventure",
            tags=["fantasy", "magic", "adventure"],
            thumbnail_url="https://example.com/fantasy.jpg"
        )
        
        assert result.success is True
        assert result.world_id is not None
        
        # Verify world appears in discovery
        published_worlds = discovery_manager.get_published_worlds()
        assert len(published_worlds) == 1
        assert published_worlds[0]['name'] == world_name
        assert published_worlds[0]['description'] == "An epic fantasy world with magic and adventure"
    
    def test_publish_world_duplicate_fails(self, discovery_manager, sample_user, sample_world):
        """Test that publishing the same world twice fails"""
        world_name, lore = sample_world
        
        # First publish should succeed
        result1 = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="First version",
            tags=["fantasy"]
        )
        assert result1.success is True
        
        # Second publish should fail
        result2 = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Second version",
            tags=["fantasy"]
        )
        assert result2.success is False
        assert "already published" in result2.error_message.lower()
    
    def test_get_published_worlds_with_filters(self, discovery_manager, sample_user):
        """Test filtering published worlds by tags and search"""
        # Create and publish multiple worlds
        worlds_data = [
            ("Fantasy World", ["fantasy", "magic"], "Epic fantasy realm"),
            ("Sci-Fi Station", ["sci-fi", "space"], "Futuristic space station"),
            ("Modern City", ["modern", "urban"], "Contemporary urban setting")
        ]
        
        with patch.object(discovery_manager, '_world_exists', return_value=True):
            for world_name, tags, desc in worlds_data:
                discovery_manager.publish_world(
                    world_name=world_name,
                    user_id=sample_user.id,
                    description=desc,
                    tags=tags
                )
        
        # Test tag filtering
        fantasy_worlds = discovery_manager.get_published_worlds(tags=["fantasy"])
        assert len(fantasy_worlds) == 1
        assert fantasy_worlds[0]['name'] == "Fantasy World"
        
        # Test search
        space_worlds = discovery_manager.get_published_worlds(search_query="space")
        assert len(space_worlds) == 1
        assert space_worlds[0]['name'] == "Sci-Fi Station"
        
        # Test multiple filters
        all_worlds = discovery_manager.get_published_worlds()
        assert len(all_worlds) == 3
    
    def test_rate_world(self, discovery_manager, sample_user, sample_world):
        """Test world rating functionality"""
        world_name, lore = sample_world
        
        # Publish world first
        publish_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        world_id = publish_result.world_id
        
        # Rate the world
        rating_result = discovery_manager.rate_world(
            world_id=world_id,
            user_id=sample_user.id,
            rating=5,
            review_text="Amazing world!"
        )
        
        assert rating_result is True
        
        # Check rating appears in world stats
        world_stats = discovery_manager.get_world_stats(world_id)
        assert world_stats['avg_rating'] == 5.0
        assert world_stats['total_ratings'] == 1
    
    def test_create_play_session(self, discovery_manager, sample_user, sample_world):
        """Test play session creation"""
        world_name, lore = sample_world
        
        # Publish world first
        publish_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        world_id = publish_result.world_id
        
        # Create play session
        session_result = discovery_manager.create_play_session(
            world_id=world_id,
            user_id=sample_user.id,
            session_name="My Adventure",
            privacy_setting="private"
        )
        
        assert session_result.success is True
        assert session_result.session_id is not None
        
        # Verify session exists
        user_sessions = discovery_manager.get_user_sessions(sample_user.id)
        assert len(user_sessions) == 1
        assert user_sessions[0]['session_name'] == "My Adventure"
    
    def test_world_stats_tracking(self, discovery_manager, sample_user, sample_world):
        """Test that world stats are tracked properly"""
        world_name, lore = sample_world
        
        # Publish world
        publish_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        world_id = publish_result.world_id
        
        # Create multiple sessions to test stats
        for i in range(3):
            discovery_manager.create_play_session(
                world_id=world_id,
                user_id=sample_user.id,
                session_name=f"Session {i+1}",
                privacy_setting="private"
            )
        
        # Check world stats
        stats = discovery_manager.get_world_stats(world_id)
        assert stats['total_sessions'] == 3
        assert stats['active_sessions'] == 3  # All sessions start as active


class TestWorldDiscoveryUI:
    """Test the UI components for world discovery"""
    
    def test_world_card_data_structure(self, discovery_manager, sample_user, sample_world):
        """Test that world cards have all required data for beautiful UI"""
        world_name, lore = sample_world
        
        # Publish world with rich metadata
        discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="A beautiful fantasy world with rich lore",
            tags=["fantasy", "magic", "adventure"],
            thumbnail_url="https://example.com/fantasy.jpg"
        )
        
        worlds = discovery_manager.get_published_worlds()
        world_card = worlds[0]
        
        # Check all required fields for beautiful cards
        required_fields = [
            'id', 'name', 'description', 'tags', 'thumbnail_url',
            'creator_username', 'avg_rating', 'total_ratings',
            'total_sessions', 'published_at', 'featured'
        ]
        
        for field in required_fields:
            assert field in world_card, f"World card missing field: {field}"
        
        # Check data types
        assert isinstance(world_card['tags'], list)
        assert isinstance(world_card['avg_rating'], (int, float))
        assert isinstance(world_card['total_ratings'], int) 
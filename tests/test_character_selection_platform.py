"""
Test Character Selection Platform

Comprehensive tests for R3-0.9 Character Selection & Session Management
"""

import pytest
import sqlite3
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Import the modules we'll be testing
from backend.app.services.world.world_discovery import WorldDiscoveryManager
from backend.app.services.character.character import CharacterManager
from backend.app.services.character.models import CharacterCore, Personality, Relationship
from backend.app.services.world.world import WorldManager
from backend.app.core.database.models import User
from backend.app.services.auth.models import UserRole


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Create users table for testing
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                is_active BOOLEAN DEFAULT 1,
                email_verified BOOLEAN DEFAULT 0
            )
        """)
        
        # Insert test user
        cursor.execute("""
            INSERT INTO users (id, username, email, hashed_password, role, is_active)
            VALUES (1, 'testcreator', 'test@example.com', 'hashed_password', 'creator', 1)
        """)
        
        conn.commit()
    
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_user():
    """Create a sample user for testing"""
    return User(
        id=1,
        username="testcreator",
        email="test@example.com",
        role=UserRole.CREATOR,
        is_active=True
    )


@pytest.fixture
def sample_world_data(tmp_path):
    """Create sample world data"""
    world_name = "Fantasy Realm"
    worlds_root = tmp_path / "content" / "worlds"
    world_path = worlds_root / world_name
    
    # Create world structure
    world_path.mkdir(parents=True, exist_ok=True)
    
    # Create world_lore.json
    lore_data = {
        "meta": {"version": 1, "created": "2025-01-14"},
        "facts": {"magic_system": "Elemental magic exists", "currency": "Gold coins"},
        "timeline": [{"year": 1000, "event": "The Great War ended"}],
        "factions": [],
        "places": []
    }
    
    with open(world_path / "world_lore.json", 'w') as f:
        json.dump(lore_data, f)
    
    return world_name, str(worlds_root)


@pytest.fixture
def sample_character_data():
    """Create sample character data"""
    return CharacterCore(
        name="Elara the Wise",
        description="A powerful wizard with deep knowledge of ancient magic",
        scenario="You meet Elara in her magical tower library",
        backstory="Once a student of the Arcane Academy, now a master wizard",
        appearance="Tall woman with silver hair and glowing blue eyes",
        personality_traits=Personality(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.8,
            neuroticism=0.3
        ),
        goals=["Preserve ancient knowledge", "Train new mages"],
        relationships=[
            Relationship(name="Council of Mages", affinity=75),
            Relationship(name="Dark Sorcerers", affinity=-80)
        ],
        tags=["wizard", "wise", "magical", "teacher"]
    )


class TestCharacterDiscoveryExtensions:
    """Test character publishing and discovery database extensions"""
    
    def test_character_discovery_tables_created(self, temp_db):
        """Test that character discovery tables are created"""
        discovery_manager = WorldDiscoveryManager(db_path=temp_db)
        
        # Check that additional character tables exist
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
        # Should have the original tables plus new character tables
        expected_tables = [
            'published_worlds', 'world_stats', 'world_ratings', 
            'play_sessions', 'session_characters',
            'published_characters', 'character_stats', 'character_ratings'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} should be created"
    
    def test_publish_character_success(self, temp_db, sample_user, sample_world_data, sample_character_data):
        """Test successful character publishing"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # First publish the world
        world_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Epic fantasy world",
            tags=["fantasy", "magic"]
        )
        world_id = world_result.world_id
        
        # Create character in world
        char_manager = CharacterManager(world_manager=WorldManager())
        char_manager.set_current_world(world_name)
        
        # Save character to world
        world_path = Path(worlds_root) / world_name
        char_manager.save_character(sample_character_data, world_path)
        
        # Mock the character exists check for testing
        with patch.object(discovery_manager, '_character_exists', return_value=True):
            # Publish character
            result = discovery_manager.publish_character(
                world_id=world_id,
                character_name=sample_character_data.name,
                creator_user_id=sample_user.id,
                description="A wise and powerful wizard",
                tags=["wizard", "mentor", "magical"]
            )
            
            assert result.success is True
            assert result.character_id is not None
            
            # Verify character appears in discovery
            characters = discovery_manager.get_published_characters(world_id=world_id)
            assert len(characters) == 1
            assert characters[0]['name'] == sample_character_data.name
    
    def test_character_rating_system(self, temp_db, sample_user, sample_world_data, sample_character_data):
        """Test character rating functionality"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # Setup world and character
        world_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        
        # Mock the character exists check for testing
        with patch.object(discovery_manager, '_character_exists', return_value=True):
            publish_result = discovery_manager.publish_character(
                world_id=world_result.world_id,
                character_name=sample_character_data.name,
                creator_user_id=sample_user.id,
                description="Test character",
                tags=["test"]
            )
            character_id = publish_result.character_id
            
            # Rate the character
            rating_result = discovery_manager.rate_character(
                character_id=character_id,
                user_id=sample_user.id,
                rating=5,
                review_text="Amazing character!"
            )
            
            assert rating_result is True
            
            # Check rating appears in character stats
            char_stats = discovery_manager.get_character_stats(character_id)
            assert char_stats['avg_rating'] == 5.0
            assert char_stats['total_ratings'] == 1


class TestSessionManager:
    """Test session management functionality"""
    
    def test_session_creation_with_characters(self, temp_db, sample_user, sample_world_data):
        """Test creating a session and adding characters"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # Publish world
        world_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        
        # Create session
        session_result = discovery_manager.create_play_session(
            world_id=world_result.world_id,
            user_id=sample_user.id,
            session_name="Epic Adventure",
            privacy_setting="private"
        )
        
        assert session_result.success is True
        session_id = session_result.session_id
        
        # Add characters to session
        character_data = {"name": "Test Character", "type": "wizard"}
        add_result = discovery_manager.add_character_to_session(
            session_id=session_id,
            character_name="Test Character",
            character_data=character_data
        )
        
        assert add_result is True
        
        # Verify character in session
        session_characters = discovery_manager.get_session_characters(session_id)
        assert len(session_characters) == 1
        assert session_characters[0]['character_name'] == "Test Character"
    
    def test_session_state_management(self, temp_db, sample_user, sample_world_data):
        """Test session state transitions"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # Setup world and session
        world_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        
        session_result = discovery_manager.create_play_session(
            world_id=world_result.world_id,
            user_id=sample_user.id,
            session_name="State Test",
            privacy_setting="private"
        )
        session_id = session_result.session_id
        
        # Test state transitions
        assert discovery_manager.update_session_status(session_id, "active") is True
        assert discovery_manager.update_session_status(session_id, "paused") is True
        assert discovery_manager.update_session_status(session_id, "completed") is True
        
        # Verify final state
        sessions = discovery_manager.get_user_sessions(sample_user.id)
        assert len(sessions) == 1
        assert sessions[0]['status'] == 'completed'


class TestCharacterSelectionInterface:
    """Test character selection UI functionality"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock streamlit for UI testing"""
        with patch('streamlit.session_state', {}) as mock_st:
            yield mock_st
    
    def test_character_discovery_filtering(self, temp_db, sample_user, sample_world_data):
        """Test character filtering by personality traits"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # Setup world
        world_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        world_id = world_result.world_id
        
        # Publish multiple characters with different traits
        characters = [
            {"name": "Brave Knight", "tags": ["brave", "noble", "warrior"]},
            {"name": "Shy Scholar", "tags": ["intelligent", "quiet", "academic"]},
            {"name": "Cheerful Bard", "tags": ["musical", "friendly", "entertaining"]}
        ]
        
        # Mock the character exists check for testing
        with patch.object(discovery_manager, '_character_exists', return_value=True):
            for char in characters:
                discovery_manager.publish_character(
                    world_id=world_id,
                    character_name=char["name"],
                    creator_user_id=sample_user.id,
                    description=f"Character: {char['name']}",
                    tags=char["tags"]
                )
            
            # Test filtering by tags
            friendly_chars = discovery_manager.get_published_characters(
                world_id=world_id,
                tags=["friendly"]
            )
            assert len(friendly_chars) == 1
            assert friendly_chars[0]['name'] == "Cheerful Bard"
            
            warrior_chars = discovery_manager.get_published_characters(
                world_id=world_id,
                tags=["warrior"]
            )
            assert len(warrior_chars) == 1
            assert warrior_chars[0]['name'] == "Brave Knight"
    
    def test_character_compatibility_hints(self, temp_db, sample_user, sample_world_data):
        """Test character compatibility analysis"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # Setup world
        world_result = discovery_manager.publish_world(
            world_name=world_name,
            user_id=sample_user.id,
            description="Test world",
            tags=["test"]
        )
        world_id = world_result.world_id
        
        # Create complementary characters
        characters = [
            {"name": "Warrior", "tags": ["brave", "protective", "helpful"]},
            {"name": "Healer", "tags": ["caring", "supportive", "friendly"]},
            {"name": "Villain", "tags": ["evil", "antagonistic"]}
        ]
        
        character_ids = []
        # Mock the character exists check for testing
        with patch.object(discovery_manager, '_character_exists', return_value=True):
            for char in characters:
                result = discovery_manager.publish_character(
                    world_id=world_id,
                    character_name=char["name"],
                    creator_user_id=sample_user.id,
                    description=f"Character: {char['name']}",
                    tags=char["tags"]
                )
                character_ids.append(result.character_id)
            
            # Test compatibility analysis
            compatibility = discovery_manager.analyze_character_compatibility([character_ids[0], character_ids[1]])
            assert compatibility.score >= 0.5  # Warrior and Healer should be compatible or neutral
            
            conflict = discovery_manager.analyze_character_compatibility([character_ids[0], character_ids[2]])
            assert conflict.score >= 0.3  # Conflict can be interesting for storytelling


# Additional tests for immersive features would go here
class TestImmersiveFeatures:
    """Test immersive character selection features"""
    
    def test_character_preview_generation(self, temp_db, sample_user, sample_world_data, sample_character_data):
        """Test character preview chat generation"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # This would test the preview chat functionality
        # Implementation details depend on the chat system integration
        pass
    
    def test_narrative_framing_system(self, temp_db, sample_world_data):
        """Test narrative framing for character selection"""
        world_name, worlds_root = sample_world_data
        discovery_manager = WorldDiscoveryManager(db_path=temp_db, worlds_root=worlds_root)
        
        # Test that character selection can be framed narratively
        # For now, just test that the discovery manager was created successfully
        assert discovery_manager is not None
        
        # Additional narrative testing would go here in a full implementation
        pass 
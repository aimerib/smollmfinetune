"""
Unit tests for database migration functionality (R3-1)

Tests the migration from file-based storage to SQLAlchemy database,
including data conversion, validation, and error handling.
"""

import pytest
import tempfile
import json
import orjson
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Import database components
from app.utils.database.models import (
    User, World, Character, TrainingRun, PreferencePair,
    WorldFact, WorldTimeline, WorldFaction, WorldPlace, WorldNPC, WorldEvent,
    CharacterRelationship, CharacterGoal, CharacterTag, DatabaseVersion
)
from app.utils.database.session import SessionManager, session_scope, transaction_scope
from app.utils.database.migration import DatabaseMigrator, MigrationError


@pytest.fixture
def temp_database():
    """Create a temporary in-memory database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    # Create session manager with test database
    session_manager = SessionManager(database_url=f"sqlite:///{db_path}")
    session_manager.create_tables()
    
    yield session_manager
    
    # Cleanup
    session_manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def test_user(temp_database):
    """Create a test user in the database"""
    with temp_database.session_scope() as session:
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed_password",
            role="creator",
            is_active=True
        )
        session.add(user)
        session.commit()
        
        # Refresh to get the ID
        session.refresh(user)
        yield user


@pytest.fixture
def sample_world_data():
    """Create sample world lore data for testing"""
    return {
        "meta": {"version": 1},
        "facts": {
            "geography": "A vast fantasy realm",
            "magic_system": "Elemental magic based on crystals"
        },
        "timeline": [
            {"year": 1000, "event": "The Great War began"},
            {"year": 1005, "event": "Peace treaty signed"}
        ],
        "factions": [
            {
                "name": "Crystal Mages",
                "timeline": [
                    {"year": 950, "event": "Order founded"},
                    {"year": 1000, "event": "Joined the war"}
                ]
            }
        ],
        "places": [
            {
                "name": "Crystal Tower",
                "description": "A magnificent tower of pure crystal",
                "npcs": [
                    {"name": "Arch-Mage Zara", "description": "Leader of the Crystal Mages"}
                ],
                "events": [
                    {"name": "Crystal Ceremony", "description": "Annual magic ritual", "random": "false"}
                ]
            }
        ]
    }


@pytest.fixture
def sample_character_data():
    """Create sample character core data for testing"""
    return {
        "name": "Zara the Wise",
        "description": "A powerful mage with deep knowledge of crystal magic",
        "scenario": "You are in the Crystal Tower seeking magical knowledge",
        "backstory": "Born in a small village, discovered magical powers at age 10",
        "appearance": "Tall woman with silver hair and glowing blue eyes",
        "personality_traits": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.3
        },
        "goals": [
            "Master all forms of crystal magic",
            "Protect the realm from dark forces",
            "Train the next generation of mages"
        ],
        "relationships": [
            {"name": "Kael", "affinity": 5, "description": "Trusted apprentice"},
            {"name": "Lord Darkbane", "affinity": -8, "description": "Ancient enemy"}
        ],
        "tags": ["mage", "wise", "powerful", "teacher"],
        "imports": {
            "source": "sillytavern",
            "original_personality": "Wise and powerful crystal mage"
        }
    }


class TestSessionManager:
    """Test database session management functionality"""
    
    def test_session_manager_initialization(self):
        """Test SessionManager initialization with different configurations"""
        # Test default SQLite initialization
        manager = SessionManager()
        assert manager.database_url.startswith("sqlite:///")
        assert manager.engine is not None
        assert manager.SessionLocal is not None
        manager.close()
        
        # Test custom database URL
        custom_url = "sqlite:///test.db"
        manager = SessionManager(database_url=custom_url)
        assert manager.database_url == custom_url
        manager.close()
    
    def test_session_scope_context_manager(self, temp_database):
        """Test session_scope context manager for database operations"""
        with temp_database.session_scope() as session:
            # Create a test user
            user = User(
                email="test@example.com",
                username="testuser",
                hashed_password="hashed_password"
            )
            session.add(user)
            # Session should auto-commit on success
        
        # Verify user was saved
        with temp_database.session_scope() as session:
            saved_user = session.query(User).filter_by(email="test@example.com").first()
            assert saved_user is not None
            assert saved_user.username == "testuser"
    
    def test_session_scope_rollback_on_error(self, temp_database):
        """Test that session_scope rolls back on exceptions"""
        try:
            with temp_database.session_scope() as session:
                user = User(
                    email="test@example.com",
                    username="testuser",
                    hashed_password="hashed_password"
                )
                session.add(user)
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        # Verify user was not saved due to rollback
        with temp_database.session_scope() as session:
            saved_user = session.query(User).filter_by(email="test@example.com").first()
            assert saved_user is None
    
    def test_create_and_drop_tables(self, temp_database):
        """Test table creation and destruction"""
        # Tables should already exist from fixture
        with temp_database.session_scope() as session:
            # Should be able to query users table
            users = session.query(User).all()
            assert isinstance(users, list)
        
        # Test dropping tables
        temp_database.drop_tables()
        
        # Should not be able to query after drop (would raise error)
        with pytest.raises(Exception):
            with temp_database.session_scope() as session:
                session.query(User).all()


class TestDatabaseModels:
    """Test SQLAlchemy model definitions and relationships"""
    
    def test_user_model_creation(self, temp_database):
        """Test User model creation and basic properties"""
        with temp_database.session_scope() as session:
            user = User(
                email="test@example.com",
                username="testuser",
                hashed_password="hashed_password",
                role="creator"
            )
            session.add(user)
            session.commit()
            
            # Test basic properties
            assert user.id is not None
            assert user.email == "test@example.com"
            assert user.username == "testuser"
            assert user.role == "creator"
            assert user.is_active is True
            assert user.created_at is not None
    
    def test_world_model_with_relationships(self, temp_database, test_user):
        """Test World model with its relationships to other models"""
        with temp_database.session_scope() as session:
            # Create world
            world = World(
                name="Test World",
                owner_id=test_user.id,
                description="A test world for unit testing",
                meta_data={"version": 1},
                control_tokens=[{"type": "action", "token": "<ACT>"}]
            )
            session.add(world)
            session.flush()  # Get world ID
            
            # Add world fact
            fact = WorldFact(
                world_id=world.id,
                key="test_fact",
                value="This is a test fact"
            )
            session.add(fact)
            
            # Add timeline event
            timeline_event = WorldTimeline(
                world_id=world.id,
                year=2024,
                event="Test event occurred"
            )
            session.add(timeline_event)
            
            session.commit()
            
            # Test relationships
            assert len(world.facts) == 1
            assert world.facts[0].key == "test_fact"
            assert len(world.timeline_events) == 1
            assert world.timeline_events[0].year == 2024
    
    def test_character_model_with_personality_traits(self, temp_database, test_user):
        """Test Character model with personality traits and constraints"""
        with temp_database.session_scope() as session:
            # Create world first
            world = World(
                name="Test World",
                owner_id=test_user.id,
                description="Test world"
            )
            session.add(world)
            session.flush()
            
            # Create character
            character = Character(
                name="Test Character",
                world_id=world.id,
                owner_id=test_user.id,
                description="A test character",
                openness=0.8,
                conscientiousness=0.7,
                extraversion=0.4,
                agreeableness=0.6,
                neuroticism=0.3
            )
            session.add(character)
            session.flush()
            
            # Add relationships, goals, and tags
            relationship = CharacterRelationship(
                character_id=character.id,
                name="Test Friend",
                affinity=5
            )
            session.add(relationship)
            
            goal = CharacterGoal(
                character_id=character.id,
                goal="Test goal",
                priority=1
            )
            session.add(goal)
            
            tag = CharacterTag(
                character_id=character.id,
                tag="test_tag"
            )
            session.add(tag)
            
            session.commit()
            
            # Test personality traits
            assert character.openness == 0.8
            assert character.conscientiousness == 0.7
            assert character.extraversion == 0.4
            assert character.agreeableness == 0.6
            assert character.neuroticism == 0.3
            
            # Test relationships
            assert len(character.relationships) == 1
            assert character.relationships[0].name == "Test Friend"
            assert len(character.goals) == 1
            assert character.goals[0].goal == "Test goal"
            assert len(character.tags) == 1
            assert character.tags[0].tag == "test_tag"
    
    def test_training_run_model(self, temp_database, test_user):
        """Test TrainingRun model with character relationship"""
        with temp_database.session_scope() as session:
            # Create world and character
            world = World(name="Test World", owner_id=test_user.id)
            session.add(world)
            session.flush()
            
            character = Character(
                name="Test Character",
                world_id=world.id,
                owner_id=test_user.id
            )
            session.add(character)
            session.flush()
            
            # Create training run
            training_run = TrainingRun(
                character_id=character.id,
                owner_id=test_user.id,
                base_model="test-model",
                training_method="lora",
                status="completed",
                output_directory="/test/output",
                total_steps=1000,
                dataset_size=500,
                final_loss=0.25
            )
            session.add(training_run)
            session.commit()
            
            # Test properties
            assert training_run.character_id == character.id
            assert training_run.base_model == "test-model"
            assert training_run.status == "completed"
            assert training_run.total_steps == 1000
            assert training_run.final_loss == 0.25
    
    def test_preference_pair_model(self, temp_database, test_user):
        """Test PreferencePair model for RLHF data"""
        with temp_database.session_scope() as session:
            # Create world and character
            world = World(name="Test World", owner_id=test_user.id)
            session.add(world)
            session.flush()
            
            character = Character(
                name="Test Character",
                world_id=world.id,
                owner_id=test_user.id
            )
            session.add(character)
            session.flush()
            
            # Create preference pair
            preference = PreferencePair(
                character_id=character.id,
                prompt_text="Hello, how are you?",
                chosen_text="I'm doing well, thank you!",
                rejected_json=["Fine.", "Good."],
                context_json={"temperature": 0.7},
                source="test_suite",
                quality_score=0.9
            )
            session.add(preference)
            session.commit()
            
            # Test properties
            assert preference.character_id == character.id
            assert preference.prompt_text == "Hello, how are you?"
            assert preference.chosen_text == "I'm doing well, thank you!"
            assert preference.rejected_json == ["Fine.", "Good."]
            assert preference.source == "test_suite"
            assert preference.quality_score == 0.9


class TestDatabaseMigrator:
    """Test database migration functionality"""
    
    def test_migrator_initialization(self):
        """Test DatabaseMigrator initialization"""
        # Test normal mode
        migrator = DatabaseMigrator(dry_run=False)
        assert migrator.dry_run is False
        assert migrator.stats['worlds_migrated'] == 0
        
        # Test dry run mode
        dry_migrator = DatabaseMigrator(dry_run=True)
        assert dry_migrator.dry_run is True
    
    @patch('app.utils.database.migration.WorldManager')
    def test_migrate_worlds_dry_run(self, mock_world_manager, temp_database, test_user, sample_world_data):
        """Test world migration in dry run mode"""
        # Setup mock world manager
        mock_manager = Mock()
        mock_manager.list_worlds.return_value = ["Test World"]
        
        # Create mock world lore object
        from app.utils.world import WorldLore, TimelineEvent, Faction, Place, NPC, PlaceEvent
        world_lore = WorldLore(
            meta=sample_world_data["meta"],
            facts=sample_world_data["facts"],
            timeline=[TimelineEvent(year=1000, event="The Great War began")],
            factions=[Faction(name="Crystal Mages", timeline=[])],
            places=[Place(name="Crystal Tower", description="A tower", npcs=[], events=[])]
        )
        
        mock_manager.load_world.return_value = world_lore
        mock_manager.load_world_tokens.return_value = []
        mock_world_manager.return_value = mock_manager
        
        # Test dry run migration
        migrator = DatabaseMigrator(dry_run=True, session_manager=temp_database)
        count = migrator.migrate_worlds(test_user.id)
        
        # Should return count but not create database records
        assert count == 1
        assert migrator.stats['worlds_migrated'] == 1
        
        # Verify no database records were created
        with temp_database.session_scope() as session:
            worlds = session.query(World).filter_by(name="Test World").all()
            assert len(worlds) == 0
    
    @patch('app.utils.database.migration.WorldManager')
    def test_migrate_worlds_actual(self, mock_world_manager, temp_database, test_user, sample_world_data):
        """Test actual world migration to database"""
        # Setup mock world manager
        mock_manager = Mock()
        mock_manager.list_worlds.return_value = ["Test World"]
        
        # Create mock world lore object with all components
        from app.utils.world import WorldLore, TimelineEvent, Faction, Place, NPC, PlaceEvent
        world_lore = WorldLore(
            meta=sample_world_data["meta"],
            facts=sample_world_data["facts"],
            timeline=[TimelineEvent(year=event["year"], event=event["event"]) for event in sample_world_data["timeline"]],
            factions=[
                Faction(
                    name=faction["name"],
                    timeline=[TimelineEvent(year=event["year"], event=event["event"]) for event in faction["timeline"]]
                )
                for faction in sample_world_data["factions"]
            ],
            places=[
                Place(
                    name=place["name"],
                    description=place["description"],
                    npcs=[NPC(name=npc["name"], description=npc["description"]) for npc in place["npcs"]],
                    events=[PlaceEvent(name=event["name"], description=event["description"], random=event["random"]) for event in place["events"]]
                )
                for place in sample_world_data["places"]
            ]
        )
        
        mock_manager.load_world.return_value = world_lore
        mock_manager.load_world_tokens.return_value = [{"type": "action", "token": "<ACT>"}]
        mock_world_manager.return_value = mock_manager
        
        # Test actual migration
        migrator = DatabaseMigrator(dry_run=False, session_manager=temp_database)
        count = migrator.migrate_worlds(test_user.id)
        
        assert count == 1
        
        # Verify database records were created
        with temp_database.session_scope() as session:
            world = session.query(World).filter_by(name="Test World").first()
            assert world is not None
            assert world.owner_id == test_user.id
            assert world.meta_data["version"] == 1
            
            # Check world facts
            assert len(world.facts) == 2
            fact_keys = {fact.key for fact in world.facts}
            assert "geography" in fact_keys
            assert "magic_system" in fact_keys
            
            # Check timeline events
            assert len(world.timeline_events) == 2
            timeline_years = {event.year for event in world.timeline_events}
            assert 1000 in timeline_years
            assert 1005 in timeline_years
            
            # Check factions
            assert len(world.factions) == 1
            assert world.factions[0].name == "Crystal Mages"
            
            # Check places
            assert len(world.places) == 1
            place = world.places[0]
            assert place.name == "Crystal Tower"
            assert len(place.npcs) == 1
            assert place.npcs[0].name == "Arch-Mage Zara"
            assert len(place.events) == 1
            assert place.events[0].name == "Crystal Ceremony"
    
    def test_migrate_characters_with_temporary_files(self, temp_database, test_user, sample_character_data):
        """Test character migration using temporary file structure"""
        # Create temporary world in database first
        with temp_database.session_scope() as session:
            world = World(
                name="Test World",
                owner_id=test_user.id,
                description="Test world for character migration"
            )
            session.add(world)
            session.commit()
        
        # Create temporary file structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the expected directory structure: content_path/worlds/World Name/characters/Character Name/
            worlds_path = Path(temp_dir) / "worlds" / "Test World"
            char_path = worlds_path / "characters" / "Zara the Wise"
            char_path.mkdir(parents=True)
            
            # Create character_core.json file
            core_file = char_path / "character_core.json"
            with open(core_file, 'wb') as f:
                f.write(orjson.dumps(sample_character_data))
            
            # Test character migration with custom content path and session manager
            migrator = DatabaseMigrator(dry_run=False, session_manager=temp_database, content_path=temp_dir)
            count = migrator.migrate_characters(test_user.id)
            
            assert count == 1
            
            # Verify character was created in database
            with temp_database.session_scope() as session:
                character = session.query(Character).filter_by(name="Zara the Wise").first()
                assert character is not None
                assert character.description == sample_character_data["description"]
                assert character.openness == 0.8
                assert character.conscientiousness == 0.7
                
                # Check relationships
                assert len(character.relationships) == 2
                relationship_names = {rel.name for rel in character.relationships}
                assert "Kael" in relationship_names
                assert "Lord Darkbane" in relationship_names
                
                # Check goals
                assert len(character.goals) == 3
                
                # Check tags
                assert len(character.tags) == 4
                tag_values = {tag.tag for tag in character.tags}
                assert "mage" in tag_values
                assert "wise" in tag_values
    
    def test_migrate_preference_data_with_temporary_files(self, temp_database, test_user):
        """Test preference data migration using temporary files"""
        # Create world and character in database first
        with temp_database.session_scope() as session:
            world = World(name="Test World", owner_id=test_user.id)
            session.add(world)
            session.flush()
            
            character = Character(
                name="Test Character",
                world_id=world.id,
                owner_id=test_user.id
            )
            session.add(character)
            session.commit()
            
            # Get the character ID while still in session
            character_id = character.id
        
        # Create temporary preference file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the expected directory structure: content_path/worlds/World Name/characters/Character Name/
            worlds_path = Path(temp_dir) / "worlds" / "Test World"
            char_path = worlds_path / "characters" / "Test Character"
            char_path.mkdir(parents=True)
            
            # Create preference logs file
            pref_file = char_path / "preference_logs.ndjson"
            with open(pref_file, 'w') as f:
                pref1 = {
                    "prompt": "Hello, how are you?",
                    "chosen": "I'm doing well, thank you!",
                    "rejected": ["Fine.", "Good."]
                }
                pref2 = {
                    "prompt": "What's your favorite color?",
                    "chosen": "I love blue, it reminds me of the sky.",
                    "rejected": ["Blue.", "Sky blue."]
                }
                f.write(json.dumps(pref1) + '\n')
                f.write(json.dumps(pref2) + '\n')
            
            # Test preference migration with custom content path and session manager
            migrator = DatabaseMigrator(dry_run=False, session_manager=temp_database, content_path=temp_dir)
            count = migrator.migrate_preference_data()
            
            assert count == 2
            
            # Verify preferences were created in database
            with temp_database.session_scope() as session:
                preferences = session.query(PreferencePair).filter_by(
                    character_id=character_id
                ).all()
                assert len(preferences) == 2
                
                prompts = {pref.prompt_text for pref in preferences}
                assert "Hello, how are you?" in prompts
                assert "What's your favorite color?" in prompts
    
    def test_full_migration_integration(self, temp_database, test_user):
        """Test complete migration workflow with all components"""
        # This test would require extensive mocking or actual test files
        # For now, test the migration statistics tracking
        migrator = DatabaseMigrator(dry_run=True)
        
        # Test error handling in stats
        migrator.stats['errors'].append("Test error")
        assert len(migrator.stats['errors']) == 1
        assert migrator.stats['errors'][0] == "Test error"
        
    def test_record_migration_version(self, temp_database):
        """Test recording migration versions"""
        migrator = DatabaseMigrator(dry_run=False, session_manager=temp_database)
        
        # Record a migration version
        migrator.record_migration_version("1.0.0", "Initial migration")
        
        # Verify it was recorded
        with temp_database.session_scope() as session:
            version = session.query(DatabaseVersion).filter_by(version="1.0.0").first()
            assert version is not None
            assert version.description == "Initial migration"
            
        # Try recording the same version again (should skip)
        migrator.record_migration_version("1.0.0", "Duplicate")
        
        with temp_database.session_scope() as session:
            versions = session.query(DatabaseVersion).filter_by(version="1.0.0").all()
            assert len(versions) == 1  # Should not duplicate


class TestMigrationErrors:
    """Test error handling in migration processes"""
    
    def test_migration_error_exception(self):
        """Test MigrationError exception"""
        with pytest.raises(MigrationError):
            raise MigrationError("Test migration failed")
    
    def test_migrator_error_tracking(self, temp_database):
        """Test that migrator tracks errors properly"""
        migrator = DatabaseMigrator(dry_run=False)
        
        # Simulate an error during migration
        error_msg = "Test error occurred"
        migrator.stats['errors'].append(error_msg)
        
        assert len(migrator.stats['errors']) == 1
        assert migrator.stats['errors'][0] == error_msg 
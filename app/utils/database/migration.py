"""
Database migration utilities for the Character Creation Platform.

Handles migration from file-based data to database records and provide Alembic integration for schema migrations.
"""

import os
import json
import logging
import orjson
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import (
    User, World, Character, TrainingRun, PreferencePair,
    WorldFact, WorldTimeline, WorldFaction, WorldPlace, WorldNPC, WorldEvent,
    CharacterRelationship, CharacterGoal, CharacterTag, DatabaseVersion
)
from .session import get_session_manager, session_scope
from ..world import WorldManager, WorldLore
from ..character.character import CharacterManager
from ..character.models import CharacterCore

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration operations fail"""
    pass


class DatabaseMigrator:
    """
    Handles migration from file-based storage to database storage.
    
    Provides utilities to convert existing JSON files, character folders,
    and training outputs to database records while preserving data integrity.
    """
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize the database migrator.
        
        Args:
            dry_run: If True, only simulate migration without making changes
        """
        self.dry_run = dry_run
        self.session_manager = get_session_manager()
        
        # Statistics
        self.stats = {
            'worlds_migrated': 0,
            'characters_migrated': 0,
            'training_runs_migrated': 0,
            'preference_pairs_migrated': 0,
            'errors': []
        }
        
        logger.info(f"Database migrator initialized (dry_run={dry_run})")
    
    def migrate_all(self, user_id: int = 1) -> Dict[str, Any]:
        """
        Migrate all file-based data to database.
        
        Args:
            user_id: ID of user to assign ownership to (default: 1)
            
        Returns:
            Migration statistics and results
        """
        logger.info("Starting full migration from file-based to database storage")
        
        try:
            # Reset statistics
            self.stats = {
                'worlds_migrated': 0,
                'characters_migrated': 0,
                'training_runs_migrated': 0,
                'preference_pairs_migrated': 0,
                'errors': []
            }
            
            # Migrate worlds first (characters depend on worlds)
            self.migrate_worlds(user_id)
            
            # Migrate characters and their associated data
            self.migrate_characters(user_id)
            
            # Migrate training runs
            self.migrate_training_runs(user_id)
            
            # Migrate preference data
            self.migrate_preference_data()
            
            logger.info(f"Migration completed successfully: {self.stats}")
            return self.stats
            
        except Exception as e:
            error_msg = f"Migration failed: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise MigrationError(error_msg) from e
    
    def migrate_worlds(self, user_id: int) -> int:
        """
        Migrate all worlds from file-based storage to database.
        
        Args:
            user_id: ID of user to assign ownership to
            
        Returns:
            Number of worlds migrated
        """
        logger.info("Migrating worlds from file-based storage")
        
        world_manager = WorldManager()
        world_names = world_manager.list_worlds()
        
        migrated_count = 0
        
        with session_scope() as session:
            for world_name in world_names:
                try:
                    # Check if world already exists in database
                    existing_world = session.query(World).filter_by(
                        name=world_name, owner_id=user_id
                    ).first()
                    
                    if existing_world:
                        logger.info(f"World '{world_name}' already exists in database, skipping")
                        continue
                    
                    # Load world data from files
                    world_lore = world_manager.load_world(world_name)
                    if not world_lore:
                        logger.warning(f"Could not load world '{world_name}', skipping")
                        continue
                    
                    # Load control tokens
                    control_tokens = world_manager.load_world_tokens(world_name)
                    
                    if not self.dry_run:
                        # Create world record
                        world = World(
                            name=world_name,
                            owner_id=user_id,
                            description=f"Migrated world: {world_name}",
                            version=world_lore.meta.get('version', 1),
                            meta_data=world_lore.meta,
                            control_tokens=control_tokens
                        )
                        session.add(world)
                        session.flush()  # Get the world ID
                        
                        # Migrate world facts
                        for key, value in world_lore.facts.items():
                            fact = WorldFact(
                                world_id=world.id,
                                key=key,
                                value=value
                            )
                            session.add(fact)
                        
                        # Migrate timeline events
                        for event in world_lore.timeline:
                            timeline_event = WorldTimeline(
                                world_id=world.id,
                                year=event.year,
                                event=event.event
                            )
                            session.add(timeline_event)
                        
                        # Migrate factions
                        for faction in world_lore.factions:
                            faction_record = WorldFaction(
                                world_id=world.id,
                                name=faction.name,
                                description=f"Faction in {world_name}",
                                timeline_events=[
                                    {"year": event.year, "event": event.event}
                                    for event in faction.timeline
                                ]
                            )
                            session.add(faction_record)
                        
                        # Migrate places
                        for place in world_lore.places:
                            place_record = WorldPlace(
                                world_id=world.id,
                                name=place.name,
                                description=place.description
                            )
                            session.add(place_record)
                            session.flush()  # Get the place ID
                            
                            # Migrate NPCs
                            for npc in place.npcs:
                                npc_record = WorldNPC(
                                    place_id=place_record.id,
                                    name=npc.name,
                                    description=npc.description
                                )
                                session.add(npc_record)
                            
                            # Migrate events
                            for event in place.events:
                                event_record = WorldEvent(
                                    place_id=place_record.id,
                                    name=event.name,
                                    description=event.description,
                                    is_random=(event.random.lower() == 'true')
                                )
                                session.add(event_record)
                    
                    migrated_count += 1
                    logger.info(f"Successfully migrated world: {world_name}")
                    
                except Exception as e:
                    error_msg = f"Failed to migrate world '{world_name}': {e}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
        
        self.stats['worlds_migrated'] = migrated_count
        logger.info(f"Migrated {migrated_count} worlds to database")
        return migrated_count
    
    def migrate_characters(self, user_id: int) -> int:
        """
        Migrate all characters from file-based storage to database.
        
        Args:
            user_id: ID of user to assign ownership to
            
        Returns:
            Number of characters migrated
        """
        logger.info("Migrating characters from file-based storage")
        
        migrated_count = 0
        
        with session_scope() as session:
            # Get all worlds from database
            worlds = session.query(World).filter_by(owner_id=user_id).all()
            
            for world in worlds:
                try:
                    # Find character directories for this world
                    world_path = Path("content/worlds") / world.name
                    characters_path = world_path / "characters"
                    
                    if not characters_path.exists():
                        continue
                    
                    for char_folder in characters_path.iterdir():
                        if not char_folder.is_dir():
                            continue
                        
                        core_file = char_folder / "character_core.json"
                        if not core_file.exists():
                            continue
                        
                        # Check if character already exists
                        existing_char = session.query(Character).filter_by(
                            name=char_folder.name, world_id=world.id
                        ).first()
                        
                        if existing_char:
                            logger.info(f"Character '{char_folder.name}' already exists, skipping")
                            continue
                        
                        # Load character data
                        with open(core_file, 'rb') as f:
                            core_data = orjson.loads(f.read())
                        
                        if not self.dry_run:
                            # Create character record
                            character = Character(
                                name=char_folder.name,
                                world_id=world.id,
                                owner_id=user_id,
                                description=core_data.get('description', ''),
                                scenario=core_data.get('scenario', ''),
                                backstory=core_data.get('backstory', ''),
                                appearance=core_data.get('appearance', ''),
                                core_data_json=core_data,
                                imports_data=core_data.get('imports', {})
                            )
                            
                            # Extract personality traits
                            personality = core_data.get('personality_traits', {})
                            character.openness = personality.get('openness', 0.5)
                            character.conscientiousness = personality.get('conscientiousness', 0.5)
                            character.extraversion = personality.get('extraversion', 0.5)
                            character.agreeableness = personality.get('agreeableness', 0.5)
                            character.neuroticism = personality.get('neuroticism', 0.5)
                            
                            session.add(character)
                            session.flush()  # Get character ID
                            
                            # Migrate relationships
                            for relationship in core_data.get('relationships', []):
                                if isinstance(relationship, dict):
                                    rel_record = CharacterRelationship(
                                        character_id=character.id,
                                        name=relationship.get('name', ''),
                                        affinity=relationship.get('affinity', 0),
                                        description=relationship.get('description', '')
                                    )
                                    session.add(rel_record)
                            
                            # Migrate goals
                            for i, goal in enumerate(core_data.get('goals', [])):
                                goal_record = CharacterGoal(
                                    character_id=character.id,
                                    goal=goal,
                                    priority=i + 1
                                )
                                session.add(goal_record)
                            
                            # Migrate tags
                            for tag in core_data.get('tags', []):
                                tag_record = CharacterTag(
                                    character_id=character.id,
                                    tag=tag
                                )
                                session.add(tag_record)
                        
                        migrated_count += 1
                        logger.info(f"Successfully migrated character: {char_folder.name}")
                
                except Exception as e:
                    error_msg = f"Failed to migrate characters for world '{world.name}': {e}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
        
        self.stats['characters_migrated'] = migrated_count
        logger.info(f"Migrated {migrated_count} characters to database")
        return migrated_count
    
    def migrate_training_runs(self, user_id: int) -> int:
        """
        Migrate training runs from training_output directory to database.
        
        Args:
            user_id: ID of user to assign ownership to
            
        Returns:
            Number of training runs migrated
        """
        logger.info("Migrating training runs from training_output directory")
        
        training_output_path = Path("training_output")
        if not training_output_path.exists():
            logger.info("No training_output directory found, skipping training run migration")
            return 0
        
        migrated_count = 0
        
        with session_scope() as session:
            # Find all character training directories
            for char_dir in training_output_path.iterdir():
                if not char_dir.is_dir():
                    continue
                
                try:
                    # Find character in database
                    character = session.query(Character).filter_by(
                        name=char_dir.name, owner_id=user_id
                    ).first()
                    
                    if not character:
                        logger.warning(f"Character '{char_dir.name}' not found in database, skipping training run")
                        continue
                    
                    # Look for training metadata
                    metadata_file = char_dir / "training_metadata.json"
                    if not metadata_file.exists():
                        continue
                    
                    # Check if training run already exists
                    existing_run = session.query(TrainingRun).filter_by(
                        character_id=character.id,
                        output_directory=str(char_dir)
                    ).first()
                    
                    if existing_run:
                        logger.info(f"Training run for '{char_dir.name}' already exists, skipping")
                        continue
                    
                    # Load training metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if not self.dry_run:
                        # Create training run record
                        training_run = TrainingRun(
                            character_id=character.id,
                            owner_id=user_id,
                            base_model=metadata.get('base_model', 'unknown'),
                            training_method=metadata.get('training_method', 'lora'),
                            status='completed',  # Assume completed if metadata exists
                            output_directory=str(char_dir),
                            sft_adapter_path=str(char_dir) if (char_dir / "adapter.safetensors").exists() else None,
                            metrics_json=metadata,
                            config_json=metadata,
                            total_steps=metadata.get('total_steps'),
                            dataset_size=metadata.get('dataset_size'),
                            completed_at=datetime.now(timezone.utc)
                        )
                        
                        # Check for RLHF adapter
                        rlhf_grpo_path = char_dir / "rlhf_output" / "adapter_grpo"
                        rlhf_ppo_path = char_dir / "rlhf_output" / "adapter_ppo"
                        
                        if rlhf_grpo_path.exists():
                            training_run.rlhf_adapter_path = str(rlhf_grpo_path)
                        elif rlhf_ppo_path.exists():
                            training_run.rlhf_adapter_path = str(rlhf_ppo_path)
                        
                        session.add(training_run)
                    
                    migrated_count += 1
                    logger.info(f"Successfully migrated training run for: {char_dir.name}")
                
                except Exception as e:
                    error_msg = f"Failed to migrate training run for '{char_dir.name}': {e}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
        
        self.stats['training_runs_migrated'] = migrated_count
        logger.info(f"Migrated {migrated_count} training runs to database")
        return migrated_count
    
    def migrate_preference_data(self) -> int:
        """
        Migrate preference data from preference_logs.ndjson files to database.
        
        Returns:
            Number of preference pairs migrated
        """
        logger.info("Migrating preference data from NDJSON files")
        
        migrated_count = 0
        
        with session_scope() as session:
            # Find all characters with preference logs
            characters = session.query(Character).all()
            
            for character in characters:
                try:
                    # Find preference logs file
                    world_path = Path("content/worlds") / character.world.name
                    char_path = world_path / "characters" / character.name
                    pref_file = char_path / "preference_logs.ndjson"
                    
                    if not pref_file.exists():
                        continue
                    
                    # Count existing preferences
                    existing_count = session.query(PreferencePair).filter_by(
                        character_id=character.id
                    ).count()
                    
                    if existing_count > 0:
                        logger.info(f"Character '{character.name}' already has {existing_count} preferences, skipping")
                        continue
                    
                    # Load and migrate preference data
                    with open(pref_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            
                            try:
                                pref_data = json.loads(line)
                                
                                if not self.dry_run:
                                    preference_pair = PreferencePair(
                                        character_id=character.id,
                                        prompt_text=pref_data.get('prompt', ''),
                                        chosen_text=pref_data.get('chosen', ''),
                                        rejected_json=pref_data.get('rejected', []),
                                        context_json=pref_data.get('context', {}),
                                        source='migration'
                                    )
                                    session.add(preference_pair)
                                
                                migrated_count += 1
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in {pref_file} line {line_num}: {e}")
                                continue
                    
                    logger.info(f"Successfully migrated preferences for: {character.name}")
                
                except Exception as e:
                    error_msg = f"Failed to migrate preferences for '{character.name}': {e}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    continue
        
        self.stats['preference_pairs_migrated'] = migrated_count
        logger.info(f"Migrated {migrated_count} preference pairs to database")
        return migrated_count
    
    def create_legacy_import_functions(self):
        """Create legacy import functions for manual data import."""
        # This would be implemented as utility functions for manual import
        # For now, we'll rely on the full migration
        pass
    
    def record_migration_version(self, version: str, description: str = None):
        """
        Record a migration version in the database.
        
        Args:
            version: Migration version string
            description: Optional description of the migration
        """
        if self.dry_run:
            logger.info(f"Would record migration version: {version}")
            return
        
        with session_scope() as session:
            try:
                # Check if version already exists
                existing = session.query(DatabaseVersion).filter_by(version=version).first()
                if existing:
                    logger.info(f"Migration version {version} already recorded")
                    return
                
                # Record new migration version
                migration_record = DatabaseVersion(
                    version=version,
                    description=description or f"Migration to version {version}"
                )
                session.add(migration_record)
                logger.info(f"Recorded migration version: {version}")
                
            except Exception as e:
                logger.error(f"Failed to record migration version {version}: {e}")
                raise 
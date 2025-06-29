#!/usr/bin/env python3
"""
Migration script for Character Creation Platform

Migrates from file-based storage to SQLAlchemy database and demonstrates
the new functionality. This is the main entry point for R3-1 migration.
"""

import os
import sys
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from utils.database import (
    init_database, DatabaseMigrator, 
    DatabaseWorldManager, session_scope,
    User, World, Character
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_user() -> int:
    """Create a demo user for the migration"""
    try:
        with session_scope() as session:
            # Check if demo user already exists
            existing_user = session.query(User).filter_by(username="demo_user").first()
            
            if existing_user:
                logger.info(f"Demo user already exists with ID: {existing_user.id}")
                return existing_user.id
            
            # Create new demo user
            demo_user = User(
                email="demo@example.com",
                username="demo_user",
                hashed_password="demo_password_hash",
                role="creator",
                is_active=True
            )
            session.add(demo_user)
            session.commit()
            
            logger.info(f"Created demo user with ID: {demo_user.id}")
            return demo_user.id
            
    except Exception as e:
        logger.error(f"Failed to create demo user: {e}")
        return 1  # Fallback to user ID 1


def run_migration(user_id: int = 1, dry_run: bool = False):
    """Run the complete migration process"""
    logger.info(f"Starting database migration (dry_run={dry_run})")
    
    try:
        # Initialize database and create tables
        session_manager = init_database(create_tables=True)
        logger.info("Database initialized successfully")
        
        # Run the migration
        migrator = DatabaseMigrator(dry_run=dry_run)
        stats = migrator.migrate_all(user_id=user_id)
        
        # Record migration version
        migrator.record_migration_version("1.0.0", "Initial migration from file-based storage")
        
        logger.info("Migration completed successfully!")
        logger.info(f"Migration statistics: {stats}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def test_database_functionality(user_id: int = 1):
    """Test the new database functionality"""
    logger.info("Testing database functionality...")
    
    try:
        # Test DatabaseWorldManager
        logger.info("Testing DatabaseWorldManager...")
        world_manager = DatabaseWorldManager(user_id=user_id)
        
        # Create a test world
        test_world_name = "Test Database World"
        if world_manager.create_world(test_world_name):
            logger.info(f"✅ Successfully created world: {test_world_name}")
        else:
            logger.info(f"⚠️ World '{test_world_name}' already exists")
        
        # List worlds
        worlds = world_manager.list_worlds()
        logger.info(f"✅ Listed {len(worlds)} worlds: {worlds}")
        
        # Load a world
        if worlds:
            world_lore = world_manager.load_world(worlds[0])
            if world_lore:
                logger.info(f"✅ Successfully loaded world: {worlds[0]}")
                logger.info(f"   - Facts: {len(world_lore.facts)}")
                logger.info(f"   - Timeline events: {len(world_lore.timeline)}")
                logger.info(f"   - Factions: {len(world_lore.factions)}")
                logger.info(f"   - Places: {len(world_lore.places)}")
            else:
                logger.warning(f"Failed to load world: {worlds[0]}")
        
        # Test tokens
        tokens = world_manager.get_current_tokens()
        logger.info(f"✅ Retrieved {len(tokens)} control tokens")
        
        logger.info("Database functionality test completed successfully!")
        
    except Exception as e:
        logger.error(f"Database functionality test failed: {e}")
        raise


def show_migration_summary():
    """Show a summary of what was migrated"""
    logger.info("=== Migration Summary ===")
    
    try:
        with session_scope() as session:
            # Count migrated entities
            user_count = session.query(User).count()
            world_count = session.query(World).count()
            character_count = session.query(Character).count()
            
            logger.info(f"📊 Database Contents:")
            logger.info(f"   - Users: {user_count}")
            logger.info(f"   - Worlds: {world_count}")
            logger.info(f"   - Characters: {character_count}")
            
            # Show some sample data
            if world_count > 0:
                worlds = session.query(World).limit(3).all()
                logger.info(f"📋 Sample Worlds:")
                for world in worlds:
                    logger.info(f"   - {world.name} (v{world.version})")
            
            if character_count > 0:
                characters = session.query(Character).limit(3).all()
                logger.info(f"📋 Sample Characters:")
                for char in characters:
                    logger.info(f"   - {char.name} in world {char.world.name}")
            
    except Exception as e:
        logger.error(f"Failed to show migration summary: {e}")


def main():
    """Main migration script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Character Creation Platform to database")
    parser.add_argument("--dry-run", action="store_true", help="Run migration in dry-run mode")
    parser.add_argument("--test-only", action="store_true", help="Only test functionality, don't migrate")
    parser.add_argument("--user-id", type=int, default=None, help="User ID to assign ownership to")
    
    args = parser.parse_args()
    
    try:
        # Create or get demo user
        user_id = args.user_id or create_demo_user()
        
        if not args.test_only:
            # Run the migration
            stats = run_migration(user_id=user_id, dry_run=args.dry_run)
            
            if not args.dry_run:
                show_migration_summary()
        
        # Test the functionality
        test_database_functionality(user_id=user_id)
        
        logger.info("🎉 Migration and testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
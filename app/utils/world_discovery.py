"""
World Discovery Manager

Handles world publishing, discovery, ratings, and play session management.
Extends the auth SQLite database with discovery-focused tables.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PublishResult:
    """Result of world publishing operation"""
    success: bool
    world_id: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class SessionResult:
    """Result of play session creation"""
    success: bool
    session_id: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class CharacterPublishResult:
    """Result of character publishing operation"""
    success: bool
    character_id: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class CompatibilityAnalysis:
    """Result of character compatibility analysis"""
    score: float  # 0.0 to 1.0, higher = more compatible
    reasoning: str
    relationship_type: str  # "complementary", "conflicting", "neutral"


class WorldDiscoveryManager:
    """
    Manages world discovery, publishing, ratings, and play sessions
    
    Extends the existing auth SQLite database with discovery tables
    """
    
    def __init__(self, db_path: str = "platform.db", worlds_root: Optional[str] = None):
        """
        Initialize WorldDiscoveryManager
        
        Args:
            db_path: Path to SQLite database file
            worlds_root: Path to worlds directory
        """
        self.db_path = db_path
        self.worlds_root = Path(worlds_root or "content/worlds")
        
        # Initialize discovery tables
        self._init_discovery_tables()
    
    def _init_discovery_tables(self):
        """Create discovery-related database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table if it doesn't exist (for testing compatibility)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TEXT
                )
            """)
            
            # Published worlds table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS published_worlds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_name TEXT UNIQUE NOT NULL,
                    creator_user_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    tags TEXT NOT NULL,  -- JSON array
                    thumbnail_url TEXT,
                    published_at TEXT NOT NULL,
                    featured BOOLEAN DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (creator_user_id) REFERENCES users (id)
                )
            """)
            
            # World statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS world_stats (
                    world_id INTEGER PRIMARY KEY,
                    total_sessions INTEGER DEFAULT 0,
                    active_sessions INTEGER DEFAULT 0,
                    avg_session_duration REAL DEFAULT 0.0,
                    total_ratings INTEGER DEFAULT 0,
                    avg_rating REAL DEFAULT 0.0,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (world_id) REFERENCES published_worlds (id)
                )
            """)
            
            # World ratings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS world_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    review_text TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (world_id) REFERENCES published_worlds (id),
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(world_id, user_id)
                )
            """)
            
            # Play sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS play_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    session_name TEXT NOT NULL,
                    privacy_setting TEXT NOT NULL DEFAULT 'private',
                    status TEXT NOT NULL DEFAULT 'setup',
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (world_id) REFERENCES published_worlds (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Session characters table (for character selection)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL,
                    character_data TEXT,  -- JSON character data
                    added_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES play_sessions (id)
                )
            """)
            
            # Published characters table (NEW)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS published_characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    world_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL,
                    creator_user_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    tags TEXT NOT NULL,  -- JSON array
                    character_data TEXT,  -- JSON character core data
                    published_at TEXT NOT NULL,
                    featured BOOLEAN DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (world_id) REFERENCES published_worlds (id),
                    FOREIGN KEY (creator_user_id) REFERENCES users (id),
                    UNIQUE(world_id, character_name)
                )
            """)
            
            # Character statistics table (NEW)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS character_stats (
                    character_id INTEGER PRIMARY KEY,
                    download_count INTEGER DEFAULT 0,
                    session_count INTEGER DEFAULT 0,
                    interaction_count INTEGER DEFAULT 0,
                    total_ratings INTEGER DEFAULT 0,
                    avg_rating REAL DEFAULT 0.0,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (character_id) REFERENCES published_characters (id)
                )
            """)
            
            # Character ratings table (NEW)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS character_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    character_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    review_text TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (character_id) REFERENCES published_characters (id),
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(character_id, user_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published_worlds_creator ON published_worlds(creator_user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published_worlds_featured ON published_worlds(featured)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_world_ratings_world ON world_ratings(world_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_play_sessions_user ON play_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_play_sessions_world ON play_sessions(world_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published_characters_world ON published_characters(world_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published_characters_creator ON published_characters(creator_user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_character_ratings_character ON character_ratings(character_id)")
            
            conn.commit()
            logger.info("Discovery database tables initialized successfully")
    
    def _get_table_names(self) -> List[str]:
        """Get list of table names in database (for testing)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor.fetchall()]
    
    def _world_exists(self, world_name: str) -> bool:
        """Check if a world exists in the file system"""
        world_path = self.worlds_root / world_name
        return world_path.exists() and (world_path / "world_lore.json").exists()
    
    def _character_exists(self, world_name: str, character_name: str) -> bool:
        """Check if a character exists in the world file system"""
        char_path = self.worlds_root / world_name / "characters" / character_name
        return char_path.exists() and (char_path / "character_core.json").exists()
    
    def publish_world(
        self,
        world_name: str,
        user_id: int,
        description: str,
        tags: List[str],
        thumbnail_url: Optional[str] = None
    ) -> PublishResult:
        """
        Publish a world for discovery
        
        Args:
            world_name: Name of the world to publish
            user_id: ID of the user publishing the world
            description: World description
            tags: List of tags for categorization
            thumbnail_url: Optional thumbnail image URL
            
        Returns:
            PublishResult with success status and world ID
        """
        try:
            # Check if world exists in file system
            if not self._world_exists(world_name):
                return PublishResult(
                    success=False,
                    error_message=f"World '{world_name}' not found in file system"
                )
            
            # Check if already published
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM published_worlds WHERE world_name = ? AND is_active = 1",
                    (world_name,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    return PublishResult(
                        success=False,
                        error_message=f"World '{world_name}' is already published"
                    )
                
                # Insert published world
                now = datetime.now(timezone.utc)
                cursor.execute("""
                    INSERT INTO published_worlds 
                    (world_name, creator_user_id, description, tags, thumbnail_url, published_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    world_name, user_id, description, json.dumps(tags),
                    thumbnail_url, now.isoformat()
                ))
                
                world_id = cursor.lastrowid
                
                # Initialize world stats
                cursor.execute("""
                    INSERT INTO world_stats 
                    (world_id, last_updated)
                    VALUES (?, ?)
                """, (world_id, now.isoformat()))
                
                conn.commit()
                
                logger.info(f"Published world '{world_name}' with ID {world_id}")
                
                return PublishResult(
                    success=True,
                    world_id=world_id
                )
                
        except Exception as e:
            logger.error(f"Failed to publish world '{world_name}': {e}")
            return PublishResult(
                success=False,
                error_message=f"Failed to publish world: {str(e)}"
            )
    
    def publish_character(
        self,
        world_id: int,
        character_name: str,
        creator_user_id: int,
        description: str,
        tags: List[str],
        character_data: Optional[Dict[str, Any]] = None
    ) -> CharacterPublishResult:
        """
        Publish a character for discovery in a world
        
        Args:
            world_id: ID of the world the character belongs to
            character_name: Name of the character
            creator_user_id: ID of the user publishing the character
            description: Character description
            tags: List of tags for categorization
            character_data: Optional character core data
            
        Returns:
            CharacterPublishResult with success status and character ID
        """
        try:
            # Verify world exists
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT world_name FROM published_worlds WHERE id = ? AND is_active = 1",
                    (world_id,)
                )
                world_result = cursor.fetchone()
                
                if not world_result:
                    return CharacterPublishResult(
                        success=False,
                        error_message=f"World with ID {world_id} not found"
                    )
                
                world_name = world_result[0]
                
                # Check if character exists in file system
                if not self._character_exists(world_name, character_name):
                    return CharacterPublishResult(
                        success=False,
                        error_message=f"Character '{character_name}' not found in world '{world_name}'"
                    )
                
                # Check if already published
                cursor.execute(
                    "SELECT id FROM published_characters WHERE world_id = ? AND character_name = ? AND is_active = 1",
                    (world_id, character_name)
                )
                existing = cursor.fetchone()
                
                if existing:
                    return CharacterPublishResult(
                        success=False,
                        error_message=f"Character '{character_name}' is already published in this world"
                    )
                
                # Insert published character
                now = datetime.now(timezone.utc)
                cursor.execute("""
                    INSERT INTO published_characters 
                    (world_id, character_name, creator_user_id, description, tags, character_data, published_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    world_id, character_name, creator_user_id, description, 
                    json.dumps(tags), json.dumps(character_data) if character_data else None, 
                    now.isoformat()
                ))
                
                character_id = cursor.lastrowid
                
                # Initialize character stats
                cursor.execute("""
                    INSERT INTO character_stats 
                    (character_id, last_updated)
                    VALUES (?, ?)
                """, (character_id, now.isoformat()))
                
                conn.commit()
                
                logger.info(f"Published character '{character_name}' with ID {character_id}")
                
                return CharacterPublishResult(
                    success=True,
                    character_id=character_id
                )
                
        except Exception as e:
            logger.error(f"Failed to publish character '{character_name}': {e}")
            return CharacterPublishResult(
                success=False,
                error_message=f"Failed to publish character: {str(e)}"
            )
    
    def get_published_worlds(
        self,
        search_query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        featured_only: bool = False,
        sort_by: str = "popular",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get published worlds with filtering options
        
        Args:
            search_query: Optional search term
            tags: Optional list of tags to filter by
            featured_only: Only return featured worlds
            sort_by: Sort method ("popular", "recent", "rating")
            limit: Maximum number of results
            
        Returns:
            List of world dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT 
                        pw.id,
                        pw.world_name as name,
                        pw.description,
                        pw.tags,
                        pw.thumbnail_url,
                        pw.published_at,
                        pw.featured,
                        u.username as creator_username,
                        COALESCE(ws.total_sessions, 0) as total_sessions,
                        COALESCE(ws.avg_rating, 0.0) as avg_rating,
                        COALESCE(ws.total_ratings, 0) as total_ratings
                    FROM published_worlds pw
                    LEFT JOIN users u ON pw.creator_user_id = u.id
                    LEFT JOIN world_stats ws ON pw.id = ws.world_id
                    WHERE pw.is_active = 1
                """
                
                params = []
                
                # Add search filter
                if search_query:
                    query += " AND (pw.world_name LIKE ? OR pw.description LIKE ?)"
                    search_term = f"%{search_query}%"
                    params.extend([search_term, search_term])
                
                # Add featured filter
                if featured_only:
                    query += " AND pw.featured = 1"
                
                # Add tag filter
                if tags:
                    for tag in tags:
                        query += " AND pw.tags LIKE ?"
                        params.append(f'%"{tag}"%')
                
                # Add sorting
                if sort_by == "popular":
                    query += " ORDER BY COALESCE(ws.total_sessions, 0) DESC"
                elif sort_by == "recent":
                    query += " ORDER BY pw.published_at DESC"
                elif sort_by == "rating":
                    query += " ORDER BY COALESCE(ws.avg_rating, 0.0) DESC"
                
                query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries and parse tags
                worlds = []
                for row in rows:
                    world_dict = dict(row)
                    world_dict['tags'] = json.loads(world_dict['tags'])
                    worlds.append(world_dict)
                
                return worlds
                
        except Exception as e:
            logger.error(f"Failed to get published worlds: {e}")
            return []

    def get_published_characters(
        self,
        world_id: int,
        search_query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "popular",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get published characters in a world with filtering options
        
        Args:
            world_id: ID of the world to get characters from
            search_query: Optional search term
            tags: Optional list of tags to filter by
            sort_by: Sort method ("popular", "recent", "rating")
            limit: Maximum number of results
            
        Returns:
            List of character dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT 
                        pc.id,
                        pc.character_name as name,
                        pc.description,
                        pc.tags,
                        pc.character_data,
                        pc.published_at,
                        pc.featured,
                        u.username as creator_username,
                        COALESCE(cs.download_count, 0) as download_count,
                        COALESCE(cs.session_count, 0) as session_count,
                        COALESCE(cs.avg_rating, 0.0) as avg_rating,
                        COALESCE(cs.total_ratings, 0) as total_ratings
                    FROM published_characters pc
                    LEFT JOIN users u ON pc.creator_user_id = u.id
                    LEFT JOIN character_stats cs ON pc.id = cs.character_id
                    WHERE pc.world_id = ? AND pc.is_active = 1
                """
                
                params = [world_id]
                
                # Add search filter
                if search_query:
                    query += " AND (pc.character_name LIKE ? OR pc.description LIKE ?)"
                    search_term = f"%{search_query}%"
                    params.extend([search_term, search_term])
                
                # Add tag filter
                if tags:
                    for tag in tags:
                        query += " AND pc.tags LIKE ?"
                        params.append(f'%"{tag}"%')
                
                # Add sorting
                if sort_by == "popular":
                    query += " ORDER BY COALESCE(cs.session_count, 0) DESC"
                elif sort_by == "recent":
                    query += " ORDER BY pc.published_at DESC"
                elif sort_by == "rating":
                    query += " ORDER BY COALESCE(cs.avg_rating, 0.0) DESC"
                
                query += f" LIMIT {limit}"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries and parse tags
                characters = []
                for row in rows:
                    char_dict = dict(row)
                    char_dict['tags'] = json.loads(char_dict['tags'])
                    if char_dict['character_data']:
                        char_dict['character_data'] = json.loads(char_dict['character_data'])
                    characters.append(char_dict)
                
                return characters
                
        except Exception as e:
            logger.error(f"Failed to get published characters for world {world_id}: {e}")
            return []

    def rate_world(self, world_id: int, user_id: int, rating: int, review_text: Optional[str] = None) -> bool:
        """Rate a world"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc)
                
                # Insert or update rating
                cursor.execute("""
                    INSERT OR REPLACE INTO world_ratings 
                    (world_id, user_id, rating, review_text, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (world_id, user_id, rating, review_text, now.isoformat()))
                
                # Update world stats
                cursor.execute("""
                    UPDATE world_stats SET
                        total_ratings = (
                            SELECT COUNT(*) FROM world_ratings WHERE world_id = ?
                        ),
                        avg_rating = (
                            SELECT AVG(rating) FROM world_ratings WHERE world_id = ?
                        ),
                        last_updated = ?
                    WHERE world_id = ?
                """, (world_id, world_id, now.isoformat(), world_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to rate world: {e}")
            return False

    def rate_character(self, character_id: int, user_id: int, rating: int, review_text: Optional[str] = None) -> bool:
        """Rate a character"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc)
                
                # Insert or update rating
                cursor.execute("""
                    INSERT OR REPLACE INTO character_ratings 
                    (character_id, user_id, rating, review_text, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (character_id, user_id, rating, review_text, now.isoformat()))
                
                # Update character stats
                cursor.execute("""
                    UPDATE character_stats SET
                        total_ratings = (
                            SELECT COUNT(*) FROM character_ratings WHERE character_id = ?
                        ),
                        avg_rating = (
                            SELECT AVG(rating) FROM character_ratings WHERE character_id = ?
                        ),
                        last_updated = ?
                    WHERE character_id = ?
                """, (character_id, character_id, now.isoformat(), character_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to rate character: {e}")
            return False
    
    def get_world_stats(self, world_id: int) -> Dict[str, Any]:
        """Get statistics for a world"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM world_stats WHERE world_id = ?
                """, (world_id,))
                
                row = cursor.fetchone()
                return dict(row) if row else {}
                
        except Exception as e:
            logger.error(f"Failed to get world stats: {e}")
            return {}

    def get_character_stats(self, character_id: int) -> Dict[str, Any]:
        """Get statistics for a character"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM character_stats WHERE character_id = ?
                """, (character_id,))
                
                row = cursor.fetchone()
                return dict(row) if row else {}
                
        except Exception as e:
            logger.error(f"Failed to get character stats: {e}")
            return {}

    def create_play_session(
        self,
        world_id: int,
        user_id: int,
        session_name: str,
        privacy_setting: str = "private"
    ) -> SessionResult:
        """
        Create a new play session
        
        Args:
            world_id: ID of the world to play in
            user_id: ID of the user creating the session
            session_name: Name for the session
            privacy_setting: Privacy setting (private, friends, public)
            
        Returns:
            SessionResult with success status and session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc)
                
                # Create session
                cursor.execute("""
                    INSERT INTO play_sessions 
                    (world_id, user_id, session_name, privacy_setting, created_at, last_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (world_id, user_id, session_name, privacy_setting, now.isoformat(), now.isoformat()))
                
                session_id = cursor.lastrowid
                
                # Update world stats
                cursor.execute("""
                    UPDATE world_stats SET
                        total_sessions = total_sessions + 1,
                        active_sessions = active_sessions + 1,
                        last_updated = ?
                    WHERE world_id = ?
                """, (now.isoformat(), world_id))
                
                conn.commit()
                
                logger.info(f"Created play session {session_id} for world {world_id}")
                
                return SessionResult(
                    success=True,
                    session_id=session_id
                )
                
        except Exception as e:
            logger.error(f"Failed to create play session: {e}")
            return SessionResult(
                success=False,
                error_message=f"Failed to create session: {str(e)}"
            )

    def update_session_status(self, session_id: int, status: str) -> bool:
        """Update session status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc)
                
                cursor.execute("""
                    UPDATE play_sessions SET
                        status = ?,
                        last_active = ?
                    WHERE id = ?
                """, (status, now.isoformat(), session_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")
            return False

    def add_character_to_session(self, session_id: int, character_name: str, character_data: Dict[str, Any]) -> bool:
        """Add a character to a play session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc)
                
                cursor.execute("""
                    INSERT INTO session_characters 
                    (session_id, character_name, character_data, added_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, character_name, json.dumps(character_data), now.isoformat()))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to add character to session: {e}")
            return False

    def get_session_characters(self, session_id: int) -> List[Dict[str, Any]]:
        """Get characters in a play session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM session_characters WHERE session_id = ?
                    ORDER BY added_at ASC
                """, (session_id,))
                
                rows = cursor.fetchall()
                characters = []
                for row in rows:
                    char_dict = dict(row)
                    if char_dict['character_data']:
                        char_dict['character_data'] = json.loads(char_dict['character_data'])
                    characters.append(char_dict)
                
                return characters
                
        except Exception as e:
            logger.error(f"Failed to get session characters: {e}")
            return []
    
    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get play sessions for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        ps.*,
                        pw.world_name,
                        pw.description as world_description
                    FROM play_sessions ps
                    JOIN published_worlds pw ON ps.world_id = pw.id
                    WHERE ps.user_id = ? AND ps.is_active = 1
                    ORDER BY ps.last_active DESC
                """, (user_id,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []

    def analyze_character_compatibility(self, character_ids: List[int]) -> CompatibilityAnalysis:
        """Analyze compatibility between selected characters"""
        try:
            if len(character_ids) < 2:
                return CompatibilityAnalysis(
                    score=1.0,
                    reasoning="Single character selected",
                    relationship_type="neutral"
                )
            
            # Get character data
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                placeholders = ",".join("?" * len(character_ids))
                cursor.execute(f"""
                    SELECT pc.*, cs.avg_rating, cs.session_count
                    FROM published_characters pc
                    LEFT JOIN character_stats cs ON pc.id = cs.character_id
                    WHERE pc.id IN ({placeholders})
                """, character_ids)
                
                characters = [dict(row) for row in cursor.fetchall()]
            
            # Simple compatibility analysis based on tags
            all_tags = []
            for char in characters:
                if char['tags']:
                    all_tags.extend(json.loads(char['tags']))
            
            # Analyze tag overlap and conflicts
            positive_tags = ['friendly', 'helpful', 'caring', 'supportive', 'noble']
            negative_tags = ['evil', 'antagonistic', 'hostile', 'cruel', 'selfish']
            
            positive_count = sum(1 for tag in all_tags if tag in positive_tags)
            negative_count = sum(1 for tag in all_tags if tag in negative_tags)
            
            if positive_count > negative_count:
                score = 0.7 + (positive_count * 0.1)
                relationship_type = "complementary"
                reasoning = "Characters have complementary positive traits"
            elif negative_count > positive_count:
                score = 0.3 + (negative_count * 0.1)  # Conflict can be interesting
                relationship_type = "conflicting"
                reasoning = "Characters have conflicting traits that create dramatic tension"
            else:
                score = 0.5
                relationship_type = "neutral"
                reasoning = "Characters have balanced traits"
            
            return CompatibilityAnalysis(
                score=min(1.0, score),
                reasoning=reasoning,
                relationship_type=relationship_type
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze character compatibility: {e}")
            return CompatibilityAnalysis(
                score=0.5,
                reasoning="Unable to analyze compatibility",
                relationship_type="neutral"
            ) 
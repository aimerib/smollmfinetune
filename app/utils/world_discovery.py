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
class WorldDiscoveryResult:
    """Result of world discovery operations"""
    success: bool
    worlds: List[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.worlds is None:
            self.worlds = []


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
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published_worlds_creator ON published_worlds(creator_user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_published_worlds_featured ON published_worlds(featured)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_world_ratings_world ON world_ratings(world_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_play_sessions_user ON play_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_play_sessions_world ON play_sessions(world_id)")
            
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
            logger.error(f"Failed to publish world: {e}")
            return PublishResult(
                success=False,
                error_message=f"Failed to publish world: {str(e)}"
            )
    
    def get_published_worlds(
        self,
        search_query: str = "",
        tags: List[str] = None,
        featured_only: bool = False,
        sort_by: str = "popular",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get published worlds with filtering and search
        
        Args:
            search_query: Text search in world names and descriptions
            tags: Filter by tags
            featured_only: Only return featured worlds
            sort_by: Sort order (popular, recent, rating)
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
    
    def rate_world(
        self,
        world_id: int,
        user_id: int,
        rating: int,
        review_text: Optional[str] = None
    ) -> bool:
        """
        Rate a world
        
        Args:
            world_id: ID of the world to rate
            user_id: ID of the user rating
            rating: Rating from 1-5
            review_text: Optional review text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not (1 <= rating <= 5):
                return False
            
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
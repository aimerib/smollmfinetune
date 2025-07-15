"""
User Preferences Manager for Multimodal Studio

Provides comprehensive user preference management including:
- User settings persistence
- Workspace configuration management
- Theme and UI customization
- Keyboard shortcuts configuration
- Auto-save and backup settings
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class ExportFormat(str, Enum):
    HUGGINGFACE = "huggingface"
    JSONL = "jsonl"
    PYTORCH = "pytorch"
    CUSTOM = "custom"

@dataclass
class UserPreferences:
    """User preferences data structure"""
    user_id: str
    theme: Theme = Theme.LIGHT
    auto_save_interval: int = 30  # seconds
    notifications_enabled: bool = True
    keyboard_shortcuts_enabled: bool = True
    default_export_format: ExportFormat = ExportFormat.JSONL
    workspace_layout: Dict[str, Any] = None
    language: str = "en"
    timezone: str = "UTC"
    dashboard_widgets: List[str] = None
    advanced_mode: bool = False
    performance_mode: bool = False
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.workspace_layout is None:
            self.workspace_layout = self._default_workspace_layout()
        if self.dashboard_widgets is None:
            self.dashboard_widgets = self._default_dashboard_widgets()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def _default_workspace_layout(self) -> Dict[str, Any]:
        """Default workspace layout configuration"""
        return {
            "sidebar_width": 250,
            "panel_layout": "horizontal",
            "show_minimap": True,
            "show_line_numbers": True,
            "word_wrap": True,
            "font_size": 14,
            "font_family": "Monaco, 'Courier New', monospace",
            "tab_size": 2,
            "auto_close_brackets": True,
            "highlight_active_line": True,
            "show_whitespace": False
        }
    
    def _default_dashboard_widgets(self) -> List[str]:
        """Default dashboard widgets"""
        return [
            "recent_jobs",
            "system_metrics",
            "quality_overview",
            "export_status",
            "performance_summary"
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['theme'] = self.theme.value
        data['default_export_format'] = self.default_export_format.value
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create from dictionary"""
        # Convert string enums back to enum objects
        if 'theme' in data:
            data['theme'] = Theme(data['theme'])
        if 'default_export_format' in data:
            data['default_export_format'] = ExportFormat(data['default_export_format'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)

@dataclass
class WorkspaceConfig:
    """Workspace configuration data structure"""
    id: str
    name: str
    user_id: str
    description: str = ""
    configuration: Dict[str, Any] = None
    is_default: bool = False
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkspaceConfig':
        """Create from dictionary"""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)

@dataclass
class KeyboardShortcut:
    """Keyboard shortcut configuration"""
    id: str
    name: str
    description: str
    key_combination: str
    action: str
    category: str
    enabled: bool = True
    user_defined: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyboardShortcut':
        """Create from dictionary"""
        return cls(**data)

class PreferencesStorage:
    """Handles persistence of preferences data"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.preferences_dir = storage_path / "preferences"
        self.workspaces_dir = storage_path / "workspaces"
        self.shortcuts_dir = storage_path / "shortcuts"
        
        for dir_path in [self.preferences_dir, self.workspaces_dir, self.shortcuts_dir]:
            dir_path.mkdir(exist_ok=True)
    
    async def save_preferences(self, preferences: UserPreferences) -> None:
        """Save user preferences to storage"""
        file_path = self.preferences_dir / f"{preferences.user_id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(preferences.to_dict(), f, indent=2)
            logger.debug(f"Saved preferences for user {preferences.user_id}")
        except Exception as e:
            logger.error(f"Failed to save preferences for user {preferences.user_id}: {e}")
            raise
    
    async def load_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Load user preferences from storage"""
        file_path = self.preferences_dir / f"{user_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return UserPreferences.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load preferences for user {user_id}: {e}")
            return None
    
    async def save_workspace(self, workspace: WorkspaceConfig) -> None:
        """Save workspace configuration"""
        file_path = self.workspaces_dir / f"{workspace.id}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(workspace.to_dict(), f, indent=2)
            logger.debug(f"Saved workspace {workspace.id}")
        except Exception as e:
            logger.error(f"Failed to save workspace {workspace.id}: {e}")
            raise
    
    async def load_workspace(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Load workspace configuration"""
        file_path = self.workspaces_dir / f"{workspace_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return WorkspaceConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load workspace {workspace_id}: {e}")
            return None
    
    async def load_user_workspaces(self, user_id: str) -> List[WorkspaceConfig]:
        """Load all workspaces for a user"""
        workspaces = []
        
        for file_path in self.workspaces_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if data.get('user_id') == user_id:
                    workspace = WorkspaceConfig.from_dict(data)
                    workspaces.append(workspace)
                    
            except Exception as e:
                logger.error(f"Failed to load workspace from {file_path}: {e}")
                continue
        
        return workspaces
    
    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete workspace configuration"""
        file_path = self.workspaces_dir / f"{workspace_id}.json"
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted workspace {workspace_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
            return False
    
    async def save_shortcuts(self, user_id: str, shortcuts: List[KeyboardShortcut]) -> None:
        """Save keyboard shortcuts configuration"""
        file_path = self.shortcuts_dir / f"{user_id}.json"
        
        try:
            data = [shortcut.to_dict() for shortcut in shortcuts]
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved shortcuts for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save shortcuts for user {user_id}: {e}")
            raise
    
    async def load_shortcuts(self, user_id: str) -> List[KeyboardShortcut]:
        """Load keyboard shortcuts configuration"""
        file_path = self.shortcuts_dir / f"{user_id}.json"
        
        if not file_path.exists():
            return self._default_shortcuts()
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return [KeyboardShortcut.from_dict(item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load shortcuts for user {user_id}: {e}")
            return self._default_shortcuts()
    
    def _default_shortcuts(self) -> List[KeyboardShortcut]:
        """Default keyboard shortcuts"""
        return [
            KeyboardShortcut(
                id="help",
                name="Show Help",
                description="Display keyboard shortcuts help",
                key_combination="?",
                action="show_help",
                category="general"
            ),
            KeyboardShortcut(
                id="new_job",
                name="New Job",
                description="Create a new generation job",
                key_combination="Ctrl+N",
                action="new_job",
                category="jobs"
            ),
            KeyboardShortcut(
                id="save",
                name="Save",
                description="Save current work",
                key_combination="Ctrl+S",
                action="save",
                category="general"
            ),
            KeyboardShortcut(
                id="preferences",
                name="Preferences",
                description="Open preferences dialog",
                key_combination="Ctrl+,",
                action="show_preferences",
                category="general"
            ),
            KeyboardShortcut(
                id="fullscreen",
                name="Toggle Fullscreen",
                description="Toggle fullscreen mode",
                key_combination="F11",
                action="toggle_fullscreen",
                category="view"
            ),
            KeyboardShortcut(
                id="next_tab",
                name="Next Tab",
                description="Switch to next tab",
                key_combination="Ctrl+Tab",
                action="next_tab",
                category="navigation"
            ),
            KeyboardShortcut(
                id="prev_tab",
                name="Previous Tab",
                description="Switch to previous tab",
                key_combination="Ctrl+Shift+Tab",
                action="prev_tab",
                category="navigation"
            ),
            KeyboardShortcut(
                id="pause_resume",
                name="Pause/Resume",
                description="Pause or resume current job",
                key_combination="Space",
                action="pause_resume",
                category="jobs"
            ),
            KeyboardShortcut(
                id="cancel_job",
                name="Cancel Job",
                description="Cancel selected job",
                key_combination="Delete",
                action="cancel_job",
                category="jobs"
            ),
            KeyboardShortcut(
                id="search",
                name="Search",
                description="Open search dialog",
                key_combination="Ctrl+F",
                action="search",
                category="general"
            )
        ]

class UserPreferencesManager:
    """Main preferences management service"""
    
    def __init__(self):
        self.storage = PreferencesStorage(Path("data/user_preferences"))
        self.preferences_cache: Dict[str, UserPreferences] = {}
        self.workspaces_cache: Dict[str, List[WorkspaceConfig]] = {}
        
    async def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating defaults if needed"""
        
        # Check cache first
        if user_id in self.preferences_cache:
            return self.preferences_cache[user_id]
        
        # Try to load from storage
        preferences = await self.storage.load_preferences(user_id)
        
        if preferences is None:
            # Create default preferences
            preferences = UserPreferences(user_id=user_id)
            await self.save_preferences(preferences)
        
        # Cache and return
        self.preferences_cache[user_id] = preferences
        return preferences
    
    async def save_preferences(self, preferences: UserPreferences) -> UserPreferences:
        """Save user preferences"""
        preferences.updated_at = datetime.utcnow()
        
        await self.storage.save_preferences(preferences)
        
        # Update cache
        self.preferences_cache[preferences.user_id] = preferences
        
        logger.info(f"Updated preferences for user {preferences.user_id}")
        return preferences
    
    async def update_preference(
        self,
        user_id: str,
        key: str,
        value: Any
    ) -> UserPreferences:
        """Update a specific preference"""
        preferences = await self.get_preferences(user_id)
        
        if hasattr(preferences, key):
            setattr(preferences, key, value)
            return await self.save_preferences(preferences)
        else:
            raise ValueError(f"Unknown preference key: {key}")
    
    async def reset_preferences(self, user_id: str) -> UserPreferences:
        """Reset user preferences to defaults"""
        preferences = UserPreferences(user_id=user_id)
        return await self.save_preferences(preferences)
    
    async def get_workspaces(self, user_id: str) -> List[WorkspaceConfig]:
        """Get all workspaces for a user"""
        
        # Check cache first
        if user_id in self.workspaces_cache:
            return self.workspaces_cache[user_id]
        
        # Load from storage
        workspaces = await self.storage.load_user_workspaces(user_id)
        
        # Sort by updated_at descending
        workspaces.sort(key=lambda w: w.updated_at, reverse=True)
        
        # Cache and return
        self.workspaces_cache[user_id] = workspaces
        return workspaces
    
    async def create_workspace(
        self,
        user_id: str,
        name: str,
        configuration: Dict[str, Any],
        description: str = "",
        tags: List[str] = None
    ) -> WorkspaceConfig:
        """Create new workspace configuration"""
        
        workspace_id = str(uuid.uuid4())
        
        workspace = WorkspaceConfig(
            id=workspace_id,
            name=name,
            user_id=user_id,
            description=description,
            configuration=configuration,
            tags=tags or []
        )
        
        await self.storage.save_workspace(workspace)
        
        # Update cache
        if user_id in self.workspaces_cache:
            self.workspaces_cache[user_id].insert(0, workspace)
        else:
            self.workspaces_cache[user_id] = [workspace]
        
        logger.info(f"Created workspace {workspace_id} for user {user_id}")
        return workspace
    
    async def update_workspace(
        self,
        workspace_id: str,
        user_id: str,
        name: str,
        configuration: Dict[str, Any],
        description: str = "",
        tags: List[str] = None
    ) -> Optional[WorkspaceConfig]:
        """Update existing workspace configuration"""
        
        workspace = await self.storage.load_workspace(workspace_id)
        
        if not workspace or workspace.user_id != user_id:
            return None
        
        # Update fields
        workspace.name = name
        workspace.configuration = configuration
        workspace.description = description
        workspace.tags = tags or []
        workspace.updated_at = datetime.utcnow()
        
        await self.storage.save_workspace(workspace)
        
        # Update cache
        if user_id in self.workspaces_cache:
            for i, cached_workspace in enumerate(self.workspaces_cache[user_id]):
                if cached_workspace.id == workspace_id:
                    self.workspaces_cache[user_id][i] = workspace
                    break
        
        logger.info(f"Updated workspace {workspace_id}")
        return workspace
    
    async def delete_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Delete workspace configuration"""
        
        workspace = await self.storage.load_workspace(workspace_id)
        
        if not workspace or workspace.user_id != user_id:
            return False
        
        success = await self.storage.delete_workspace(workspace_id)
        
        if success:
            # Update cache
            if user_id in self.workspaces_cache:
                self.workspaces_cache[user_id] = [
                    w for w in self.workspaces_cache[user_id]
                    if w.id != workspace_id
                ]
            
            logger.info(f"Deleted workspace {workspace_id}")
        
        return success
    
    async def set_default_workspace(self, workspace_id: str, user_id: str) -> bool:
        """Set a workspace as the default for a user"""
        
        workspaces = await self.get_workspaces(user_id)
        
        workspace_found = False
        for workspace in workspaces:
            if workspace.id == workspace_id:
                workspace.is_default = True
                workspace_found = True
            else:
                workspace.is_default = False
            
            workspace.updated_at = datetime.utcnow()
            await self.storage.save_workspace(workspace)
        
        if workspace_found:
            # Update cache
            self.workspaces_cache[user_id] = workspaces
            logger.info(f"Set workspace {workspace_id} as default for user {user_id}")
        
        return workspace_found
    
    async def get_keyboard_shortcuts(self, user_id: str) -> List[KeyboardShortcut]:
        """Get keyboard shortcuts for user"""
        return await self.storage.load_shortcuts(user_id)
    
    async def update_keyboard_shortcuts(
        self,
        user_id: str,
        shortcuts: List[KeyboardShortcut]
    ) -> None:
        """Update keyboard shortcuts for user"""
        await self.storage.save_shortcuts(user_id, shortcuts)
        logger.info(f"Updated keyboard shortcuts for user {user_id}")
    
    async def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for backup or migration"""
        
        preferences = await self.get_preferences(user_id)
        workspaces = await self.get_workspaces(user_id)
        shortcuts = await self.get_keyboard_shortcuts(user_id)
        
        return {
            'user_id': user_id,
            'exported_at': datetime.utcnow().isoformat(),
            'preferences': preferences.to_dict(),
            'workspaces': [w.to_dict() for w in workspaces],
            'shortcuts': [s.to_dict() for s in shortcuts],
            'version': '1.0'
        }
    
    async def import_user_data(
        self,
        user_id: str,
        data: Dict[str, Any],
        overwrite: bool = False
    ) -> bool:
        """Import user data from backup"""
        
        try:
            # Validate data structure
            if data.get('version') != '1.0':
                raise ValueError("Unsupported data version")
            
            # Import preferences
            if 'preferences' in data:
                if overwrite or user_id not in self.preferences_cache:
                    preferences = UserPreferences.from_dict(data['preferences'])
                    preferences.user_id = user_id  # Ensure correct user ID
                    await self.save_preferences(preferences)
            
            # Import workspaces
            if 'workspaces' in data:
                if overwrite:
                    # Delete existing workspaces
                    existing_workspaces = await self.get_workspaces(user_id)
                    for workspace in existing_workspaces:
                        await self.delete_workspace(workspace.id, user_id)
                
                # Import new workspaces
                for workspace_data in data['workspaces']:
                    workspace = WorkspaceConfig.from_dict(workspace_data)
                    workspace.user_id = user_id  # Ensure correct user ID
                    workspace.id = str(uuid.uuid4())  # Generate new ID
                    await self.storage.save_workspace(workspace)
            
            # Import shortcuts
            if 'shortcuts' in data:
                shortcuts = [KeyboardShortcut.from_dict(s) for s in data['shortcuts']]
                await self.update_keyboard_shortcuts(user_id, shortcuts)
            
            # Clear cache to force reload
            if user_id in self.preferences_cache:
                del self.preferences_cache[user_id]
            if user_id in self.workspaces_cache:
                del self.workspaces_cache[user_id]
            
            logger.info(f"Imported user data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import user data for user {user_id}: {e}")
            return False
    
    async def cleanup_old_data(self, days: int = 365) -> None:
        """Cleanup old preferences data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # This is a placeholder for cleanup logic
        # In practice, you might want to archive old workspaces
        # or remove preferences for inactive users
        
        logger.info(f"Cleanup would remove data older than {cutoff_date}")
    
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's preferences and usage"""
        
        preferences = await self.get_preferences(user_id)
        workspaces = await self.get_workspaces(user_id)
        shortcuts = await self.get_keyboard_shortcuts(user_id)
        
        return {
            'preferences_last_updated': preferences.updated_at.isoformat(),
            'total_workspaces': len(workspaces),
            'default_workspace': next(
                (w.name for w in workspaces if w.is_default),
                None
            ),
            'custom_shortcuts': len([s for s in shortcuts if s.user_defined]),
            'theme': preferences.theme.value,
            'advanced_mode': preferences.advanced_mode,
            'auto_save_interval': preferences.auto_save_interval
        } 
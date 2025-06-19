import unittest
import tempfile
import shutil
from pathlib import Path
from app.utils.world import WorldManager, WorldLore, Faction, TimelineEvent, Place, NPC, PlaceEvent
from app.utils.character.character import CharacterManager


class TestWorldManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.world_manager = WorldManager(worlds_root=self.temp_dir)
        
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_world(self):
        """Test world creation"""
        # Test successful creation
        result = self.world_manager.create_world("Test World")
        self.assertTrue(result)
        
        # Check directory structure was created
        world_path = Path(self.temp_dir) / "Test World"
        self.assertTrue(world_path.exists())
        self.assertTrue((world_path / "characters").exists())
        self.assertTrue((world_path / "world_lore.json").exists())
        
        # Test duplicate creation fails gracefully
        result = self.world_manager.create_world("Test World")
        self.assertFalse(result)
        
        # Test empty name raises error
        with self.assertRaises(ValueError):
            self.world_manager.create_world("")
            
        with self.assertRaises(ValueError):
            self.world_manager.create_world("   ")
    
    def test_list_worlds(self):
        """Test world listing"""
        # Initially empty
        worlds = self.world_manager.list_worlds()
        self.assertEqual(worlds, [])
        
        # Create some worlds
        self.world_manager.create_world("World A")
        self.world_manager.create_world("World B")
        
        worlds = self.world_manager.list_worlds()
        self.assertEqual(sorted(worlds), ["World A", "World B"])
    
    def test_world_lore_with_faction_timeline(self):
        """Test creating, saving, and loading world with faction timeline"""
        world_name = "Fantasy World"
        
        # Create world
        self.assertTrue(self.world_manager.create_world(world_name))
        
        # Create complex lore with faction timelines
        moonfall_faction = Faction(
            name="Moonfall",
            timeline=[
                TimelineEvent(year=310, event="Moonfall rebellion"),
                TimelineEvent(year=315, event="Treaty of Shadows")
            ]
        )
        
        sunblade_faction = Faction(
            name="Sunblade",
            timeline=[
                TimelineEvent(year=300, event="Formation of Sunblade Order"),
                TimelineEvent(year=320, event="Great Purge")
            ]
        )
        
        # Create place with NPCs and events
        whispering_peaks = Place(
            name="Whispering Peaks",
            description="Tall, misty mountains shrouded in ancient magic",
            npcs=[
                NPC(name="Elara", description="Queen of the Dwarven folk, wise and ancient"),
                NPC(name="Thorin", description="Master blacksmith of the peaks")
            ],
            events=[
                PlaceEvent(name="Moonlight Festival", description="Every 100 years the Dwarven folk celebrate", random="true"),
                PlaceEvent(name="Trade Gathering", description="Monthly meeting of merchants", random="false")
            ]
        )
        
        # Create world lore
        lore = WorldLore(
            meta={"version": 1},
            facts={
                "magic_system": "Elemental runes",
                "currency": "Golden Crowns",
                "technology_level": "Medieval with magic"
            },
            factions=[moonfall_faction, sunblade_faction],
            timeline=[
                TimelineEvent(year=300, event="Founding of Solaris"),
                TimelineEvent(year=305, event="Discovery of the First Rune")
            ],
            places=[whispering_peaks]
        )
        
        # Save the lore
        self.world_manager.current_world = world_name
        self.world_manager.current_lore = lore
        result = self.world_manager.save_world_lore()
        self.assertTrue(result)
        
        # Version should be auto-incremented to 2
        self.assertEqual(lore.meta["version"], 2)
        
        # Create new manager instance to test loading
        new_manager = WorldManager(worlds_root=self.temp_dir)
        loaded_lore = new_manager.load_world(world_name)
        
        # Verify loaded lore matches saved lore
        self.assertIsNotNone(loaded_lore)
        self.assertEqual(loaded_lore.meta["version"], 2)
        
        # Check facts
        self.assertEqual(loaded_lore.facts["magic_system"], "Elemental runes")
        self.assertEqual(loaded_lore.facts["currency"], "Golden Crowns")
        
        # Check main timeline
        self.assertEqual(len(loaded_lore.timeline), 2)
        self.assertEqual(loaded_lore.timeline[0].year, 300)
        self.assertEqual(loaded_lore.timeline[0].event, "Founding of Solaris")
        
        # Check factions and their timelines
        self.assertEqual(len(loaded_lore.factions), 2)
        
        moonfall = next(f for f in loaded_lore.factions if f.name == "Moonfall")
        self.assertEqual(len(moonfall.timeline), 2)
        self.assertEqual(moonfall.timeline[0].year, 310)
        self.assertEqual(moonfall.timeline[0].event, "Moonfall rebellion")
        
        sunblade = next(f for f in loaded_lore.factions if f.name == "Sunblade")
        self.assertEqual(len(sunblade.timeline), 2)
        self.assertEqual(sunblade.timeline[1].year, 320)
        self.assertEqual(sunblade.timeline[1].event, "Great Purge")
        
        # Check places
        self.assertEqual(len(loaded_lore.places), 1)
        place = loaded_lore.places[0]
        self.assertEqual(place.name, "Whispering Peaks")
        self.assertEqual(len(place.npcs), 2)
        self.assertEqual(place.npcs[0].name, "Elara")
        self.assertEqual(len(place.events), 2)
        self.assertEqual(place.events[0].random, "true")
    
    def test_version_auto_increment(self):
        """Test that version auto-increments on save"""
        world_name = "Version Test"
        self.world_manager.create_world(world_name)
        
        # Load and modify
        lore = self.world_manager.load_world(world_name)
        self.assertEqual(lore.meta["version"], 1)
        
        # Add some content and save
        lore.facts["test"] = "value"
        result = self.world_manager.save_world_lore()
        self.assertTrue(result)
        self.assertEqual(lore.meta["version"], 2)
        
        # Save again
        lore.facts["test2"] = "value2"
        result = self.world_manager.save_world_lore()
        self.assertTrue(result)
        self.assertEqual(lore.meta["version"], 3)
    
    def test_get_world_paths(self):
        """Test world path helper methods"""
        world_name = "Path Test"
        self.world_manager.create_world(world_name)
        
        world_path = self.world_manager.get_world_path(world_name)
        expected_path = Path(self.temp_dir) / world_name
        self.assertEqual(world_path, expected_path)
        
        characters_path = self.world_manager.get_characters_path(world_name)
        expected_characters_path = expected_path / "characters"
        self.assertEqual(characters_path, expected_characters_path)
    
    def test_load_nonexistent_world(self):
        """Test loading a world that doesn't exist"""
        result = self.world_manager.load_world("Nonexistent World")
        self.assertIsNone(result)
    
    def test_save_without_current_world(self):
        """Test saving when no current world is set"""
        result = self.world_manager.save_world_lore()
        self.assertFalse(result)
    
    def test_character_manager_integration(self):
        """Test that CharacterManager integrates correctly with WorldManager"""
        # Create a character manager with our world manager
        char_manager = CharacterManager(world_manager=self.world_manager)
        
        # Should create default world automatically
        worlds = self.world_manager.list_worlds()
        self.assertEqual(len(worlds), 1)
        self.assertEqual(worlds[0], "Default World")
        self.assertEqual(char_manager.get_current_world(), "Default World")
        
        # Create a custom world with lore
        self.world_manager.create_world("Fantasy Realm")
        lore = WorldLore(
            meta={"version": 1},
            facts={"magic_system": "Elemental runes", "setting": "High fantasy"},
            timeline=[TimelineEvent(year=1000, event="Age of Heroes begins")]
        )
        self.world_manager.current_world = "Fantasy Realm"
        self.world_manager.current_lore = lore
        self.world_manager.save_world_lore()
        
        # Switch character manager to this world
        result = char_manager.set_current_world("Fantasy Realm")
        self.assertTrue(result)
        self.assertEqual(char_manager.get_current_world(), "Fantasy Realm")
        
        # Test world lore retrieval
        world_lore = char_manager.get_world_lore()
        self.assertIsNotNone(world_lore)
        self.assertEqual(world_lore["facts"]["magic_system"], "Elemental runes")
        self.assertEqual(len(world_lore["timeline"]), 1)
        self.assertEqual(world_lore["timeline"][0]["year"], 1000)
        
        # Test character card generation with world context
        test_card = {
            "name": "Aria the Mage",
            "description": "A powerful elementalist",
            "personality": "Wise but impulsive"
        }
        char_manager.load_character_card(test_card)
        
        # Generate card block with world context
        card_block = char_manager.make_card_block(include_world_context=True)
        self.assertIn("### <CHAR_CARD>", card_block)
        self.assertIn("Name: Aria the Mage", card_block)
        self.assertIn("### <WORLD_CONTEXT>", card_block)
        self.assertIn("World: Fantasy Realm", card_block)
        self.assertIn("magic_system: Elemental runes", card_block)
        self.assertIn("Year 1000: Age of Heroes begins", card_block)
        self.assertIn("<|endofworld|>", card_block)
        
        # Test without world context
        card_block_no_context = char_manager.make_card_block(include_world_context=False)
        self.assertNotIn("### <WORLD_CONTEXT>", card_block_no_context)
        
        # Test character summary includes world
        summary = char_manager.get_character_summary()
        self.assertEqual(summary["current_world"], "Fantasy Realm")


if __name__ == '__main__':
    unittest.main() 
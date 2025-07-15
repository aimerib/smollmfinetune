import unittest
import pytest
import tempfile

from backend.app.services.character.models import CharacterCore, Personality, Relationship, llm_estimate_big5
from backend.app.services.character.character import CharacterManager
from backend.app.services.world.world import WorldManager


class TestCharacterCoreModels(unittest.TestCase):
    """Test CharacterCore Pydantic models"""
    
    def test_personality_model(self):
        """Test Personality model with default values"""
        personality = Personality()
        
        # Check default values
        self.assertEqual(personality.openness, 0.5)
        self.assertEqual(personality.conscientiousness, 0.5)
        self.assertEqual(personality.extraversion, 0.5)
        self.assertEqual(personality.agreeableness, 0.5)
        self.assertEqual(personality.neuroticism, 0.5)
    
    def test_personality_validation(self):
        """Test Personality model validation"""
        # Valid values
        personality = Personality(
            openness=0.8,
            conscientiousness=0.3,
            extraversion=1.0,
            agreeableness=0.0,
            neuroticism=0.2
        )
        
        self.assertEqual(personality.openness, 0.8)
        self.assertEqual(personality.conscientiousness, 0.3)
        
        # Test invalid values are caught by Pydantic
        with self.assertRaises(ValueError):
            Personality(openness=1.5)  # Should fail validation
        
        with self.assertRaises(ValueError):
            Personality(neuroticism=-0.1)  # Should fail validation
    
    def test_relationship_model(self):
        """Test Relationship model"""
        rel = Relationship(name="Alice", affinity=75)
        
        self.assertEqual(rel.name, "Alice")
        self.assertEqual(rel.affinity, 75)
        
        # Test validation
        with self.assertRaises(ValueError):
            Relationship(name="Bob", affinity=150)  # Should fail validation
        
        with self.assertRaises(ValueError):
            Relationship(name="Charlie", affinity=-150)  # Should fail validation
    
    def test_character_core_model(self):
        """Test CharacterCore model with all fields"""
        personality = Personality(openness=0.7, conscientiousness=0.6)
        relationships = [
            Relationship(name="Friend", affinity=50),
            Relationship(name="Rival", affinity=-30)
        ]
        
        core = CharacterCore(
            name="Test Character",
            description="A test character",
            scenario="Testing scenario",
            backstory="Born in tests",
            appearance="Tall and pixelated",
            personality_traits=personality,
            goals=["Pass tests", "Be helpful"],
            relationships=relationships,
            tags=["test", "example"],
            imports={"source": "test_suite"}
        )
        
        self.assertEqual(core.name, "Test Character")
        self.assertEqual(core.description, "A test character")
        self.assertEqual(core.appearance, "Tall and pixelated")
        self.assertEqual(len(core.goals), 2)
        self.assertEqual(len(core.relationships), 2)
        self.assertEqual(core.personality_traits.openness, 0.7)
    
    def test_character_core_defaults(self):
        """Test CharacterCore with minimal required fields"""
        core = CharacterCore(
            name="Minimal Character",
            description="Basic description"
        )
        
        self.assertEqual(core.name, "Minimal Character")
        self.assertEqual(core.scenario, "")
        self.assertEqual(core.backstory, "")
        self.assertEqual(core.appearance, "")
        self.assertEqual(len(core.goals), 0)
        self.assertEqual(len(core.relationships), 0)
        self.assertEqual(len(core.tags), 0)
        self.assertIsInstance(core.personality_traits, Personality)


class TestCharacterManager(unittest.TestCase):
    """Test CharacterManager with CharacterCore functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.world_manager = WorldManager(worlds_root=self.temp_dir)
        self.char_manager = CharacterManager(world_manager=self.world_manager)
        
        # Create test world
        self.world_manager.create_world("Test World")
        self.char_manager.set_current_world("Test World")
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sillytavern_import(self):
        """Test importing SillyTavern card to CharacterCore"""
        # Sample SillyTavern card data
        card_data = {
            "name": "Test Hero",
            "description": "A brave adventurer with a sword. They are skilled in combat and magic.",
            "personality": "Confident, brave, caring",
            "scenario": "Standing in a tavern looking for work",
            "mes_example": "*draws sword confidently* I'm ready for any challenge!"
        }
        
        # This would normally require an async context, but for testing we'll mock it
        # In a real scenario, we'd need to handle the async call to llm_estimate_big5
        # For now, let's test the synchronous parts
        
        # Test validation first
        is_valid, error = self.char_manager.validate_character_card(card_data)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Test legacy loading
        loaded_card = self.char_manager.load_character_card(card_data)
        self.assertEqual(loaded_card["name"], "Test Hero")
        self.assertIn("brave adventurer", loaded_card["description"])
    
    def test_character_save_load(self):
        """Test saving and loading CharacterCore"""
        # Create a test CharacterCore
        personality = Personality(openness=0.8, extraversion=0.7)
        relationships = [Relationship(name="Mentor", affinity=80)]
        
        core = CharacterCore(
            name="Saved Character",
            description="A character for save/load testing",
            scenario="In a test environment",
            backstory="Created for testing purposes",
            appearance="Looks like test data",
            personality_traits=personality,
            goals=["Be saved", "Be loaded"],
            relationships=relationships,
            tags=["test", "save", "load"],
            imports={"source": "test"}
        )
        
        # Save the character
        success = self.char_manager.save_character(core)
        self.assertTrue(success)
        
        # Check that files were created
        world_path = self.world_manager.get_world_path("Test World")
        char_folder = world_path / "characters" / "Saved Character"
        
        self.assertTrue(char_folder.exists())
        self.assertTrue((char_folder / "character_core.json").exists())
        self.assertTrue((char_folder / "assets").exists())
        
        # Load the character back
        loaded_core = self.char_manager.load_character_core(char_folder)
        self.assertIsNotNone(loaded_core)
        self.assertEqual(loaded_core.name, core.name)
        self.assertEqual(loaded_core.description, core.description)
        self.assertEqual(loaded_core.personality_traits.openness, 0.8)
        self.assertEqual(len(loaded_core.relationships), 1)
        self.assertEqual(loaded_core.relationships[0].name, "Mentor")
    
    def test_list_characters_in_world(self):
        """Test listing characters in a world"""
        # Initially no characters
        characters = self.char_manager.list_characters_in_world()
        self.assertEqual(len(characters), 0)
        
        # Create and save a character
        core = CharacterCore(
            name="Listed Character",
            description="A character for list testing"
        )
        
        self.char_manager.save_character(core)
        
        # Should now have one character
        characters = self.char_manager.list_characters_in_world()
        self.assertEqual(len(characters), 1)
        self.assertEqual(characters[0], "Listed Character")
    
    def test_character_core_card_block(self):
        """Test enhanced card block generation with CharacterCore"""
        personality = Personality(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.9,
            agreeableness=0.7,
            neuroticism=0.3
        )
        relationships = [Relationship(name="Best Friend", affinity=90)]
        
        core = CharacterCore(
            name="Block Test Character",
            description="Used for testing card blocks",
            scenario="Testing card generation",
            backstory="Born in a unit test",
            appearance="Clearly defined features",
            personality_traits=personality,
            goals=["Generate good blocks", "Pass tests"],
            relationships=relationships
        )
        
        # Generate card block
        card_block = self.char_manager.make_card_block(character_core=core, include_world_context=False)
        
        # Check that expected elements are present
        self.assertIn("Block Test Character", card_block)
        self.assertIn("Appearance: Clearly defined features", card_block)
        self.assertIn("Personality (Big-5):", card_block)
        self.assertIn("0.8", card_block)  # Openness value
        self.assertIn("0.9", card_block)  # Extraversion value
        self.assertIn("Goals: Generate good blocks, Pass tests", card_block)
        self.assertIn("Best Friend (+90)", card_block)
    
    def test_character_summary_core(self):
        """Test character summary with CharacterCore"""
        core = CharacterCore(
            name="Summary Character",
            description="For summary testing",
            appearance="Well-defined",
            backstory="Has a backstory",
            goals=["Goal 1", "Goal 2"],
            tags=["tag1", "tag2", "tag3"]
        )
        
        summary = self.char_manager.get_character_summary(character_core=core)
        
        self.assertEqual(summary['name'], "Summary Character")
        self.assertEqual(summary['format'], 'character_core')
        self.assertTrue(summary['has_description'])
        self.assertTrue(summary['has_appearance'])
        self.assertTrue(summary['has_backstory'])
        self.assertEqual(summary['goals_count'], 2)
        self.assertEqual(summary['tags_count'], 3)
    
    def test_backward_compatibility(self):
        """Test that legacy methods still work"""
        # Test with legacy card format
        legacy_card = {
            "name": "Legacy Character",
            "description": "Old format character",
            "personality": "Traditional, old-school",
            "scenario": "In the past",
            "mes_example": "Hello from the past!"
        }
        
        loaded_card = self.char_manager.load_character_card(legacy_card)
        self.assertEqual(loaded_card["name"], "Legacy Character")
        
        # Should still generate card blocks
        card_block = self.char_manager.make_card_block()
        self.assertIn("Legacy Character", card_block)
        self.assertIn("Traditional, old-school", card_block)
        
        # Summary should work
        summary = self.char_manager.get_character_summary()
        self.assertEqual(summary['format'], 'legacy')


class AsyncTestCharacterCore(unittest.IsolatedAsyncioTestCase):
    """Async tests for CharacterCore functionality"""
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_estimate_big5_confident_character(self):
        """Test Big Five estimation for a confident character"""
        description = "A confident and outgoing character who loves new experiences and meeting people"
        mes_example = "*speaks boldly* Let's try something completely new! I love meeting new people and exploring different ideas."
        
        personality = await llm_estimate_big5(description, mes_example)
        
        # Should return a valid Personality object
        self.assertIsInstance(personality, Personality)
        
        # All values should be in valid range
        self.assertGreaterEqual(personality.openness, 0.0)
        self.assertLessEqual(personality.openness, 1.0)
        self.assertGreaterEqual(personality.conscientiousness, 0.0)
        self.assertLessEqual(personality.conscientiousness, 1.0)
        self.assertGreaterEqual(personality.extraversion, 0.0)
        self.assertLessEqual(personality.extraversion, 1.0)
        self.assertGreaterEqual(personality.agreeableness, 0.0)
        self.assertLessEqual(personality.agreeableness, 1.0)
        self.assertGreaterEqual(personality.neuroticism, 0.0)
        self.assertLessEqual(personality.neuroticism, 1.0)
        
        # Should reflect confident, outgoing traits
        self.assertGreater(personality.extraversion, 0.6)  # Should be high for outgoing
        self.assertGreater(personality.openness, 0.6)     # Should be high for new experiences
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_estimate_big5_introverted_character(self):
        """Test Big Five estimation for an introverted character"""
        description = "A quiet, thoughtful researcher who prefers working alone and values routine"
        mes_example = "*speaks softly* I prefer to work alone on my research. Routine helps me focus better."
        
        personality = await llm_estimate_big5(description, mes_example)
        
        # Should return a valid Personality object
        self.assertIsInstance(personality, Personality)
        
        # Should reflect introverted, organized traits
        self.assertLess(personality.extraversion, 0.5)      # Should be low for introverted
        self.assertGreater(personality.conscientiousness, 0.6)  # Should be high for routine-oriented
        
        # All values should still be in valid range
        for trait_name in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            trait_value = getattr(personality, trait_name)
            self.assertGreaterEqual(trait_value, 0.0)
            self.assertLessEqual(trait_value, 1.0)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_llm_estimate_big5_edge_cases(self):
        """Test Big Five estimation with minimal input"""
        description = "A person"
        mes_example = "Hello."
        
        personality = await llm_estimate_big5(description, mes_example)
        
        # Should still return a valid Personality object even with minimal input
        self.assertIsInstance(personality, Personality)
        
        # All values should be in valid range
        for trait_name in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            trait_value = getattr(personality, trait_name)
            self.assertGreaterEqual(trait_value, 0.0)
            self.assertLessEqual(trait_value, 1.0)


if __name__ == '__main__':
    unittest.main() 
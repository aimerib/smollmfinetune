import pytest
import json
import tempfile
import shutil
import torch
from pathlib import Path
from pydantic import ValidationError
from backend.app.narrative_engine.data_schema import Turn, DatasetSample


class TestDataSchema:
    """Test the Pydantic schema models for the Narrative Engine dataset format"""
    
    def test_turn_validation_basic(self):
        """Test basic Turn model validation"""
        # Valid turn
        turn_data = {
            "sender": "user",
            "text": "Hello there!",
            "channel": "text"
        }
        
        turn = Turn(**turn_data)
        assert turn.sender == "user"
        assert turn.text == "Hello there!"
        assert turn.channel == "text"
        assert turn.action is None
    
    def test_turn_validation_assistant_action(self):
        """Test Turn with action channel and structured action"""
        turn_data = {
            "sender": "assistant", 
            "text": "I'll check the weather for you.",
            "channel": "action",
            "action": {
                "tool": "weather_api",
                "parameters": {"location": "San Francisco"}
            }
        }
        
        turn = Turn(**turn_data)
        assert turn.sender == "assistant"
        assert turn.channel == "action"
        assert turn.action["tool"] == "weather_api"
    
    def test_turn_validation_invalid_sender(self):
        """Test that invalid sender raises ValidationError"""
        turn_data = {
            "sender": "invalid_sender",  # Should only be 'user' or 'assistant'
            "text": "Hello",
            "channel": "text"
        }
        
        with pytest.raises(ValidationError):
            Turn(**turn_data)
    
    def test_turn_validation_invalid_channel(self):
        """Test that invalid channel raises ValidationError"""
        turn_data = {
            "sender": "assistant",
            "text": "Hello",
            "channel": "invalid_channel"  # Should only be 'text' or 'action'
        }
        
        with pytest.raises(ValidationError):
            Turn(**turn_data)
    
    def test_dataset_sample_validation(self):
        """Test DatasetSample validation with complete example"""
        sample_data = {
            "session_id": "test_session_123",
            "persona_mix": {
                "DetectiveNoir": 0.7,
                "CosmicHorror": 0.3
            },
            "memory_slots": [
                "The character remembers their first case",
                "The user mentioned they like mysteries"
            ],
            "turns": [
                {
                    "sender": "user",
                    "text": "Tell me about your most interesting case.",
                    "channel": "text"
                },
                {
                    "sender": "assistant", 
                    "text": "Ah, let me tell you about the Blackwater incident...",
                    "channel": "text"
                },
                {
                    "sender": "assistant",
                    "text": "Let me search my case files for details.",
                    "channel": "action",
                    "action": {
                        "tool": "memory_search",
                        "query": "Blackwater incident details"
                    }
                }
            ]
        }
        
        sample = DatasetSample(**sample_data)
        assert sample.session_id == "test_session_123"
        assert len(sample.persona_mix) == 2
        assert sample.persona_mix["DetectiveNoir"] == 0.7
        assert len(sample.memory_slots) == 2
        assert len(sample.turns) == 3
        assert sample.turns[2].channel == "action"
    
    def test_dataset_sample_validation_empty_turns(self):
        """Test that DatasetSample requires at least one turn"""
        sample_data = {
            "session_id": "test_session",
            "persona_mix": {"Default": 1.0},
            "memory_slots": [],
            "turns": []  # Empty turns should fail validation
        }
        
        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(**sample_data)
        
        # Check that the error is about minimum length
        assert "at least 1 item" in str(exc_info.value) or "min_length" in str(exc_info.value)
    
    def test_dataset_sample_forbids_extra_fields(self):
        """Test that DatasetSample forbids extra fields (strict schema)"""
        sample_data = {
            "session_id": "test_session",
            "persona_mix": {"Default": 1.0},
            "memory_slots": [],
            "turns": [
                {
                    "sender": "user",
                    "text": "Hello",
                    "channel": "text"
                }
            ],
            "extra_field": "should_not_be_allowed"  # This should be forbidden
        }
        
        with pytest.raises(ValidationError) as exc_info:
            DatasetSample(**sample_data)
        
        # Check that the error mentions forbidden/extra fields
        error_msg = str(exc_info.value).lower()
        assert "extra" in error_msg or "forbidden" in error_msg or "not permitted" in error_msg


class TestCharacterConversion:
    """Test conversion from character folder format to DatasetSample format"""
    
    def setup_method(self):
        """Set up test character folder structure"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test character folder structure
        self.char_folder = self.temp_dir / "characters" / "TestDetective"
        self.char_folder.mkdir(parents=True)
        
        # Create character_core.json
        character_core = {
            "name": "Detective Morgan",
            "description": "A hardboiled detective with a mysterious past and keen intuition.",
            "scenario": "Working on a case involving supernatural elements in 1940s noir city.",
            "backstory": "Former police officer turned private investigator after witnessing unexplained events.",
            "appearance": "Tall, weathered face, always wears a trench coat and fedora.",
            "personality_traits": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.4,
                "agreeableness": 0.6,
                "neuroticism": 0.3
            },
            "goals": [
                "Solve the current supernatural case",
                "Uncover the truth about their past",
                "Protect innocent people"
            ],
            "relationships": [
                {"name": "Chief O'Brien", "affinity": 50},
                {"name": "Mysterious Client", "affinity": -20}
            ],
            "tags": ["detective", "noir", "supernatural"],
            "imports": {"source": "manual_creation", "version": "1.0"}
        }
        
        with open(self.char_folder / "character_core.json", 'w') as f:
            json.dump(character_core, f, indent=2)
        
        # Create mes_example.txt
        mes_example = """User: Tell me about your latest case.
Detective Morgan: *adjusts fedora and leans back in chair* Well, stranger, this one's different from the usual cheating spouses and missing persons. Started with what looked like a simple robbery, but the evidence... *taps fingers on desk* ...it doesn't add up. Witness says the thief just vanished into thin air. Course, witnesses lie, but the security footage... that's harder to explain.

User: What do you think is really going on?
Detective Morgan: *lights cigarette and stares out rain-streaked window* In this business, you learn that the impossible usually just means you're not asking the right questions. Could be staged, could be inside job with some clever tricks. But... *exhales smoke* ...there's something else. Something that makes my skin crawl when I walk past that crime scene. Call it detective instinct, call it superstition, but I've learned to trust that feeling.

User: Are you scared?
Detective Morgan: *chuckles grimly* Scared? Kid, I've been scared since the day I hung up my badge. Fear keeps you alive in this line of work. But scared enough to walk away? *meets your eyes* Not a chance. This city's got enough shadows without me adding to them by running."""
        
        with open(self.char_folder / "mes_example.txt", 'w') as f:
            f.write(mes_example)
        
        # Create assets folder
        (self.char_folder / "assets").mkdir()
    
    def teardown_method(self):
        """Clean up test files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_character_to_narrative_sample(self):
        """Test conversion from character folder to DatasetSample"""
        # This test will fail initially because the conversion function doesn't exist yet
        from scripts.convert_to_narrative_format import convert_character_to_dataset_sample
        
        # Call the conversion function (doesn't exist yet - will cause ImportError)
        dataset_sample = convert_character_to_dataset_sample(self.char_folder)
        
        # Validate the output is a proper DatasetSample
        assert isinstance(dataset_sample, DatasetSample)
        assert dataset_sample.session_id.startswith("char_")
        assert "DetectiveNoir" in dataset_sample.persona_mix or "Detective" in dataset_sample.persona_mix
        assert len(dataset_sample.turns) >= 2  # Should have user + assistant pairs
        
        # Check that character information is properly converted
        # Session ID should contain character name
        assert "detective_morgan" in dataset_sample.session_id.lower()
        
        # Should have meaningful dialogue content
        assistant_turns = [turn.text for turn in dataset_sample.turns if turn.sender == "assistant"]
        assert len(assistant_turns) >= 1
        assert any(len(turn) > 10 for turn in assistant_turns)  # Should have substantial content
        
        # Validate structure
        DatasetSample(**dataset_sample.model_dump())  # Should not raise ValidationError
    
    def test_character_conversion_with_missing_files(self):
        """Test conversion handles missing mes_example.txt gracefully"""
        from scripts.convert_to_narrative_format import convert_character_to_dataset_sample
        
        # Remove mes_example.txt
        (self.char_folder / "mes_example.txt").unlink()
        
        # Should still work but with synthetic conversation
        dataset_sample = convert_character_to_dataset_sample(self.char_folder)
        
        assert isinstance(dataset_sample, DatasetSample)
        assert len(dataset_sample.turns) >= 1
    
    def test_character_conversion_invalid_folder(self):
        """Test conversion fails gracefully with invalid character folder"""
        from scripts.convert_to_narrative_format import convert_character_to_dataset_sample
        
        invalid_folder = self.temp_dir / "nonexistent"
        
        with pytest.raises((FileNotFoundError, ValueError)):
            convert_character_to_dataset_sample(invalid_folder)


class TestDatasetProcessor:
    """Test the DatasetProcessor for tokenization and loss masking"""
    
    def setup_method(self):
        """Set up test data"""
        self.sample_data = DatasetSample(
            session_id="test_session_123",
            persona_mix={"DetectiveNoir": 0.7, "CosmicHorror": 0.3},
            memory_slots=[
                "Character is a detective with noir style",
                "The user is asking about cases"
            ],
            turns=[
                Turn(sender="user", text="What's your latest case?", channel="text"),
                Turn(sender="assistant", text="Let me tell you about a mysterious disappearance.", channel="text"),
                Turn(sender="assistant", text="I need to search my case files for details.", channel="action", 
                     action={"tool": "search", "query": "recent cases"})
            ]
        )
    
    def test_dataset_processor_import(self):
        """Test that DatasetProcessor can be imported"""
        # This will fail initially because DatasetProcessor doesn't exist
        from backend.app.narrative_engine.data_pipeline import DatasetProcessor
        assert DatasetProcessor is not None
    
    def test_dataset_processor_instantiation(self):
        """Test DatasetProcessor can be instantiated with a tokenizer"""
        from backend.app.narrative_engine.data_pipeline import DatasetProcessor
        from transformers import AutoTokenizer
        
        # Use a small tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        processor = DatasetProcessor(tokenizer=tokenizer)
        
        assert processor is not None
        assert processor.tokenizer == tokenizer
    
    def test_dataset_processor_output(self):
        """Test DatasetProcessor produces correctly shaped tensors with loss masks"""
        from backend.app.narrative_engine.data_pipeline import DatasetProcessor
        from transformers import AutoTokenizer
        
        # Initialize processor
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        processor = DatasetProcessor(tokenizer=tokenizer)
        
        # Process the sample
        result = processor.process_sample(self.sample_data)
        
        # Check output structure
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "loss_mask" in result
        
        # Check tensor shapes
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]
        loss_mask = result["loss_mask"]
        
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(loss_mask, torch.Tensor)
        
        # All tensors should have same sequence length
        seq_len = input_ids.shape[0]
        assert attention_mask.shape[0] == seq_len
        assert loss_mask.shape[0] == seq_len
        
        # loss_mask should be binary (0 or 1)
        assert torch.all((loss_mask == 0) | (loss_mask == 1))
        
        # loss_mask should have different values for text vs action channels
        # (action turns should have different mask values)
        assert loss_mask.sum() > 0  # Should have some tokens marked for loss
        assert loss_mask.sum() < seq_len  # Should not mark all tokens
    
    def test_dataset_processor_loss_mask_channels(self):
        """Test that loss_mask correctly identifies text vs action channels"""
        from backend.app.narrative_engine.data_pipeline import DatasetProcessor
        from transformers import AutoTokenizer
        
        # Initialize processor
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        processor = DatasetProcessor(tokenizer=tokenizer)
        
        # Process the sample
        result = processor.process_sample(self.sample_data)
        
        # Get the loss mask and channel information
        loss_mask = result["loss_mask"]
        channel_mask = result.get("channel_mask")  # Should indicate text (0) vs action (1) 
        
        # Should have channel information
        assert channel_mask is not None
        assert isinstance(channel_mask, torch.Tensor)
        assert channel_mask.shape[0] == loss_mask.shape[0]
        
        # Channel mask should be binary
        assert torch.all((channel_mask == 0) | (channel_mask == 1))
        
        # Should have both text (0) and action (1) tokens since we have both channel types
        assert 0 in channel_mask
        assert 1 in channel_mask
    
    def test_dataset_processor_batch_processing(self):
        """Test DatasetProcessor can handle multiple samples"""
        from backend.app.narrative_engine.data_pipeline import DatasetProcessor
        from transformers import AutoTokenizer
        
        # Initialize processor
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        processor = DatasetProcessor(tokenizer=tokenizer)
        
        # Create batch of samples
        samples = [self.sample_data, self.sample_data]  # Use same sample twice for simplicity
        
        # Process batch
        batch_result = processor.process_batch(samples)
        
        # Check batch structure
        assert isinstance(batch_result, dict)
        assert "input_ids" in batch_result
        assert "attention_mask" in batch_result
        assert "loss_mask" in batch_result
        
        # Should have batch dimension
        assert batch_result["input_ids"].shape[0] == 2  # Batch size 2
        assert batch_result["attention_mask"].shape[0] == 2
        assert batch_result["loss_mask"].shape[0] == 2 
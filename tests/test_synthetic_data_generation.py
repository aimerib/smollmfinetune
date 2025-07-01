"""
Tests for Synthetic Data Generation (R4-2.5)
Following TDD approach - testing the synthetic conversation generation system
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from narrative_engine.data_schema import DatasetSample, Turn


class TestSyntheticDataGeneration:
    """Test synthetic conversation generation system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Sample character for testing
        self.test_character = {
            "name": "Test Scholar",
            "description": "A knowledgeable researcher with expertise in ancient texts.",
            "personality_traits": {
                "openness": 0.9,
                "conscientiousness": 0.8,
                "extraversion": 0.4,
                "agreeableness": 0.7,
                "neuroticism": 0.2
            },
            "goals": ["Discover ancient knowledge", "Share wisdom with others"],
            "tags": ["scholar", "academic", "research"],
            "scenario": "Working in a library researching ancient manuscripts"
        }
    
    def test_synthetic_generator_import(self):
        """Test that SyntheticDataGenerator can be imported"""
        # This will fail initially because the module doesn't exist
        from scripts.generate_synthetic_conversations import SyntheticDataGenerator
        assert SyntheticDataGenerator is not None
    
    def test_synthetic_generator_instantiation(self):
        """Test SyntheticDataGenerator can be instantiated with configuration"""
        from scripts.generate_synthetic_conversations import SyntheticDataGenerator
        
        config = {
            "conversation_length": 6,
            "action_frequency": 0.3,
            "scenario_diversity": "high",
            "api_provider": "openai"
        }
        
        generator = SyntheticDataGenerator(config=config)
        assert generator is not None
        assert generator.config.conversation_length == 6
        assert generator.config.action_frequency == 0.3
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_generate_single_conversation(self):
        """Test generating a single synthetic conversation"""
        from scripts.generate_synthetic_conversations import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        
        result = await generator.generate_conversation(self.test_character)
        
        # Validate the result is a DatasetSample
        assert isinstance(result, DatasetSample)
        assert result.session_id.startswith("synthetic_")
        assert len(result.turns) >= 2  # Should have at least user and assistant turns
        
        # Should have meaningful conversation content
        conversation_text = " ".join([turn.text for turn in result.turns])
        assert len(conversation_text.strip()) > 20  # Substantial content
        
        # Should have both user and assistant turns
        senders = {turn.sender for turn in result.turns}
        assert "user" in senders
        assert "assistant" in senders
        
        # Validate persona mix was created
        assert len(result.persona_mix) > 0
        assert sum(result.persona_mix.values()) == pytest.approx(1.0, abs=1e-6)
        
        # Should reflect the scholar character
        assert any("scholar" in turn.text.lower() or "research" in turn.text.lower() 
                  for turn in result.turns if turn.sender == "assistant")
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_generate_diverse_scenarios(self):
        """Test generating conversations with different scenario templates"""
        from scripts.generate_synthetic_conversations import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(config={"scenario_diversity": "high"})
        
        # Test different scenario types
        scenarios = ["research_session", "teaching_moment"]
        
        results = []
        for scenario in scenarios:
            result = await generator.generate_conversation(
                self.test_character, scenario_type=scenario
            )
            
            assert isinstance(result, DatasetSample)
            results.append(result)
        
        # Results should be different for different scenarios
        if len(results) >= 2:
            conv1_text = " ".join([t.text for t in results[0].turns])
            conv2_text = " ".join([t.text for t in results[1].turns])
            
            # Should have some differences in content
            assert conv1_text != conv2_text
    
    def test_conversation_templates_loading(self):
        """Test loading and validation of conversation templates"""
        from scripts.generate_synthetic_conversations import ConversationTemplates
        
        templates = ConversationTemplates()
        
        # Should have default templates
        assert len(templates.get_templates()) > 0
        
        # Each template should have required fields
        for template in templates.get_templates():
            assert "name" in template
            assert "description" in template
            assert "prompts" in template
            assert "expected_actions" in template
            
            # Prompts should be a list
            assert isinstance(template["prompts"], list)
            assert len(template["prompts"]) > 0
    
    def test_action_frequency_control(self):
        """Test that action frequency parameter controls action channel usage"""
        from scripts.generate_synthetic_conversations import SyntheticDataGenerator
        
        # Low action frequency
        low_action_gen = SyntheticDataGenerator(config={"action_frequency": 0.1})
        
        # High action frequency  
        high_action_gen = SyntheticDataGenerator(config={"action_frequency": 0.8})
        
        # Generators should have different action frequency settings
        assert low_action_gen.config.action_frequency == 0.1
        assert high_action_gen.config.action_frequency == 0.8
    
    
    @pytest.mark.slow
    @pytest.mark.llm  
    @pytest.mark.evaluation
    async def test_batch_generation(self):
        """Test generating multiple conversations in batch"""
        from scripts.generate_synthetic_conversations import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        
        # Generate smaller batch for real LLM calls
        results = await generator.generate_batch(
            characters=[self.test_character], 
            conversations_per_character=2  # Reduced from 3 for real LLM testing
        )
        
        # Should generate 2 conversations for 1 character
        assert len(results) == 2
        assert all(isinstance(sample, DatasetSample) for sample in results)
        
        # Each conversation should be unique
        session_ids = [sample.session_id for sample in results]
        assert len(set(session_ids)) == len(session_ids)  # All unique
        
        # All should have substantial content
        for sample in results:
            assert len(sample.turns) >= 2
            conversation_text = " ".join([turn.text for turn in sample.turns])
            assert len(conversation_text.strip()) > 20
    
    def test_quality_scoring_metrics(self):
        """Test quality scoring for generated conversations"""
        from scripts.generate_synthetic_conversations import ConversationQualityScorer
        
        scorer = ConversationQualityScorer()
        
        # Create a sample conversation
        sample = DatasetSample(
            session_id="quality_test",
            persona_mix={"Scholar": 1.0},
            memory_slots=["Character is knowledgeable"],
            turns=[
                Turn(sender="user", text="What's your expertise?", channel="text"),
                Turn(sender="assistant", text="I specialize in ancient manuscripts and historical research.", channel="text"),
                Turn(sender="assistant", text="Let me search my database for relevant examples.", channel="action", 
                     action={"tool": "search", "query": "manuscript examples"}),
                Turn(sender="user", text="That's impressive!", channel="text"),
                Turn(sender="assistant", text="Thank you! I find the work quite rewarding.", channel="text")
            ]
        )
        
        score = scorer.score_conversation(sample)
        
        # Score should be a dict with various metrics
        assert isinstance(score, dict)
        assert "overall_quality" in score
        assert "dialogue_coherence" in score
        assert "action_integration" in score
        assert "character_consistency" in score
        
        # Overall quality should be between 0 and 1
        assert 0 <= score["overall_quality"] <= 1
    
    def test_data_export_integration(self):
        """Test that generated data exports to unified dataset format"""
        from scripts.generate_synthetic_conversations import SyntheticDataExporter
        
        # Create sample synthetic data
        samples = [
            DatasetSample(
                session_id="export_test_1",
                persona_mix={"Scholar": 1.0},
                memory_slots=["Test memory 1"],
                turns=[Turn(sender="user", text="Hello", channel="text")]
            ),
            DatasetSample(
                session_id="export_test_2", 
                persona_mix={"Scholar": 1.0},
                memory_slots=["Test memory 2"],
                turns=[Turn(sender="user", text="Hi there", channel="text")]
            )
        ]
        
        exporter = SyntheticDataExporter()
        output_file = self.temp_dir / "exported_dataset.jsonl"
        
        # Export the data
        result = exporter.export_to_jsonl(samples, output_file)
        
        assert result is True
        assert output_file.exists()
        
        # Verify file content
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            # Each line should be valid JSON representing a DatasetSample
            for line in lines:
                data = json.loads(line)
                # Should be able to reconstruct DatasetSample
                sample = DatasetSample(**data)
                assert sample.session_id.startswith("export_test_")


class TestDataCollectionInfrastructure:
    """Test data collection and annotation infrastructure"""
    
    def test_conversation_annotator_import(self):
        """Test that ConversationAnnotator can be imported"""
        # This will fail initially 
        from scripts.generate_synthetic_conversations import ConversationAnnotator
        assert ConversationAnnotator is not None
    
    def test_annotation_interface_creation(self):
        """Test creating annotation interface for existing conversations"""
        from scripts.generate_synthetic_conversations import ConversationAnnotator
        
        annotator = ConversationAnnotator()
        
        # Sample raw conversation
        raw_conversation = [
            {"role": "user", "content": "Can you help me find information about quantum physics?"},
            {"role": "assistant", "content": "I'll search my knowledge base for quantum physics information."},
            {"role": "user", "content": "What did you find?"},
            {"role": "assistant", "content": "Quantum physics deals with the behavior of matter and energy at the smallest scales."}
        ]
        
        # Should be able to create annotation interface
        interface = annotator.create_annotation_interface(raw_conversation)
        
        assert interface is not None
        assert "conversation_id" in interface
        assert "turns_to_annotate" in interface
        assert len(interface["turns_to_annotate"]) == 4
    
    def test_action_annotation_suggestions(self):
        """Test automatic suggestions for action annotations"""
        from scripts.generate_synthetic_conversations import ActionAnnotationSuggester
        
        suggester = ActionAnnotationSuggester()
        
        # Text that suggests an action
        text = "Let me search my database for information about quantum entanglement"
        
        suggestions = suggester.suggest_actions(text)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest a search action
        search_suggestion = next((s for s in suggestions if s["action_type"] == "search"), None)
        assert search_suggestion is not None
        assert "confidence" in search_suggestion
        assert 0 <= search_suggestion["confidence"] <= 1


class TestDataQualityAndCuration:
    """Test data quality assessment and curation tools"""
    
    def test_duplicate_detection(self):
        """Test detection and removal of duplicate conversations"""
        from scripts.generate_synthetic_conversations import DuplicateDetector
        
        detector = DuplicateDetector()
        
        # Create samples with some duplicates
        samples = [
            DatasetSample(
                session_id="dup_test_1",
                persona_mix={"Scholar": 1.0},
                memory_slots=["Memory 1"],
                turns=[Turn(sender="user", text="Hello", channel="text")]
            ),
            DatasetSample(
                session_id="dup_test_2",
                persona_mix={"Scholar": 1.0}, 
                memory_slots=["Memory 1"],
                turns=[Turn(sender="user", text="Hello", channel="text")]  # Duplicate
            ),
            DatasetSample(
                session_id="dup_test_3",
                persona_mix={"Scholar": 1.0},
                memory_slots=["Memory 2"],
                turns=[Turn(sender="user", text="Hi there", channel="text")]  # Unique
            )
        ]
        
        unique_samples = detector.remove_duplicates(samples)
        
        # Should remove one duplicate
        assert len(unique_samples) == 2
        
        # Should keep the first occurrence
        session_ids = {sample.session_id for sample in unique_samples}
        assert "dup_test_1" in session_ids
        assert "dup_test_3" in session_ids
    
    def test_dataset_balance_analysis(self):
        """Test analysis of dataset balance (text vs action channels)"""
        from scripts.generate_synthetic_conversations import DatasetBalanceAnalyzer
        
        analyzer = DatasetBalanceAnalyzer()
        
        # Create imbalanced dataset (mostly text)
        samples = [
            DatasetSample(
                session_id=f"balance_test_{i}",
                persona_mix={"Scholar": 1.0},
                memory_slots=["Memory"],
                turns=[
                    Turn(sender="user", text="Hello", channel="text"),
                    Turn(sender="assistant", text="Hi", channel="text")
                ]
            ) for i in range(10)
        ]
        
        # Add one action-heavy conversation
        samples.append(DatasetSample(
            session_id="balance_test_action",
            persona_mix={"Scholar": 1.0},
            memory_slots=["Memory"],
            turns=[
                Turn(sender="user", text="Search for info", channel="text"),
                Turn(sender="assistant", text="Searching...", channel="action", 
                     action={"tool": "search", "query": "info"})
            ]
        ))
        
        balance_report = analyzer.analyze_balance(samples)
        
        assert isinstance(balance_report, dict)
        assert "text_channel_ratio" in balance_report
        assert "action_channel_ratio" in balance_report
        assert "balance_score" in balance_report
        
        # Should detect text-heavy imbalance
        assert balance_report["text_channel_ratio"] > 0.8
        assert balance_report["action_channel_ratio"] < 0.2 
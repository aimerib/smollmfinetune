---
# R4-7: Preference Data Collection (Reward & Self-Play)
Status: **Todo**
Ring: R4
Created: 2025-06-19
Updated: 2025-01-16 (Triple-Head Architecture Integration)
---

## Goal
Create the infrastructure to generate and collect preference data across all three model heads: a "Triple-Head Self-Play Harness" for automated data generation and a "Multi-Dimensional Reward Collector" UI for human feedback on generation quality, emotional control, and memory formation.

## Context
To align the triple-head model with human preferences (DPO), we need to collect preferences for each head independently and in coordination. This involves generating conversational snippets where humans can evaluate content quality (generation head), emotional appropriateness (control head), and memory consistency (memory head). This task builds the tools for comprehensive triple-head preference collection.

## Acceptance Criteria

### Triple-Head Self-Play Harness:
- [ ] **Enhanced Script**: `scripts/run_triple_head_self_play.py` supporting all three model heads
- [ ] **Multi-Head Conversation**: Load two instances of the triple-head NarrativeLLM to converse with full head outputs
- [ ] **Head-Specific Logging**: Log generation content, control tokens, and memory formations separately
- [ ] **Coordinated Evaluation**: Generate scenarios where all three heads can be evaluated simultaneously
- [ ] **Head Performance Tracking**: Monitor each head's contribution to conversation quality
- [ ] **Judge Model Evaluation**: Use our judge model harness to evaluate early model performance for quick and automated iterations.

### NEW: Judge Model Integration Pipeline:
- [ ] **Automated Quality Scoring**: Integration with R4-1.2 Judge Model Service for rapid evaluation
- [ ] **Head-Specific Judge Prompts**: Specialized evaluation prompts for generation, control, and memory heads
- [ ] **Quality Filtering**: Automated filtering of conversation pairs based on judge model scores
- [ ] **Evaluation Metrics**: Standardized scoring for content quality, emotional appropriateness, and memory consistency
- [ ] **Human-Judge Correlation**: Track correlation between judge model scores and human preferences for calibration
- [ ] **Batch Processing**: Efficient processing of large conversation datasets through judge model pipeline

### Multi-Dimensional Reward Collector:
- [ ] **Enhanced UI**: Streamlit page `pages/triple_head_reward_labeling.py` with head-specific evaluation
- [ ] **Generation Head Evaluation**: Rate content quality, creativity, and factual accuracy
- [ ] **Control Head Evaluation**: Rate emotional appropriateness, personality consistency, mood matching
- [ ] **Memory Head Evaluation**: Rate memory accuracy, consistency, and formation quality
- [ ] **Coordinated Preference Collection**: Allow evaluation of how well all three heads work together
- [ ] **Head-Specific Storage**: Save preferences for each head in structured format (`generation_preferences.jsonl`, `control_preferences.jsonl`, `memory_preferences.jsonl`)

### NEW: Advanced Preference Collection Features:
- [ ] **Memory Consistency Evaluation**: Compare character memories across conversations for consistency
- [ ] **Emotional Arc Rating**: Evaluate how well control head maintains emotional continuity
- [ ] **Cross-Head Harmony Scoring**: Rate how well generation, control, and memory heads coordinate
- [ ] **Context-Aware Preferences**: Collect preferences that account for conversation context and character state
- [ ] **Multi-Evaluator Support**: Allow multiple humans to rate the same outputs for inter-rater reliability

### Enhanced Self-Play Scenarios:
- [ ] **Memory Formation Scenarios**: Conversations designed to test memory head performance
- [ ] **Emotional Challenge Scenarios**: Situations requiring sophisticated emotional control
- [ ] **Generation Quality Tests**: Prompts that challenge content generation capabilities
- [ ] **Cross-Head Coordination Tests**: Complex scenarios requiring all three heads to work together

## Implementation Notes
```text
• TDD Instructions:
  - Triple-Head Self-Play: Write tests that run the enhanced self-play script. Mock all three head outputs with deterministic responses. Assert that separate log files are created for generation, control, and memory outputs.
  - Multi-Head Reward Collector: Create tests that verify each head can be evaluated independently. Simulate rating generation quality, emotional appropriateness, and memory accuracy separately.
  - Cross-Head Evaluation: Test scenarios where all three heads are rated together for coordination quality.
  - Head-Specific Storage: Verify that preferences are stored with proper head attribution and can be loaded for training each head independently.
```

## Enhanced Checklist / Steps
1. **NEW**: Create triple-head self-play harness with head-specific conversation generation
2. **NEW**: Implement multi-head model loading and coordination
3. **ENHANCED**: Add head-specific logging functionality for all three outputs
4. **NEW**: Integrate judge model pipeline for automated quality assessment
   - Create head-specific judge prompts for each model head
   - Implement batch processing through R4-1.2 Judge Model Service
   - Add quality score thresholds for filtering conversations
   - Track judge-human correlation metrics for calibration
5. **NEW**: Create comprehensive Streamlit page for triple-head evaluation
6. **NEW**: Implement generation head preference collection (content quality, creativity)
7. **NEW**: Implement control head preference collection (emotional appropriateness, personality)
8. **NEW**: Implement memory head preference collection (accuracy, consistency)
9. **NEW**: Add cross-head coordination evaluation interface
10. **NEW**: Add judge model scores to preference collection UI for human evaluator context
11. **ENHANCED**: Create head-specific preference storage with proper attribution
12. **NEW**: Implement memory consistency checking across conversations
13. **NEW**: Add emotional arc evaluation tools
14. **NEW**: Create advanced scenario generation for each head type
15. **ENHANCED**: Write comprehensive tests covering all three heads and judge model integration
16. **NEW**: Add multi-evaluator support and inter-rater reliability tracking

## Head-Specific Preference Collection

### Generation Head Preferences:
- Content quality and coherence
- Factual accuracy and consistency
- Creative expression and engagement
- Writing style and tone appropriateness

### Control Head Preferences:
- Emotional appropriateness for context
- Personality consistency over time
- Mood transitions and emotional arcs
- Control token effectiveness

### Memory Head Preferences:
- Memory formation accuracy
- Information retention consistency
- Memory retrieval appropriateness
- Long-term narrative continuity

### Cross-Head Coordination Preferences:
- How well all three heads work together
- Consistency between head outputs
- Overall character coherence
- Narrative flow and engagement

## References
- **Enhanced Integration**: R4-6 Triple-Head SFT Pipeline for training data format
- **Memory System**: R4-5 External Memory API for memory evaluation
- **Control Tokens**: R1-10 Control Token Vocabulary for emotional evaluation
- **Training Pipeline**: R4-8 DPO Training for multi-head preference learning
- Original proposal §5.2: Reward Modeling (now enhanced for triple-head) 

## Judge Model Integration Implementation

### Automated Evaluation Pipeline:
```python
# Example workflow integration
conversation_pairs = self_play_harness.generate_conversations()
judge_scores = judge_model_service.evaluate_triple_head(conversation_pairs)
filtered_pairs = filter_by_quality_threshold(conversation_pairs, judge_scores)
human_evaluation_queue = prioritize_for_human_review(filtered_pairs)
```

### Head-Specific Judge Prompts:
- **Generation Head Judge**: "Evaluate this response for content quality, coherence, and factual accuracy. Rate 1-10."
- **Control Head Judge**: "Assess emotional appropriateness and personality consistency. Does the character's emotional state match the context?"
- **Memory Head Judge**: "Evaluate memory formation accuracy. Are important details being remembered correctly? Is there consistency with previous memories?"

### Quality Filtering Strategy:
- **Threshold-Based**: Only send conversations with judge scores above certain thresholds to human evaluators
- **Diverse Sampling**: Ensure human evaluators see range of quality levels for calibration
- **Edge Case Focus**: Prioritize conversations where judge model uncertainty is high
- **Performance Tracking**: Monitor correlation between judge scores and human preferences

### Integration with Existing Infrastructure:
- Leverage **R4-1.2 Judge Model Service** API for standardized evaluation
- Use existing conversation logging format from self-play harness
- Integrate judge scores into preference collection UI for human context
- Store judge-human correlation data for continuous improvement 
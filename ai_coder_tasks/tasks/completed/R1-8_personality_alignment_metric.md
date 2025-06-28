---
# R1-8  Big-Five Personality Alignment Metric
Status: **Complete**
Ring: R1
Created: 2025-06-18
Completed: 2025-01-24
---

## Goal
Using an LLM-as-a-judge, evaluate how well a generated response aligns with the character's target **Big-Five personality profile** (O, C, E, A, N) and produce a 0-1 score.

## Acceptance Criteria
- [x] Create new module `utils/evaluation/personality_metric.py`.
- [x] Implement `calculate_personality_alignment(response: str, big_five_scores: Dict[str, float]) -> float`.
      • This function constructs a prompt for a judge model.
      • The prompt will contain the generated `response` and the target `big_five_scores`.
      • It asks the judge to return a JSON object with a single key, `"alignment_score"`, from 0.0 to 1.0.
      • The function should parse the JSON and return the float.
- [x] The judge prompt must include a rubric explaining what high and low scores for each of the Big-Five traits look like in conversation.
- [x] Integrate this new metric into the `TrainingQualityTracker` in `utils/metrics.py`.
- [x] The score (`avg_personality_alignment`) is logged to WandB during training.
- [x] Unit test mocks the LLM call and verifies that the score is correctly parsed and logged.

## Implementation Summary
Successfully implemented the Big-Five personality alignment metric with:

1. **Module Created**: `app/utils/evaluation/personality_metric.py` with:
   - `calculate_personality_alignment()` function that uses LLM-as-judge
   - Detailed rubric for evaluating all five personality traits
   - Input validation and error handling
   - Optional cached version for performance optimization

2. **Detailed Rubric**: Enhanced prompt includes comprehensive descriptions for each trait at high/medium/low levels, helping the LLM judge make accurate assessments.

3. **TrainingQualityTracker Integration**: Added methods:
   - `add_personality_alignment_score()` to track scores
   - `get_wandb_metrics()` to format metrics for logging
   - `log_to_wandb()` for direct WandB integration
   - Personality alignment monitoring in `get_training_health()`

4. **Training Pipeline Integration**: ✅ **BONUS** - Fully integrated into training loop:
   - `TrainingCallback.on_evaluate()` automatically calls personality alignment evaluation
   - Only evaluates when character has `big_five_scores` defined
   - Scores are logged to WandB during training
   - Status updates include personality alignment metrics
   - Training dashboard will show real-time personality alignment scores

5. **Comprehensive Testing**: 
   - 12 unit tests covering all functionality
   - 4 integration tests verifying training pipeline hooks
   - Mocked LLM calls for reliable testing
   - Validation tests for edge cases

**Ready for UI Integration**: The personality alignment metric is now fully functional in both standalone evaluation and live training monitoring. Task R1-9 can proceed with UI display of these metrics.

## Implementation Notes
The LLM-as-a-judge prompt is critical. Here is a starting point:

```
System: You are an expert personality psychologist. Your task is to evaluate a dialogue snippet.

User:
Please evaluate the following response based on the provided Big-Five personality profile. The profile scores range from 0.0 (low) to 1.0 (high).

**Big-Five Profile:**
- Openness: {{O_score}} (High: curious, imaginative; Low: conventional, cautious)
- Conscientiousness: {{C_score}} (High: organized, disciplined; Low: spontaneous, careless)
- Extraversion: {{E_score}} (High: outgoing, energetic; Low: solitary, reserved)
- Agreeableness: {{A_score}} (High: compassionate, cooperative; Low: antagonistic, competitive)
- Neuroticism: {{N_score}} (High: anxious, sensitive; Low: secure, confident)

**Response to evaluate:**
"{{response_text}}"

Based on this, provide a single alignment score from 0.0 (not at all aligned) to 1.0 (perfectly aligned). Return ONLY a JSON object with your score.

Example:
{"alignment_score": 0.8}
```

## References
- Relies on R1-2 (Character Core Structure with Big-Five) and R1-5 (Personality Radar Chart).
- Will be displayed in the UI as per R1-9. 
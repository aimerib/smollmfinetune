---
# R1-9  UI for Advanced Model Metrics
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Display the new **Personality Alignment** and **Lore Adherence** metrics in the UI, including a "Personality Drift" radar chart that compares the authored character profile against the model's generated output.

## Acceptance Criteria

### 1. New Metric: Lore Adherence
- [ ] Create `utils/evaluation/lore_metric.py` with `calculate_lore_adherence(response: str, lore_fact: str) -> float`.
- [ ] This function uses an LLM-as-a-judge to score if the `response` correctly uses or contradicts the provided `lore_fact`.
- [ ] The score is integrated into `TrainingQualityTracker` and logged to WandB as `avg_lore_adherence`.
- [ ] Unit test mocks the LLM call.

### 2. Training Dashboard UI (`page_training_dashboard` in `app.py`)
- [ ] Add columns for "Personality Alignment" and "Lore Adherence" to the training metrics table.
- [ ] These columns will display the `avg_personality_alignment` and `avg_lore_adherence` scores from the training run.

### 3. Model Comparison UI (`page_model_comparison` in `app.py`)
- [ ] Add a new section: **Personality Drift Analysis**.
- [ ] This section displays a **Plotly Scatterpolar (radar) chart**.
- [ ] The chart will have two traces:
      - **Authored Personality**: The character's Big-Five scores from `character_core.json`.
      - **Generated Personality**: The average Big-Five scores calculated by running the new `personality_metric` over a sample of the model's outputs.
- [ ] A button "Run Drift Analysis" will trigger the calculation over ~50 sample generations.

### 4. Character Deep Dive (`render_consistency_deep_dive` in `app.py`)
- [ ] Enhance this function to include the Personality Drift radar chart when viewing a trained model's metrics.

## Implementation Notes
- The "Generated Personality" profile for the drift chart will require a new function, e.g., `evaluate_generated_personality(model, character, num_samples=50) -> Dict[str, float]`, which generates responses and aggregates the personality scores from the judge model.
- The Lore Adherence judge prompt should be simple: "Did the response correctly incorporate or contradict this fact? Score 0.0 (contradicted), 0.5 (ignored), 1.0 (incorporated)."

## References
- ✅ **Depends on R1-8 (Personality Alignment Metric)** - COMPLETED with full training pipeline integration.
- Directly enhances visualizations mentioned in R1-5 (Personality Radar Chart).

## Ready for Implementation
The `calculate_personality_alignment()` function is now fully implemented in `app/utils/evaluation/personality_metric.py` and integrated into the training pipeline. The training dashboard already tracks `avg_personality_alignment` metrics in real-time. UI components can import and use this function directly. 
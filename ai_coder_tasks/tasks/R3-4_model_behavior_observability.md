# R3-4: Model Behavior Observability

- **Ring:** R3
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R3-3, R4-1

---

## 1. Goal

To build a suite of internal tools and dashboards for observing and debugging the behavior of the language model during inference. This provides the fine-grained insight needed to understand *why* a model makes a certain choice.

---

## 2. Why? (The Story)

When a character says something brilliant or something bizarre, "I don't know" is not an acceptable answer. As engineers building a new engine (R4), we need to be able to pop the hood and see what's going on. Is the model attending to the right parts of the prompt? Were the probabilities for the chosen token razor-thin or a landslide? Answering these questions requires more than just looking at the final text output. This task is about building the microscope that will allow us to diagnose model behavior, which is essential for both debugging our R3 model and iterating on the R4 architecture.

---

## 3. How? (The Implementation)

1.  **Modify Inference Pipeline to Log Intermediates:**
    -   Update the `app/utils/inference.py` (or equivalent generation manager) to capture and log intermediate model data.
    -   This requires modifying the generation call to output hidden states, attention weights, and token probabilities (`output_hidden_states=True`, `output_attentions=True`).
    -   **Performance Warning:** This should be an opt-in feature, controlled by a debug flag, as it adds performance overhead.

2.  **Create a Log Storage Strategy:** ⚡ ENHANCED BY R3-1.5 INFRASTRUCTURE
    -   Define a schema for storing this rich observability data.
    -   ✅ **Structured logging system ready** from R3-1.5 error handling infrastructure
    -   ✅ **Health monitoring framework** can track observability system performance
    -   This could be as simple as structured JSON files saved to a specific directory, linked by a request ID, or a more robust solution using a dedicated logging database (e.g., Elasticsearch, Loki).

3.  **Build an "Inference Inspector" Page:**
    -   Create a new Streamlit page: `app/pages/inference_inspector.py`.
    -   This page will allow a developer to enter a request ID from a past interaction.
    -   It will fetch the corresponding observability logs.

4.  **Develop Visualization Components:** ⚡ ENHANCED BY R3-1.5 INFRASTRUCTURE
    -   Within the inspector page, create components to visualize the data:
        -   **Attention Visualizer:** Use a library like `bertviz` or a custom `matplotlib`/`plotly` solution to create heatmaps showing attention from each token to every other token, layer by layer. This helps answer "What part of the prompt was the model looking at?".
        -   **Token Probability Viewer:** For a given position, show a bar chart of the top-K token probabilities. This helps understand the model's certainty and what other choices it considered.
        -   ✅ **Progress indicators ready** for long visualization operations
        -   ✅ **Error boundaries** will handle visualization failures gracefully

---

## 4. How to Test?

-   **Unit Tests:**
    -   Test the modified inference function to ensure it correctly returns intermediate states when the debug flag is enabled and behaves normally otherwise.
    -   Test the logging mechanism to ensure it writes data in the correct format.
-   **UI Tests (`tests/ui/test_inference_inspector.py`):**
    -   Create an `AppTest` for the new page.
    -   Use fixture data (a sample observability log file) to test that the page loads correctly.
    -   Test that the visualization components render without errors when fed the sample data. 
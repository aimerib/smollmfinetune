# R3-3: Data Analysis and Hypothesis Validation

- **Ring:** R3
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Medium
- **Related-Tasks:** R3-2.5, R4-0.5

---

## 1. Goal

To analyze the data collected from the R0-R3 platform (using the `smollm2`-based model) to identify its core weaknesses and formulate a data-driven hypothesis that will define the requirements for the R4 Narrative Engine.

---

## 2. Why? (The Story)

We've successfully built a devkit and are generating character interactions. But before we build a brand-new, expensive engine (R4), we must be rigorous scientists. We need to look at the data from our current engine and ask: "Where does it fall short?" "What are the most common ways the illusion of character is broken?" This task is about moving from anecdotal evidence to a quantitative, data-backed report card for our current model. The output of this task will be the foundational document that justifies and guides the entire R4 effort.

---

## 3. How? (The Implementation)

1.  **Develop Data Extraction Scripts:**
    - Create Python scripts (likely in `/notebooks` or as a utility in `app/utils/dataset`) to pull interaction data from the production/staging database (as defined in R3-1).
    - Data to extract: conversation logs, character metadata, world context, user feedback if any.

2.  **Define Key Failure-Mode Metrics:**
    - Work with the "Product Owner" (you!) to define categories of failures. Examples:
        - **Memory Failure:** Forgetting a key fact from N turns ago.
        - **Personality Drift:** Acting out of character based on their Big-Five traits.
        - **Goal Inconsistency:** Contradicting a previously stated goal.
        - **Repetition:** Getting stuck in a conversational loop.
        - **Lore Contradiction:** Violating an established world rule.

3.  **Build Analysis Tooling (Jupyter Notebooks):**
    - Create a suite of Jupyter notebooks to process the extracted data.
    - Use libraries like `pandas`, `matplotlib`, `seaborn` to analyze and visualize the frequency of the defined failure modes.
    - Generate plots showing "Personality Drift vs. Conversation Length" or "Memory Failure Rate by Character."

4.  **Produce the Hypothesis Report:**
    - Synthesize the findings into a markdown report (`R3-3_validation_report.md`).
    - The report should clearly state the findings, e.g., "Our analysis of 10,000 conversations shows the current model has a 30% chance of memory failure after 50 turns."
    - Conclude with a clear, actionable hypothesis for R4, e.g., "Therefore, the R4 Narrative Engine's primary architectural goal is to reduce long-term memory failure to <5% and will incorporate an external memory module to achieve this."

---

## 4. How to Test?

-   This is a data analysis task, not a feature with traditional unit tests.
-   **Testing is via validation:**
    -   The analysis scripts should be runnable and produce verifiable outputs (e.g., CSV files, plots).
    -   The final report must be reviewed and signed off. The logic in the notebooks should be clear and commented.
    -   A small, sample dataset can be created in `tests/fixtures` to test the data processing logic of the scripts. 
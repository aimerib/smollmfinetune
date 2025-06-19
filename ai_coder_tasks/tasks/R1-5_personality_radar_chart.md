---
# R1-5  Personality Radar Component
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Build an interactive **Personality Editor** composed of five sliders and a live-updating Plotly Scatterpolar (radar) chart; returns updated Big-5 dict and logs preference when AI estimates are accepted.

## Component API
```python
def render_personality_editor(core: CharacterCore, key_prefix="pers_", show_ai_btn=True) -> Personality:
    """Render sliders + radar, mutate core in-place, return updated personality dict."""
```
• Placed in `app/components/personality_editor.py`.

## Acceptance Criteria
- [ ] Slider step 0.05, default values from `core.personality_traits`.
- [ ] Radar chart updates on slider move (use `st.plotly_chart(fig, key=key_prefix)` with `use_container_width=True`).
- [ ] Optional AI button "Estimate from examples"; emits 5 scores, shows diff, and on accept updates sliders and logs a preference event `{type:"big5_estimate", old:..., new:...}`.
- [ ] Component used in R1-4 Character Management tab and in Model Comparison page to plot target vs generated.
- [ ] Tooltip next to each personality trait to describe what they mean, and how it could affect a character

## Implementation Notes
```python
import plotly.graph_objects as go
labels = ["Openness","Conscientiousness","Extraversion","Agreeableness","Neuroticism"]

def build_fig(values):
    vals = list(values)+[values[0]]
    return go.Figure(go.Scatterpolar(r=vals, theta=labels+labels[:1], fill='toself'))
```
Use `fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])))`.

## Steps
1. Create component file, implement sliders & chart.
2. Wire AI estimate button (calls helper in `utils/character/analysis.py`).
3. Add preference logging call.
4. Import component into Character Management UI.

## Dependencies
- Pydantic `Personality` model from R1-2.
- Preference logging (R1-3).

## References
- Overview §4. 
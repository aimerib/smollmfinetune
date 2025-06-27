---
# R5-6: The Director's Chair
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Build a production-ready real-time training system that enables creators to improve characters through natural conversation and correction, with invisible background optimization and immediate feedback.

## Context
Traditional ML training workflows are too complex for content creators. This system provides an intuitive interface where creators simply talk to characters and make corrections, while sophisticated machine learning happens transparently in the background.

## Acceptance Criteria

### Production Training Infrastructure:
- [ ] High-throughput preference pair collection with intelligent batching
- [ ] GPU-optimized micro-training pipeline with <2 minute cycle times
- [ ] Atomic model updates with zero-downtime deployment
- [ ] Training queue management with priority and fairness scheduling
- [ ] Comprehensive monitoring and alerting for training pipeline health

### Advanced Creator Interface:
- [ ] **Conversation Studio**: Multi-turn conversation editing with branching
- [ ] **Performance Analytics**: Real-time character improvement metrics
- [ ] **Quality Assurance**: Automated validation of character responses
- [ ] **Version Control**: Character model versioning with rollback capabilities
- [ ] **Collaboration Tools**: Multi-creator workflows with conflict resolution

### ML Operations at Scale:
- [ ] Distributed training across multiple GPUs for faster iteration
- [ ] Intelligent sample selection for maximum training efficiency
- [ ] Curriculum learning based on creator expertise and character complexity
- [ ] A/B testing framework for comparing training approaches
- [ ] Cost optimization with spot instance usage and smart scheduling

### User Experience:
- [ ] Sub-second response time for character interactions
- [ ] Visual progress indicators for ongoing improvements
- [ ] Smart suggestions for character enhancement opportunities
- [ ] Integration with analytics to track creator satisfaction
- [ ] Mobile-optimized interface for on-the-go character improvement

## Implementation Notes
```text
• TDD Instructions:
  - Red (UI): Using AppTest, create a test for director_mode.py. Simulate a conversation. Use the API to find and trigger the edit callback on an assistant message. Mock the handle_live_correction backend function and assert that it was called with the correct (prompt, original, edited) arguments.
  - Green (UI): Implement the editable chat UI and the callback logic.
  - Red (Backend): Write a unit test for handle_live_correction. Assert that it correctly formats the preference pair and adds it to a mocked queue.
  - Green (Backend): Implement the handle_live_correction function.
  - Red/Green (Worker): Write an integration test for the training worker. Pre-populate the DPO queue with a few dummy preference pairs. Start the worker. Assert that the DPO training script (mocked) is eventually called with the correct arguments (e.g., path to the character's adapter).
```

## Checklist / Steps
1. Create director_mode.py with elegant chat interface
2. Implement editable assistant messages functionality
3. Create handle_live_correction backend function
4. Implement preference pair formatting and queueing
5. Create real-time training worker process
6. Integrate with DPO pipeline from R4-8
7. Add adapter updating and hot-swapping
8. Write comprehensive tests for live training workflow
9. Add progress indicators and training status UI

## References
A radical simplification and enhancement of the entire creator workflow, building on the DPO pipeline from R4-8. 
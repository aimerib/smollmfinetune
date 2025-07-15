---
# R4-5: Narrative Engine - External Memory API Integration
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

## Goal
Integrate the Narrative-LLM with an external vector database for long-term memory retrieval via a well-defined API, enabling the model to pull in relevant context during generation. Additionally, implement a dual-layer memory architecture with working memory (internal) and long-term memory (external), supporting both memory formation and retrieval with surprise-weighted persistence.

## Context
The model's architecture includes a cross-attention layer designed to incorporate external memories. This task builds the client-side logic for fetching that memory and feeding it into the model. This is critical for narrative coherence over long conversations.

We're implementing TWO parallel approaches:
- **Method A (Memory Tokens)**: Use control tokens to annotate memory operations
- **Method B (Memory Head)**: Add a third head to the model for memory vector outputs

## Acceptance Criteria
### Phase 1 - Mock Implementation & Architecture:
- [ ] A `MockMemoryClient` class is created that returns random/fixed vectors for development
- [ ] The model's forward pass is updated to accept external memory states (using mock data initially)
- [ ] Cross-attention layer integration is tested with dummy memory vectors
- [ ] **NEW**: Pydantic schemas for memory annotations (`MemoryAnnotation`, `MemoryFormation`, etc.)
- [ ] **NEW**: Memory-specific control tokens added to vocabulary (Method A)
- [ ] **NEW**: Third head architecture implemented in NarrativeLLM (Method B)

### Phase 2 - SQLite Backend & Dataset Generation:
- [ ] SQLite database with vector plugin (sqlite-vss or sqlite-vec) is set up
- [ ] A `MemoryAPIClient` class is created that interfaces with SQLite
- [ ] Client has methods for `add_memory` and `retrieve_memories(session_id, query_text, k)`
- [ ] Replace mock client with real SQLite-backed implementation
- [ ] **NEW**: Synthetic memory generation pipeline using OpenAI structured outputs
- [ ] **NEW**: Dataset augmentation to include both memory tokens AND memory head labels
- [ ] **NEW**: Surprise detection and emotional valence calculation

### Phase 3 - Integration & Evaluation:
- [ ] The model's training pipeline includes memory retrieval during forward passes
- [ ] Memory embeddings are correctly passed to the cross-attention layer
- [ ] End-to-end testing with real conversational memory
- [ ] **NEW**: Parallel training of both Method A and Method B
- [ ] **NEW**: Evaluation framework comparing memory coherence between methods
- [ ] **NEW**: Memory formation triggers based on surprise scores

## Implementation Notes
```text
• Memory Architecture:
  - Working Memory: Active embeddings for current conversation context
  - Long-term Memory: External vector DB for persistent storage
  - Surprise Engine: Calculates unexpectedness to weight memory importance
  - Emotional Valence: -1 to +1 scale affecting memory persistence

• Method A (Tokens):
  - Add tokens: <memory_form>, <memory_recall>, <memory_importance_high/med/low>
  - Integrate with existing control head infrastructure
  - Should converge faster for initial testing

• Method B (Memory Head):
  - Third output head on model
  - Outputs: memory_vector (embedding) + metadata (importance, valence, type)
  - More complex but potentially more expressive

• TDD Instructions:
 - Red (Failing Test - API Client): In tests/narrative_engine/test_memory_client.py, write a test that 
  attempts to instantiate MemoryAPIClient and call .retrieve_memories(). This will fail.
  - Green (Passing Test - API Client): Create the class and method stubs to make the test pass.
  - Red (Failing Test - Mocked API Call): Use a library like pytest-https or requests-mock to mock the Memory 
  API. Write a test where you call retrieve_memories, the mock API returns a predefined JSON payload, and you 
  assert that the client correctly parses this payload into the expected data structure (e.g., a list of 
  strings or embeddings).
  - Green (Passing Test - Mocked API Call): Implement the client's request and parsing logic to satisfy the 
  test.
  - Integration Test: Add a test to tests/narrative_engine/test_model.py that verifies the NarrativeLLM's 
  forward pass can accept and process the external_memory_states tensor without error.
  - Red: Create test for MemoryAnnotation Pydantic model
  - Green: Implement the schema with all required fields
  - Red: Test synthetic memory generation with OpenAI
  - Green: Implement structured output generation
  - Red: Test both memory methods in model forward pass
  - Green: Implement dual approaches in parallel
```

## Checklist / Steps
1. Define Pydantic schemas for memory data structures
2. Create memory-specific control tokens for Method A
3. Implement third head architecture for Method B
4. Build synthetic memory generation pipeline
5. Create MemoryAPIClient class with HTTP request handling
6. Implement memory retrieval and parsing logic
7. Update model's forward pass to support both methods
8. Create parallel training pipelines for A/B testing
9. Write comprehensive tests for all components
10. Integrate with cross-attention layer in model architecture
11. Build evaluation metrics for memory coherence
12. Document results and recommend best approach

## References
- Proposal §4: Cross-Attention to External Memory
- Proposal §9: Work-Package Skeleton (Item 3)
- R5-3: Living Interface (memory visualization integration) 
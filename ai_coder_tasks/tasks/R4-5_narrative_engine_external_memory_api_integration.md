---
# R4-5: Narrative Engine - External Memory API Integration
Status: **Todo**
Ring: R4
Created: 2025-06-19
---

## Goal
Integrate the Narrative-LLM with an external vector database for long-term memory retrieval via a well-defined API, enabling the model to pull in relevant context during generation.

## Context
The model's architecture includes a cross-attention layer designed to incorporate external memories. This task builds the client-side logic for fetching that memory and feeding it into the model. This is critical for narrative coherence over long conversations.

## Acceptance Criteria
### Phase 1 - Mock Implementation:
- [ ] A `MockMemoryClient` class is created that returns random/fixed vectors for development
- [ ] The model's forward pass is updated to accept external memory states (using mock data initially)
- [ ] Cross-attention layer integration is tested with dummy memory vectors

### Phase 2 - SQLite Backend:
- [ ] SQLite database with vector plugin (sqlite-vss or sqlite-vec) is set up
- [ ] A `MemoryAPIClient` class is created that interfaces with SQLite
- [ ] Client has methods for `add_memory` and `retrieve_memories(session_id, query_text, k)`
- [ ] Replace mock client with real SQLite-backed implementation

### Phase 3 - Integration:
- [ ] The model's training pipeline includes memory retrieval during forward passes
- [ ] Memory embeddings are correctly passed to the cross-attention layer
- [ ] End-to-end testing with real conversational memory

## Implementation Notes
```text
• TDD Instructions:
  - Red (Failing Test - API Client): In tests/narrative_engine/test_memory_client.py, write a test that attempts to instantiate MemoryAPIClient and call .retrieve_memories(). This will fail.
  - Green (Passing Test - API Client): Create the class and method stubs to make the test pass.
  - Red (Failing Test - Mocked API Call): Use a library like pytest-https or requests-mock to mock the Memory API. Write a test where you call retrieve_memories, the mock API returns a predefined JSON payload, and you assert that the client correctly parses this payload into the expected data structure (e.g., a list of strings or embeddings).
  - Green (Passing Test - Mocked API Call): Implement the client's request and parsing logic to satisfy the test.
  - Integration Test: Add a test to tests/narrative_engine/test_model.py that verifies the NarrativeLLM's forward pass can accept and process the external_memory_states tensor without error.
```

## Checklist / Steps
1. Define OpenAPI specification for Memory API endpoints
2. Create MemoryAPIClient class with HTTP request handling
3. Implement memory retrieval and parsing logic
4. Update model's forward pass to accept external memory states
5. Write comprehensive tests for all components
6. Integrate with cross-attention layer in model architecture

## References
- Proposal §4: Cross-Attention to External Memory
- Proposal §9: Work-Package Skeleton (Item 3) 
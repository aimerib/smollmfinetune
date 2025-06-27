---
# R5-5: The Digital Ecology Engine
Status: **Todo**
Ring: R5
Created: 2025-06-19
---

## Goal
Simulate "offline" interactions between characters within a world, allowing the world state and character relationships to evolve emergently, creating a truly living, persistent digital ecosystem.

## Context
Inspired by a biologist's view of an ecosystem, this feature makes the world feel alive even when the player is away. By simulating interactions based on characters' intrinsic goals and personalities, the platform generates emergent narratives and a dynamic history, making the world richer and more believable each time the user returns.

## Acceptance Criteria
- [ ] Simulation Script: A new script, `scripts/run_ecology_simulation.py`, is created. It's designed to be run periodically as a background job (e.g., nightly cron job).
- [ ] Simulation Logic:
  - [ ] The script selects a world at random.
  - [ ] It selects two characters from that world.
  - [ ] It runs a simplified, non-interactive simulation of an encounter between them, using their goals and relationship data to seed an LLM prompt.
  - [ ] The LLM generates a summary of the outcome (e.g., `"outcome": "Theft attempt failed"`, `"relationship_change": {"rivalry": "+10"}`, `"new_world_fact": "A strange light was seen near the Citadel vault."`).
- [ ] World State Mutation: The script uses the WorldManager and CharacterManager to atomically update the canonical world state based on the simulation outcome. This includes updating relationship scores in `character_core.json` and adding new events to `world_lore.json`.
- [ ] Thread Safety: The Manager classes are reviewed and updated to ensure their file/database write operations are thread-safe to prevent race conditions.

## Implementation Notes
```text
• TDD Instructions:
  - Red (Simulation Logic): In tests/scripts/test_ecology_simulation.py, create a temporary world with two dummy characters. Run the simulation function. Mock the LLM call to return a deterministic outcome JSON. Assert that the function correctly parses this outcome.
  - Green (Simulation Logic): Implement the core simulation and outcome-parsing logic.
  - Red (State Mutation): In an integration test, run the full script on a temporary directory. After the script finishes, load the character and world files from that directory and assert that their contents have been modified exactly as specified by the mocked LLM outcome.
  - Green (State Mutation): Implement the logic that calls the Manager classes to save the updated data to disk/DB.
```

## Checklist / Steps
1. Create ecology simulation script framework
2. Implement world and character selection logic
3. Create LLM-driven interaction simulation
4. Implement outcome parsing and interpretation
5. Update WorldManager and CharacterManager for thread safety
6. Add atomic world state mutation functionality
7. Create background job scheduling capability
8. Write comprehensive tests for simulation logic
9. Add logging and monitoring for ecosystem changes

## References
Directly builds upon the Agent Core (R5-4) concept but applies it between NPCs. 
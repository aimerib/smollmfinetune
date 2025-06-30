# R4-12: Runtime State Manager

- **Ring:** R4
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Large
- **Related-Tasks:** R3-1, R5-4, R5-5

---

## 1. Goal

To design and implement a robust, scalable, and persistent state management system for the narrative runtime. This system will track the state of all entities, characters, and the environment over time, allowing for dynamic and emergent world simulation.

---

## 2. Why? (The Story)

Our R0-R3 worlds are largely static. The lore is a fixed document. But for a true "Digital Ecology" (R5-5) where agents are proactive (R5-4), the world itself must be alive. If a character chops down a tree, the tree needs to *stay chopped down*. If an NPC moves from one town to another, the system needs to know their new location. The `RuntimeStateManager` is the single source of truth for the "now" of the game world. It's the difference between a scripted puppet show and a dynamic simulation.

---

## 3. How? (The Implementation)

1.  **Technology Selection & Design:**
    -   Evaluate different backend technologies for state management. This goes beyond the simple file-based approach.
    -   **Option A (Event Sourcing):** Store a log of all *actions* (events) that have ever happened. The current state is derived by replaying the events. This is excellent for auditing and time-travel debugging. Libraries like `eventsourcing` in Python could be used.
    -   **Option B (Document Database):** Use a NoSQL database like MongoDB or a persistent in-memory store like Redis with JSON documents representing the state of each entity. This is often faster for direct state lookups.
    -   The design must support atomic updates to prevent race conditions (e.g., two characters trying to pick up the same item at the same time).

2.  **Schema Definition:**
    -   Define a clear JSON schema for world and entity state. This will include:
        -   `entity_id`, `type` (character, item, location)
        -   `location`, `inventory`, `status_effects`
        -   `relationships` (e.g., `{target_id: "Rival", affinity: -50}`)
        -   Timestamps for last update.

3.  **Implement the `StateManager` API:**
    -   Create a new class, `StateManager`, in the `narrative_engine` module.
    -   It will expose methods like:
        -   `get_state(entity_id)`: Retrieve the current state of an entity.
        -   `update_state(entity_id, new_state_patch)`: Update parts of an entity's state.
        -   `query_by_location(location_id)`: Find all entities in a given location.
        -   `commit_transaction(list_of_updates)`: Apply multiple changes atomically.
        -   `get_recent_events(since_timestamp)`: Retrieve events for an agent's perception.

4.  **Integration with `WorldManager`:**
    -   Refactor the existing `WorldManager` to be the loader for the *initial* state of the world (from the lore files).
    -   Once loaded, the `StateManager` takes over for all dynamic updates during runtime.

---

## 4. How to Test?

-   **Unit Tests (`tests/narrative_engine/test_state_manager.py`):**
    -   This component MUST have comprehensive unit tests.
    -   Mock the chosen backend (or use an in-memory version like `fakeredis`).
    -   Test all API methods: create, read, update, delete (CRuD).
    -   Test transactional integrity: ensure that partial updates in a failed transaction are rolled back.
    -   Test query logic: ensure `query_by_location` returns correct and complete results.
-   **Integration Tests:**
    -   Test the interaction between `WorldManager` and `StateManager` to ensure the initial world state is loaded correctly. 

---

## COMPLETION SUMMARY

### What Was Built

Successfully implemented a comprehensive Runtime State Manager with the following components:

1. **Core Architecture**:
   - `EntityState` dataclass representing any entity (character, item, location) with support for:
     - Location tracking
     - Inventory management
     - Status effects
     - Relationships with affinity scores
     - Memory storage (integrated with our memory system!)
     - Custom data fields for extensibility
   - Thread-safe in-memory backend with full CRUD operations
   - Event logging system tracking all state changes

2. **Key Features**:
   - **Atomic Transactions**: Multi-entity updates that either all succeed or all fail with automatic rollback
   - **Atomic Updates**: Thread-safe read-modify-write operations preventing race conditions
   - **Complex Queries**: Query entities by type, location, or custom filters
   - **Event Sourcing**: Complete audit trail of all state changes with timestamps
   - **Memory Integration**: Characters can store and retrieve memories from our memory system
   - **World Integration**: Seamless loading of initial state from WorldManager

3. **API Methods Implemented**:
   - `create_entity()`, `get_entity()`, `update_entity()`, `delete_entity()`
   - `query_by_location()`, `query()` with custom filters
   - `execute_transaction()` for atomic multi-entity updates
   - `atomic_update()` for thread-safe modifications
   - `get_recent_events()` for event history
   - `load_world_state()` for WorldManager integration
   - `add_memory_to_character()`, `get_character_memories()` for memory system

4. **Test Coverage**:
   - 15 comprehensive unit tests covering all functionality
   - 2 integration tests demonstrating full world simulation
   - Tests include concurrency handling, transaction rollback, and memory integration
   - Beautiful integration test showing Clara exploring a world, forming memories, and building relationships

### Technical Decisions

- **Backend Choice**: Started with in-memory backend for development, designed to be swappable (Redis, MongoDB ready)
- **Concurrency**: Used Python's `threading.RLock` for thread safety
- **Serialization**: Full support for datetime serialization/deserialization
- **Deep Copying**: All get operations return deep copies to prevent accidental mutations

### Integration Points

The StateManager beautifully integrates with:
- **WorldManager**: Loads initial static world state
- **Memory System**: Stores character memories with importance, surprise, and emotional valence
- **Future Systems**: Ready for proactive agents (R5-4) and digital ecology (R5-5)

### Example Usage

```python
# Create a living world
state_manager = StateManager()
clara = EntityState(
    entity_id="clara_001",
    entity_type="character",
    location="Village Square",
    memories=[],
    relationships={}
)
state_manager.create_entity(clara)

# Clara forms a memory
memory = {
    "content": "The Elder Tree knew my name!",
    "importance": 1.0,
    "surprise": 0.95,
    "emotional_valence": 0.9
}
state_manager.add_memory_to_character("clara_001", memory)

# The world persists and evolves!
```

The Runtime State Manager is now the beating heart of our living world, ready to track every movement, every relationship change, and every precious memory our characters form. 🌟 
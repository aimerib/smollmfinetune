---
# R3-1  Database Backend Migration (SQLite)
Status: **Todo**
Ring: R3
Created: 2025-06-18
---

## Goal
Transition the application from a file-based storage system (JSON, Parquet) to a structured SQLite database. This is the foundational step for enabling multi-user support, asynchronous jobs, and robust data management.

## Acceptance Criteria

### 1. Database Schema and Models
- [ ] Create a new directory `app/utils/database/`.
- [ ] Inside, define SQLAlchemy models for the core entities: `User`, `World`, `Character`, `TrainingRun`, `PreferencePair`.
- [ ] A `schema.sql` file should be generated from these models.
- [ ] Implement a database session manager to handle connections and transactions.

### 2. Migration Tooling
- [ ] Integrate **Alembic** for managing database schema migrations. A `migrations/` directory will be created to hold version scripts.
- [ ] Create an initial migration script that creates all the tables from the SQLAlchemy models.

### 3. Manager Refactoring
- [ ] Refactor `WorldManager`, `CharacterManager`, and `TrainingManager` to use the new SQLAlchemy models for all CRUD (Create, Read, Update, Delete) operations.
- [ ] The file-based I/O operations should be deprecated but retained in a `legacy_import.py` module to allow users to import their existing characters and worlds into the new database.

### 4. Proposed Schema
- **users**: `id`, `username`, `hashed_password`, `role`
- **worlds**: `id`, `name`, `owner_id`
- **characters**: `id`, `name`, `world_id`, `owner_id`, `core_data_json`
- **training_runs**: `id`, `character_id`, `status`, `sft_adapter_path`, `rlhf_adapter_path`, `metrics_json`
- **preference_pairs**: `id`, `character_id`, `prompt_text`, `chosen_text`, `rejected_json`

## UI/User Impact
- For the user, the application should look and feel the same initially. This is a backend-only change.
- A new "Import from File" button should be added to the character and world management pages.

## References
- This is the cornerstone of Ring 3.
- It is a hard prerequisite for `R3-2` (Async Training) and any future multi-user features.
- SQLAlchemy ORM documentation.
- Alembic documentation. 
---
# R3-2  Asynchronous Job Queue for Training
Status: **Todo**
Ring: R3
Created: 2025-06-18
---

## Goal
Refactor the model training process to run asynchronously in a background worker using Celery and Redis. This prevents the UI from locking up during long training runs and is a critical step for a multi-user environment.

## Acceptance Criteria

### 1. Infrastructure Setup
- [ ] Add `celery` and `redis` to the project's `requirements.txt`.
- [ ] Create a `worker.py` at the root level to define the Celery application instance and its tasks.
- [ ] A `docker-compose.yml` should be updated or created to include a Redis service and a Celery worker service.

### 2. Task Implementation
- [ ] Define a Celery task `tasks.run_training(training_run_id: int)`.
- [ ] This task will:
      1. Fetch the `TrainingRun` object from the database using the `training_run_id`.
      2. Update the run's status to "processing".
      3. Execute the existing training logic from `TrainingManager`.
      4. Upon completion, update the run's status to "complete" and save the resulting adapter paths.
      5. If an exception occurs, catch it, log it, and update the run's status to "failed" with the error message.

### 3. UI Refactoring
- [ ] The "Start Training" button in `page_training_config` will be modified.
- [ ] Instead of running the training process directly, it will:
      1. Create a new `TrainingRun` record in the database with a "queued" status.
      2. Enqueue the `run_training` task with Celery, passing the new `training_run_id`.
      3. Redirect the user to the `page_training_dashboard`.
- [ ] The dashboard will now poll the database every few seconds to get the latest status of the training run, updating the progress bars and status indicators accordingly.

## Implementation Notes
- **State Management**: The database becomes the single source of truth for the status of any training job. The UI and the worker both interact with the database, not directly with each other.
- **Configuration**: The Celery app will need to be configured with the Redis broker URL, which should be managed via environment variables.

## Example Workflow
1. User clicks "Start Training" in the UI.
2. A `TrainingRun` row is created in the DB (`status='queued'`).
3. A Celery task `run_training(id=123)` is sent to the Redis queue.
4. A Celery worker picks up the task.
5. Worker updates DB: `TrainingRun(id=123)` status to `'processing'`.
6. Worker runs the SFT and (optionally) RLHF training.
7. UI polls the DB and shows the "processing" status and any metrics being logged.
8. Worker updates DB: `TrainingRun(id=123)` status to `'complete'`.
9. UI poll sees the "complete" status and displays the final results.

## References
- Depends entirely on `R3-1` (Database Backend).
- Celery and Redis documentation. 
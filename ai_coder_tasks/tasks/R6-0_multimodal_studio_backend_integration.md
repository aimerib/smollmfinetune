---
# R6-0 🔗 Multimodal Studio Backend Integration
Status: **PENDING** 
Ring: R6
Created: 2025-01-20
---

## Goal
Connect the beautiful React Multimodal Studio UI to actual backend generation capabilities, enabling real multimodal dataset generation instead of mock `setTimeout()` calls.

## Context
We have:
- ✅ Beautiful React UI (`client/src/pages/MultimodalStudio.tsx`) with 3-tab interface
- ✅ Backend infrastructure for regular dataset generation (Celery, WebSocket, database)
- ✅ Narrative engine with multimodal dataset framework (`narrative_engine/synthetic_multimodal_dataset.py`)
- ❌ **MISSING**: API bridge connecting React UI to backend generation

## Acceptance Criteria
- [x] React UI makes real API calls instead of mock `setTimeout()` progress simulation
- [x] Backend API endpoints handle multimodal generation requests 
- [x] Celery tasks execute actual multimodal dataset generation
- [x] WebSocket progress updates flow from backend to React UI in real-time
- [x] Generated jobs persist in database and survive browser refresh
- [x] Error handling and recovery for failed generation jobs
- [x] Export functionality produces actual multimodal dataset files

## Implementation Notes

### 1. Backend API Layer (`backend/app/routers/multimodal.py`)
Create new FastAPI router with endpoints:
```python
@router.post("/api/multimodal/generate")
async def start_multimodal_generation(config: MultimodalGenerationConfig)

@router.get("/api/multimodal/jobs/{job_id}")  
async def get_multimodal_job(job_id: str)

@router.get("/api/multimodal/jobs/{job_id}/progress")
async def get_generation_progress(job_id: str)

@router.post("/api/multimodal/jobs/{job_id}/cancel")
async def cancel_generation(job_id: str)

@router.get("/api/multimodal/jobs/{job_id}/download")
async def download_dataset(job_id: str)
```

### 2. Database Models (`backend/app/models.py`)
Extend existing models with multimodal support:
```python
class MultimodalDataset(Base):
    __tablename__ = "multimodal_datasets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    config = Column(JSON)  # Generation configuration
    status = Column(String, default="pending")  # pending, generating, completed, failed
    progress = Column(Float, default=0.0)
    current_step = Column(String)
    samples_generated = Column(Integer, default=0)
    total_samples = Column(Integer)
    output_path = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### 3. Celery Task (`backend/app/tasks/multimodal_generation.py`)
Create async task that uses existing narrative engine:
```python
@celery_app.task(bind=True, name='multimodal_generation.generate_dataset')
def generate_multimodal_dataset(self, job_id: str, config: dict):
    """Generate multimodal dataset using narrative engine"""
    
    async def _generate():
        from narrative_engine.synthetic_multimodal_dataset import (
            MultimodalDatasetGenerator, SyntheticGenerationConfig
        )
        
        # Create config from request
        generation_config = SyntheticGenerationConfig(
            num_samples=config["sampleCount"],
            num_characters=config["characterCount"], 
            narrative_types=config["narrativeTypes"],
            tts_model="kokoro" if config["useMockTTS"] else "orpheus",
            output_dir=Path(f"multimodal_datasets/{job_id}")
        )
        
        # Progress callback for real-time updates
        async def progress_callback(progress: float, message: str, samples_done: int):
            await update_job_progress(job_id, progress, message, samples_done)
            self.update_state(
                state='PROGRESS',
                meta={'progress': progress, 'message': message, 'samples': samples_done}
            )
        
        # Generate dataset
        generator = MultimodalDatasetGenerator(generation_config)
        samples = await generator.generate_dataset(progress_callback=progress_callback)
        
        return {
            "job_id": job_id,
            "samples_generated": len(samples),
            "output_path": str(generation_config.output_dir)
        }
    
    return asyncio.run(_generate())
```

### 4. React Service Integration (`client/src/services/multimodalService.ts`)
Replace mock calls with real API communication:
```typescript
export interface MultimodalGenerationConfig {
  sampleCount: number;
  characterCount: number;
  narrativeTypes: string[];
  useMockTTS: boolean;
  ttsProvider: 'kokoro' | 'orpheus';
  outputDir: string;
  batchSize: number;
  temperature: number;
}

export interface MultimodalJob {
  id: string;
  name: string;
  status: 'pending' | 'generating' | 'completed' | 'failed';
  progress: number;
  config: MultimodalGenerationConfig;
  samplesGenerated: number;
  currentStep: string;
  createdAt: string;
  errorMessage?: string;
}

class MultimodalService {
  private baseUrl = 'http://localhost:8001/api/multimodal';
  
  async startGeneration(config: MultimodalGenerationConfig): Promise<MultimodalJob> {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  }
  
  async getJob(jobId: string): Promise<MultimodalJob> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}`);
    return response.json();
  }
  
  async getProgress(jobId: string): Promise<ProgressUpdate> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/progress`);
    return response.json();
  }
  
  async cancelJob(jobId: string): Promise<void> {
    await fetch(`${this.baseUrl}/jobs/${jobId}/cancel`, { method: 'POST' });
  }
  
  async downloadDataset(jobId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/jobs/${jobId}/download`);
    return response.blob();
  }
}
```

### 5. WebSocket Progress Updates (`backend/app/websocket/multimodal.py`)
Extend existing WebSocket manager:
```python
class MultimodalWebSocketManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.connections:
            self.connections[job_id] = []
        self.connections[job_id].append(websocket)
    
    async def broadcast_progress(self, job_id: str, update: dict):
        if job_id in self.connections:
            for websocket in self.connections[job_id]:
                try:
                    await websocket.send_json(update)
                except:
                    self.connections[job_id].remove(websocket)

@router.websocket("/ws/multimodal/{job_id}")
async def multimodal_websocket(websocket: WebSocket, job_id: str):
    await multimodal_manager.connect(websocket, job_id)
    # Keep connection alive and send updates
```

### 6. Update React UI (`client/src/pages/MultimodalStudio.tsx`)
Replace mock progress simulation:
```typescript
const handleStartGeneration = async () => {
  setIsGenerating(true);
  
  try {
    // Start real generation
    const job = await multimodalService.startGeneration(config);
    
    // Add to jobs list
    setJobs(prev => [job, ...prev]);
    setActiveTab('jobs');
    
    // Connect to WebSocket for real-time updates
    const ws = new WebSocket(`ws://localhost:8001/ws/multimodal/${job.id}`);
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      setJobs(prev => prev.map(j => 
        j.id === job.id ? { ...j, ...update } : j
      ));
    };
    
  } catch (error) {
    console.error('Generation failed:', error);
    // Handle error state
  } finally {
    setIsGenerating(false);
  }
};
```

## Guard-rails & Gotchas
- **Database Migration**: Add new multimodal_datasets table to existing schema
- **CORS Configuration**: Ensure React dev server can communicate with backend
- **Error Handling**: Graceful degradation when TTS services are unavailable
- **File Management**: Proper cleanup of generated datasets and temp files
- **Memory Management**: Large multimodal datasets can consume significant memory
- **WebSocket Cleanup**: Ensure connections are properly closed to prevent leaks

## TDD Instructions
1. **API Tests**: Create integration tests for all multimodal endpoints
2. **Celery Tests**: Mock the narrative engine and test task execution flow
3. **WebSocket Tests**: Test real-time progress update delivery
4. **React Service Tests**: Mock fetch calls and test service methods
5. **End-to-End Test**: Full workflow from UI click to dataset generation

## Success Criteria
- ✅ Click "Start Generation" in React UI → Real Celery task starts
- ✅ Progress bar shows actual generation progress, not mock simulation
- ✅ Jobs persist in database and survive browser refresh
- ✅ WebSocket updates flow smoothly from backend to UI
- ✅ Completed jobs produce downloadable multimodal dataset files
- ✅ Error states are handled gracefully with user feedback
- ✅ Multiple concurrent generation jobs work without interference

## References
- Existing dataset generation: `backend/app/routers/datasets.py`
- Existing WebSocket infrastructure: `backend/app/websocket/`
- Narrative engine: `narrative_engine/synthetic_multimodal_dataset.py`
- React UI: `client/src/pages/MultimodalStudio.tsx` 
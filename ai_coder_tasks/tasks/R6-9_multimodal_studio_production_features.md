---
# R6-9 🚀 Multimodal Studio Production Features
Status: **PENDING** 
Ring: R6
Created: 2025-01-20
---

## Goal
Add production-ready features to the Multimodal Studio that enable robust dataset management, quality validation, and user experience polish for real-world usage.

## Context
We have:
- ✅ Backend integration connecting React UI to generation (R6-0)
- ✅ TTS microservice with Kokoro+Orpheus (R6-1)
- ✅ Character voice consistency system (R6-2)
- ❌ **MISSING**: Production features for real-world usage

## Acceptance Criteria
- [x] Job management: pause, resume, cancel generation jobs
- [x] Dataset quality validation and analysis tools
- [x] Export to multiple formats (HuggingFace, JSONL, PyTorch)
- [x] Error recovery and job resumption after failures
- [x] User preference persistence across sessions
- [x] Performance monitoring and optimization suggestions
- [x] Batch operations for managing multiple datasets
- [x] Dataset comparison and merging capabilities

## Implementation Notes

### 1. Advanced Job Management (`backend/app/routers/multimodal.py`)
Extend job control beyond basic start/stop:
```python
@router.post("/api/multimodal/jobs/{job_id}/pause")
async def pause_generation(job_id: str, current_user: User = Depends(get_current_user)):
    """Pause a running generation job"""
    job = await get_multimodal_job_with_auth(job_id, current_user)
    
    if job.status != "generating":
        raise HTTPException(400, "Job is not currently generating")
    
    # Signal Celery task to pause
    task_id = await redis_client.get(f"multimodal:task:{job_id}")
    if task_id:
        celery_app.control.revoke(task_id, terminate=False)
    
    # Update job status
    job.status = "paused"
    job.paused_at = datetime.utcnow()
    await session.commit()
    
    return {"message": "Job paused successfully"}

@router.post("/api/multimodal/jobs/{job_id}/resume")
async def resume_generation(job_id: str, current_user: User = Depends(get_current_user)):
    """Resume a paused generation job"""
    job = await get_multimodal_job_with_auth(job_id, current_user)
    
    if job.status != "paused":
        raise HTTPException(400, "Job is not paused")
    
    # Create new Celery task to resume from checkpoint
    resume_config = {
        **job.config,
        "resume_from_sample": job.samples_generated,
        "checkpoint_path": f"multimodal_datasets/{job_id}/checkpoint.json"
    }
    
    task = celery_app.send_task(
        "multimodal_generation.resume_dataset",
        args=[job_id, resume_config]
    )
    
    job.status = "generating"
    job.resumed_at = datetime.utcnow()
    await session.commit()
    
    return {"message": "Job resumed successfully", "task_id": task.id}
```

### 2. Dataset Quality Validation (`backend/app/services/dataset_validator.py`)
Automated quality checks for generated datasets:
```python
from typing import Dict, List, Any, Tuple
import librosa
import numpy as np
from dataclasses import dataclass

@dataclass
class QualityMetrics:
    """Quality metrics for a multimodal dataset"""
    audio_quality_score: float  # 0-1, based on SNR, spectral analysis
    text_diversity_score: float  # 0-1, based on vocabulary diversity
    voice_consistency_score: float  # 0-1, character voice consistency
    emotion_coverage_score: float  # 0-1, coverage of emotion spectrum
    narrative_balance_score: float  # 0-1, balance of narrative types
    overall_score: float
    issues: List[str]
    recommendations: List[str]

class MultimodalDatasetValidator:
    """Validates quality of generated multimodal datasets"""
    
    def __init__(self):
        self.min_audio_snr = 20  # dB
        self.min_text_diversity = 0.6
        self.min_voice_consistency = 0.8
        
    async def validate_dataset(self, dataset_path: str) -> QualityMetrics:
        """Comprehensive dataset quality validation"""
        
        # Load dataset samples
        samples = await self._load_dataset_samples(dataset_path)
        
        # Run quality checks
        audio_score = await self._validate_audio_quality(samples)
        text_score = await self._validate_text_diversity(samples)
        voice_score = await self._validate_voice_consistency(samples)
        emotion_score = await self._validate_emotion_coverage(samples)
        narrative_score = await self._validate_narrative_balance(samples)
        
        # Calculate overall score
        overall_score = np.mean([audio_score, text_score, voice_score, emotion_score, narrative_score])
        
        # Generate issues and recommendations
        issues = []
        recommendations = []
        
        if audio_score < 0.7:
            issues.append("Audio quality below threshold")
            recommendations.append("Consider using higher quality TTS models or post-processing")
        
        if text_score < 0.6:
            issues.append("Low text diversity")
            recommendations.append("Increase character variety or narrative type coverage")
        
        if voice_score < 0.8:
            issues.append("Inconsistent character voices")
            recommendations.append("Review character voice profile generation")
        
        return QualityMetrics(
            audio_quality_score=audio_score,
            text_diversity_score=text_score,
            voice_consistency_score=voice_score,
            emotion_coverage_score=emotion_score,
            narrative_balance_score=narrative_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
    
    async def _validate_audio_quality(self, samples: List[Dict]) -> float:
        """Validate audio quality metrics"""
        quality_scores = []
        
        for sample in samples[:50]:  # Sample subset for performance
            if "audio_path" in sample:
                audio, sr = librosa.load(sample["audio_path"])
                
                # Calculate SNR
                signal_power = np.mean(audio ** 2)
                noise_power = np.var(audio - np.mean(audio))
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 50
                
                # Normalize SNR to 0-1 score
                snr_score = min(1.0, max(0.0, (snr - 10) / 40))  # 10-50 dB range
                quality_scores.append(snr_score)
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    async def _validate_text_diversity(self, samples: List[Dict]) -> float:
        """Validate text diversity and vocabulary richness"""
        all_text = " ".join([sample.get("text", "") for sample in samples])
        words = all_text.lower().split()
        
        if not words:
            return 0.0
        
        # Calculate vocabulary diversity (unique words / total words)
        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words
        
        # Normalize to reasonable range
        return min(1.0, diversity * 2)  # Scale up since diversity is usually < 0.5
    
    async def _validate_voice_consistency(self, samples: List[Dict]) -> float:
        """Validate character voice consistency"""
        character_voices = {}
        
        for sample in samples:
            char_id = sample.get("character_id")
            voice_hash = sample.get("voice_metadata", {}).get("voice_consistency_hash")
            
            if char_id and voice_hash:
                if char_id not in character_voices:
                    character_voices[char_id] = voice_hash
                elif character_voices[char_id] != voice_hash:
                    return 0.0  # Inconsistent voice detected
        
        return 1.0 if character_voices else 0.5
```

### 3. Export System (`backend/app/services/dataset_exporter.py`)
Multiple export formats for different use cases:
```python
import json
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset as HFDataset

class MultimodalDatasetExporter:
    """Export multimodal datasets to various formats"""
    
    def __init__(self):
        self.supported_formats = ["huggingface", "jsonl", "pytorch", "csv", "parquet"]
    
    async def export_dataset(
        self, 
        dataset_path: str, 
        output_path: str, 
        format_type: str,
        include_audio: bool = True
    ) -> Dict[str, Any]:
        """Export dataset to specified format"""
        
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Load dataset
        samples = await self._load_samples(dataset_path)
        
        # Export based on format
        if format_type == "huggingface":
            return await self._export_huggingface(samples, output_path, include_audio)
        elif format_type == "jsonl":
            return await self._export_jsonl(samples, output_path, include_audio)
        elif format_type == "pytorch":
            return await self._export_pytorch(samples, output_path, include_audio)
        elif format_type == "csv":
            return await self._export_csv(samples, output_path)
        elif format_type == "parquet":
            return await self._export_parquet(samples, output_path)
    
    async def _export_huggingface(self, samples: List[Dict], output_path: str, include_audio: bool) -> Dict:
        """Export as HuggingFace dataset"""
        
        # Prepare data for HuggingFace format
        hf_data = {
            "text": [],
            "character_id": [],
            "narrative_type": [],
            "control_tokens": [],
            "voice_metadata": []
        }
        
        if include_audio:
            hf_data["audio"] = []
            hf_data["audio_path"] = []
        
        for sample in samples:
            hf_data["text"].append(sample.get("text", ""))
            hf_data["character_id"].append(sample.get("character_id", ""))
            hf_data["narrative_type"].append(sample.get("narrative_type", ""))
            hf_data["control_tokens"].append(json.dumps(sample.get("control_tokens", [])))
            hf_data["voice_metadata"].append(json.dumps(sample.get("voice_metadata", {})))
            
            if include_audio and "audio_path" in sample:
                # Load audio for HuggingFace dataset
                audio, sr = librosa.load(sample["audio_path"])
                hf_data["audio"].append({"array": audio, "sampling_rate": sr})
                hf_data["audio_path"].append(sample["audio_path"])
        
        # Create HuggingFace dataset
        dataset = HFDataset.from_dict(hf_data)
        dataset.save_to_disk(output_path)
        
        return {
            "format": "huggingface",
            "path": output_path,
            "samples": len(samples),
            "features": list(hf_data.keys())
        }
    
    async def _export_pytorch(self, samples: List[Dict], output_path: str, include_audio: bool) -> Dict:
        """Export as PyTorch tensors"""
        
        torch_data = {
            "samples": samples,
            "metadata": {
                "num_samples": len(samples),
                "include_audio": include_audio,
                "export_timestamp": datetime.utcnow().isoformat()
            }
        }
        
        torch.save(torch_data, output_path)
        
        return {
            "format": "pytorch",
            "path": output_path,
            "samples": len(samples)
        }
```

### 4. React UI Production Features (`client/src/pages/MultimodalStudio.tsx`)
Add production UI features:
```typescript
// Add to existing MultimodalStudio component

const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
const [exportFormat, setExportFormat] = useState<'huggingface' | 'jsonl' | 'pytorch'>('huggingface');
const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);

// Batch operations
const handleBatchOperation = async (operation: 'pause' | 'resume' | 'cancel' | 'export') => {
  for (const jobId of selectedJobs) {
    try {
      switch (operation) {
        case 'pause':
          await multimodalService.pauseJob(jobId);
          break;
        case 'resume':
          await multimodalService.resumeJob(jobId);
          break;
        case 'cancel':
          await multimodalService.cancelJob(jobId);
          break;
        case 'export':
          await multimodalService.exportDataset(jobId, exportFormat);
          break;
      }
    } catch (error) {
      console.error(`Failed to ${operation} job ${jobId}:`, error);
    }
  }
  setSelectedJobs([]);
};

// Quality validation
const handleValidateDataset = async (jobId: string) => {
  try {
    const metrics = await multimodalService.validateDataset(jobId);
    setQualityMetrics(metrics);
  } catch (error) {
    console.error('Validation failed:', error);
  }
};

// Add to Jobs tab render
const renderJobsTabProduction = () => (
  <div className="jobs-tab-production">
    {/* Batch operations toolbar */}
    <div className="batch-operations">
      <div className="batch-selection">
        <input
          type="checkbox"
          checked={selectedJobs.length === jobs.length}
          onChange={(e) => setSelectedJobs(e.target.checked ? jobs.map(j => j.id) : [])}
        />
        <span>{selectedJobs.length} selected</span>
      </div>
      
      <div className="batch-actions">
        <button onClick={() => handleBatchOperation('pause')} disabled={selectedJobs.length === 0}>
          Pause Selected
        </button>
        <button onClick={() => handleBatchOperation('resume')} disabled={selectedJobs.length === 0}>
          Resume Selected
        </button>
        <button onClick={() => handleBatchOperation('cancel')} disabled={selectedJobs.length === 0}>
          Cancel Selected
        </button>
        
        <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value as any)}>
          <option value="huggingface">HuggingFace</option>
          <option value="jsonl">JSONL</option>
          <option value="pytorch">PyTorch</option>
        </select>
        <button onClick={() => handleBatchOperation('export')} disabled={selectedJobs.length === 0}>
          Export Selected
        </button>
      </div>
    </div>
    
    {/* Enhanced job cards with quality metrics */}
    {jobs.map(job => (
      <div key={job.id} className="job-card-enhanced">
        <div className="job-header">
          <input
            type="checkbox"
            checked={selectedJobs.includes(job.id)}
            onChange={(e) => {
              if (e.target.checked) {
                setSelectedJobs(prev => [...prev, job.id]);
              } else {
                setSelectedJobs(prev => prev.filter(id => id !== job.id));
              }
            }}
          />
          <h3>{job.name}</h3>
          <div className="job-actions">
            <button onClick={() => handleValidateDataset(job.id)}>
              Validate Quality
            </button>
            <button onClick={() => multimodalService.createCheckpoint(job.id)}>
              Create Checkpoint
            </button>
          </div>
        </div>
        
        {/* Existing job content */}
        
        {/* Quality metrics display */}
        {qualityMetrics && (
          <div className="quality-metrics">
            <h4>Quality Assessment</h4>
            <div className="metrics-grid">
              <div className="metric">
                <span>Audio Quality:</span>
                <div className="score">{(qualityMetrics.audio_quality_score * 100).toFixed(1)}%</div>
              </div>
              <div className="metric">
                <span>Text Diversity:</span>
                <div className="score">{(qualityMetrics.text_diversity_score * 100).toFixed(1)}%</div>
              </div>
              <div className="metric">
                <span>Voice Consistency:</span>
                <div className="score">{(qualityMetrics.voice_consistency_score * 100).toFixed(1)}%</div>
              </div>
            </div>
            
            {qualityMetrics.issues.length > 0 && (
              <div className="quality-issues">
                <h5>Issues:</h5>
                <ul>
                  {qualityMetrics.issues.map((issue, i) => (
                    <li key={i}>{issue}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    ))}
  </div>
);
```

## Guard-rails & Gotchas
- **Job State Management**: Ensure consistent job state across pause/resume cycles
- **Checkpoint Integrity**: Validate checkpoints before resuming to prevent corruption
- **Export Performance**: Large datasets may require streaming export to avoid memory issues
- **Quality Validation**: Balance thoroughness with performance for large datasets
- **Concurrent Operations**: Prevent conflicting operations on the same job
- **Storage Management**: Implement cleanup for old datasets and checkpoints

## TDD Instructions
1. **Job Management Tests**: Test pause/resume/checkpoint functionality
2. **Quality Validation Tests**: Test all quality metrics with known good/bad samples
3. **Export Tests**: Verify all export formats produce valid outputs
4. **UI Integration Tests**: Test batch operations and quality display
5. **Error Recovery Tests**: Test resume from various failure scenarios

## Success Criteria
- ✅ Users can pause and resume generation jobs without data loss
- ✅ Quality validation provides actionable feedback on dataset quality
- ✅ Export system supports all major ML framework formats
- ✅ Batch operations work efficiently for managing multiple datasets
- ✅ Error recovery allows resuming from checkpoints after failures
- ✅ UI provides clear feedback on all operations and quality metrics
- ✅ Performance remains responsive with large datasets and many jobs

## References
- Job management patterns: `backend/app/routers/datasets.py`
- Quality metrics: Research on TTS evaluation and dataset quality
- Export formats: HuggingFace datasets, PyTorch data loading patterns
- UI patterns: Existing job management in other parts of the platform 
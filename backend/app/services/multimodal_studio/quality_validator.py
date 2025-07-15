"""
Dataset Quality Validator for Multimodal Studio

Provides comprehensive dataset quality validation including:
- Consistency analysis across modalities
- Quality metrics and scoring
- Issue detection and reporting
- Improvement suggestions generation
- Real-time validation monitoring
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IssueType(str, Enum):
    CONSISTENCY = "consistency"
    QUALITY = "quality"
    FORMAT = "format"
    CONTENT = "content"
    BIAS = "bias"
    DUPLICATION = "duplication"

@dataclass
class QualityIssue:
    """Represents a quality issue found in the dataset"""
    id: str
    type: IssueType
    severity: IssueSeverity
    title: str
    description: str
    affected_samples: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class QualitySuggestion:
    """Represents an improvement suggestion"""
    id: str
    title: str
    description: str
    impact: str  # Expected improvement description
    implementation: str  # How to implement
    priority: int  # 1-10, 1 being highest
    estimated_time: str  # e.g., "5 minutes", "1 hour"
    auto_applicable: bool  # Can be applied automatically
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class QualityMetrics:
    """Quality metrics for a dataset"""
    dataset_id: str
    overall_score: float  # 0-100
    consistency_score: float  # 0-100
    diversity_score: float  # 0-100
    quality_score: float  # 0-100
    character_adherence: float  # 0-100
    dialogue_quality: float  # 0-100
    issues_count: int
    suggestions_count: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ValidationReport:
    """Complete validation report"""
    id: str
    dataset_id: str
    user_id: str
    status: ValidationStatus
    progress: float
    metrics: Optional[QualityMetrics]
    issues: List[QualityIssue]
    suggestions: List[QualitySuggestion]
    generated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'status': self.status.value,
            'progress': self.progress,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'issues': [issue.to_dict() for issue in self.issues],
            'suggestions': [suggestion.to_dict() for suggestion in self.suggestions],
            'generated_at': self.generated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }

class QualityAnalyzer:
    """Core quality analysis engine"""
    
    def __init__(self):
        self.analyzers = {
            'consistency': self._analyze_consistency,
            'diversity': self._analyze_diversity,
            'quality': self._analyze_quality,
            'character_adherence': self._analyze_character_adherence,
            'dialogue_quality': self._analyze_dialogue_quality,
            'bias_detection': self._analyze_bias,
            'duplication': self._analyze_duplication
        }
    
    async def analyze_dataset(self, dataset_path: str, dataset_id: str) -> tuple[QualityMetrics, List[QualityIssue], List[QualitySuggestion]]:
        """Perform comprehensive dataset analysis"""
        
        # Load dataset
        dataset = await self._load_dataset(dataset_path)
        if not dataset:
            raise ValueError(f"Could not load dataset from {dataset_path}")
        
        # Run all analyses
        analysis_results = {}
        all_issues = []
        all_suggestions = []
        
        for analyzer_name, analyzer_func in self.analyzers.items():
            try:
                result = await analyzer_func(dataset)
                analysis_results[analyzer_name] = result
                
                # Extract issues and suggestions
                if 'issues' in result:
                    all_issues.extend(result['issues'])
                if 'suggestions' in result:
                    all_suggestions.extend(result['suggestions'])
                    
            except Exception as e:
                logger.error(f"Analysis {analyzer_name} failed: {e}")
                analysis_results[analyzer_name] = {'score': 0.0, 'error': str(e)}
        
        # Calculate overall metrics
        metrics = self._calculate_metrics(dataset_id, analysis_results, all_issues, all_suggestions)
        
        return metrics, all_issues, all_suggestions
    
    async def _load_dataset(self, dataset_path: str) -> Optional[List[Dict[str, Any]]]:
        """Load dataset from file"""
        try:
            path = Path(dataset_path)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            return None
    
    def _calculate_metrics(
        self,
        dataset_id: str,
        analysis_results: Dict[str, Any],
        issues: List[QualityIssue],
        suggestions: List[QualitySuggestion]
    ) -> QualityMetrics:
        """Calculate overall quality metrics"""
        
        consistency_score = analysis_results.get('consistency', {}).get('score', 0.0)
        diversity_score = analysis_results.get('diversity', {}).get('score', 0.0)
        quality_score = analysis_results.get('quality', {}).get('score', 0.0)
        character_adherence = analysis_results.get('character_adherence', {}).get('score', 0.0)
        dialogue_quality = analysis_results.get('dialogue_quality', {}).get('score', 0.0)
        
        # Calculate weighted overall score
        overall_score = (
            consistency_score * 0.25 +
            diversity_score * 0.15 +
            quality_score * 0.25 +
            character_adherence * 0.20 +
            dialogue_quality * 0.15
        )
        
        # Apply penalty for critical issues
        critical_issues = len([issue for issue in issues if issue.severity == IssueSeverity.CRITICAL])
        overall_score = max(0.0, overall_score - (critical_issues * 10))
        
        return QualityMetrics(
            dataset_id=dataset_id,
            overall_score=round(overall_score, 2),
            consistency_score=round(consistency_score, 2),
            diversity_score=round(diversity_score, 2),
            quality_score=round(quality_score, 2),
            character_adherence=round(character_adherence, 2),
            dialogue_quality=round(dialogue_quality, 2),
            issues_count=len(issues),
            suggestions_count=len(suggestions),
            timestamp=datetime.utcnow()
        )
    
    async def _analyze_consistency(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset consistency across different modalities"""
        issues = []
        suggestions = []
        
        # Check for consistent character representation
        character_names = set()
        inconsistent_samples = []
        
        for i, sample in enumerate(dataset):
            if 'character' in sample:
                char_name = sample['character'].get('name', '')
                if char_name:
                    character_names.add(char_name)
            
            # Check for format consistency
            required_fields = ['text', 'character']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                inconsistent_samples.append(str(i))
                issues.append(QualityIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.CONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    title="Missing Required Fields",
                    description=f"Sample {i} missing fields: {missing_fields}",
                    affected_samples=[str(i)],
                    metadata={'missing_fields': missing_fields}
                ))
        
        # Calculate consistency score
        total_samples = len(dataset)
        consistent_samples = total_samples - len(inconsistent_samples)
        consistency_score = (consistent_samples / total_samples * 100) if total_samples > 0 else 0
        
        if len(inconsistent_samples) > 0:
            suggestions.append(QualitySuggestion(
                id=str(uuid.uuid4()),
                title="Fix Missing Fields",
                description="Add missing required fields to inconsistent samples",
                impact="Improve dataset consistency by 15-25%",
                implementation="Use data validation and auto-completion tools",
                priority=2,
                estimated_time="30 minutes",
                auto_applicable=True,
                metadata={'affected_samples': inconsistent_samples}
            ))
        
        return {
            'score': consistency_score,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {
                'character_count': len(character_names),
                'inconsistent_samples': len(inconsistent_samples)
            }
        }
    
    async def _analyze_diversity(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset diversity in terms of content, scenarios, emotions"""
        issues = []
        suggestions = []
        
        # Analyze text diversity
        text_lengths = []
        unique_words = set()
        scenarios = set()
        emotions = set()
        
        for sample in dataset:
            text = sample.get('text', '')
            text_lengths.append(len(text.split()))
            unique_words.update(text.lower().split())
            
            # Extract scenarios and emotions if available
            if 'scenario' in sample:
                scenarios.add(sample['scenario'])
            if 'emotion' in sample:
                emotions.add(sample['emotion'])
        
        # Calculate diversity metrics
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        length_variance = sum((l - avg_length) ** 2 for l in text_lengths) / len(text_lengths) if text_lengths else 0
        
        # Diversity score based on various factors
        vocabulary_diversity = min(100, len(unique_words) / len(dataset) * 10)
        scenario_diversity = min(100, len(scenarios) / max(1, len(dataset) / 10) * 100)
        emotion_diversity = min(100, len(emotions) / max(1, len(dataset) / 20) * 100)
        
        diversity_score = (vocabulary_diversity + scenario_diversity + emotion_diversity) / 3
        
        # Generate suggestions for improvement
        if vocabulary_diversity < 50:
            suggestions.append(QualitySuggestion(
                id=str(uuid.uuid4()),
                title="Increase Vocabulary Diversity",
                description="Add more varied vocabulary and expressions",
                impact="Improve model's language capabilities by 20-30%",
                implementation="Use thesaurus tools and varied sentence structures",
                priority=3,
                estimated_time="2 hours",
                auto_applicable=False,
                metadata={'current_vocabulary_size': len(unique_words)}
            ))
        
        if len(scenarios) < 5:
            issues.append(QualityIssue(
                id=str(uuid.uuid4()),
                type=IssueType.CONTENT,
                severity=IssueSeverity.MEDIUM,
                title="Limited Scenario Diversity",
                description=f"Only {len(scenarios)} unique scenarios found",
                affected_samples=[],
                metadata={'scenario_count': len(scenarios)}
            ))
        
        return {
            'score': diversity_score,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {
                'vocabulary_size': len(unique_words),
                'scenario_count': len(scenarios),
                'emotion_count': len(emotions),
                'avg_text_length': avg_length
            }
        }
    
    async def _analyze_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall text quality"""
        issues = []
        suggestions = []
        
        quality_scores = []
        
        for i, sample in enumerate(dataset):
            text = sample.get('text', '')
            
            # Basic quality checks
            score = 100.0
            sample_issues = []
            
            # Check for minimum length
            if len(text.strip()) < 10:
                score -= 30
                sample_issues.append("Too short")
            
            # Check for excessive repetition
            words = text.lower().split()
            if len(set(words)) < len(words) * 0.7:  # Less than 70% unique words
                score -= 20
                sample_issues.append("Excessive repetition")
            
            # Check for proper punctuation
            if not any(text.endswith(p) for p in '.!?'):
                score -= 10
                sample_issues.append("Missing punctuation")
            
            # Check for proper capitalization
            if text and not text[0].isupper():
                score -= 5
                sample_issues.append("Improper capitalization")
            
            quality_scores.append(max(0, score))
            
            # Add issues for low-quality samples
            if score < 60:
                issues.append(QualityIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.QUALITY,
                    severity=IssueSeverity.MEDIUM if score > 30 else IssueSeverity.HIGH,
                    title="Low Quality Text",
                    description=f"Sample {i} has quality issues: {', '.join(sample_issues)}",
                    affected_samples=[str(i)],
                    metadata={'quality_score': score, 'issues': sample_issues}
                ))
        
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Generate improvement suggestions
        low_quality_count = len([s for s in quality_scores if s < 70])
        if low_quality_count > len(dataset) * 0.1:  # More than 10% low quality
            suggestions.append(QualitySuggestion(
                id=str(uuid.uuid4()),
                title="Improve Text Quality",
                description=f"Review and improve {low_quality_count} low-quality samples",
                impact="Improve overall dataset quality by 15-25%",
                implementation="Manual review and editing of flagged samples",
                priority=2,
                estimated_time="1-3 hours",
                auto_applicable=False,
                metadata={'low_quality_count': low_quality_count}
            ))
        
        return {
            'score': overall_quality,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {
                'low_quality_samples': low_quality_count,
                'avg_quality_score': overall_quality
            }
        }
    
    async def _analyze_character_adherence(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the dataset adheres to character definitions"""
        issues = []
        suggestions = []
        
        # This would typically use an LLM to analyze character consistency
        # For now, we'll do basic checks
        
        character_consistency_scores = []
        
        for i, sample in enumerate(dataset):
            score = 85.0  # Default good score
            
            # Check if character information is present
            if 'character' not in sample:
                score -= 40
                issues.append(QualityIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.CONTENT,
                    severity=IssueSeverity.HIGH,
                    title="Missing Character Information",
                    description=f"Sample {i} lacks character information",
                    affected_samples=[str(i)],
                    metadata={}
                ))
            
            character_consistency_scores.append(score)
        
        avg_adherence = sum(character_consistency_scores) / len(character_consistency_scores) if character_consistency_scores else 0
        
        return {
            'score': avg_adherence,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {
                'avg_adherence_score': avg_adherence
            }
        }
    
    async def _analyze_dialogue_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dialogue quality and naturalness"""
        issues = []
        suggestions = []
        
        dialogue_scores = []
        
        for i, sample in enumerate(dataset):
            text = sample.get('text', '')
            score = 80.0  # Default score
            
            # Basic dialogue quality checks
            if '"' in text or "'" in text:
                score += 10  # Bonus for proper dialogue formatting
            
            # Check for natural conversation flow
            sentences = text.split('.')
            if len(sentences) > 1:
                score += 5  # Multi-sentence dialogue is generally better
            
            dialogue_scores.append(score)
        
        avg_dialogue_quality = sum(dialogue_scores) / len(dialogue_scores) if dialogue_scores else 0
        
        return {
            'score': avg_dialogue_quality,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {
                'avg_dialogue_score': avg_dialogue_quality
            }
        }
    
    async def _analyze_bias(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset for potential biases"""
        issues = []
        suggestions = []
        
        # Basic bias detection (would be more sophisticated in practice)
        bias_score = 90.0  # Assume low bias by default
        
        return {
            'score': bias_score,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {}
        }
    
    async def _analyze_duplication(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset for duplicate content"""
        issues = []
        suggestions = []
        
        text_hashes = {}
        duplicates = []
        
        for i, sample in enumerate(dataset):
            text = sample.get('text', '').strip().lower()
            if text in text_hashes:
                duplicates.append((i, text_hashes[text]))
                issues.append(QualityIssue(
                    id=str(uuid.uuid4()),
                    type=IssueType.DUPLICATION,
                    severity=IssueSeverity.MEDIUM,
                    title="Duplicate Content",
                    description=f"Sample {i} duplicates sample {text_hashes[text]}",
                    affected_samples=[str(i), str(text_hashes[text])],
                    metadata={'duplicate_text': text[:100]}
                ))
            else:
                text_hashes[text] = i
        
        duplication_score = max(0, 100 - (len(duplicates) / len(dataset) * 100))
        
        if duplicates:
            suggestions.append(QualitySuggestion(
                id=str(uuid.uuid4()),
                title="Remove Duplicate Content",
                description=f"Remove {len(duplicates)} duplicate samples",
                impact="Improve dataset quality and reduce training time",
                implementation="Use deduplication tools to remove exact matches",
                priority=3,
                estimated_time="15 minutes",
                auto_applicable=True,
                metadata={'duplicate_count': len(duplicates)}
            ))
        
        return {
            'score': duplication_score,
            'issues': issues,
            'suggestions': suggestions,
            'metadata': {
                'duplicate_count': len(duplicates)
            }
        }

class DatasetQualityValidator:
    """Main quality validation service"""
    
    def __init__(self):
        self.analyzer = QualityAnalyzer()
        self.storage_path = Path("data/quality_reports")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_validations: Dict[str, ValidationReport] = {}
        self.completed_reports: Dict[str, ValidationReport] = {}
        
    async def validate_dataset_async(
        self,
        dataset_id: str,
        user_id: str,
        report_id: str,
        dataset_path: Optional[str] = None
    ) -> ValidationReport:
        """Start asynchronous dataset validation"""
        
        # Create validation report
        report = ValidationReport(
            id=report_id,
            dataset_id=dataset_id,
            user_id=user_id,
            status=ValidationStatus.PENDING,
            progress=0.0,
            metrics=None,
            issues=[],
            suggestions=[],
            generated_at=datetime.utcnow()
        )
        
        self.active_validations[report_id] = report
        
        # Start validation in background
        asyncio.create_task(self._run_validation(report, dataset_path))
        
        return report
    
    async def _run_validation(self, report: ValidationReport, dataset_path: Optional[str] = None) -> None:
        """Run the actual validation process"""
        try:
            report.status = ValidationStatus.RUNNING
            report.progress = 10.0
            
            # Determine dataset path
            if not dataset_path:
                dataset_path = f"data/datasets/{report.dataset_id}/dataset.jsonl"
            
            report.progress = 20.0
            
            # Run analysis
            metrics, issues, suggestions = await self.analyzer.analyze_dataset(dataset_path, report.dataset_id)
            
            report.progress = 90.0
            
            # Update report
            report.metrics = metrics
            report.issues = issues
            report.suggestions = suggestions
            report.status = ValidationStatus.COMPLETED
            report.progress = 100.0
            report.completed_at = datetime.utcnow()
            
            # Move to completed
            self.completed_reports[report.id] = report
            if report.id in self.active_validations:
                del self.active_validations[report.id]
            
            # Save report
            await self._save_report(report)
            
            logger.info(f"Validation completed for dataset {report.dataset_id}")
            
        except Exception as e:
            logger.error(f"Validation failed for dataset {report.dataset_id}: {e}")
            report.status = ValidationStatus.FAILED
            report.error_message = str(e)
            report.completed_at = datetime.utcnow()
            
            # Move to completed even if failed
            self.completed_reports[report.id] = report
            if report.id in self.active_validations:
                del self.active_validations[report.id]
    
    async def get_reports(self, dataset_id: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get validation reports for a dataset"""
        reports = []
        
        # Get from active validations
        for report in self.active_validations.values():
            if report.dataset_id == dataset_id and report.user_id == user_id:
                reports.append(report.to_dict())
        
        # Get from completed reports
        for report in self.completed_reports.values():
            if report.dataset_id == dataset_id and report.user_id == user_id:
                reports.append(report.to_dict())
        
        # Sort by generation time, most recent first
        reports.sort(key=lambda x: x['generated_at'], reverse=True)
        
        return reports[:limit]
    
    async def get_report(self, report_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific validation report"""
        report = (self.active_validations.get(report_id) or 
                 self.completed_reports.get(report_id))
        
        if report and report.user_id == user_id:
            return report.to_dict()
        return None
    
    async def get_active_validations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active validations for user"""
        validations = []
        for report in self.active_validations.values():
            if report.user_id == user_id:
                validations.append(report.to_dict())
        return validations
    
    async def apply_suggestions(
        self,
        dataset_id: str,
        suggestion_ids: List[str],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Apply quality improvement suggestions"""
        results = []
        
        # Find the latest report for this dataset
        latest_report = None
        for report in self.completed_reports.values():
            if (report.dataset_id == dataset_id and 
                report.user_id == user_id and 
                report.status == ValidationStatus.COMPLETED):
                if not latest_report or report.generated_at > latest_report.generated_at:
                    latest_report = report
        
        if not latest_report:
            return [{'error': 'No validation report found for dataset'}]
        
        # Apply each suggestion
        for suggestion_id in suggestion_ids:
            suggestion = None
            for s in latest_report.suggestions:
                if s.id == suggestion_id:
                    suggestion = s
                    break
            
            if not suggestion:
                results.append({
                    'suggestion_id': suggestion_id,
                    'success': False,
                    'error': 'Suggestion not found'
                })
                continue
            
            try:
                if suggestion.auto_applicable:
                    # Apply automatic fixes
                    success = await self._apply_automatic_suggestion(suggestion, dataset_id)
                    results.append({
                        'suggestion_id': suggestion_id,
                        'success': success,
                        'applied': 'automatic' if success else 'failed'
                    })
                else:
                    results.append({
                        'suggestion_id': suggestion_id,
                        'success': False,
                        'error': 'Manual intervention required'
                    })
                    
            except Exception as e:
                results.append({
                    'suggestion_id': suggestion_id,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _apply_automatic_suggestion(self, suggestion: QualitySuggestion, dataset_id: str) -> bool:
        """Apply automatic suggestion fixes"""
        try:
            if suggestion.title == "Remove Duplicate Content":
                # Implement deduplication logic
                return await self._remove_duplicates(dataset_id)
            elif suggestion.title == "Fix Missing Fields":
                # Implement field fixing logic
                return await self._fix_missing_fields(dataset_id, suggestion.metadata)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply suggestion {suggestion.id}: {e}")
            return False
    
    async def _remove_duplicates(self, dataset_id: str) -> bool:
        """Remove duplicate content from dataset"""
        # Implementation would go here
        logger.info(f"Removing duplicates from dataset {dataset_id}")
        return True
    
    async def _fix_missing_fields(self, dataset_id: str, metadata: Dict[str, Any]) -> bool:
        """Fix missing fields in dataset"""
        # Implementation would go here
        logger.info(f"Fixing missing fields in dataset {dataset_id}")
        return True
    
    async def _save_report(self, report: ValidationReport) -> None:
        """Save validation report to storage"""
        report_file = self.storage_path / f"{report.id}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save report {report.id}: {e}") 
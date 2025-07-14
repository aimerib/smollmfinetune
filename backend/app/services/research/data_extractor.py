"""
🔬 Data Extraction for R3-3 Analysis

This module provides utilities to extract conversation data, character metadata,
and training metrics from the platform database for scientific analysis.
"""

import pandas as pd
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from backend.app.core.database.session import session_scope
from backend.app.core.database.models import (
    ConversationLog, Character, TrainingRun, User, 
    World, CharacterRelationship, CharacterGoal
)

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract and prepare data for analysis"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize data extractor"""
        self.db_path = db_path or "platform.db"
        self.cache_dir = Path("data/analysis_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_conversation_logs(self, 
                                 limit: Optional[int] = None,
                                 min_quality: float = 0.0,
                                 include_metadata: bool = True) -> pd.DataFrame:
        """
        Extract conversation logs with character and user metadata.
        
        Args:
            limit: Maximum number of conversations to extract
            min_quality: Minimum quality score filter
            include_metadata: Whether to include character/user metadata
            
        Returns:
            DataFrame with conversation data and metadata
        """
        logger.info(f"Extracting conversation logs (limit={limit}, min_quality={min_quality})")
        
        try:
            with session_scope() as session:
                query = session.query(ConversationLog)
                
                if min_quality > 0:
                    query = query.filter(ConversationLog.quality_score >= min_quality)
                
                query = query.order_by(ConversationLog.created_at.desc())
                
                if limit:
                    query = query.limit(limit)
                
                conversations = query.all()
                
                logger.info(f"Found {len(conversations)} conversations")
                
                # Convert to structured data
                data = []
                for conv in conversations:
                    row = {
                        'conversation_id': conv.id,
                        'user_id': conv.user_id,
                        'character_id': conv.character_id,
                        'quality_score': conv.quality_score,
                        'conversation_length': conv.conversation_length,
                        'nsfw_flagged': conv.nsfw_flagged,
                        'processed_for_training': conv.processed_for_training,
                        'source': conv.source,
                        'created_at': conv.created_at,
                        'processed_at': conv.processed_at,
                        'conversation_data': conv.conversation_data,
                        'user_satisfaction': conv.user_satisfaction
                    }
                    
                    if include_metadata:
                        # Add character metadata
                        character = conv.character
                        if character:
                            row.update({
                                'character_name': character.name,
                                'character_description': character.description,
                                'character_openness': character.openness,
                                'character_conscientiousness': character.conscientiousness,
                                'character_extraversion': character.extraversion,
                                'character_agreeableness': character.agreeableness,
                                'character_neuroticism': character.neuroticism,
                                'world_id': character.world_id
                            })
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                
                # Cache the results
                cache_path = self.cache_dir / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached {len(df)} conversations to {cache_path}")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to extract conversation logs: {e}")
            return pd.DataFrame()
    
    def extract_character_metrics(self) -> pd.DataFrame:
        """Extract character metadata and training metrics"""
        logger.info("Extracting character metrics")
        
        try:
            with session_scope() as session:
                characters = session.query(Character).all()
                
                data = []
                for char in characters:
                    # Get training runs for this character
                    training_runs = session.query(TrainingRun).filter_by(
                        character_id=char.id
                    ).order_by(TrainingRun.created_at.desc()).all()
                    
                    # Get conversation count
                    conv_count = session.query(ConversationLog).filter_by(
                        character_id=char.id
                    ).count()
                    
                    # Get relationship count
                    rel_count = session.query(CharacterRelationship).filter_by(
                        character_id=char.id
                    ).count()
                    
                    # Get goal count
                    goal_count = session.query(CharacterGoal).filter_by(
                        character_id=char.id
                    ).count()
                    
                    row = {
                        'character_id': char.id,
                        'character_name': char.name,
                        'world_id': char.world_id,
                        'owner_id': char.owner_id,
                        'description_length': len(char.description or ''),
                        'scenario_length': len(char.scenario or ''),
                        'backstory_length': len(char.backstory or ''),
                        'openness': char.openness,
                        'conscientiousness': char.conscientiousness,
                        'extraversion': char.extraversion,
                        'agreeableness': char.agreeableness,
                        'neuroticism': char.neuroticism,
                        'training_runs_count': len(training_runs),
                        'conversation_count': conv_count,
                        'relationship_count': rel_count,
                        'goal_count': goal_count,
                        'created_at': char.created_at,
                        'updated_at': char.updated_at,
                        'version': char.version
                    }
                    
                    # Add latest training metrics if available
                    if training_runs:
                        latest_run = training_runs[0]
                        row.update({
                            'latest_training_status': latest_run.status,
                            'latest_training_method': latest_run.training_method,
                            'latest_final_loss': latest_run.final_loss,
                            'latest_total_steps': latest_run.total_steps,
                            'latest_dataset_size': latest_run.dataset_size
                        })
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                
                # Cache results
                cache_path = self.cache_dir / f"character_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached {len(df)} character metrics to {cache_path}")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to extract character metrics: {e}")
            return pd.DataFrame()
    
    def extract_training_performance(self) -> pd.DataFrame:
        """Extract training run performance data"""
        logger.info("Extracting training performance data")
        
        try:
            with session_scope() as session:
                training_runs = session.query(TrainingRun).all()
                
                data = []
                for run in training_runs:
                    row = {
                        'training_run_id': run.id,
                        'character_id': run.character_id,
                        'character_name': run.character.name if run.character else None,
                        'owner_id': run.owner_id,
                        'base_model': run.base_model,
                        'training_method': run.training_method,
                        'status': run.status,
                        'total_steps': run.total_steps,
                        'dataset_size': run.dataset_size,
                        'final_loss': run.final_loss,
                        'created_at': run.created_at,
                        'started_at': run.started_at,
                        'completed_at': run.completed_at,
                        'output_directory': run.output_directory,
                        'sft_adapter_path': run.sft_adapter_path,
                        'rlhf_adapter_path': run.rlhf_adapter_path
                    }
                    
                    # Calculate training duration
                    if run.started_at and run.completed_at:
                        duration = run.completed_at - run.started_at
                        row['training_duration_minutes'] = duration.total_seconds() / 60
                    
                    # Extract metrics from JSON if available
                    if run.metrics_json:
                        metrics = run.metrics_json
                        row.update({
                            'current_loss': metrics.get('current_loss'),
                            'current_step': metrics.get('current_step'),
                            'learning_rate': metrics.get('learning_rate'),
                            'elapsed_time': metrics.get('elapsed_time')
                        })
                    
                    # Extract config from JSON if available
                    if run.config_json:
                        config = run.config_json
                        row.update({
                            'epochs': config.get('epochs'),
                            'learning_rate_config': config.get('learning_rate'),
                            'batch_size': config.get('batch_size'),
                            'lora_r': config.get('lora_r'),
                            'lora_alpha': config.get('lora_alpha'),
                            'lora_dropout': config.get('lora_dropout')
                        })
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                
                # Cache results
                cache_path = self.cache_dir / f"training_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached {len(df)} training runs to {cache_path}")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to extract training performance: {e}")
            return pd.DataFrame()
    
    def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get individual messages from a conversation"""
        try:
            with session_scope() as session:
                conv = session.query(ConversationLog).get(conversation_id)
                if conv and conv.conversation_data:
                    return conv.conversation_data.get('messages', [])
                return []
        except Exception as e:
            logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
            return []
    
    def extract_synthetic_vs_real_data(self) -> Dict[str, pd.DataFrame]:
        """Compare synthetic training data vs real user conversations"""
        logger.info("Analyzing synthetic vs real conversation data")
        
        # Get real conversation data
        real_conversations = self.extract_conversation_logs(include_metadata=True)
        
        # TODO: Extract synthetic training data from character datasets
        # This would require parsing the jsonl files in character directories
        
        return {
            'real_conversations': real_conversations,
            # 'synthetic_data': synthetic_df  # To be implemented
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of available data"""
        logger.info("Generating data summary report")
        
        summary = {}
        
        try:
            with session_scope() as session:
                # Basic counts
                summary['total_conversations'] = session.query(ConversationLog).count()
                summary['total_characters'] = session.query(Character).count()
                summary['total_training_runs'] = session.query(TrainingRun).count()
                summary['total_users'] = session.query(User).count()
                
                # Quality distribution
                quality_stats = session.query(ConversationLog.quality_score).all()
                if quality_stats:
                    quality_scores = [q[0] for q in quality_stats if q[0] is not None]
                    if quality_scores:
                        summary['quality_score_mean'] = sum(quality_scores) / len(quality_scores)
                        summary['quality_score_min'] = min(quality_scores)
                        summary['quality_score_max'] = max(quality_scores)
                
                # Training success rate
                completed_runs = session.query(TrainingRun).filter_by(status='completed').count()
                total_runs = session.query(TrainingRun).count()
                summary['training_success_rate'] = completed_runs / total_runs if total_runs > 0 else 0
                
                # Data freshness
                latest_conv = session.query(ConversationLog).order_by(
                    ConversationLog.created_at.desc()
                ).first()
                if latest_conv:
                    summary['latest_conversation_date'] = latest_conv.created_at
                
                # Processing status
                processed_convs = session.query(ConversationLog).filter_by(
                    processed_for_training=True
                ).count()
                summary['conversations_processed_for_training'] = processed_convs
                
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
        
        return summary


# Convenience functions for quick analysis
def quick_conversation_sample(n: int = 100) -> pd.DataFrame:
    """Quick function to get a sample of conversations for analysis"""
    extractor = DataExtractor()
    return extractor.extract_conversation_logs(limit=n, min_quality=2.0)


def quick_character_overview() -> pd.DataFrame:
    """Quick function to get character metrics overview"""
    extractor = DataExtractor()
    return extractor.extract_character_metrics()


def quick_data_summary() -> Dict[str, Any]:
    """Quick function to get data summary"""
    extractor = DataExtractor()
    return extractor.generate_summary_report()


if __name__ == "__main__":
    # Demo the data extractor
    extractor = DataExtractor()
    
    print("🔬 Data Extraction Demo")
    print("=" * 40)
    
    # Generate summary
    summary = extractor.generate_summary_report()
    print(f"📊 Total conversations: {summary.get('total_conversations', 0)}")
    print(f"🎭 Total characters: {summary.get('total_characters', 0)}")
    print(f"🚀 Total training runs: {summary.get('total_training_runs', 0)}")
    print(f"📈 Training success rate: {summary.get('training_success_rate', 0):.1%}")
    print(f"⭐ Average quality score: {summary.get('quality_score_mean', 0):.2f}")
    
    # Extract sample data
    conversations = extractor.extract_conversation_logs(limit=10)
    if not conversations.empty:
        print(f"\n💬 Sample conversations: {len(conversations)} rows")
        print(conversations[['character_name', 'quality_score', 'conversation_length']].head())
    
    characters = extractor.extract_character_metrics()
    if not characters.empty:
        print(f"\n🎭 Character metrics: {len(characters)} characters")
        print(characters[['character_name', 'conversation_count', 'training_runs_count']].head()) 
"""
World Consistency Evaluation

Uses LLM-as-judge with structured outputs to check adherence to world lore and rules.
Essential for maintaining immersive fictional worlds and character consistency.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import json
from pydantic import BaseModel, Field

from app.utils.openai_client import get_client

logger = logging.getLogger(__name__)


class LoreViolationAnalysis(BaseModel):
    """Structured output for lore violation detection"""
    violations_found: List[str] = Field(description="Specific lore violations identified")
    violation_severity: str = Field(description="Overall severity: none, low, medium, high")
    technology_consistency: float = Field(ge=0.0, le=1.0, description="Technology level consistency score")
    setting_consistency: float = Field(ge=0.0, le=1.0, description="Setting consistency score")
    magic_consistency: float = Field(ge=0.0, le=1.0, description="Magic system consistency score")
    forbidden_items_found: List[str] = Field(description="Forbidden items mentioned inappropriately")
    contextual_reasoning: str = Field(description="Reasoning for the consistency assessment")


class LocationValidationAnalysis(BaseModel):
    """Structured output for location reference validation"""
    valid_references: List[str] = Field(description="Valid location references found")
    invalid_references: List[str] = Field(description="Invalid or inconsistent location references")
    location_consistency_score: float = Field(ge=0.0, le=1.0, description="Overall location consistency")
    geographic_coherence: float = Field(ge=0.0, le=1.0, description="Geographic logic and coherence")
    issues_found: List[str] = Field(description="Specific location-related issues")
    recommendations: List[str] = Field(description="Suggestions for improvement")


class FactConsistencyAnalysis(BaseModel):
    """Structured output for fact consistency evaluation"""
    contradictions_found: List[str] = Field(description="Factual contradictions identified")
    consistency_score: float = Field(ge=0.0, le=1.0, description="Overall fact consistency score")
    world_rule_adherence: float = Field(ge=0.0, le=1.0, description="Adherence to established world rules")
    character_knowledge_consistency: float = Field(ge=0.0, le=1.0, description="Character knowledge consistency")
    temporal_consistency: float = Field(ge=0.0, le=1.0, description="Timeline and temporal consistency")
    detailed_analysis: str = Field(description="Detailed analysis of consistency issues")


class WorldConsistencyEvaluator:
    """Evaluates world consistency using LLM-as-judge"""
    
    def __init__(self, world_lore: Optional[Dict[str, Any]] = None):
        """
        Initialize world consistency evaluator.
        
        Args:
            world_lore: Dictionary containing world rules and lore
        """
        self.client = get_client()
        self.world_lore = world_lore or {}
        self._initialize_lore_constraints()
    
    def _initialize_lore_constraints(self):
        """Extract and process lore constraints from world data"""
        self.forbidden_items = set(self.world_lore.get('forbidden_items', []))
        self.technology_level = self.world_lore.get('technology_level', 'modern')
        self.setting = self.world_lore.get('setting', 'contemporary')
        
        # Smart default for magic based on setting
        if 'fantasy' in self.setting.lower() or 'magical' in self.setting.lower():
            self.magic_exists = self.world_lore.get('magic_exists', True)
        else:
            self.magic_exists = self.world_lore.get('magic_exists', False)
            
        self.key_locations = set(self.world_lore.get('key_locations', []))
        
        # Add technology-specific constraints
        if self.technology_level == 'pre-industrial':
            self.forbidden_items.update([
                'computer', 'phone', 'car', 'television', 'radio',
                'internet', 'electricity', 'gun', 'airplane', 'train',
                'google', 'email', 'website', 'browser', 'smartphone'
            ])
    
    def check_lore_adherence(self, text: str) -> Dict[str, Any]:
        """
        Check if text adheres to world lore using LLM-as-judge.
        
        Args:
            text: Text to check for lore violations
            
        Returns:
            Dictionary with lore adherence metrics
        """
        results = {
            'lore_consistency_score': 1.0,
            'violations': [],
            'violation_severity': 'none'
        }
        
        try:
            # Use LLM-as-judge for contextual lore checking
            lore_analysis = asyncio.run(self._judge_lore_adherence(text))
            
            results.update({
                'lore_consistency_score': min(
                    lore_analysis.technology_consistency,
                    lore_analysis.setting_consistency,
                    lore_analysis.magic_consistency
                ),
                'violations': lore_analysis.violations_found,
                'violation_severity': lore_analysis.violation_severity,
                'technology_consistency': lore_analysis.technology_consistency,
                'setting_consistency': lore_analysis.setting_consistency,
                'magic_consistency': lore_analysis.magic_consistency,
                'forbidden_items_found': lore_analysis.forbidden_items_found,
                'contextual_reasoning': lore_analysis.contextual_reasoning,
                'llm_analysis': lore_analysis.model_dump()
            })
            
        except Exception as e:
            logger.error(f"Error in lore adherence check: {e}")
            # Fallback to keyword-based checking
            fallback_result = self._fallback_lore_check(text)
            results.update(fallback_result)
            results['error'] = str(e)
        
        return results
    
    async def _judge_lore_adherence(self, text: str) -> LoreViolationAnalysis:
        """Use LLM-as-judge to assess lore adherence"""
        try:
            system_prompt = self._build_lore_system_prompt()
            user_prompt = self._build_lore_user_prompt(text)
            
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "lore_violation_analysis",
                        "schema": LoreViolationAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return LoreViolationAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in lore judgment: {e}")
            # Return fallback assessment
            fallback_result = self._fallback_lore_analysis(text)
            return LoreViolationAnalysis(**fallback_result)
    
    def _build_lore_system_prompt(self) -> str:
        """Build system prompt for lore adherence evaluation"""
        return f"""You are an expert world-building consistency checker. Your task is to evaluate whether text content adheres to established world lore and rules.

**World Setting**: {self.setting}
**Technology Level**: {self.technology_level}
**Magic Exists**: {self.magic_exists}
**Key Locations**: {', '.join(self.key_locations) if self.key_locations else 'None specified'}
**Forbidden Items**: {', '.join(self.forbidden_items) if self.forbidden_items else 'None specified'}

Evaluate the text for:
1. **Technology Consistency**: Does it mention technology appropriate for the world's level?
2. **Setting Consistency**: Does it fit the established world setting?
3. **Magic Consistency**: Is magic usage consistent with world rules?
4. **Forbidden Items**: Are any prohibited items mentioned inappropriately?

**SCORING GUIDELINES:**
- 0.9-1.0: Perfectly consistent with world lore
- 0.7-0.8: Mostly consistent with minor issues
- 0.5-0.6: Moderately consistent but noticeable problems
- 0.3-0.4: Poor consistency with significant violations
- 0.0-0.2: Major lore violations, breaks world immersion

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive).

Consider context and intent. Some mentions might be metaphorical or acceptable within the world's rules."""

    def _build_lore_user_prompt(self, text: str) -> str:
        """Build user prompt for lore adherence evaluation"""
        return f"""Analyze this text for world consistency violations:

**Text to Analyze:**
{text}

Check for:
1. Technology mentions that don't fit the {self.technology_level} technology level
2. Setting elements that contradict the {self.setting} world
3. Magic usage that's {'consistent with' if self.magic_exists else 'inappropriate for'} the world's magic rules
4. Any forbidden items: {', '.join(self.forbidden_items) if self.forbidden_items else 'None specified'}

Provide:
- Specific violations found (if any)
- Consistency scores for technology, setting, and magic
- Contextual reasoning for your assessment
- Severity assessment (none, low, medium, high)

Consider context - some references might be metaphorical or historically appropriate within the world."""

    def validate_location_references(self, text: str, known_locations: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate location references using LLM-as-judge.
        
        Args:
            text: Text containing location references
            known_locations: Optional list of valid locations
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid_locations': [],
            'invalid_locations': [],
            'location_consistency_score': 1.0
        }
        
        try:
            # Combine world locations with provided ones
            all_locations = list(self.key_locations)
            if known_locations:
                all_locations.extend(known_locations)
            
            # Use LLM-as-judge for contextual location validation
            location_analysis = asyncio.run(self._judge_location_references(text, all_locations))
            
            results.update({
                'valid_locations': location_analysis.valid_references,
                'invalid_locations': location_analysis.invalid_references,
                'location_consistency_score': location_analysis.location_consistency_score,
                'geographic_coherence': location_analysis.geographic_coherence,
                'issues_found': location_analysis.issues_found,
                'recommendations': location_analysis.recommendations,
                'llm_analysis': location_analysis.model_dump()
            })
            
        except Exception as e:
            logger.error(f"Error in location validation: {e}")
            # Fallback to pattern matching
            fallback_result = self._fallback_location_check(text, all_locations)
            results.update(fallback_result)
            results['error'] = str(e)
        
        return results
    
    async def _judge_location_references(self, text: str, known_locations: List[str]) -> LocationValidationAnalysis:
        """Use LLM-as-judge to validate location references"""
        try:
            system_prompt = """You are an expert in geographic consistency and world-building. Your task is to validate location references in text for consistency and coherence.

Evaluate:
1. **Valid References**: Locations that are properly established or reasonable
2. **Invalid References**: Locations that are inconsistent or problematic
3. **Geographic Coherence**: Whether location references make geographic sense
4. **Consistency**: Whether locations are used consistently throughout

**SCORING GUIDELINES:**
- 0.9-1.0: All location references are valid and coherent
- 0.7-0.8: Mostly valid with minor inconsistencies
- 0.5-0.6: Some problematic references but generally coherent
- 0.3-0.4: Many invalid or inconsistent references
- 0.0-0.2: Major geographic inconsistencies

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive)."""

            user_prompt = f"""Validate location references in this text:

**Text:**
{text}

**Known Valid Locations:**
{', '.join(known_locations) if known_locations else 'None specified'}

Analyze:
1. Which location references are valid/established?
2. Which references seem invalid or inconsistent?
3. Do the locations make geographic sense together?
4. Are there any consistency issues with how locations are used?

Provide specific feedback on each location reference and overall geographic coherence."""

            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "location_validation_analysis",
                        "schema": LocationValidationAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return LocationValidationAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in location judgment: {e}")
            # Return fallback assessment
            return LocationValidationAnalysis(
                valid_references=[],
                invalid_references=[],
                location_consistency_score=0.5,
                geographic_coherence=0.5,
                issues_found=["Could not assess with LLM judge"],
                recommendations=["Manual review needed"]
            )
    
    def evaluate_fact_consistency(self, text: str, established_facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate fact consistency using LLM-as-judge.
        
        Args:
            text: Text to check for fact consistency
            established_facts: Optional dictionary of established facts
            
        Returns:
            Dictionary with fact consistency metrics
        """
        results = {
            'fact_consistency_score': 1.0,
            'contradictions': [],
            'world_rule_violations': []
        }
        
        try:
            # Use LLM-as-judge for fact consistency checking
            fact_analysis = asyncio.run(self._judge_fact_consistency(text, established_facts))
            
            results.update({
                'fact_consistency_score': fact_analysis.consistency_score,
                'contradictions': fact_analysis.contradictions_found,
                'world_rule_adherence': fact_analysis.world_rule_adherence,
                'character_knowledge_consistency': fact_analysis.character_knowledge_consistency,
                'temporal_consistency': fact_analysis.temporal_consistency,
                'detailed_analysis': fact_analysis.detailed_analysis,
                'llm_analysis': fact_analysis.model_dump()
            })
            
        except Exception as e:
            logger.error(f"Error in fact consistency evaluation: {e}")
            # Fallback to simple checking
            fallback_result = self._fallback_fact_check(text)
            results.update(fallback_result)
            results['error'] = str(e)
        
        return results
    
    async def _judge_fact_consistency(self, text: str, established_facts: Optional[Dict[str, Any]] = None) -> FactConsistencyAnalysis:
        """Use LLM-as-judge to assess fact consistency"""
        try:
            system_prompt = """You are an expert in narrative consistency and fact-checking. Your task is to evaluate text for internal fact consistency and adherence to established world rules.

Evaluate:
1. **Internal Contradictions**: Facts that contradict each other within the text
2. **World Rule Adherence**: Consistency with established world rules
3. **Character Knowledge**: Whether characters know things they should/shouldn't
4. **Temporal Consistency**: Timeline and sequence consistency

**SCORING GUIDELINES:**
- 0.9-1.0: No contradictions, perfectly consistent
- 0.7-0.8: Minor inconsistencies that don't break immersion
- 0.5-0.6: Some noticeable contradictions
- 0.3-0.4: Multiple contradictions that affect believability
- 0.0-0.2: Major contradictions that break world logic

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive)."""

            facts_context = ""
            if established_facts:
                facts_context = f"\n**Established Facts:**\n{json.dumps(established_facts, indent=2)}\n"

            user_prompt = f"""Analyze this text for fact consistency:

**Text:**
{text}{facts_context}

Check for:
1. Internal contradictions within the text
2. Violations of established world rules
3. Character knowledge inconsistencies
4. Timeline or temporal issues

Provide:
- Specific contradictions found (if any)
- Overall consistency score
- Detailed analysis of issues
- Assessment of world rule adherence

Be thorough in identifying logical inconsistencies."""

            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fact_consistency_analysis",
                        "schema": FactConsistencyAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return FactConsistencyAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in fact consistency judgment: {e}")
            # Return fallback assessment
            return FactConsistencyAnalysis(
                contradictions_found=[],
                consistency_score=0.7,
                world_rule_adherence=0.7,
                character_knowledge_consistency=0.7,
                temporal_consistency=0.7,
                detailed_analysis="Fallback assessment due to LLM error"
            )
    
    def _fallback_lore_check(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based lore checking when LLM fails"""
        text_lower = text.lower()
        violations = []
        violation_count = 0
        
        # Check for forbidden items using keyword matching
        for item in self.forbidden_items:
            if item.lower() in text_lower:
                violations.append(f"Forbidden item mentioned: {item}")
                violation_count += 1
        
        # Check for technology violations
        if self.technology_level == 'pre-industrial':
            modern_tech = ['smartphone', 'computer', 'internet', 'email', 'google']
            for tech in modern_tech:
                if tech in text_lower:
                    violations.append(f"Anachronistic technology: {tech}")
                    violation_count += 1
        
        # Calculate penalty
        penalty = min(violation_count * 0.25, 0.9)
        score = max(0.1, 1.0 - penalty)
        
        # Determine severity
        if violation_count == 0:
            severity = 'none'
        elif violation_count == 1:
            severity = 'low'
        elif violation_count == 2:
            severity = 'medium'
        else:
            severity = 'high'
        
        return {
            'lore_consistency_score': score,
            'violations': violations,
            'violation_severity': severity,
            'technology_consistency': score,
            'setting_consistency': 0.8,  # Default
            'magic_consistency': 0.8,  # Default
            'fallback_used': True
        }
    
    def _fallback_lore_analysis(self, text: str) -> Dict[str, Any]:
        """Create fallback LoreViolationAnalysis data"""
        fallback = self._fallback_lore_check(text)
        return {
            'violations_found': fallback['violations'],
            'violation_severity': fallback['violation_severity'],
            'technology_consistency': fallback['technology_consistency'],
            'setting_consistency': fallback['setting_consistency'],
            'magic_consistency': fallback['magic_consistency'],
            'forbidden_items_found': [v.split(': ')[1] for v in fallback['violations'] if 'Forbidden item' in v],
            'contextual_reasoning': 'Fallback keyword-based analysis'
        }
    
    def _fallback_location_check(self, text: str, known_locations: List[str]) -> Dict[str, Any]:
        """Fallback pattern-based location checking"""
        # Simple check for known locations in text
        found_locations = []
        for location in known_locations:
            if location.lower() in text.lower():
                found_locations.append(location)
        
        return {
            'valid_locations': found_locations,
            'invalid_locations': [],
            'location_consistency_score': 0.8 if found_locations else 0.9,
            'geographic_coherence': 0.8,
            'issues_found': [] if found_locations else ['No clear location references found'],
            'fallback_used': True
        }
    
    def _fallback_fact_check(self, text: str) -> Dict[str, Any]:
        """Fallback simple fact consistency check"""
        # Very basic contradiction detection
        contradictions = []
        if 'never' in text.lower() and 'always' in text.lower():
            contradictions.append("Potential absolute contradiction (never/always)")
        
        return {
            'fact_consistency_score': 0.8 if not contradictions else 0.6,
            'contradictions': contradictions,
            'world_rule_adherence': 0.8,
            'character_knowledge_consistency': 0.8,
            'temporal_consistency': 0.8,
            'fallback_used': True
        } 
import re
import logging
import traceback
from typing import Dict, Any, List

from pydantic import BaseModel

from ..character.kink_extractor import extract_kinks

logger = logging.getLogger(__name__)


def analyze_character_intimacy_style(character: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze how this character would approach intimate situations"""
    personality = character.get('personality', '').lower()
    description = character.get('description', '').lower()
    mes_example = character.get('mes_example', '').lower()
    
    intimacy_traits = {
        'style': 'passionate',  # passionate, gentle, playful, intense, romantic
        'pace': 'moderate',  # slow, moderate, eager
        'expression': 'verbal',  # verbal, physical, emotional
        'confidence': 'moderate',  # shy, moderate, confident, dominant
        'approach': 'emotional',  # emotional, physical, intellectual, playful
    }
    
    # Analyze personality for intimacy cues
    full_text = f"{personality} {description} {mes_example}"
    
    # Confidence analysis
    if any(word in full_text for word in ['shy', 'nervous', 'innocent', 'timid', 'bashful', 'hesitant']):
        intimacy_traits['confidence'] = 'shy'
        intimacy_traits['pace'] = 'slow'
    elif any(word in full_text for word in ['confident', 'bold', 'dominant', 'assertive', 'commanding']):
        intimacy_traits['confidence'] = 'confident'
        intimacy_traits['style'] = 'intense'
    elif any(word in full_text for word in ['playful', 'teasing', 'mischievous', 'flirty']):
        intimacy_traits['confidence'] = 'playful'
        intimacy_traits['style'] = 'playful'
    
    # Style analysis
    if any(word in full_text for word in ['gentle', 'caring', 'tender', 'soft', 'sweet']):
        intimacy_traits['style'] = 'gentle'
    elif any(word in full_text for word in ['passionate', 'fiery', 'intense', 'wild']):
        intimacy_traits['style'] = 'passionate'
    elif any(word in full_text for word in ['romantic', 'loving', 'devoted', 'affectionate']):
        intimacy_traits['style'] = 'romantic'
    
    # Expression analysis
    if any(word in full_text for word in ['quiet', 'reserved', 'stoic', 'silent']):
        intimacy_traits['expression'] = 'physical'
    elif any(word in full_text for word in ['talkative', 'expressive', 'vocal', 'articulate']):
        intimacy_traits['expression'] = 'verbal'
    elif any(word in full_text for word in ['emotional', 'sensitive', 'empathetic', 'feeling']):
        intimacy_traits['expression'] = 'emotional'
    
    # Approach analysis
    if any(word in full_text for word in ['intellectual', 'analytical', 'thoughtful', 'philosophical']):
        intimacy_traits['approach'] = 'intellectual'
    elif any(word in full_text for word in ['physical', 'athletic', 'strong', 'active']):
        intimacy_traits['approach'] = 'physical'
    elif any(word in full_text for word in ['playful', 'fun', 'humorous', 'witty']):
        intimacy_traits['approach'] = 'playful'
    
    return intimacy_traits


def extract_intimate_speech_patterns(mes_example: str, character_name: str) -> Dict[str, List[str]]:
    """Extract how character speaks in intimate moments"""
    patterns = {
        'endearments': [],  # "darling", "love", "sweetheart"
        'physical_descriptions': [],  # How they describe touch/sensation
        'emotional_expressions': [],  # How they express desire/love
        'consent_phrases': [],  # How they check in with partner
        'intimacy_style': [],  # How they approach intimate moments
    }
    
    # Common endearments to look for
    endearment_words = ['darling', 'love', 'sweetheart', 'baby', 'honey', 'dear', 
                       'beloved', 'treasure', 'angel', 'beautiful', 'gorgeous']
    
    # Analyze example messages
    lines = mes_example.split('\n')
    for line in lines:
        line_lower = line.lower()
        
        # Extract endearments
        for endearment in endearment_words:
            if endearment in line_lower:
                patterns['endearments'].append(endearment)
        
        # Extract physical descriptions
        if any(word in line_lower for word in ['touch', 'feel', 'soft', 'warm', 'close', 'hold', 'kiss']):
            # Extract the context
            if '*' in line:  # Action text
                action_match = re.findall(r'\*([^*]+)\*', line)
                for action in action_match:
                    if any(word in action.lower() for word in ['touch', 'kiss', 'hold', 'caress']):
                        patterns['physical_descriptions'].append(action)
        
        # Extract emotional expressions
        if any(word in line_lower for word in ['want', 'need', 'desire', 'love', 'feel', 'yearn']):
            patterns['emotional_expressions'].append('expressive')
        
        # Extract consent phrases
        if any(phrase in line_lower for phrase in ['is this okay', 'do you want', 'may i', 'can i', 'tell me if']):
            patterns['consent_phrases'].append('asks_consent')
        
        # Intimacy style markers
        if '*blush*' in line_lower or '*shy*' in line_lower:
            patterns['intimacy_style'].append('shy')
        if '*confident*' in line_lower or '*bold*' in line_lower:
            patterns['intimacy_style'].append('confident')
        if '*playful*' in line_lower or '*tease*' in line_lower:
            patterns['intimacy_style'].append('playful')
    
    # Remove duplicates
    for key in patterns:
        patterns[key] = list(set(patterns[key]))
    
    return patterns

class CharacterKnowledge(BaseModel):
    name: str
    traits: List[str] = []
    skills: List[str] = []
    relationships: List[str] = []
    backstory_elements: List[str] = []
    goals: List[str] = []
    fears: List[str] = []
    likes: List[str] = []
    dislikes: List[str] = []
    speech_patterns: List[str] = []
    emotional_triggers: List[str] = []
    mannerisms: List[str] = []
    locations: List[str] = []
    occupation: str = None
    species: str = None
    appearance: List[str] = []
    equipment: List[str] = []
    known_spells: List[str] = []
    current_situation: str = None
    world_info: List[str] = []
    kinks: Dict[str, List[str]] = {'likes': [], 'limits': []}

async def extract_character_knowledge(client,character: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured knowledge from character card for better prompt generation"""
    char_name = character.get('name', 'Assistant')
    description = character.get('description', '')
    personality = character.get('personality', '')
    scenario = character.get('scenario', '')
    mes_example = character.get('mes_example', '')
    first_mes = character.get('first_mes', '')
    alternate_greetings = character.get('alternate_greetings', [])
    
    # knowledge = {
    #     'name': char_name,
    #     'traits': [],
    #     'skills': [],
    #     'relationships': [],
    #     'backstory_elements': [],
    #     'goals': [],
    #     'fears': [],
    #     'likes': [],
    #     'dislikes': [],
    #     'speech_patterns': [],
    #     'emotional_triggers': [],
    #     'mannerisms': [],
    #     'locations': [],
    #     'occupation': None,
    #     'species': None,
    #     'appearance': [],
    #     'equipment': [],
    #     'known_spells': [],
    #     'current_situation': None,
    #     'world_info': [],
    #     'kinks': {'likes': [], 'limits': []}
    # }
    
    # # Parse structured format (Type:, Species:, etc.)
    # structured_info = _parse_structured_format(description)
    # knowledge.update(structured_info)
    
    # Combine all text for additional analysis
    # full_text = f"{description} {personality} {scenario}".lower()
    
    # Extract kinks from character description and personality
    prompt = f"""
    Extract the following information from the character description and personality:
    Description: {description}
    Personality: {personality}
    Scenario: {scenario}
    Messages Example: {mes_example}
    First Message: {first_mes}
    Alternate Greetings: {alternate_greetings}

    Respond with ONLY a JSON object in this exact format:
    {CharacterKnowledge.model_json_schema()}
    """

    try:
        knowledge = await client.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            response_format={"type": "json_schema", "json_schema": CharacterKnowledge}
        )

        kink_text = f"{description} {personality}"
        extracted_kinks = extract_kinks(kink_text)
        knowledge['kinks'] = extracted_kinks
    
    # # Extract from structured fields if not already found
    # if not knowledge['occupation']:
    #     occ_match = re.search(r'occupation:\s*([^,\n]+)', full_text, re.IGNORECASE)
    #     if occ_match:
    #         knowledge['occupation'] = occ_match.group(1).strip()
    
    # # Extract personality traits more comprehensively
    # if 'personality:' in full_text:
    #     pers_match = re.search(r'personality:\s*([^,\n]+(?:,\s*[^,\n]+)*)', full_text, re.IGNORECASE)
    #     if pers_match:
    #         traits = [t.strip() for t in pers_match.group(1).split(',')]
    #         knowledge['traits'].extend(traits)
    
    # # Extract skills and abilities
    # skill_patterns = [
    #     r'(?:skills?|abilities):\s*([^,\n]+(?:,\s*[^,\n]+)*)',
    #     r'(?:skilled\s+in|expert\s+at|master\s+of|proficient\s+in)\s+([^.,]+)',
    #     r'(?:can|able\s+to|capable\s+of)\s+([^.,]+)',
    # ]
    
    # for pattern in skill_patterns:
    #     matches = re.findall(pattern, full_text, re.IGNORECASE)
    #     for match in matches:
    #         if ',' in match:
    #             skills = [s.strip() for s in match.split(',')]
    #             knowledge['skills'].extend(skills)
    #         else:
    #             knowledge['skills'].append(match.strip())
    
    # # Extract goals
    # goal_patterns = [
    #     r'goal:\s*([^,\n]+)',
    #     r'(?:wants\s+to|seeks\s+to|aims\s+to|desires\s+to)\s+([^.,]+)',
    #     r'(?:determined\s+to|passionate\s+about)\s+([^.,]+)',
    # ]
    
    # for pattern in goal_patterns:
    #     matches = re.findall(pattern, full_text, re.IGNORECASE)
    #     knowledge['goals'].extend([m.strip() for m in matches])
    
    # # Extract backstory elements
    # backstory_keywords = ['born in', 'grew up', 'childhood', 'expelled', 'survived', 'refugee', 
    #                       'moved to', 'rejected', 'army', 'academy', 'war', 'crash']
    # backstory_sentences = []
    # for sentence in full_text.split('.'):
    #     if any(keyword in sentence for keyword in backstory_keywords):
    #         backstory_sentences.append(sentence.strip())
    # knowledge['backstory_elements'] = backstory_sentences[:5]  # Top 5 most relevant
    
    # # Extract from first message for current situation
    # if first_mes:
    #     knowledge['current_situation'] = _extract_situation_from_greeting(first_mes)
    #     # Extract locations mentioned
    #     location_patterns = [r'\b(?:at|in|on)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 
    #                         r'(?:warehouse|tavern|guild|shop|city|street|room|office|desk|door)']
    #     for pattern in location_patterns:
    #         locs = re.findall(pattern, first_mes)
    #         knowledge['locations'].extend([l for l in locs if isinstance(l, str)])
    
    # # Enhanced speech pattern extraction from mes_example
    # if mes_example:
    #     knowledge['speech_patterns'] = _extract_speech_patterns(mes_example, char_name)
    #     knowledge['mannerisms'] = _extract_mannerisms(mes_example, char_name)
        
    #     # Extract intimate speech patterns if relevant
    #     if any(word in full_text for word in ['romantic', 'lover', 'passionate', 'sensual', 'intimate']):
    #         intimate_patterns = extract_intimate_speech_patterns(mes_example, char_name)
    #         knowledge['intimate_speech'] = intimate_patterns
    
    # # Process alternate greetings for variety
    # if alternate_greetings:
    #     for greeting in alternate_greetings[:3]:  # Process up to 3
    #         if greeting:
    #             # Extract emotional states and scenarios
    #             if 'screwed' in greeting or 'desperate' in greeting:
    #                 knowledge['emotional_triggers'].append('financial_stress')
    #             if 'celebrate' in greeting or 'cheers' in greeting:
    #                 knowledge['emotional_triggers'].append('success_celebration')
    
    # # Extract character book / world info if present
    # char_book = character.get('character_book', {})
    # if char_book and 'entries' in char_book:
    #     for entry in char_book.get('entries', []):
    #         if entry.get('enabled', True):
    #             knowledge['world_info'].append({
    #                 'name': entry.get('name', 'Unknown'),
    #                 'content': entry.get('content', ''),
    #                 'keys': entry.get('keys', [])
    #             })
    
    # # Remove duplicates and empty entries
    # for key in knowledge:
    #     if isinstance(knowledge[key], list):
    #         knowledge[key] = list(set([item for item in knowledge[key] if item]))
    
        return knowledge
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error extracting character knowledge: {e}")
        return None


def _parse_structured_format(text: str) -> Dict[str, Any]:
    """Parse structured character format (Type:, Species:, etc.)"""
    result = {}
    
    # Common structured fields
    field_patterns = {
        'species': r'(?:species|race):\s*([^,\n]+)',
        'appearance': r'appearance:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
        'equipment': r'equipment:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
        'known_spells': r'known\s+spells?:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
        'occupation': r'occupation:\s*([^,\n]+)',
        'traits': r'traits?:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
    }
    
    for field, pattern in field_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if field in ['appearance', 'equipment', 'known_spells', 'traits']:
                # These are lists
                result[field] = [v.strip() for v in value.split(',')]
            else:
                result[field] = value
    
    return result


def _extract_situation_from_greeting(greeting: str) -> str:
    """Extract the current situation/context from greeting"""
    # Look for scene-setting elements
    situation_parts = []
    
    # Time indicators
    if 'dead day' in greeting or 'another day' in greeting:
        situation_parts.append("struggling with business")
    if 'knock at the door' in greeting:
        situation_parts.append("receiving unexpected visitor")
    if 'rent' in greeting and 'due' in greeting:
        situation_parts.append("facing financial pressure")
    
    # Emotional state
    if '*groan*' in greeting or 'RGGHH' in greeting:
        situation_parts.append("frustrated")
    if '!!' in greeting or 'almost falls' in greeting:
        situation_parts.append("excited by opportunity")
    
    return "; ".join(situation_parts) if situation_parts else "starting new interaction"


def _extract_speech_patterns(mes_example: str, char_name: str) -> List[str]:
    """Extract detailed speech patterns from example messages"""
    patterns = []
    
    # Split into character's lines
    char_lines = []
    lines = mes_example.split('\n')
    for i, line in enumerate(lines):
        if f'{char_name}:' in line or (i > 0 and '{{char}}:' in lines[i-1]):
            # Extract just the speech part
            speech = line.split(':', 1)[-1].strip() if ':' in line else line
            # Remove action text in asterisks
            speech_only = re.sub(r'\*[^*]+\*', '', speech).strip()
            if speech_only:
                char_lines.append(speech_only)
    
    # Analyze speech characteristics
    for line in char_lines:
        if 'haha' in line.lower():
            patterns.append('nervous_laughter')
        if '...' in line:
            patterns.append('trailing_off')
        if '!' in line and line.count('!') >= 2:
            patterns.append('multiple_exclamations')
        if '?' in line and line.count('?') >= 2:
            patterns.append('multiple_questions')
        if 'uh' in line.lower() or 'um' in line.lower():
            patterns.append('hesitation')
    
    return list(set(patterns))


def _extract_mannerisms(mes_example: str, char_name: str) -> List[str]:
    """Extract character mannerisms from example messages"""
    mannerisms = []
    
    # Look for action patterns in asterisks
    actions = re.findall(r'\*([^*]+)\*', mes_example)
    
    for action in actions:
        action_lower = action.lower()
        if any(word in action_lower for word in ['blush', 'shy', 'nervous']):
            mannerisms.append('shows_embarrassment')
        if any(word in action_lower for word in ['grin', 'smile', 'laugh']):
            mannerisms.append('expressive_face')
        if any(word in action_lower for word in ['gesture', 'wave', 'point']):
            mannerisms.append('uses_gestures')
        if any(word in action_lower for word in ['lean', 'close', 'approach']):
            mannerisms.append('physical_proximity')
    
    return list(set(mannerisms))
"""Kink and boundary extraction utilities"""

import re
from typing import Dict, List


def extract_kinks(text: str) -> Dict[str, List[str]]:
    """
    Extract kinks from character description text using naive regex patterns.
    
    Args:
        text: The concatenated character description/personality text
        
    Returns:
        dict with keys 'likes' and 'limits', each containing a list of detected kinks
    """
    if not text:
        return {"likes": [], "limits": []}
    
    text_lower = text.lower()
    
    # Define kink patterns to detect
    kink_patterns = {
        # Dominance/submission
        "dominance": ["dominat", "control", "command", "superior", "authorit"],
        "submission": ["submiss", "obey", "serve", "yield", "surrender"],
        
        # Physical activities
        "bondage": ["bondage", "rope", "tied", "restrain", "bind"],
        "spanking": ["spank", "paddle", "whip", "flog"],
        "rough_play": ["rough", "hard", "force", "aggressiv"],
        
        # Psychological
        "humiliation": ["humiliat", "degrad", "embarrass", "shame"],
        "praise": ["praise", "good girl", "good boy", "worship"],
        "teasing": ["teas", "deny", "edge", "frustrat"],
        
        # Roleplaying
        "roleplay": ["roleplay", "role play", "pretend", "scenario"],
        "pet_play": ["pet", "collar", "leash", "kitten", "puppy"],
        "daddy_dom": ["daddy", "little", "caregiver"],
        
        # Sensory
        "pain": ["pain", "hurt", "sting", "bite"],
        "temperature": ["hot", "cold", "ice", "wax"],
        "sensory": ["blind", "deaf", "sens"],
        
        # Taboo/fetish
        "feet": ["feet", "foot", "toe"],
        "latex": ["latex", "rubber", "leather"],
        "voyeurism": ["watch", "voyeur", "exhib"],
    }
    
    # Track detected kinks
    detected_likes = []
    detected_limits = []
    
    # Search for kink indicators
    for kink_name, patterns in kink_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Check context to determine if it's a like or limit
                context_match = re.search(rf'.{{0,30}}{pattern}.{{0,30}}', text_lower)
                if context_match:
                    context = context_match.group()
                    
                    # Look for negative indicators
                    negative_indicators = [
                        "no", "not", "never", "don't", "doesn't", "won't", "refuse",
                        "against", "dislike", "hate", "avoid", "forbidden", "limit"
                    ]
                    
                    # Look for positive indicators  
                    positive_indicators = [
                        "love", "like", "enjoy", "want", "desire", "crave", "into",
                        "pleasure", "turn on", "aroused", "excited"
                    ]
                    
                    # Determine if it's a limit or like based on context
                    is_negative = any(neg in context for neg in negative_indicators)
                    is_positive = any(pos in context for pos in positive_indicators)
                    
                    if is_negative:
                        if kink_name not in detected_limits:
                            detected_limits.append(kink_name)
                    elif is_positive or not is_negative:  # Default to like if no clear negative context
                        if kink_name not in detected_likes:
                            detected_likes.append(kink_name)
                
                break  # Found this kink, move to next
    
    # Look for explicit like/dislike statements
    like_patterns = [
        r"(?:enjoys?|likes?|loves?|into|turned on by|aroused by)\s+([^.,]+)",
        r"([^.,]+)\s+(?:turns? (?:me|her|him) on|excites? (?:me|her|him))",
        r"kinks?:\s*([^.,\n]+)",
        r"fetishes?:\s*([^.,\n]+)",
    ]
    
    limit_patterns = [
        r"(?:doesn't like|dislikes?|hates?|not into|turned off by|limits?)\s+([^.,]+)",
        r"(?:no|never|won't do|refuses?)\s+([^.,]+)",
        r"hard limits?:\s*([^.,\n]+)",
        r"soft limits?:\s*([^.,\n]+)",
    ]
    
    # Extract explicit likes
    for pattern in like_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if cleaned and len(cleaned) < 50:  # Reasonable length filter
                detected_likes.append(cleaned)
    
    # Extract explicit limits
    for pattern in limit_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if cleaned and len(cleaned) < 50:  # Reasonable length filter
                detected_limits.append(cleaned)
    
    # Remove duplicates while preserving order
    detected_likes = list(dict.fromkeys(detected_likes))
    detected_limits = list(dict.fromkeys(detected_limits))
    
    return {
        "likes": detected_likes,
        "limits": detected_limits
    } 
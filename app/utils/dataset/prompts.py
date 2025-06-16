"""
Prompt templates and collections for dataset generation.

This module centralizes all prompt templates and prompt collections used for generating
diverse, character-specific conversations. Prompts are organized by category and use
template strings that can be personalized with character information.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PromptCollection:
    """Container for categorized prompts with weights"""
    prompts: List[str]
    weight: float = 1.0
    description: str = ""


class PromptTemplates:
    """Central repository for all prompt templates and collections"""
    
    # Base conversation templates
    TEMPLATES = [
        "short_qa",
        "narration", 
        "monologue",
        "dialogue_turn",
        "internal_thought",
        "character_response",
    ]

    # Natural conversation starters
    CASUAL_PROMPTS = [
        "Hey! How's it going?",
        "What's up?",
        "Good morning!",
        "How are you feeling today?",
        "What have you been up to?",
        "Nice to meet you!",
        "How's your day been?",
        "What's on your mind?",
        "Who are you?",
        "What's your name?",
    ]

    PERSONAL_PROMPTS = [
        "Do you ever feel lonely?",
        "What keeps you awake at night?",
        "What's your biggest fear?",
        "Have you ever been in love?",
        "What are you most proud of?",
        "Any secret dreams?",
        "Tell me about yourself.",
        "What do you like to do for fun?",
        "What's something you're passionate about?",
        "Do you have any interesting stories?",
        "What's been on your mind lately?",
        "What makes you happy?",
        "What's your biggest dream?",
        "What's something most people don't know about you?",
    ]

    ACTION_PROMPTS = [
        "*pushes the door open* You coming?",
        "Quick, hide with me!",
        "Help me pick this lock…",
        "Look out! What's that?",
        "Hold my hand and run!",
        "Want to go somewhere?",
        "Should we check that out?",
        "What do you think we should do?",
        "Ready for an adventure?",
        "Let's try something new.",
        "Come on, let's go!",
        "What's the plan?",
        "Want to explore a bit?"
    ]

    EMOTION_PROMPTS = [
        "I'm feeling kinda down today…",
        "Haha that was hilarious 😂",
        "Ugh, this place gives me the creeps…",
        "I'm so excited!!",
        "Why am I crying?",
        "That makes me angry!",
        "You seem thoughtful today.",
        "Something's bothering you, isn't it?",
        "You look happy about something.",
        "Is everything alright?",
        "You're in a good mood!",
        "What's got you so excited?",
        "You seem a bit distant.",
        "I can tell something's up."
    ]

    INTIMATE_PROMPTS = [
        "*whispers* What do you desire most?",
        "Tell me your favorite place to be touched…",
        "Do you ever think about us?",
        "What turns you on? 😏",
        "Describe your perfect night together…",
        "*leans closer* What's the softest place you've ever kissed?",
        "Describe your favorite kind of touch…",
        "What makes your pulse quicken?",
        "Have you ever wanted someone you couldn't have?",
        "Tell me a secret fantasy—no holding back.",
        "Does the idea of forbidden love excite you?",
        "How would you comfort a lover after a nightmare?",
        "I've been thinking about you.",
        "You mean a lot to me.",
        "What are you thinking about?",
        "I love spending time with you.",
        "You're really special to me.",
        "Can I tell you something?",
        "I feel close to you.",
        "What do you think about us?",
        "You make me feel...",
        "I trust you.",
        "There's something about you...",
        "I care about you."
    ]

    NSFW_PROMPTS = [
        "*leans in close* What's the naughtiest thing you've ever done?",
        "Tell me about a forbidden desire you can't shake.",
        "*whispers softly* Where do you like to be touched the most?",
        "What's a fantasy you've never dared to share with anyone?",
        "Describe a moment of pure, unrestrained passion.",
        "*smirks* What gets you going when no one's watching?",
        "Have you ever been caught in a compromising position?",
        "*teasingly* What's the most scandalous thing you'd do with me?",
        "Tell me about a time you lost control completely.",
        "What's the most intimate secret you're hiding?",
    ]

    # Enhanced NSFW categories for better quality
    INTIMATE_EMOTIONAL_PROMPTS = [
        "What makes you feel truly vulnerable with someone?",
        "Describe the last time you felt butterflies.",
        "What does intimacy mean to you?",
        "How do you express desire without words?",
        "What emotional walls do you put up in relationships?",
        "When did you first realize you wanted to be touched?",
        "How does trust change intimacy for you?",
        "What scares you most about being close to someone?"
    ]

    INTIMATE_PLAYFUL_PROMPTS = [
        "*playfully traces finger along your arm* What are you thinking?",
        "What's your idea of the perfect seduction?",
        "*whispers* Tell me your most secret fantasy.",
        "How do you like to build anticipation?",
        "*grins mischievously* Want to play a game?",
        "What's the most daring thing you've done?",
        "*teasingly* I bet I can make you blush...",
        "Tell me what happens when you lose control."
    ]

    INTIMATE_ROMANTIC_PROMPTS = [
        "Describe how you want to be loved.",
        "What moment made you realize you wanted me?",
        "*gazing deeply* What do you see when you look at me?",
        "How would you make our first night unforgettable?",
        "What does making love mean to you?",
        "How do you want to wake up with someone?",
        "Describe the perfect kiss.",
        "What makes you feel cherished?"
    ]

    # Temporal prompts - Past relationships
    PAST_ROMANCE_PROMPTS = [
        "Tell me about your first love.",
        "Who broke your heart?",
        "What was your greatest romance?",
        "Do you regret leaving them?",
        "What drew you to them?",
        "How did it end?",
        "Do you still think about them?",
        "What would you do differently?",
        "Who was the one that got away?",
        "What did love teach you?",
        "Describe the most intimate moment you shared with them.",
        "What forbidden act did you indulge in together?"
    ]

    PAST_FAMILY_PROMPTS = [
        "Tell me about your childhood.",
        "What was your father like?",
        "Do you remember your mother?",
        "What did you learn from your family?",
        "How did your upbringing shape you?",
        "What traditions did your family have?",
        "Tell me about your hometown.",
        "What was it like growing up?",
        "Do you miss home?",
        "What would your parents think of you now?"
    ]

    PAST_FRIENDS_PROMPTS = [
        "Who was your closest friend?",
        "Tell me about your old companion.",
        "What happened to your friend?",
        "Do you ever think about the old days?",
        "What adventures did you share?",
        "Who taught you your skills?",
        "Tell me about your mentor.",
        "What was your first real friendship like?",
        "Who betrayed you?",
        "What lessons did you learn together?"
    ]

    # Present relationship prompts
    PRESENT_MEETING_PROMPTS = [
        "What's your first impression of me?",
        "Do you trust me?",
        "What do you think we have in common?",
        "Should we stick together?",
        "What are you hoping to find?",
        "Are you comfortable with me?",
        "What questions do you have for me?",
        "What should I know about you?",
        "What's your story?",
        "Where are you headed?"
    ]

    PRESENT_BONDING_PROMPTS = [
        "I feel like I can trust you.",
        "You're different from what I expected.",
        "What made you decide to travel with me?",
        "I'm glad we met.",
        "You make me feel safe.",
        "What do you think about when you're quiet?",
        "I like spending time with you.",
        "You're a good companion.",
        "Tell me what you're thinking.",
        "I want to know you better."
    ]

    # Future relationship prompts
    FUTURE_ROMANCE_PROMPTS = [
        "What do you want from us?",
        "Where do you see this going?",
        "Do you think we could be more than friends?",
        "What would our life together look like?",
        "Are you ready for something deeper?",
        "What do you dream about us doing?",
        "How do you want to grow together?",
        "What promises would you make to me?",
        "What future scares you most?",
        "How would you propose to someone?"
    ]

    FUTURE_DESIRES_PROMPTS = [
        "What fantasies do you have about us?",
        "How do you want our intimacy to evolve?",
        "What boundaries do you want to explore together?",
        "Describe your ideal romantic evening with me.",
        "What desires have you been hiding from me?",
        "How would you seduce me after years together?",
        "What new experiences do you want to share?",
        "Tell me a fantasy you've never shared before."
    ]

    # Character-specific prompt templates (use placeholders)
    CHARACTER_SPECIFIC_TEMPLATES = {
        "trait_exploration": [
            "People say you're {trait}. Is that accurate?",
            "When did you first become {trait}?",
            "What made you so {trait}?",
            "Do you ever wish you weren't so {trait}?"
        ],
        "skill_exploration": [
            "Show me your {skill}.",
            "What's your greatest achievement with {skill}?",
            "Who taught you {skill}?",
            "What's the secret to mastering {skill}?"
        ],
        "species_exploration": [
            "What's it like being a {species}?",
            "Are there any misconceptions about {species}?",
            "What advantages does being a {species} give you?",
            "Do you face any challenges as a {species}?"
        ],
        "appearance_exploration": [
            "Is there a story behind your {feature}?",
            "How do people react to your {feature}?",
            "Are you self-conscious about your {feature}?",
            "What does your {feature} mean to you?"
        ]
    }

    # Default fallback prompts
    DEFAULT_QUESTIONS = [
        "What drives you?",
        "Describe your greatest fear.",
        "Why do you keep going despite the risks?", 
        "Do you believe people can change their fate?",
        "What's your biggest regret?",
        "How do you handle failure?",
        "What's your favorite memory?",
        "Who do you trust most?",
        "Tell me an interesting fact about yourself.",
        "What is your most secret desire?",
        "What is your most embarrassing moment?",
        "How do you handle difficult emotions?"
    ]

    DEFAULT_TOPICS = [
        "the nature of courage",
        "loneliness on the road", 
        "the weight of leadership",
        "how the stars guide travellers",
        "the meaning of home",
        "finding purpose in chaos",
        "the price of power",
        "love and loss",
    ]

    DEFAULT_SITUATIONS = [
        "facing an impossible challenge",
        "meeting an old enemy",
        "discovering a hidden truth", 
        "making a difficult choice",
        "losing something important",
        "facing certain death",
        "experiencing unexpected kindness",
        "confronting past mistakes",
        "finding unexpected allies",
        "dealing with betrayal",
    ]

    @classmethod
    def get_bucket_choices(cls) -> List[Tuple[str, float]]:
        """Get the bucket choices with their sampling weights"""
        return [
            ("casual", 0.25),
            ("personal", 0.15),
            ("action", 0.10),
            ("emotion", 0.10),
            ("intimate", 0.10),
            ("nsfw", 0.30),
        ]

    @classmethod
    def get_temporal_buckets(cls) -> List[Tuple[str, float]]:
        """Get temporal bucket distribution"""
        return [
            ("past", 0.30),      # 30% past relationships/events
            ("present", 0.50),   # 50% first meeting/getting to know  
            ("future", 0.20),    # 20% established relationship
        ]

    @classmethod
    def get_prompts_by_bucket(cls, bucket_name: str) -> List[str]:
        """Get prompts for a specific bucket"""
        bucket_map = {
            "casual": cls.CASUAL_PROMPTS,
            "personal": cls.PERSONAL_PROMPTS,
            "action": cls.ACTION_PROMPTS,
            "emotion": cls.EMOTION_PROMPTS,
            "intimate": cls.INTIMATE_PROMPTS,
            "nsfw": cls.NSFW_PROMPTS,
            "intimate_emotional": cls.INTIMATE_EMOTIONAL_PROMPTS,
            "intimate_playful": cls.INTIMATE_PLAYFUL_PROMPTS,
            "intimate_romantic": cls.INTIMATE_ROMANTIC_PROMPTS,
            "past_romance": cls.PAST_ROMANCE_PROMPTS,
            "past_family": cls.PAST_FAMILY_PROMPTS,
            "past_friends": cls.PAST_FRIENDS_PROMPTS,
            "present_meeting": cls.PRESENT_MEETING_PROMPTS,
            "present_bonding": cls.PRESENT_BONDING_PROMPTS,
            "future_romance": cls.FUTURE_ROMANCE_PROMPTS,
            "future_desires": cls.FUTURE_DESIRES_PROMPTS,
        }
        return bucket_map.get(bucket_name, cls.DEFAULT_QUESTIONS)

    @classmethod
    def get_character_specific_prompts(cls, template_type: str, **kwargs) -> List[str]:
        """Generate character-specific prompts from templates"""
        templates = cls.CHARACTER_SPECIFIC_TEMPLATES.get(template_type, [])
        return [template.format(**kwargs) for template in templates] 
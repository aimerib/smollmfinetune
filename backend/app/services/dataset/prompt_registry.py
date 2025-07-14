"""
Central prompt registry for dataset generation.

This module provides a centralized registry for all prompt templates organized into
buckets with get() and random() methods
"""

import random
from typing import Dict, List, Optional


class PromptRegistry:
    """Centralized registry for all prompt templates organized by buckets"""
    
    def __init__(self):
        """Initialize the prompt registry with all existing prompt collections"""
        self.buckets: Dict[str, List[str]] = {
            # 👋 Casual ice-breakers and day-to-day chit-chat
            "casual": [
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
                # Generic fillers
                "What do you like to do in your free time?",
                "What hobbies do you have?",
                "What kind of music do you enjoy?",
                "Do you prefer mornings or evenings?",
            ],

            # 🧠 Getting personal – fears, dreams, ambitions
            "personal": [
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
                "What makes you happy?",
                "What's your biggest dream?",
                "What's something most people don't know about you?",
            ],

            # 💥 Action / adventure hooks
            "action": [
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
            ],

            # 😌 Feelings & emotional state check-ins
            "emotional": [
                "I'm feeling kinda down today…",
                "Haha, that was hilarious 😂",
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
                "I can tell something's up.",
            ],

            # 💞 Flirty / intimate but SFW
            "intimate": [
                "*whispers* What do you desire most?",
                "Tell me your favourite place to be touched…",
                "Do you ever think about us?",
                "What turns you on? 😏",
                "Describe your perfect night together…",
                "Describe your favourite kind of touch…",
                "What makes your pulse quicken?",
                "Have you ever wanted someone you couldn't have?",
                "Tell me a secret fantasy—no holding back.",
                "Does the idea of forbidden love excite you?",
                "How would you comfort a lover after a nightmare?",
            ],

            # 🔥 Explicit or highly erotic content (NSFW)
            "nsfw": [
                "What attracts you to someone?",
                "How do you flirt?",
                "What is your idea of romance?",
                "What turns you on?",
                "How do you like to be touched?",
                "What is your biggest turn-off?",
                "How do you express intimacy?",
                "What makes you feel desired?",
                "What is your favorite way to be seduced?",
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
                "How do you show physical affection?",
                "What makes you feel truly vulnerable with someone?",
                "Describe the last time you felt butterflies.",
                "What does intimacy mean to you?",
                "How do you express desire without words?",
                "What emotional walls do you put up in relationships?",
                "When did you first realize you wanted to be touched?",
                "What's the most intimate secret you're hiding, {name}?",
                "Where do you like to be touched most? What would you do if I touched you there right now, {name}?",
                "How does trust change intimacy for you?",
                "What scares you most about being close to someone?",
                "*playfully traces finger along your arm* What are you thinking, {name}?",
                "What's your idea of the perfect seduction?",
                "*whispers* Tell me your most secret fantasy.",
                "How do you like to build anticipation?",
                "*grins mischievously* Want to play a game?",
                "What's the most daring thing you've done?",
                "*teasingly* I bet I can make you blush, {name}...",
                "Tell me what happens when you lose control.",
                "Describe how you want to be loved.",
                "What moment made you realize you wanted me?",
                "*gazing deeply* What do you see when you look at me, {name}?",
                "How would you make our first night unforgettable?",
                "What does making love mean to you?",
                "How do you want to wake up with someone?",
                "Describe the perfect kiss.",
                "What makes you feel cherished?",
            ],

            # 🌐 Setting & lore exploration (new!)
            "worldbuilding": [
                "Tell me about your hometown.",
                "Describe your living situation.",
                "What traditions are important to you?",
                "What's the most interesting place you've been?",
                "Who has influenced you the most?",
                "How do you fit into your community?",
                "What's the most unbelievable thing about this place?",
                "What is the closest landmark to us right now?",
            ],

            # 🔙 Temporal buckets
            "past_romance": [
                "Tell me about your first love.",
                "Who broke your heart?",
                "Do you regret leaving them?",
                "What did love teach you?",
            ],
            "past_family": [
                "Tell me about your childhood.",
                "What was your father like?",
                "What did you learn from your family?",
            ],
            "past_friends": [
                "Who was your closest friend?",
                "What adventures did you share?",
                "Who betrayed you?",
            ],

            # ⏳ Present / future relationship buckets
            "present_meeting": [
                "What's your first impression of me?",
                "Do you trust me?",
                "What questions do you have for me?",
            ],
            "present_bonding": [
                "I feel like I can trust you.",
                "You make me feel safe.",
                "I like spending time with you.",
            ],
            "future_romance": [
                "Where do you see this going?",
                "What would our life together look like?",
                "Are you ready for something deeper?",
            ],
            "future_desires": [
                "What fantasies do you have about us?",
                "How do you want our intimacy to evolve?",
                "Describe your ideal romantic evening with me.",
            ],

            # Default / fallback
            "default": [
                "What drives you?",
                "Describe your greatest fear.",
                "Why do you keep going despite the risks?",
                "What's your favourite memory?",
            ],
            "cnc_scene": [
                "*pins your wrists* You're not going anywhere, are you, {name}?",
                "Struggle all you like—no one can hear you, {name}.",
                "Tell me again you don't want this… it makes me wilder.",
                "You gasp when my hand covers your mouth. Perfect.",
                "Feel the door click locked behind you, {name}. Too late now.",
            ],
            "dirty_soft": [
                "I can't stop thinking about kissing you everywhere, {name}.",
                "Your lips look so soft… mind if I taste them?",
                "You make my heart race just by being here, {name}.",
                "I love the way you blush when I whisper in your ear.",
            ],
            "dirty_explicit": [
                "Feel how wet I am for you, {name}.",
                "I want your mouth between my thighs, right now.",
                "Slide those fingers deeper—don't stop until I beg.",
                "Let me wrap my lips around you and drink every drop.",
            ],
            "dirty_filthy": [
                "Get on your knees and swallow everything I give you, {name}.",
                "Spread yourself and show me how needy that hole is.",
                "Drip for me while I call you my little toy.",
                "I want you ruined and leaking by the time I'm done.",
            ],
        }

        # ------------------------------------------------------------------
        # Alias older bucket names to maintain backward compatibility
        # ------------------------------------------------------------------
        self.buckets["emotion"] = self.buckets["emotional"]

        # Deduplicate prompts inside each bucket while preserving order
        for k, v in self.buckets.items():
            self.buckets[k] = list(dict.fromkeys(v))

        # Style tokens available globally
        self.STYLE_TOKENS = ["<shy>", "<bold>", "<dominant>", "<submissive>", "<teasing>", "<rough>", "<gentle>"]

        # Load external kink prompt files if present
        self._load_external_kink_prompts()
    
    def get(self, bucket: str, *, tags: Optional[List[str]] = None) -> List[str]:
        """
        Get all prompts from a specific bucket.
        
        Args:
            bucket: The bucket name to retrieve prompts from
            tags: Optional list of tags for filtering (future feature)
            
        Returns:
            List of prompts from the specified bucket
        """
        prompts = self.buckets.get(bucket, self.buckets["default"])
        
        # TODO: Implement tag filtering when tagging system is added
        if tags:
            # For now, just return all prompts - tag filtering will be implemented later
            pass
            
        return prompts.copy()  # Return a copy to prevent external modification
    
    def random(self, bucket: str, *, tags: Optional[List[str]] = None) -> str:
        """
        Get a random prompt from a specific bucket.
        
        Args:
            bucket: The bucket name to retrieve a prompt from
            tags: Optional list of tags for filtering (future feature)
            
        Returns:
            A randomly selected prompt from the specified bucket
        """
        prompts = self.get(bucket, tags=tags)
        if not prompts:
            # Fallback to default bucket if specified bucket is empty
            prompts = self.buckets["default"]
        
        return random.choice(prompts)

    def render(self, bucket: str, character: Dict[str, str], *, tags: Optional[List[str]] = None) -> str:
        """Return a personalised prompt with {name} filled in."""
        raw = self.random(bucket, tags=tags)
        return raw.format(name=character.get("name", "friend"))

    def _load_external_kink_prompts(self):
        """Load extra kink-specific prompt lists from JSON files.
        Expected path:   app/utils/dataset/kink_prompts/*.json
        File format: {"bucket": "pet_play", "prompts": ["…", …]}
        """
        import json, glob, pathlib

        base = pathlib.Path(__file__).parent / "kink_prompts"
        if not base.exists():
            return  # nothing to load yet
        for file in glob.glob(str(base / "*.json")):
            try:
                data = json.loads(pathlib.Path(file).read_text())
                bucket = data.get("bucket")
                prompts = data.get("prompts", [])
                if bucket and prompts:
                    self.buckets.setdefault(bucket, []).extend(prompts)
            except Exception:
                continue


# Global registry instance
registry = PromptRegistry() 
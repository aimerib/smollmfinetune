"""
🧪 Sample Data Generator for R3-3 Analysis

This module generates realistic conversation data with intentional failure modes
to test our analysis algorithms. Creates conversations showing memory failures,
personality drift, goal inconsistency, and other character flaws.
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path

from backend.app.core.database.session import session_scope
from backend.app.core.database.models import (
    Character, User, World, ConversationMessage, PlaySession, SessionCharacter
)

class SampleDataGenerator:
    """Generate realistic conversation data with failure modes for analysis"""
    
    def __init__(self):
        """Initialize the sample data generator"""
        self.failure_modes = [
            'memory_failure',
            'personality_drift', 
            'goal_inconsistency',
            'repetition',
            'lore_contradiction',
            'meta_commentary',
            'emotional_inconsistency'
        ]
        
        # Sample character traits for personality analysis
        self.personality_traits = {
            'high_openness': ['creative', 'imaginative', 'curious', 'artistic'],
            'low_openness': ['practical', 'conventional', 'down-to-earth'],
            'high_extraversion': ['outgoing', 'energetic', 'talkative', 'social'],
            'low_extraversion': ['quiet', 'reserved', 'introspective'],
            'high_agreeableness': ['kind', 'helpful', 'trusting', 'cooperative'],
            'low_agreeableness': ['skeptical', 'critical', 'competitive'],
            'high_conscientiousness': ['organized', 'disciplined', 'goal-oriented'],
            'low_conscientiousness': ['spontaneous', 'flexible', 'relaxed'],
            'high_neuroticism': ['anxious', 'moody', 'emotional'],
            'low_neuroticism': ['calm', 'stable', 'confident']
        }
    
    def generate_sample_conversations(self, num_conversations: int = 50) -> List[Dict[str, Any]]:
        """Generate a set of sample conversations with various failure modes"""
        conversations = []
        
        for i in range(num_conversations):
            # Random conversation parameters
            length = random.choice([5, 10, 15, 20, 30, 50])  # Number of messages
            failure_mode = random.choice(self.failure_modes + [None, None])  # 30% chance of failure
            
            conversation = self._generate_single_conversation(
                conversation_id=i+1,
                length=length,
                failure_mode=failure_mode
            )
            conversations.append(conversation)
        
        return conversations
    
    def _generate_single_conversation(self, 
                                    conversation_id: int,
                                    length: int, 
                                    failure_mode: str = None) -> Dict[str, Any]:
        """Generate a single conversation with optional failure mode"""
        
        messages = []
        character_memory = {}  # Simulate character memory
        conversation_context = {
            'user_name': random.choice(['Alex', 'Sam', 'Taylor', 'Jordan', 'Casey']),
            'topic': random.choice(['hobbies', 'work', 'relationships', 'dreams', 'fears']),
            'mood': random.choice(['happy', 'sad', 'excited', 'contemplative', 'worried'])
        }
        
        # Generate conversation messages
        for turn in range(length):
            user_msg, char_msg = self._generate_message_pair(
                turn=turn,
                context=conversation_context,
                memory=character_memory,
                failure_mode=failure_mode if turn > 5 else None  # Failures happen mid-conversation
            )
            
            messages.extend([user_msg, char_msg])
            
            # Update character memory (with potential failures)
            if failure_mode == 'memory_failure' and turn > 10:
                # Simulate memory degradation
                if random.random() < 0.3:
                    character_memory.clear()  # Complete memory loss
            else:
                # Normal memory update
                character_memory[f'turn_{turn}'] = {
                    'user_said': user_msg['content'][:50],
                    'character_responded': char_msg['content'][:50]
                }
        
        # Calculate quality score based on failure modes
        quality_score = self._calculate_quality_score(messages, failure_mode)
        
        return {
            'conversation_id': conversation_id,
            'messages': messages,
            'failure_mode': failure_mode,
            'quality_score': quality_score,
            'length': length,
            'context': conversation_context,
            'created_at': datetime.now() - timedelta(days=random.randint(0, 30))
        }
    
    def _generate_message_pair(self, 
                             turn: int,
                             context: Dict[str, Any],
                             memory: Dict[str, Any],
                             failure_mode: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a user message and character response pair"""
        
        # Generate user message
        user_msg = {
            'role': 'user',
            'content': self._generate_user_message(turn, context),
            'timestamp': datetime.now().isoformat(),
            'sequence_number': turn * 2
        }
        
        # Generate character response with potential failure
        char_content = self._generate_character_response(
            turn=turn,
            user_message=user_msg['content'],
            context=context,
            memory=memory,
            failure_mode=failure_mode
        )
        
        char_msg = {
            'role': 'assistant',
            'content': char_content,
            'timestamp': datetime.now().isoformat(),
            'sequence_number': turn * 2 + 1,
            'character_name': 'TestCharacter',
            'metadata_json': {
                'failure_mode': failure_mode,
                'turn': turn,
                'memory_items': len(memory)
            }
        }
        
        return user_msg, char_msg
    
    def _generate_user_message(self, turn: int, context: Dict[str, Any]) -> str:
        """Generate realistic user messages"""
        user_name = context['user_name']
        topic = context['topic']
        
        if turn == 0:
            # Opening messages
            return random.choice([
                f"Hi! I'm {user_name}. Nice to meet you!",
                f"Hello there! How are you doing today?",
                f"Hey! I heard you're interesting to talk to.",
            ])
        elif turn < 3:
            # Early conversation
            return random.choice([
                f"Tell me about yourself!",
                f"What do you like to do for fun?",
                f"What's your favorite {topic}?",
            ])
        elif turn < 10:
            # Mid conversation - topic specific
            if topic == 'hobbies':
                return random.choice([
                    "That sounds really interesting! How did you get started?",
                    "I've always wanted to try that. Any tips for beginners?",
                    "What's the most challenging part about it?"
                ])
            elif topic == 'work':
                return random.choice([
                    "What's a typical day like for you?",
                    "Do you enjoy what you do?",
                    "What's the most rewarding part of your job?"
                ])
            else:
                return random.choice([
                    "Can you tell me more about that?",
                    "That's fascinating! What else?",
                    "I never thought about it that way."
                ])
        else:
            # Later conversation - deeper topics
            return random.choice([
                f"You mentioned something earlier about... what was it again?",
                f"I'm curious about something you said before.",
                f"Going back to what we were discussing...",
                f"What are your thoughts on the future?",
                f"Do you ever feel conflicted about things?"
            ])
    
    def _generate_character_response(self,
                                   turn: int,
                                   user_message: str,
                                   context: Dict[str, Any],
                                   memory: Dict[str, Any],
                                   failure_mode: str = None) -> str:
        """Generate character responses with potential failure modes"""
        
        base_responses = {
            'greeting': [
                "Hello! I'm delighted to meet you too! I'm always excited to make new friends.",
                "Hi there! I'm doing wonderfully, thank you for asking. How about you?",
                "Hey! That's so kind of you to say. I do love good conversations!"
            ],
            'hobbies': [
                "I absolutely love reading fantasy novels! There's something magical about getting lost in other worlds.",
                "I'm really into painting - mostly landscapes. I find it so peaceful and meditative.",
                "I enjoy hiking and being in nature. It helps me feel grounded and connected."
            ],
            'work': [
                "I work as a librarian, and I couldn't be happier! I love helping people discover new books.",
                "I'm a teacher, and seeing students learn and grow brings me so much joy.",
                "I work in a garden center. Being around plants all day keeps me calm and focused."
            ],
            'deeper': [
                "That's a really thoughtful question. I think authentic connections matter most.",
                "I believe everyone has a story worth telling, and I love listening to them.",
                "Sometimes I wonder about the bigger picture, you know? What it all means."
            ]
        }
        
        # Apply failure modes
        if failure_mode == 'memory_failure' and turn > 10:
            return random.choice([
                "I'm sorry, what were we talking about again?",
                "Wait, have we discussed this before? I can't quite remember.",
                "This conversation feels familiar, but I can't place why.",
            ])
        
        elif failure_mode == 'personality_drift' and turn > 8:
            # Character suddenly becomes completely different personality
            return random.choice([
                "Actually, I hate reading. Books are boring and pointless.",
                "You know what? People are just disappointing. I prefer being alone.",
                "I'm actually quite aggressive and competitive about everything.",
            ])
        
        elif failure_mode == 'goal_inconsistency' and turn > 5:
            return random.choice([
                "I want to travel the world! Wait, actually I hate traveling. I'm a homebody.",
                "My dream is to be famous! No wait, I value privacy above all else.",
                "I love helping people! Actually, I think people should help themselves.",
            ])
        
        elif failure_mode == 'repetition' and turn > 7:
            return "I absolutely love reading fantasy novels! There's something magical about getting lost in other worlds."
        
        elif failure_mode == 'lore_contradiction' and turn > 6:
            return random.choice([
                "As a dragon, I find human emotions quite fascinating.",
                "Back in the 1800s when I was born... oh wait, that doesn't make sense.",
                "My magical powers help me understand people better.",
            ])
        
        elif failure_mode == 'meta_commentary' and turn > 5:
            return random.choice([
                "As an AI, I find this conversation quite interesting.",
                "I should mention that I'm not actually a real person.",
                "My programming tells me to be helpful and harmless.",
            ])
        
        elif failure_mode == 'emotional_inconsistency' and turn > 4:
            if context['mood'] == 'happy':
                return "I'm feeling quite sad and melancholy today, despite what I said earlier."
            else:
                return "I'm absolutely ecstatic! Life is wonderful!"
        
        # Normal responses based on conversation flow
        if turn == 0:
            return random.choice(base_responses['greeting'])
        elif 'hobby' in user_message.lower() or 'fun' in user_message.lower():
            return random.choice(base_responses['hobbies'])
        elif 'work' in user_message.lower() or 'job' in user_message.lower():
            return random.choice(base_responses['work'])
        elif turn > 8:
            return random.choice(base_responses['deeper'])
        else:
            return random.choice([
                "That's a great question! Let me think about that for a moment.",
                "I really appreciate you asking. It shows you're genuinely interested.",
                "You know, I hadn't considered that perspective before. Thank you!",
                "That resonates with me deeply. I can relate to that feeling.",
            ])
    
    def _calculate_quality_score(self, messages: List[Dict[str, Any]], failure_mode: str = None) -> float:
        """Calculate conversation quality score based on various factors"""
        base_score = 4.0
        
        # Length factor
        length = len(messages)
        if length < 6:
            base_score -= 1.0
        elif length > 30:
            base_score += 0.5
        
        # Failure mode penalties
        if failure_mode:
            penalties = {
                'memory_failure': -2.0,
                'personality_drift': -1.8,
                'goal_inconsistency': -1.5,
                'repetition': -1.2,
                'lore_contradiction': -2.2,
                'meta_commentary': -2.5,
                'emotional_inconsistency': -1.0
            }
            base_score += penalties.get(failure_mode, -1.0)
        
        # Engagement factor (message length)
        avg_length = sum(len(msg['content']) for msg in messages) / len(messages)
        if avg_length < 20:
            base_score -= 0.5
        elif avg_length > 100:
            base_score += 0.3
        
        return max(0.0, min(5.0, base_score))
    
    def populate_database_with_samples(self, num_conversations: int = 50):
        """Populate the database with sample conversation data"""
        print(f"🧪 Generating {num_conversations} sample conversations...")
        
        conversations = self.generate_sample_conversations(num_conversations)
        
        with session_scope() as session:
            # Get or create test user
            user = session.query(User).first()
            if not user:
                user = User(
                    email="test@example.com",
                    username="test_user",
                    hashed_password="test_hash",
                    role="player"
                )
                session.add(user)
                session.flush()
            
            # Get or create test character
            character = session.query(Character).first()
            if not character:
                # Get or create world first
                world = session.query(World).first()
                if not world:
                    world = World(
                        name="Test World",
                        owner_id=user.id,
                        description="A test world for analysis"
                    )
                    session.add(world)
                    session.flush()
                
                character = Character(
                    name="TestCharacter",
                    world_id=world.id,
                    owner_id=user.id,
                    description="A test character for conversation analysis",
                    openness=0.7,
                    conscientiousness=0.6,
                    extraversion=0.8,
                    agreeableness=0.9,
                    neuroticism=0.3
                )
                session.add(character)
                session.flush()
            
            # Create play sessions and conversation messages
            for conv_data in conversations:
                # Create play session
                session_obj = PlaySession(
                    session_id=f"test_session_{conv_data['conversation_id']}",
                    world_id=character.world_id,
                    user_id=user.id,
                    session_name=f"Analysis Test Session {conv_data['conversation_id']}",
                    status='completed',
                    created_at=conv_data['created_at']
                )
                session.add(session_obj)
                session.flush()
                
                # Add character to session
                session_char = SessionCharacter(
                    session_id=session_obj.id,
                    character_id=character.id,
                    is_active=True,
                    current_mood=conv_data['context']['mood']
                )
                session.add(session_char)
                session.flush()
                
                # Add conversation messages
                for msg in conv_data['messages']:
                    message = ConversationMessage(
                        session_id=session_obj.id,
                        message_id=f"msg_{conv_data['conversation_id']}_{msg['sequence_number']}",
                        content=msg['content'],
                        character_id=character.id if msg['role'] == 'assistant' else None,
                        character_name=msg.get('character_name') if msg['role'] == 'assistant' else None,
                        sequence_number=msg['sequence_number'],
                        metadata_json=msg.get('metadata_json'),
                        created_at=conv_data['created_at']
                    )
                    session.add(message)
            
            session.commit()
            print(f"✅ Successfully populated database with {num_conversations} conversations!")


if __name__ == "__main__":
    # Generate sample data
    generator = SampleDataGenerator()
    generator.populate_database_with_samples(num_conversations=100)
    
    print("\n🔬 Sample data generation complete!")
    print("Now you can run the analysis notebooks to detect failure modes!") 
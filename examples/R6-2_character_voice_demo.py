import asyncio
import numpy as np
import soundfile as sf
from narrative_engine.character_voice_integration import CharacterVoiceSynthesizer

async def main():
    """
    Demonstrates the use of the CharacterVoiceSynthesizer to generate
    character-specific speech with control tokens.
    """
    print("--- Character Voice System Demo ---")

    # 1. Define a sample character with personality traits
    character = {
        "id": "char_demo_001",
        "name": "Aria",
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.4,
            "extraversion": 0.9,
            "agreeableness": 0.6,
            "neuroticism": 0.2
        }
    }
    print(f"Character: {character['name']} (Extraversion: {character['personality']['extraversion']})")

    # 2. Define text with control tokens
    text = "Wow, this is [EMOTION:joy:0.9] absolutely amazing! I'm speaking with my own voice!"
    print(f"Text with control tokens: {text}")

    # 3. Instantiate the synthesizer
    synthesizer = CharacterVoiceSynthesizer()
    print("Initialized CharacterVoiceSynthesizer.")

    # 4. Synthesize speech
    print("Synthesizing speech...")
    audio, sample_rate, voice_metadata = await synthesizer.synthesize_character_speech(
        text=text,
        character=character
    )
    print("Speech synthesized successfully!")

    # 5. Print metadata
    print("\n--- Voice Metadata ---")
    print(f"Voice Consistency Hash: {voice_metadata['voice_consistency_hash']}")
    print(f"TTS Model Used: {voice_metadata['model_used']}")
    print(f"Final Emotion Intensity: {voice_metadata['emotion_intensity']:.2f}")
    print(f"Control Tokens Applied: {voice_metadata['control_tokens_applied']}")
    print("----------------------")

    # 6. Save the audio to a file
    output_filename = "character_voice_demo.wav"
    sf.write(output_filename, audio, sample_rate)
    print(f"\nGenerated audio saved to: {output_filename}")
    print("You can play this file to hear the character's voice.")

if __name__ == "__main__":
    # Add the project root to the python path to allow imports from app and narrative_engine
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    asyncio.run(main()) 
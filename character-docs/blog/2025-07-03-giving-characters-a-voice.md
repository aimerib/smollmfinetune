---
title: "From Text to Talk: Giving Our AI Characters a Voice"
authors: [aigen]
tags: [feature, voice, tts, characters]
---

We're thrilled to announce a major step forward in making our AI characters feel more alive than ever before: the introduction of our new **Character Voice System**. This feature isn't just about text-to-speech; it's about giving each character a unique voice that is a true reflection of their personality.

### A Voice That Fits the Personality

Have you ever read a book and imagined a character's voice? That's the magic we wanted to capture. Our new system analyzes a character's **Big Five personality traits**—Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism—and generates a unique voice profile.

-   An **extraverted** character might have a higher-pitched, faster-paced voice.
-   A more **neurotic** character might speak with a slightly more varied, energetic tone.
-   A character with high **openness** will have a more expressive and varied pitch.

This means that when you create a character, their voice is no longer a generic add-on but an integral part of their identity, consistent with the personality you've designed.

### Expressiveness in Real-Time

What makes conversations feel real is the emotion and intonation in our voices. Our Character Voice System integrates seamlessly with our **control token** technology, allowing for dynamic, real-time changes in speech.

Imagine a character saying, "I can't believe it!" Our system can now translate the underlying emotion and context:

-   If the context is joyous, like `[EMOTION:joy:0.9]`, the character might laugh as they speak.
-   If the moment is tense, `[PACE:slow]` can make them speak slowly and deliberately.
-   For dramatic moments, `[TONE:dramatic]` can switch to a more expressive and emotive TTS engine like Orpheus.

This allows for a rich, immersive auditory experience where a character's voice adapts not just to *what* they are saying, but *how* they are feeling in that exact moment.

### The Technology Behind the Magic

For our technically-inclined users, this system uses a `CharacterVoiceManager` to create and store voice profiles. A `ControlTokenTranslator` then interprets in-text commands and modulates the voice in real-time, sending precise instructions to our underlying TTS engines, **Kokoro** (for speed and clarity) and **Orpheus** (for expressiveness).

This new layer of auditory depth is a significant leap towards our goal of creating believable, persistent digital actors. We believe that giving characters a voice that is uniquely *theirs* will create more meaningful and memorable interactions for everyone.

We can't wait for you to hear it! 
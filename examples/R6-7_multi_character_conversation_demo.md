# Multi-Character Conversation Demo

Experience console-quality multi-character conversations with spatial audio, real-time mixing, and seamless character interactions!

## 🎮 What You'll Experience

This demo showcases our revolutionary multi-character conversation system that transforms AI character interaction from simple chatbots into immersive, console-quality audio experiences.

### Key Features You'll Try:
- **🎧 3D Spatial Audio**: Characters positioned in 3D space with HRTF processing
- **🎭 Seamless Voice Switching**: Natural conversation flow with interruptions and overlaps  
- **🌍 Environmental Effects**: Professional acoustic modeling (Studio, Room, Hall, Outdoor)
- **🎚️ Real-time Audio Mixing**: Professional-grade controls for volume, pacing, and effects
- **📊 Live Conversation Timeline**: Visual representation with real-time updates

## 🚀 Quick Start Guide

### Prerequisites
1. **Platform Running**: Ensure you have the React+FastAPI platform running
2. **Headphones Recommended**: For the full spatial audio experience
3. **2+ Characters**: Have at least two characters created in your world

### Step-by-Step Demo

#### 1. Launch the Platform
```bash
# Start the full platform with one command
./launch-client.sh
```

Wait for both services to start:
- **Backend API**: Will be running on `http://localhost:8000`
- **React Client**: Will be available at `http://localhost:3000`

#### 2. Access the Multi-Character Audio Mixer

1. Open your web browser and navigate to `http://localhost:3000`
2. From the main navigation, find and click **"Multi-Character Audio Mixer"**
3. You'll see the professional audio mixing interface

#### 3. Set Up Your Characters

**If you have existing characters:**
- They should automatically appear in the character panel
- Each character will show their name and voice provider (Kokoro, Orpheus, etc.)

**If you need to create characters:**
1. Navigate to **Character Builder**
2. Create at least 2 characters with different personalities
3. Assign voice profiles to each character
4. Return to the Multi-Character Audio Mixer

#### 4. Experience the Features

**🎧 Enable Spatial Audio:**
1. Check the "Enable 3D Audio" toggle in the Spatial Audio section
2. Put on headphones for the best experience
3. You'll see character icons positioned around a central listener position

**🗺️ Position Your Characters:**
1. Drag the character icons (🗣️) to different positions
2. Notice how the audio positioning changes as you move them
3. Try positioning one character to your left, another to your right

**🌍 Try Different Environments:**
1. In the Environmental Effects panel, change the "Acoustic Environment"
2. Switch between Studio, Room, Hall, and Outdoor
3. Notice how each environment changes the sound characteristics
4. Adjust reverb, ambient noise, and distance attenuation

#### 5. Create a Conversation

**Start a Multi-Character Dialogue:**
1. In the Conversation Timeline section, use the dialogue input
2. Select a character from the dropdown
3. Type some dialogue and click "Send"
4. Switch to a different character and add a response
5. Watch the real-time conversation timeline update

**Example Conversation to Try:**
```
Alice: "Welcome to our tavern! What brings you here on such a stormy night?"
Bob: "I'm seeking information about the ancient ruins to the north."
Alice: "Ah, those ruins... they hold many secrets and dangers."
Bob: "I'm not afraid of danger. What can you tell me about them?"
```

#### 6. Master the Controls

**🎚️ Real-time Mixing:**
- Adjust **Master Volume** for overall conversation level
- Change **Conversation Pacing** (0.5x to 2.0x speed)
- Try the **Recording** feature to capture conversations

**🎛️ Environmental Effects:**
- **Reverb Level**: Try values from 0% (dry) to 100% (cathedral-like)
- **Ambient Noise**: Add background atmosphere (0-100%)
- **Distance Attenuation**: Control how distance affects volume

#### 7. Advanced Features

**Character Interruptions:**
- Characters can interrupt each other based on personality and relationships
- Try creating rapid back-and-forth dialogue to see natural interruptions

**Emotional Contagion:**
- Characters react to each other's emotional states
- Create emotionally charged dialogue and watch characters influence each other

**Session Management:**
- The mixer tracks your active session
- Character count and session info are displayed in the header

## 🎯 Demo Scenarios to Try

### Scenario 1: Fantasy Tavern Scene
**Characters**: Innkeeper, Adventurer, Mysterious Stranger
**Environment**: Room with moderate reverb
**Setup**: Position Innkeeper center, Adventurer left, Stranger right back
**Dialogue**: Create a scene where information is exchanged about a quest

### Scenario 2: Space Station Meeting
**Characters**: Captain, Engineer, AI Assistant  
**Environment**: Hall with long reverb (metallic space station feel)
**Setup**: Captain center-front, Engineer left, AI right
**Dialogue**: Technical discussion about ship repairs

### Scenario 3: Intimate Conversation
**Characters**: Two close friends
**Environment**: Studio (intimate, dry sound)  
**Setup**: Both characters close to center
**Dialogue**: Personal, emotional conversation with natural interruptions

### Scenario 4: Outdoor Adventure
**Characters**: Explorer, Guide, Wildlife Expert
**Environment**: Outdoor with ambient noise
**Setup**: Wide positioning to simulate being spread out
**Dialogue**: Discussion about wildlife and navigation

## 🔧 Troubleshooting

### Audio Issues
**No Sound Playing:**
1. Check browser audio permissions (look for 🔇 icon in address bar)
2. Verify your volume settings
3. Ensure characters have voice profiles assigned

**Spatial Audio Not Working:**
1. **Use headphones** - spatial audio requires headphones for proper positioning
2. Make sure "Enable 3D Audio" is checked
3. Try repositioning characters to notice the difference

### Performance Issues
**Slow Response:**
1. Check your internet connection
2. Try reducing the number of active characters
3. Use "Studio" environment for best performance

**Character Not Responding:**
1. Verify WebSocket connection (check browser developer tools)
2. Ensure characters are properly configured with voice profiles
3. Try refreshing the page if connection is lost

### Browser Compatibility
- **Best Performance**: Chrome, Firefox, Safari (latest versions)
- **WebSocket Support**: Required for real-time features
- **Audio Context**: Modern browsers with Web Audio API support

## 🎓 Learning Points

After completing this demo, you'll understand:

1. **Spatial Audio Concepts**: How 3D positioning creates immersive experiences
2. **Real-time Streaming**: How WebSocket technology enables live conversations
3. **Environmental Modeling**: How acoustic environments affect audio experience
4. **Character Dynamics**: How AI characters can interact naturally with each other
5. **Professional Audio Tools**: How to use mixing controls for creative expression

## 🚀 Next Steps

**Explore More Features:**
- Try **Director's Chair** for advanced conversation monitoring
- Use **Character Builder** to create characters optimized for conversations
- Experiment with **World Builder** to create environments that enhance stories
- Generate training data with **Dataset Studio** using conversation examples

**Creative Projects:**
- Create audio dramas with multiple characters
- Develop interactive story experiences
- Design character interaction prototypes for games
- Build educational content with historical or literary characters

**Share Your Experience:**
- Record interesting conversations using the built-in recording feature
- Share feedback about what features you'd like to see next
- Join our community discussions about character AI development

## 💡 Pro Tips

1. **Headphones are Essential**: The spatial audio experience is dramatically better with headphones
2. **Start Simple**: Begin with 2 characters and gradually add more
3. **Experiment with Positioning**: Try extreme positions (far left/right) to feel the spatial effect
4. **Match Environment to Scene**: Use appropriate acoustic environments for your story setting
5. **Record Your Favorites**: Use the recording feature to save compelling conversations
6. **Character Personalities Matter**: Characters with distinct personalities create more interesting dynamics

---

**🎮 Ready to Experience the Future of AI Character Interaction?**

This Multi-Character Conversation system represents a fundamental leap forward in how we interact with AI characters. Instead of simple question-and-answer sessions, you're experiencing **living digital actors** that exist in shared virtual spaces.

Have fun exploring, and let us know what amazing conversations you create! 
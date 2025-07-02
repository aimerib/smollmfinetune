---
sidebar_position: 5
---

# React Client Guide

Welcome to our stunning production React client! This guide will help you master all the features of our beautiful character chat interface.

<div style={{background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', padding: '2rem', borderRadius: '12px', color: 'white', marginBottom: '2rem'}}>
  <h2 style={{marginTop: 0}}>Client Features</h2>
  <p>Experience AI characters like never before with our glassmorphism UI:</p>
  <ul style={{marginBottom: 0}}>
    <li>Real-time emotional state visualization</li>
    <li>Memory formation animations</li>
    <li>Beautiful particle effects and smooth transitions</li>
    <li>WebSocket-powered instant responses</li>
    <li>Mobile-first responsive design</li>
  </ul>
</div>

## Interface Overview

### Home Screen - Character Selection

:::note[Home Screen]
When you first open the client at `http://localhost:3000`, you'll see:

1. **Animated Background**: Floating gradient orbs that drift across the screen
2. **Character Cards**: Each character displayed with:
   - Custom emoji avatar
   - Name and tagline
   - Personality stats visualization
   - Shimmer effect on hover
   - Click to start chatting

3. **Quick Stats**: For each character:
   - Total conversations
   - Average rating
   - Active emotional state

:::

### Chat Interface

Once you select a character, you enter the immersive chat experience:

#### Header Section
- **Character Avatar**: Animated with emotion-based gradient
- **Character Name & Status**: Current emotional state indicator
- **Back Button**: Return to character selection
- **Options Menu**: Settings and character info

#### Chat Area
- **Message Bubbles**: 
  - User messages: Purple gradient, right-aligned
  - Character messages: Blue gradient, left-aligned
  - Smooth fade-in animations
  - Markdown support for formatting

#### Emotion Indicators
Real-time visualization of character's emotional state:
- **Happiness**: Golden gradient progress bar
- **Sadness**: Blue gradient progress bar  
- **Anger**: Red gradient progress bar
- **Curiosity**: Teal gradient progress bar
- **Excitement**: Pink gradient progress bar

#### Input Area
- **Message Input**: Auto-expanding textarea
- **Send Button**: Animated with hover effects
- **Typing Indicator**: Shows when character is "thinking"

## Getting Started

### Basic Conversation Flow

1. **Select a Character**
   ```
   Click on any character card from the home screen
   ```

2. **Start Chatting**
   ```
   Type your message and press Enter or click Send
   ```

3. **Watch Emotions**
   ```
   See real-time emotion changes as the character responds
   ```

4. **Explore Topics**
   ```
   Characters have deep knowledge - ask about their world!
   ```

### Conversation Tips

<div style={{display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '2rem'}}>
  :::tip[✅ Good Conversation Starters]
  <div>
    <ul>
      <li>"Tell me about yourself"</li>
      <li>"What's your greatest adventure?"</li>
      <li>"How are you feeling today?"</li>
      <li>"What's your favorite memory?"</li>
      <li>Ask about their special skills</li>
    </ul>
  </div>
  :::
  :::note[🎯 Character-Specific Topics]
  <div>
    <ul>
      <li><strong>Alice</strong>: Dreams, adventures, curiosity</li>
      <li><strong>Max</strong>: Technology, gadgets, future</li>
      <li><strong>Luna</strong>: Dreams, mysticism, guidance</li>
      <li>Ask about their relationships</li>
      <li>Explore their world's lore</li>
    </ul>
  </div>
  :::
</div>

## Understanding Emotions

### Emotion System

Characters express emotions through multiple channels:

1. **Visual Indicators**
   - Avatar gradient changes color
   - Progress bars show emotion intensity
   - Smooth transitions between states

2. **Behavioral Changes**
   - Happy characters are more enthusiastic
   - Sad characters may be more reflective
   - Angry characters show frustration
   - Curious characters ask questions

3. **Memory Formation**
   - Important moments trigger memory animations
   - Memories influence future conversations
   - Emotional memories are stronger

### Emotion Color Guide

```css
/* Character emotion gradients */
.happy    { background: linear-gradient(135deg, #F6D365, #FDA085); }
.sad      { background: linear-gradient(135deg, #4FACFE, #00F2FE); }
.angry    { background: linear-gradient(135deg, #FA709A, #FEE140); }
.curious  { background: linear-gradient(135deg, #A8EDEA, #FED6E3); }
.excited  { background: linear-gradient(135deg, #F093FB, #F5576C); }
.neutral  { background: linear-gradient(135deg, #E0C3FC, #8EC5FC); }
```

## Advanced Features

### Memory System

:::note[]
Characters form memories during conversations:

- **Visual Feedback**: Sparkle animation when memory forms
- **Persistence**: Memories carry across sessions
- **Influence**: Past memories affect responses
- **Types**:
  - Factual memories (things you tell them)
  - Emotional memories (how interactions felt)
  - Preference memories (your likes/dislikes)
:::

### Control Tokens

Characters may use special control tokens that trigger UI effects:

| Token | Effect | Example |
|-------|--------|---------|
| `<happy>` | Emotion shift to happy | Character becomes cheerful |
| `<thinking>` | Extended typing indicator | Deep contemplation |
| `<memory>` | Memory formation animation | Important moment |
| `<excited>` | Particle burst effect | High enthusiasm |

### Proactive Agents

Some characters can initiate conversations:

```javascript
// Characters may send messages like:
"I've been thinking about what you said earlier..."
"Something interesting just occurred to me!"
"How has your day been since we last spoke?"
```

### Multi-Character Scenes

Advanced feature for character interactions:

1. **Character Switching**: Seamlessly switch between characters
2. **Group Conversations**: Multiple characters in one chat
3. **Relationship Dynamics**: See how characters interact

## UI Customization

### Theme Options

While the default glassmorphism theme is stunning, you can customize:

```javascript
// In your .env file
REACT_APP_THEME=glass      // Default glassmorphism
REACT_APP_THEME=minimal    // Clean, minimal design
REACT_APP_THEME=dark       // Pure dark mode
REACT_APP_THEME=vibrant    // Extra colorful
```

### Performance Settings

For different devices:

```javascript
// Reduce animations on slower devices
REACT_APP_REDUCED_MOTION=true

// Disable particle effects
REACT_APP_PARTICLES=false

// Simple message rendering
REACT_APP_SIMPLE_MODE=true
```

## Mobile Experience

The client is fully responsive with mobile-specific features:

### Touch Gestures
- **Swipe left**: Return to character selection
- **Long press**: Message options (copy, react)
- **Pull to refresh**: Reload conversation

### Mobile Optimizations
- Optimized animations for battery life
- Reduced particle count
- Efficient WebSocket reconnection
- Offline message queue

## Troubleshooting

### Common Issues

<details>
<summary><strong>Messages not sending</strong></summary>

1. Check if inference server is running
2. Verify WebSocket connection in console
3. Try refreshing the page
4. Check browser console for errors

</details>

<details>
<summary><strong>Emotions not updating</strong></summary>

1. Ensure character has emotion support
2. Check if control tokens are enabled
3. Verify adapter is loaded correctly
4. Look for console warnings

</details>

<details>
<summary><strong>Slow performance</strong></summary>

1. Enable performance mode in settings
2. Reduce animation quality
3. Clear conversation history
4. Use Chrome/Firefox for best performance

</details>

### Browser Compatibility

Best experience on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile Safari (iOS 14+)
- Chrome Mobile

## Pro Tips

<div style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '2rem', borderRadius: '12px', color: 'white', marginBottom: '2rem'}}>
  <h3 style={{marginTop: 0}}>Master the Client</h3>
  <ol>
    <li><strong>Use Markdown</strong>: Format your messages with **bold**, *italic*, and `code`</li>
    <li>
      <strong>Keyboard Shortcuts</strong>:
      <ul>
        <li>Ctrl/Cmd + Enter: Send message</li>
        <li>Esc: Return to home</li>
        <li>Ctrl/Cmd + K: Quick character switch</li>
      </ul>
    </li>
    <li><strong>Easter Eggs</strong>: Try typing "show me magic" to any character!</li>
    <li><strong>Memory Triggers</strong>: Share personal details to form stronger memories</li>
    <li><strong>Emotion Influence</strong>: Your tone affects character emotions</li>
  </ol>
</div>

## Fun Interactions

### Mini-Games

Some characters support interactive experiences:

- **20 Questions**: Play guessing games
- **Story Building**: Collaborative storytelling
- **Riddles**: Solve character-specific puzzles
- **Trivia**: Test your knowledge of their world

### Special Commands

Try these with any character:

```
/stats - Show character statistics
/memory - View formed memories
/mood - Check detailed emotional state
/help - Get character-specific tips
```

## Best Practices

### For Best Conversations

1. **Be Specific**: Detailed questions get detailed answers
2. **Show Interest**: Characters respond to engagement
3. **Build Rapport**: Relationships develop over time
4. **Explore Deeply**: Don't just chat - discover their stories
5. **React Naturally**: Your emotions influence theirs

### For Best Performance

1. **Regular Cleanup**: Clear old conversations monthly
2. **Update Regularly**: Keep client updated for new features
3. **Report Issues**: Help us improve the experience
4. **Share Feedback**: Your input shapes development

## What's Next?

Now that you've mastered the client:

1. **[Advanced Features](./advanced-features)** - Unlock hidden capabilities

## 🚀 Production Deployment

For production deployment:

1. Build the client:
   ```bash
   npm run build
   ```

2. The build output will be in the `build/` directory

3. Serve using your preferred static hosting solution (Nginx, Apache, Vercel, Netlify, etc.)

## 🔧 Current Integration Status

### Chat Interface
The chat interface is currently a demonstration UI. Full integration requires:
- Connection to real inference API endpoints
- Loading actual trained character models
- Integration with RuntimePromptConstructor
- Session management with conversation history

### Director's View
The Director's View shows real-time monitoring capabilities but currently uses simulated data. Full integration requires:
- Connection to actual inference events
- Real memory formation tracking
- Emotion state updates from control tokens
- Triple-head metrics from NarrativeLLM

See the [Director's View Integration Guide](./directors-view-integration) for detailed integration plans.

## 📝 Contributing

---
:::tip[🌈 Enjoy the Magic!]
<div>
  <p>You're ready to have amazing conversations with AI characters. The beautiful interface is just the beginning - the real magic is in the connections you'll make!</p>
  <a href="/" style={{background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white', padding: '0.75rem 2rem', borderRadius: '25px', textDecoration: 'none', display: 'inline-block', marginTop: '1rem'}}>
    <span style={{fontSize: '1.2rem'}}>Start Chatting →</span>
  </a>
</div> 
:::
# R7-2: Real-Time Performance Mode
Status: **Todo**
Ring: R7
Created: 2025-01-14
---

## Goal
Create a live performance system within the unified React+FastAPI platform that enables real-time character puppeteering for streaming, gaming, and interactive entertainment, with console-quality responsiveness and professional-grade controls.

## Context
**Post-Migration**: This task assumes completion of R6-3.1, R6-3.2, R6-3.3 (Architecture Migration) and R6-9 (Multimodal Studio Production Features)

Content creators need characters that can respond instantly to live situations. This system transforms static characters into performance instruments within the Dreamcast platform, enabling streamers, VTubers, and live performers to embody AI characters while maintaining personality consistency.

**Console-Quality Performance**: Real-time character performance that rivals AAA game responsiveness, with professional-grade React controls and seamless platform integration.

## Acceptance Criteria

### Core Performance Engine (FastAPI)
- [ ] **Real-time Inference**: <100ms response time optimization
- [ ] **Streaming Generation**: Word-by-word output via WebSocket
- [ ] **Performance Mode**: Reduced context RuntimePromptConstructor
- [ ] **Load Balancing**: Graceful degradation under high load
- [ ] **Edge Deployment**: Optional edge server deployment for streamers

### React Performance Dashboard
- [ ] **Live Performance Interface**: Real-time character control dashboard
- [ ] **Emotion Wheel**: Interactive emotion control with visual feedback
- [ ] **Quick Actions**: Customizable quick response buttons
- [ ] **Character Stats**: Live personality metrics and mood display
- [ ] **Director Mode**: Override controls for breaking character when needed

### Audience Integration (React + FastAPI)
- [ ] **Chat Command Parsing**: Real-time audience interaction processing
- [ ] **Crowd Decision Making**: Polling system for audience-driven choices
- [ ] **Audience Relationship Tracking**: Persistent viewer relationship memory
- [ ] **Donation Triggers**: Special interactions for donations/bits
- [ ] **Raid/Host Responses**: Automated behavioral responses to platform events

### Streaming Platform Integration
- [ ] **OBS Plugin**: Character visualization overlay plugin
- [ ] **Streamlabs Integration**: Native Streamlabs overlay support
- [ ] **Twitch API**: Direct Twitch chat and event integration
- [ ] **YouTube Live**: YouTube Live chat and superchat integration
- [ ] **Platform Webhooks**: Generic webhook system for other platforms

### Recording & Analytics (React)
- [ ] **Session Recording**: Full state capture with timeline
- [ ] **Performance Replay**: Timeline scrubbing and playback controls
- [ ] **Highlight Generation**: AI-powered highlight reel creation
- [ ] **Training Data Export**: Export performances for character improvement
- [ ] **Performance Analytics**: Detailed metrics and improvement suggestions

## Technical Architecture Design

### React Performance Dashboard
```typescript
const LivePerformanceDashboard: React.FC = () => {
  const [character, setCharacter] = useState<Character>();
  const [performanceState, setPerformanceState] = useState<PerformanceState>();
  const [audienceMetrics, setAudienceMetrics] = useState<AudienceMetrics>();
  const [isLive, setIsLive] = useState(false);
  const wsRef = useRef<WebSocket>();
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/live-performance');
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'performance_state':
          setPerformanceState(data.state);
          break;
        case 'audience_metrics':
          setAudienceMetrics(data.metrics);
          break;
        case 'chat_command':
          handleChatCommand(data.command);
          break;
      }
    };
    
    return () => ws.close();
  }, []);
  
  const handleEmotionChange = (emotion: EmotionState) => {
    wsRef.current?.send(JSON.stringify({
      type: 'emotion_override',
      emotion: emotion,
      intensity: emotion.intensity
    }));
  };
  
  const triggerQuickAction = (action: QuickAction) => {
    wsRef.current?.send(JSON.stringify({
      type: 'quick_action',
      action: action.id,
      context: performanceState?.context
    }));
  };
  
  return (
    <div className="live-performance-dashboard">
      <PerformanceStatusPanel 
        isLive={isLive}
        character={character}
        performanceState={performanceState}
      />
      <EmotionWheelControl 
        currentEmotion={performanceState?.emotion}
        onEmotionChange={handleEmotionChange}
      />
      <QuickActionsPanel 
        actions={character?.quickActions || []}
        onActionTrigger={triggerQuickAction}
      />
      <AudienceInteractionPanel 
        metrics={audienceMetrics}
        onPollCreate={createAudiencePoll}
      />
      <DirectorModeControls 
        onBreakCharacter={handleBreakCharacter}
        onResumeCharacter={handleResumeCharacter}
      />
    </div>
  );
};
```

### FastAPI Performance Engine
```python
class LivePerformanceEngine:
    """Real-time character performance engine"""
    
    def __init__(self):
        self.performance_sessions = {}
        self.streaming_generators = {}
        self.audience_trackers = {}
        self.websocket_manager = WebSocketManager()
        
    async def start_performance_session(self, character_id: str, streamer_id: str):
        """Start a live performance session"""
        
        # Initialize performance state
        performance_state = PerformanceState(
            character_id=character_id,
            streamer_id=streamer_id,
            start_time=datetime.utcnow(),
            emotion_state=EmotionState(),
            audience_context=AudienceContext(),
            performance_mode=True
        )
        
        # Set up streaming generator with performance optimizations
        generator = StreamingCharacterGenerator(
            character_id=character_id,
            context_window=512,  # Reduced for performance
            response_timeout=100,  # 100ms max
            cache_enabled=True
        )
        
        self.performance_sessions[streamer_id] = performance_state
        self.streaming_generators[streamer_id] = generator
        
        # Start audience tracking
        audience_tracker = AudienceTracker(streamer_id)
        self.audience_trackers[streamer_id] = audience_tracker
        
        return performance_state
    
    async def process_chat_command(self, streamer_id: str, command: ChatCommand):
        """Process audience chat command in real-time"""
        
        performance_state = self.performance_sessions.get(streamer_id)
        if not performance_state:
            return
        
        # Parse command and update audience context
        audience_context = await self.parse_audience_command(command)
        performance_state.audience_context.update(audience_context)
        
        # Generate character response
        generator = self.streaming_generators[streamer_id]
        
        async for response_chunk in generator.generate_streaming_response(
            command.text,
            performance_state.to_context()
        ):
            # Stream response to performance dashboard
            await self.websocket_manager.send_to_session(streamer_id, {
                'type': 'character_response',
                'chunk': response_chunk,
                'command_id': command.id
            })
    
    async def handle_emotion_override(self, streamer_id: str, emotion: EmotionState):
        """Handle real-time emotion override from performer"""
        
        performance_state = self.performance_sessions.get(streamer_id)
        if not performance_state:
            return
        
        # Apply emotion override
        performance_state.emotion_state = emotion
        
        # Update character generator context
        generator = self.streaming_generators[streamer_id]
        await generator.update_emotion_context(emotion)
        
        # Notify dashboard of state change
        await self.websocket_manager.send_to_session(streamer_id, {
            'type': 'performance_state',
            'state': performance_state.to_dict()
        })
```

### Streaming Platform Integration
```python
class StreamingPlatformIntegration:
    """Integration with streaming platforms"""
    
    def __init__(self):
        self.twitch_client = TwitchClient()
        self.youtube_client = YouTubeClient()
        self.platform_webhooks = {}
        
    async def connect_twitch_channel(self, channel_name: str, streamer_id: str):
        """Connect to Twitch channel for live integration"""
        
        # Set up IRC connection for chat
        await self.twitch_client.connect_to_chat(channel_name)
        
        # Set up EventSub for channel events
        await self.twitch_client.subscribe_to_events(channel_name, [
            'channel.follow',
            'channel.subscribe',
            'channel.cheer',
            'channel.raid'
        ])
        
        # Forward events to performance engine
        self.twitch_client.on_chat_message = lambda msg: self.forward_chat_message(streamer_id, msg)
        self.twitch_client.on_channel_event = lambda event: self.forward_channel_event(streamer_id, event)
    
    async def forward_chat_message(self, streamer_id: str, message: TwitchMessage):
        """Forward Twitch chat message to performance engine"""
        
        chat_command = ChatCommand(
            id=message.id,
            text=message.text,
            user=message.user,
            platform='twitch',
            timestamp=message.timestamp
        )
        
        await self.performance_engine.process_chat_command(streamer_id, chat_command)
    
    async def create_obs_overlay(self, streamer_id: str, overlay_config: OverlayConfig):
        """Create OBS overlay for character visualization"""
        
        overlay_html = self.generate_overlay_html(overlay_config)
        
        # Save overlay to static files
        overlay_path = f"overlays/{streamer_id}/character_overlay.html"
        await self.save_overlay_file(overlay_path, overlay_html)
        
        # Return URL for OBS browser source
        return f"https://api.dreamcast.dev/overlays/{streamer_id}/character_overlay.html"
```

## Implementation Notes
```text
• Performance Optimizations:
  - Reduced context windows (512 tokens) for speed
  - Response caching for common interactions
  - Predictive pre-generation for likely responses
  - Edge deployment options for ultra-low latency
  
• React Architecture:
  - Real-time dashboard with WebSocket updates
  - Emotion wheel with visual feedback
  - Customizable quick action buttons
  - Professional-grade performance controls
  
• Platform Integration:
  - Native Twitch/YouTube API integration
  - OBS plugin for overlay visualization
  - Webhook system for custom platforms
  - Real-time event processing
  
• Console-Quality Features:
  - <100ms response times
  - Graceful degradation under load
  - Professional streaming tool integration
  - Comprehensive performance analytics
```

## TDD Instructions
- **Performance Tests**: Test real-time response generation and latency
- **React Tests**: Test performance dashboard and emotion controls
- **API Tests**: Test FastAPI streaming endpoints and WebSocket connections
- **Integration Tests**: Test streaming platform integrations and overlays
- **Load Tests**: Test performance under high audience load

## Checklist / Steps
1. **Build FastAPI performance engine** with streaming optimization
2. **Create React performance dashboard** with real-time controls
3. **Implement emotion wheel interface** with visual feedback
4. **Add WebSocket communication** for real-time updates
5. **Create streaming platform integrations** (Twitch, YouTube)
6. **Build OBS overlay system** with character visualization
7. **Implement audience interaction processing** and command parsing
8. **Add session recording** and replay functionality
9. **Create performance analytics** and improvement metrics
10. **Implement edge deployment** options for streamers
11. **Add comprehensive testing** for all performance features
12. **Create documentation** and streamer guides

## References
- Depends on: R6-9 (Multimodal Studio Production Features)
- Enhances: R7-3 (Analytics Dashboard - performance metrics)
- Integrates with: R7-4 (Integration Ecosystem - streaming platforms)
- Architecture: See overview.mdc architecture diagram
- Platform Integration: React+FastAPI unified architecture
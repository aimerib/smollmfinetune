# R7-3: Character Analytics Dashboard
Status: **Todo**
Ring: R7
Created: 2025-01-14
---

## Goal
Build a comprehensive analytics platform within the unified React+FastAPI platform that tracks character performance, user engagement, personality consistency, and conversation quality over time, providing creators with Netflix-level insights for character improvement.

## Context
**Post-Migration**: This task assumes completion of R6-3.1, R6-3.2, R6-3.3 (Architecture Migration) and R7-2 (Real-Time Performance Mode)

Creators currently train characters blindly, hoping they improve. This React-based analytics dashboard provides Netflix-level analytics about how characters perform in real conversations, where they succeed or fail, and how they drift over time within the Dreamcast platform.

**Console-Quality Analytics**: Professional-grade analytics dashboard that rivals commercial creative tools, with real-time insights and actionable recommendations for character improvement.

## Acceptance Criteria

### Core Metrics Collection (FastAPI)
- [ ] **Conversation Quality Scoring**: Coherence, engagement, consistency metrics
- [ ] **User Satisfaction Metrics**: Session length, return rate, ratings tracking
- [ ] **Personality Drift Tracking**: Real-time monitoring against Big Five baseline
- [ ] **Memory Formation Analytics**: Success rate and recall accuracy metrics
- [ ] **Control Token Analytics**: Usage patterns and effectiveness measurement
- [ ] **Performance Metrics**: Response time and system performance tracking

### React Analytics Dashboard
- [ ] **Real-time Metrics Display**: Live conversation and engagement metrics
- [ ] **Interactive Visualizations**: Charts, graphs, and heatmaps for data exploration
- [ ] **Customizable Views**: Personalized dashboard layouts for different use cases
- [ ] **Drill-down Analysis**: Detailed analysis of specific metrics and time periods
- [ ] **Export Functionality**: Export analytics data for external analysis

### Engagement Analytics (React)
- [ ] **User Conversation Heatmaps**: Visual representation of user activity patterns
- [ ] **Topic Clustering**: AI-powered conversation theme analysis and visualization
- [ ] **Dropout Point Analysis**: Identification and visualization of conversation end points
- [ ] **Sentiment Analysis**: Real-time user sentiment tracking throughout conversations
- [ ] **Character Favorability**: Trending favorability scores with historical data
- [ ] **Virality Metrics**: Tracking shares, recommendations, and character popularity

### Character Health Monitoring (React + FastAPI)
- [ ] **Personality Consistency Dashboard**: Real-time consistency scores with deviation alerts
- [ ] **Goal Achievement Tracking**: Visual tracking of character goal pursuit
- [ ] **Relationship Dynamics**: Interactive visualization of character relationships
- [ ] **Linguistic Metrics**: Vocabulary diversity and linguistic pattern analysis
- [ ] **Emotional Range Analysis**: Utilization analysis of emotional spectrum
- [ ] **Memory Coherence Monitoring**: Contradiction detection and coherence scoring

### A/B Testing Framework (React)
- [ ] **Experiment Management**: React interface for creating and managing A/B tests
- [ ] **Statistical Analysis**: Real-time statistical significance calculation
- [ ] **Winner Selection**: Automated winner selection with confidence intervals
- [ ] **Test Visualization**: Visual representation of test results and performance
- [ ] **Automatic Deployment**: Automated deployment of winning variants

## Technical Architecture Design

### React Analytics Dashboard
```typescript
const CharacterAnalyticsDashboard: React.FC = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData>();
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['engagement', 'quality']);
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');
  const [characterFilter, setCharacterFilter] = useState<string>('all');
  const wsRef = useRef<WebSocket>();
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/analytics');
    wsRef.current = ws;
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'metrics_update':
          setAnalyticsData(prev => ({
            ...prev,
            realTimeMetrics: data.metrics
          }));
          break;
        case 'personality_drift_alert':
          handlePersonalityDriftAlert(data.alert);
          break;
        case 'engagement_spike':
          handleEngagementSpike(data.spike);
          break;
      }
    };
    
    return () => ws.close();
  }, []);
  
  const handleMetricDrilldown = (metric: string, timePoint: Date) => {
    // Navigate to detailed metric analysis
    const detailView = {
      metric,
      timePoint,
      filters: { character: characterFilter, timeRange }
    };
    
    setAnalyticsData(prev => ({
      ...prev,
      detailView
    }));
  };
  
  return (
    <div className="analytics-dashboard">
      <DashboardHeader 
        timeRange={timeRange}
        onTimeRangeChange={setTimeRange}
        characterFilter={characterFilter}
        onCharacterFilterChange={setCharacterFilter}
      />
      <MetricsSummaryPanel 
        data={analyticsData?.summary}
        selectedMetrics={selectedMetrics}
        onMetricToggle={handleMetricToggle}
      />
      <ConversationHeatmap 
        data={analyticsData?.engagementData}
        onCellClick={handleMetricDrilldown}
      />
      <PersonalityDriftChart 
        data={analyticsData?.personalityData}
        onAlertClick={handlePersonalityDriftAlert}
      />
      <TopicClusteringVisualization 
        data={analyticsData?.topicData}
        onTopicClick={handleTopicDrilldown}
      />
      <ABTestingPanel 
        experiments={analyticsData?.experiments}
        onExperimentCreate={createNewExperiment}
      />
    </div>
  );
};
```

### FastAPI Analytics Engine
```python
class CharacterAnalyticsEngine:
    """Comprehensive analytics engine for character performance"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.conversation_analyzer = ConversationAnalyzer()
        self.personality_tracker = PersonalityTracker()
        self.engagement_analyzer = EngagementAnalyzer()
        self.websocket_manager = WebSocketManager()
        
    async def collect_conversation_metrics(self, conversation_id: str):
        """Collect comprehensive conversation metrics"""
        
        conversation = await self.get_conversation(conversation_id)
        
        metrics = {
            'quality_score': await self.calculate_quality_score(conversation),
            'engagement_score': await self.calculate_engagement_score(conversation),
            'personality_consistency': await self.check_personality_consistency(conversation),
            'memory_formation': await self.analyze_memory_formation(conversation),
            'sentiment_progression': await self.analyze_sentiment_progression(conversation),
            'topic_coherence': await self.analyze_topic_coherence(conversation)
        }
        
        # Store metrics for historical analysis
        await self.store_conversation_metrics(conversation_id, metrics)
        
        # Check for alerts
        alerts = await self.check_metric_alerts(metrics)
        if alerts:
            await self.broadcast_alerts(alerts)
        
        return metrics
    
    async def analyze_personality_drift(self, character_id: str, time_window: str = '24h'):
        """Analyze character personality drift over time"""
        
        conversations = await self.get_recent_conversations(character_id, time_window)
        
        personality_scores = []
        for conv in conversations:
            score = await self.personality_tracker.analyze_conversation(conv)
            personality_scores.append({
                'timestamp': conv.timestamp,
                'scores': score,
                'conversation_id': conv.id
            })
        
        # Calculate drift from baseline
        baseline = await self.get_character_baseline_personality(character_id)
        drift_analysis = await self.calculate_personality_drift(personality_scores, baseline)
        
        # Check for significant drift
        if drift_analysis['max_drift'] > 0.3:  # 30% drift threshold
            await self.create_personality_drift_alert(character_id, drift_analysis)
        
        return drift_analysis
    
    async def generate_engagement_heatmap(self, character_id: str, time_range: str):
        """Generate user engagement heatmap data"""
        
        conversations = await self.get_conversations_in_range(character_id, time_range)
        
        # Group conversations by hour and day
        heatmap_data = {}
        for conv in conversations:
            hour = conv.timestamp.hour
            day = conv.timestamp.strftime('%Y-%m-%d')
            
            if day not in heatmap_data:
                heatmap_data[day] = {}
            if hour not in heatmap_data[day]:
                heatmap_data[day][hour] = []
            
            heatmap_data[day][hour].append({
                'engagement_score': conv.engagement_score,
                'duration': conv.duration,
                'user_satisfaction': conv.user_satisfaction
            })
        
        # Calculate aggregated metrics for each time slot
        aggregated_heatmap = {}
        for day, hours in heatmap_data.items():
            aggregated_heatmap[day] = {}
            for hour, conversations in hours.items():
                aggregated_heatmap[day][hour] = {
                    'avg_engagement': np.mean([c['engagement_score'] for c in conversations]),
                    'conversation_count': len(conversations),
                    'avg_duration': np.mean([c['duration'] for c in conversations]),
                    'avg_satisfaction': np.mean([c['user_satisfaction'] for c in conversations])
                }
        
        return aggregated_heatmap
```

### A/B Testing Framework
```python
class ABTestingFramework:
    """A/B testing framework for character optimization"""
    
    def __init__(self):
        self.experiment_manager = ExperimentManager()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.variant_deployer = VariantDeployer()
        
    async def create_character_experiment(self, experiment_config: ExperimentConfig):
        """Create new A/B test for character variants"""
        
        experiment = {
            'id': generate_experiment_id(),
            'name': experiment_config.name,
            'character_id': experiment_config.character_id,
            'variants': experiment_config.variants,
            'success_metrics': experiment_config.success_metrics,
            'traffic_split': experiment_config.traffic_split,
            'duration': experiment_config.duration,
            'start_date': datetime.utcnow(),
            'status': 'active'
        }
        
        # Deploy variants
        for variant in experiment['variants']:
            await self.variant_deployer.deploy_variant(variant)
        
        # Start traffic splitting
        await self.start_traffic_splitting(experiment)
        
        return experiment
    
    async def analyze_experiment_results(self, experiment_id: str):
        """Analyze A/B test results with statistical significance"""
        
        experiment = await self.get_experiment(experiment_id)
        results = {}
        
        for variant in experiment['variants']:
            variant_metrics = await self.collect_variant_metrics(variant['id'])
            results[variant['id']] = {
                'metrics': variant_metrics,
                'sample_size': variant_metrics['conversation_count'],
                'conversion_rate': variant_metrics['success_rate']
            }
        
        # Calculate statistical significance
        significance_results = await self.statistical_analyzer.calculate_significance(results)
        
        # Determine winner
        winner = await self.determine_experiment_winner(results, significance_results)
        
        return {
            'experiment_id': experiment_id,
            'results': results,
            'significance': significance_results,
            'winner': winner,
            'confidence_level': significance_results['confidence']
        }
```

### Real-time Metrics Streaming
```typescript
const RealTimeMetricsStream: React.FC = () => {
  const [liveMetrics, setLiveMetrics] = useState<LiveMetrics>();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  
  useEffect(() => {
    const eventSource = new EventSource('/api/analytics/stream');
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'conversation_metrics':
          setLiveMetrics(prev => ({
            ...prev,
            conversations: data.metrics
          }));
          break;
        case 'personality_drift_alert':
          setAlerts(prev => [data.alert, ...prev.slice(0, 9)]); // Keep last 10 alerts
          break;
        case 'engagement_spike':
          handleEngagementSpike(data);
          break;
      }
    };
    
    return () => eventSource.close();
  }, []);
  
  return (
    <div className="real-time-metrics">
      <LiveMetricsPanel metrics={liveMetrics} />
      <AlertsPanel alerts={alerts} onAlertDismiss={handleAlertDismiss} />
      <EngagementSpikesPanel onSpikeClick={handleSpikeAnalysis} />
    </div>
  );
};
```

## Implementation Notes
```text
• React Analytics Architecture:
  - Professional-grade dashboard with interactive visualizations
  - Real-time updates via WebSocket and Server-Sent Events
  - Customizable dashboard layouts and metric selection
  - Responsive design for desktop, tablet, and mobile
  
• Analytics Engine:
  - Time-series database for historical metrics storage
  - Real-time stream processing for live analytics
  - Machine learning models for predictive analytics
  - Statistical analysis for A/B testing and significance
  
• Data Pipeline:
  - Event streaming from conversation engines
  - Batch processing for historical analysis
  - Real-time aggregation for live dashboards
  - Privacy-compliant data collection and storage
  
• Console-Quality Features:
  - Netflix-level analytics depth and presentation
  - Actionable insights and recommendations
  - Professional-grade data visualization
  - Comprehensive A/B testing framework
```

## TDD Instructions
- **Analytics Tests**: Test metrics collection and calculation algorithms
- **React Tests**: Test dashboard components and interactive visualizations
- **API Tests**: Test FastAPI analytics endpoints and streaming
- **A/B Testing Tests**: Test experiment framework and statistical analysis
- **Integration Tests**: Test end-to-end analytics pipeline

## Checklist / Steps
1. **Implement analytics data collection** pipeline with FastAPI
2. **Create React analytics dashboard** with interactive visualizations
3. **Build real-time metrics streaming** via WebSocket/SSE
4. **Implement personality drift tracking** and alerting system
5. **Create engagement analytics** with heatmaps and clustering
6. **Build A/B testing framework** with statistical analysis
7. **Add predictive analytics** for user behavior and character performance
8. **Implement alert system** for critical metric changes
9. **Create export functionality** for external analysis tools
10. **Add comprehensive testing** for all analytics features
11. **Implement performance optimization** for large-scale analytics
12. **Create documentation** and user guides

## References
- Depends on: R7-2 (Real-Time Performance Mode)
- Enhances: R6-9 (Multimodal Studio Production Features - performance insights)
- Integrates with: R7-4 (Integration Ecosystem - analytics APIs)
- Architecture: See overview.mdc architecture diagram
- Platform Integration: React+FastAPI unified architecture
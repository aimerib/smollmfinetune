# User Guide

This guide covers all the essential features of the Character Creation Devkit for content creators.

## Overview

The Character Creation Devkit is a comprehensive platform for creating, training, and deploying AI characters. The workflow follows these main stages:

1. **World Building** - Create rich, interconnected worlds
2. **Character Creation** - Design compelling character personalities
3. **Dataset Generation** - Create training data for your characters
4. **Model Training** - Fine-tune AI models with your character data
5. **Quality Assessment** - Evaluate character consistency and performance

## World Management

### Creating Worlds

Navigate to **World Management** (🌍) to create and edit worlds:

1. Click "New World" to create a new world
2. Add world lore in three categories:
   - **Facts**: Key-value pairs defining world truths
   - **Timeline**: Historical events and their dates
   - **Places**: Locations with NPCs and events

### Best Practices for World Building

- Define clear facts that characters must respect
- Create historical events that affect character backgrounds
- Include diverse locations for character interactions
- Use AI assistance to expand and enhance world lore

## Character Creation

### Creation Methods

#### Conversational Builder (Recommended)
- Navigate to **Conversational Builder** (🗨️)
- Answer AI questions to naturally develop your character
- Watch real-time character synthesis in the preview panel
- Most intuitive method for new creators

#### Character Management
- Navigate to **Character Management** (📋)
- Use detailed tabs for precise character editing:
  - **Profile**: Name, description, backstory, appearance
  - **Personality**: Big Five traits with interactive radar chart
  - **Goals & Relationships**: Character motivations and connections
  - **Examples**: Sample conversations demonstrating character voice

#### Character Upload
- Navigate to **Character Upload** (📁)
- Import existing SillyTavern cards
- Automatically convert and enhance with platform features

### Big Five Personality Traits

Every character has five personality dimensions (0.0-1.0):

- **Openness**: Creativity, curiosity, openness to new experiences
- **Conscientiousness**: Organization, discipline, reliability
- **Extraversion**: Social energy, assertiveness, enthusiasm
- **Agreeableness**: Compassion, cooperation, trust in others
- **Neuroticism**: Emotional reactivity, anxiety, mood stability

Use the interactive radar chart to visualize and adjust personality traits.

## Dataset Studio

### Generation Methods

Navigate to **Dataset Studio** (🎨) for training data creation:

#### Interactive Generation (Recommended)
- Collaborative batch-by-batch generation with human oversight
- Review and accept/reject each sample
- AI learns from your preferences over time
- Highest quality results

#### Experimental Methods
- **Fast Mode**: Template-based rapid generation
- **Slow Mode**: AI-curated high-quality generation
- **Factual Q&A**: Knowledge-focused conversations

### Quality Optimization

- Monitor dataset statistics and quality scores
- Use preference learning to improve AI suggestions
- Aim for 300-1000 high-quality training samples
- Ensure diversity across conversation types

## Training Pipeline

### Training Configuration

Navigate to **Training Config** (⚙️) to set up model training:

#### Supervised Fine-Tuning (SFT)
- **Epochs**: 2-5 training passes through data
- **Learning Rate**: 1e-5 to 5e-5 (conservative recommended)
- **Batch Size**: Limited by available GPU memory

#### Reinforcement Learning (RLHF)
- Optional second-stage training using preference data
- Enable if you have 100+ preference pairs
- Algorithm: GRPO (recommended) or PPO
- Refines character behavior based on creator preferences

### Monitoring Training

Switch to **Training Dashboard** (📊) to monitor progress:

- Watch real-time loss curves and quality metrics
- Monitor personality alignment and lore adherence scores
- Use pause/resume controls as needed
- Test intermediate checkpoints

## Model Testing & Evaluation

### Interactive Testing

Navigate to **Model Testing** (🧪) to test your trained character:

- Select your trained model adapter
- Try various conversation prompts
- Test personality consistency across different scenarios
- Evaluate character knowledge and world integration

### Quality Analysis

Navigate to **Model Comparison** (🔍) for comprehensive evaluation:

- Run personality drift analysis to compare authored vs generated traits
- Check personality alignment and lore adherence metrics
- Compare multiple model versions side-by-side
- Identify areas for improvement

## Advanced Features

### Control Tokens

Fine-grained content control without retraining:

- **Mood Tokens**: `<mood_happy>`, `<mood_angry>`, `<mood_sad>`
- **Action Tokens**: `<stage_whisper>`, `<stage_shout>`, `<stage_laugh>`
- **Scene Tokens**: `<scene_tavern>`, `<scene_night>`, `<scene_outdoor>`
- **Content Tokens**: `<nsfw_soft>`, `<nsfw_explicit>` (where appropriate)

### NSFW Content Handling

For mature content creators:

- Sophisticated content analysis and categorization
- Style-aware generation with appropriate boundaries
- Creator control over content types and limits
- Professional handling of adult content

### Character Intelligence

AI-powered features throughout the platform:

- Real-time character synthesis and archetype detection
- Preference learning that adapts to creator choices
- World-aware character suggestions and integration
- Automated quality assessment and consistency monitoring

## Best Practices

### Character Design
1. Start with clear personality traits using the Big Five model
2. Integrate characters meaningfully into world lore
3. Define specific goals and motivations
4. Use the conversational builder for natural character discovery

### Training Optimization
1. Generate 500-1000 high-quality training samples
2. Use interactive generation for best results
3. Monitor personality alignment during training
4. Test character consistency regularly

### Quality Assurance
1. Use personality drift analysis to check consistency
2. Verify lore adherence in character responses
3. Test characters across multiple scenarios
4. Maintain high standards throughout development

## Troubleshooting

### Common Issues

**Character feels generic**:
- Add more specific personality details and quirks
- Use AI suggestions to enhance character depth
- Include distinctive speech patterns or expertise

**Training loss not decreasing**:
- Check dataset quality and size
- Adjust learning rate (try lower values)
- Ensure proper data formatting

**Personality drift**:
- Generate more personality-focused training examples
- Use control tokens to reinforce traits
- Monitor personality alignment metrics

**Poor lore adherence**:
- Strengthen world lore documentation
- Include more world-integrated training examples
- Verify character background aligns with world facts

## Getting Help

For additional assistance:

- **Core Concepts**: Understanding platform fundamentals
- **Getting Started**: Step-by-step first character walkthrough
- **Advanced Features**: Specialized techniques and controls
- **API Reference**: Technical integration details

---

This user guide covers the essential workflows for creating compelling AI characters. For detailed technical information, see the Advanced Features documentation. 
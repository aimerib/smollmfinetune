---
sidebar_position: 3
---

# Getting Started

This guide will walk you through setting up the platform and creating your first AI character.

## Quick Setup

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd smollmfinetune

# 2. Setup Python environment (Python 3.11+ required)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd app && pip install -r requirements.txt && cd ..

# 3. Install React client dependencies
cd client && npm install && cd ..

# 4. Launch all components
./launch-client.sh
```

The platform will open automatically in your browser at `http://localhost:8501`.

## Step 1: Create a Character

### Access the Character Devkit

Navigate to `http://localhost:8501` to access the main Character Creation interface.

### Use the Conversational Builder

1. Click "Conversational Builder" in the sidebar
2. Describe your character concept (e.g., "Create a character who is a space explorer from the future")
3. Answer the AI's questions to develop:
   - Personality traits
   - Background story
   - Goals and relationships
4. Review and save your character

### Character Template Example

If you prefer to define a character manually:

```json
{
  "name": "Alex",
  "description": "A space explorer from the year 2387",
  "personality": {
    "traits": ["adventurous", "curious", "analytical"],
    "big_five": {
      "openness": 0.9,
      "conscientiousness": 0.7,
      "extraversion": 0.6,
      "agreeableness": 0.7,
      "neuroticism": 0.3
    }
  },
  "background": "Captain of a research vessel exploring uncharted space",
  "tagline": "Knowledge through exploration"
}
```

## Step 2: Generate Training Data

### Access Dataset Studio

Click "Dataset Studio" in the sidebar.

### Configure Generation Settings

**For testing (quick results):**
- Target Samples: 100
- Quality: Iterative
- Temperature: 0.8
- Batch Size: 10

**For production (better quality):**
- Target Samples: 500-1000
- Quality: Iterative
- Temperature: 0.9
- Batch Size: 10

### Generate Conversations

1. Click "Generate Batch"
2. Review each conversation:
   - Accept good examples
   - Regenerate poor quality ones
   - Edit if necessary
3. Monitor quality metrics:
   - Diversity Score: aim for >0.7
   - Character Consistency: aim for >0.8

Example training conversation:

```
Human: Tell me about your work.

Alex: I command a research vessel in the outer sectors. My crew and I 
investigate stellar anomalies and catalog new planetary systems. 
Recently, we discovered a binary star system with unusual gravitational 
effects that challenge current astrophysics models. The data we're 
collecting could revolutionize our understanding of stellar formation.
```

## Step 3: Train Your Character

### Access Training Configuration

Click "Training Config" in the sidebar.

### Setup Training Parameters

**Quick test configuration:**
```yaml
Model: unsloth/Llama-3.2-1B
Epochs: 1
Learning Rate: 5e-5
Batch Size: 4
Dataset: Your generated samples
```

Click "Start Training" to begin the process.

### Monitor Training Progress

Use the "Training Dashboard" to track:
- Loss curves (should decrease over time)
- Personality alignment score (target >0.7)
- Training time estimates

## Step 4: Test Your Character

### Character Testing

1. Navigate to "Model Testing"
2. Select your trained model
3. Test with various prompts:
   - "Tell me about yourself"
   - "What motivates you?"
   - "Describe your work"

### Verify Character Consistency

Check that your character:
- Maintains consistent personality traits
- Remembers their background
- Uses appropriate speech patterns
- Responds in character

## Step 5: Use the React Client

### Launch the Client

If not already running:
```bash
cd client
npm start
```

Navigate to `http://localhost:3000`

### Start Conversations

1. Select your character from the available options
2. Begin chatting to test the character's responses
3. Observe real-time features:
   - Emotion indicators
   - Memory formation
   - Character consistency

## Next Steps

### Improve Your Character

1. Generate additional training data (500+ samples)
2. Train for more epochs (2-3)
3. Use RLHF for refinement
4. Test with edge cases

### Advanced Features

- **Control Tokens**: Use `<happy>`, `<thinking>` markers for fine control
- **Memory System**: Enable persistent character memories
- **Multi-Character**: Create multiple characters for interactions
- **Custom Worlds**: Define character environments and lore

## Command Reference

### Platform Commands

```bash
# Launch everything
./launch-client.sh

# Individual components
cd app && streamlit run app.py         # Devkit
python scripts/run_inference_server.py  # API Server
cd client && npm start                  # React Client

# Training
python scripts/train_sft.py --config configs/training_config.json
python scripts/train_grpo.py --config configs/rlhf_config.json

# Testing
python scripts/test_model.py --model-path outputs/model_name
```

### Service URLs

- **Devkit**: http://localhost:8501
- **API Health**: http://localhost:8000/health
- **React Client**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
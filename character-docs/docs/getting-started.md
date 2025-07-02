---
sidebar_position: 3
---

# Getting Started

This guide will walk you through setting up the platform and creating your first AI character using our new modern web interface.

<div style={{background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', padding: '2rem', borderRadius: '12px', color: 'white', marginBottom: '2rem'}}>
  <h2 style={{marginTop: 0}}>🆕 New Web Platform</h2>
  <p>We've completely redesigned the character creation experience with a stunning React interface and powerful FastAPI backend. Experience faster workflows, real-time updates, and a beautiful UI!</p>
</div>

## Quick Setup

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd smollmfinetune

# 2. Launch everything with Docker (recommended)
./start-devkit.sh

# The platform will open automatically at http://localhost:3000
```

### Manual Setup (Alternative)

If you prefer to run services manually:

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend setup (new terminal)
cd client
npm install
npm start

# Redis (new terminal)
docker run -d -p 6379:6379 redis:alpine

# Celery worker (new terminal)
cd backend
celery -A app.celery_app worker --loglevel=info
```

## Step 1: Create Your Account

Navigate to `http://localhost:3000` and you'll see our new stunning interface:

1. Click **"Get Started"** on the landing page
2. Register with your email and password
3. You'll be automatically logged in to the Creator Dashboard

## Step 2: Create a World

Before creating characters, let's build their world:

1. From the Creator Dashboard, click **"Create World"**
2. Navigate to the **World Builder**
3. Fill in the world details across the tabs:
   - **Overview**: Name, description, and setting
   - **Rules**: Physical laws, magic systems, technology level
   - **History**: Major events and timeline
   - **Cultures**: Different societies and their customs
   - **Locations**: Important places in your world
4. Click **"Save World"** when complete

### Example World: Neo-Tokyo 2089

```json
{
  "name": "Neo-Tokyo 2089",
  "description": "A cyberpunk metropolis where technology and tradition clash",
  "setting": "Post-climate disaster Japan, 2089. Sea levels have risen, and humanity lives in vertical mega-cities.",
  "rules": {
    "Technology": "Neural implants are common, AI assistants are standard",
    "Society": "Corporate feudalism with traditional Japanese values",
    "Environment": "Perpetual rain, neon-lit streets, vertical living"
  }
}
```

## Step 3: Create a Character

Now let's bring a character to life in your world:

1. Click **"Create Character"** from the dashboard
2. In the **Character Builder**, you'll see a beautiful 3-step wizard:

### Step 1: Basic Information
- Name your character
- Add a tagline (their essence in one line)
- Select their world
- Write a description

### Step 2: Personality
Use the interactive sliders to set Big Five traits:
- **Openness**: Creativity and curiosity
- **Conscientiousness**: Organization and discipline
- **Extraversion**: Social energy
- **Agreeableness**: Cooperation and trust
- **Neuroticism**: Emotional stability

Watch the **personality radar chart** update in real-time!

### Step 3: Background
- Write their backstory
- Define their goals
- List important relationships
- Add unique traits or quirks

## Step 4: Generate Training Data

Head to the **Dataset Studio** to create conversations:

1. Select your character
2. Configure generation parameters:
   ```
   Target Samples: 500
   Temperature: 0.8
   Batch Size: 10
   Quality Mode: Iterative
   ```
3. Add conversation topics (optional):
   - Character's expertise
   - Emotional scenarios
   - World-specific situations
4. Click **"Start Generation"**
5. Watch real-time progress with our WebSocket updates!

### Live Progress Tracking

The new interface shows:
- Progress bar with percentage
- Current topic being generated
- Real-time conversation count
- Pause/resume controls

## Step 5: Train Your Character

Once dataset generation is complete:

1. Navigate to **Training Config**
2. Select your dataset
3. Configure training:
   ```yaml
   Model: unsloth/Llama-3.2-1B
   Epochs: 3
   Learning Rate: 5e-5
   Batch Size: 4
   ```
4. Click **"Start Training"**
5. Monitor progress in the training dashboard

### Training Dashboard Features

- Live loss curves
- Personality alignment scores
- Estimated time remaining
- GPU utilization metrics
- Pause/resume capabilities

## Step 6: Test Your Character

After training completes:

1. Go to **Character Chat**
2. Select your trained character
3. Start conversing!
4. Watch the emotion indicators change in real-time
5. See memory formation animations when important moments occur

### Testing Prompts

Try these to test different aspects:
- "Tell me about yourself" (personality check)
- "What's your greatest fear?" (emotional depth)
- "Describe your world" (world knowledge)
- "What are your goals?" (motivation check)

## Step 7: Export and Deploy

Ready to share your character?

1. Navigate to **Export Center**
2. Select export format:
   - **Runtime Packet**: For use with our inference engine
   - **GGUF**: For llama.cpp compatibility
   - **HuggingFace**: For model hub upload
3. Configure export settings
4. Click **"Export Character"**

## What's New in v2.0

### UI/UX Improvements
- 🎨 **Stunning Design**: Slate and orange theme with glassmorphism effects
- ⚡ **Real-time Updates**: WebSocket integration for live progress
- 📱 **Fully Responsive**: Works beautifully on all devices
- 🎯 **Intuitive Workflows**: Step-by-step wizards guide you

### Performance Enhancements
- 🚀 **5-10x Faster**: Async operations throughout
- 💾 **Smart Caching**: Redis integration reduces load times
- 🔄 **Background Tasks**: Celery handles long-running operations
- 📊 **Live Metrics**: Real-time performance monitoring

### New Features
- 🌍 **World Builder**: Create rich, detailed worlds
- 📊 **Dataset Studio**: Advanced generation controls
- 🎯 **Training Dashboard**: Professional monitoring tools
- 💬 **WebSocket Progress**: Live updates for all operations

## Tips for Success

<div style={{display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginBottom: '2rem'}}>
  :::tip[World Building]
  <div>
    <ul>
      <li>Start with a clear vision</li>
      <li>Define unique rules that affect characters</li>
      <li>Create rich history for depth</li>
      <li>Add diverse cultures for variety</li>
    </ul>
  </div>
  :::
  :::tip[Character Creation]
  <div>
    <ul>
      <li>Make personality traits distinctive</li>
      <li>Connect backstory to world lore</li>
      <li>Define clear motivations</li>
      <li>Add unique speech patterns</li>
    </ul>
  </div>
  :::
</div>

## Troubleshooting

### Common Issues

<details>
<summary><strong>Can't connect to backend</strong></summary>

Make sure all services are running:
```bash
# Check backend
curl http://localhost:8000/health

# Check Redis
redis-cli ping

# Check frontend
curl http://localhost:3000
```
</details>

<details>
<summary><strong>WebSocket connection failed</strong></summary>

Ensure your `.env` file has correct WebSocket URL:
```
REACT_APP_WS_URL=ws://localhost:8000
```
</details>

<details>
<summary><strong>Generation stuck at 0%</strong></summary>

Check if Celery worker is running:
```bash
ps aux | grep celery
# If not running, start it:
celery -A backend.app.celery_app worker
```
</details>

## Next Steps

Now that you've created your first character:

1. **[Training Guide](./training-guide)** - Deep dive into advanced training
2. **[Client Guide](./client-guide)** - Master the chat interface
3. **[API Reference](./api-reference)** - Integrate with your own apps

---
:::tip[🎉 Congratulations!]
<div>
  <p>You've successfully created your first AI character on our new platform! The modern interface makes the process faster and more enjoyable than ever.</p>
  <a href="./training-guide" style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', padding: '0.75rem 2rem', borderRadius: '25px', textDecoration: 'none', display: 'inline-block', marginTop: '1rem'}}>
    <span style={{fontSize: '1.2rem'}}>Learn Advanced Training →</span>
  </a>
</div> 
:::
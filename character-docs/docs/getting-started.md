# Getting Started

This guide will walk you through setting up the Character Creation Devkit and creating your first AI character from start to finish.

## Installation

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for models and data
- **GPU**: Optional but recommended for faster training

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd smollmfinetune
   ```

2. **Install Dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

3. **Start the Application**
   ```bash
   # Recommended: Use the startup script
   ./startup.sh
   
   # Alternative: Direct Streamlit launch
   streamlit run app.py
   ```

4. **Open Your Browser**
   Navigate to `http://localhost:8501`

### First Launch

When you first open the application, you'll see the main navigation sidebar with several options. The platform automatically creates a default world for you to start with.

## Your First Character: A Complete Walkthrough

Let's create your first character from concept to trained model. We'll create "Aria", a wise but playful magical librarian.

### Step 1: World Setup (Optional)

While you can use the default world, let's create a custom world for our character:

1. **Navigate to World Management** (🌍 in sidebar)
2. **Create New World**:
   - Name: "Mystical Academy"
   - Click "Create World"

3. **Add World Lore**:
   - **Facts Tab**: Add key-value facts:
     - `magic_system`: "Elemental magic through enchanted crystals"
     - `setting`: "Ancient academy floating above the clouds"
     - `technology_level`: "Medieval with magical enhancements"
   
   - **Timeline Tab**: Add historical events:
     - Year: 1200, Event: "Academy founded by the Circle of Mages"
     - Year: 1450, Event: "Great Library construction completed"
   
   - **Places Tab**: Add locations:
     - Name: "Grand Library"
     - Description: "A vast repository of magical knowledge spanning seven floating towers"

4. **Save Changes**: Your world lore is now established

### Step 2: Character Creation

Now let's create our character using the conversational builder:

1. **Navigate to Conversational Builder** (🗨️ in sidebar)

2. **Start the Conversation**:
   The AI will ask: "Let's create your character together. What's their name?"
   
   **You respond**: "Her name is Aria. She's a librarian at a magical academy."

3. **Follow the AI's Questions**:
   The AI will ask follow-up questions. Here are example responses:
   
   - **About her personality**: "She's incredibly wise and knowledgeable, but has a playful side. She loves surprising students with unexpected magical demonstrations while teaching."
   
   - **About her background**: "She's been at the academy for decades, mastering both traditional scholarship and practical magic. She has a special affinity for crystal magic."
   
   - **About her goals**: "She wants to help students discover their magical potential and preserve ancient knowledge for future generations."

4. **Watch the Character Develop**:
   As you answer questions, watch the right panel update with:
   - Personality trait analysis
   - Character archetype detection
   - Voice consistency scoring
   - Training readiness assessment

5. **Complete the Conversation**:
   Continue until the AI indicates the character is well-developed, then click "Save Character"

### Step 3: Character Refinement

Let's enhance our character using the detailed management tools:

1. **Navigate to Character Management** (📋 in sidebar)
2. **Select Aria** from the character list

3. **Review Each Tab**:
   
   **Profile Tab**:
   - Fine-tune the description
   - Add appearance details: "Auburn hair, green eyes, always wears a crystal pendant"
   - Enhance the backstory
   
   **Personality Tab**:
   - Review the Big Five radar chart
   - Adjust traits if needed:
     - Openness: 0.85 (highly curious and creative)
     - Conscientiousness: 0.75 (organized but flexible)
     - Extraversion: 0.60 (friendly but enjoys solitude)
     - Agreeableness: 0.80 (caring and helpful)
     - Neuroticism: 0.25 (calm and stable)
   
   **Goals & Relationships Tab**:
   - Add specific goals:
     - "Help struggling students find their magical strengths"
     - "Catalog rare magical texts before they're lost"
     - "Bridge the gap between theoretical and practical magic"
   
   **Examples Tab**:
   - Add a few example conversations showing her teaching style

4. **Use AI Suggestions**:
   - Click the "✨" buttons for AI-powered enhancement suggestions
   - Accept or modify suggestions that fit your vision

### Step 4: Dataset Generation

Now let's create training data for our character:

1. **Navigate to Dataset Studio** (🎨 in sidebar)
2. **Select Interactive Generation Tab**

3. **Configure Generation**:
   - Target samples: 500 (good starting point)
   - Quality level: Iterative (balance of quality and speed)
   - NSFW handling: Appropriate for your character

4. **Start Generation**:
   - Click "Generate Batch"
   - Review generated samples as they appear
   - Accept good samples, regenerate poor ones
   - The AI learns from your preferences

5. **Monitor Quality**:
   - Watch the real-time statistics
   - Aim for diversity and character consistency
   - Continue until you have 300-500 high-quality samples

### Step 5: Model Training

Time to train your character's AI model:

1. **Navigate to Training Config** (⚙️ in sidebar)

2. **Configure Training**:
   - **SFT Settings**:
     - Epochs: 3 (start conservative)
     - Learning rate: 5e-5 (recommended)
     - Batch size: Based on your GPU (1-4)
   
   - **RLHF Settings** (if you have preference data):
     - Enable if you've marked preferences during generation
     - Algorithm: GRPO (recommended)

3. **Start Training**:
   - Click "Start Training"
   - The system will begin supervised fine-tuning

4. **Monitor Progress**:
   - Switch to Training Dashboard (📊 in sidebar)
   - Watch loss curves and quality metrics
   - Training typically takes 20-60 minutes

### Step 6: Testing Your Character

Let's see how your character performs:

1. **Navigate to Model Testing** (🧪 in sidebar)
2. **Select Your Trained Model**: Choose Aria's latest adapter

3. **Test Conversations**:
   Try these prompts:
   - "Can you help me understand crystal magic?"
   - "I'm struggling with my enchantment studies."
   - "Tell me about the history of the Grand Library."

4. **Evaluate Responses**:
   - Does Aria sound knowledgeable but approachable?
   - Are her responses consistent with her personality?
   - Does she reference world lore appropriately?

### Step 7: Quality Analysis

Assess your character's consistency:

1. **Navigate to Model Comparison** (🔍 in sidebar)
2. **Run Personality Drift Analysis**:
   - Click "Run Drift Analysis"
   - Review the radar chart comparing authored vs generated personality
   - Check individual trait consistency

3. **Review Metrics**:
   - Personality alignment score
   - Lore adherence rating
   - Voice consistency analysis

## Common First-Time Issues

### Character Feels Generic
- **Solution**: Add more specific personality details and unique traits
- **Tip**: Use the conversational builder to discover unique character aspects

### Training Loss Not Decreasing
- **Solution**: Check dataset quality and reduce learning rate
- **Tip**: Generate more diverse training examples

### Character Responses Too Short
- **Solution**: Include longer example conversations in training data
- **Tip**: Use the "Generate additional example" feature

### Personality Drift
- **Solution**: Increase personality-focused training samples
- **Tip**: Use control tokens to reinforce personality traits

## Next Steps

Congratulations! You've created your first AI character. Here's what to explore next:

1. **Advanced Features**:
   - Experiment with control tokens for fine-grained control
   - Try different training configurations
   - Explore multi-character world building

2. **Quality Optimization**:
   - Use RLHF training to refine character behavior
   - Generate more specialized training data
   - Test across diverse scenarios

3. **World Building**:
   - Add more characters to your world
   - Create complex relationship networks
   - Build rich world lore and timelines

4. **Export for Production**:
   - Export runtime packets for deployment
   - Test in different contexts
   - Prepare for integration with game engines

## Getting Help

- **User Guide**: Comprehensive feature documentation
- **Best Practices**: Tips for optimal results
- **Troubleshooting**: Solutions to common issues
- **Community**: Connect with other creators

---

**Ready to dive deeper?** Continue with the [Complete User Guide](user-guide.md) or explore [Advanced Features](advanced-features.md) 
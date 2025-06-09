# 🎭 Character AI Training Studio

A beautiful, intuitive Streamlit application for training custom character AI models using LoRA fine-tuning. Transform SillyTavern character cards into intelligent AI companions with synthetic dataset generation and real-time training monitoring.

## ✨ Features

### 🎯 Core Functionality
- **Character Card Upload**: Support for SillyTavern-compatible JSON character cards
- **Synthetic Dataset Generation**: Automated creation of training data using LM Studio
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with customizable settings
- **Real-time Training Dashboard**: Live monitoring with pause/resume functionality
- **Model Testing**: Interactive testing and comparison of trained models

### 🎨 Beautiful UI
- **Modern Design**: Gradient backgrounds, smooth animations, custom CSS styling
- **Intuitive Navigation**: Clear sidebar navigation with status indicators
- **Responsive Layout**: Works well on different screen sizes
- **Dark Theme**: Easy on the eyes with professional color scheme

### 🚀 Advanced Features
- **Quality Metrics**: Dataset diversity analysis and quality scoring
- **Training Control**: Pause, resume, or stop training at any time
- **Checkpoint Management**: Test intermediate checkpoints during training
- **Smart Recommendations**: Automatic parameter suggestions based on dataset size
- **Progress Tracking**: Real-time loss curves and training statistics

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- **Local Development**: LM Studio (for synthetic data generation on Mac)
- **Cloud Deployment**: GPU with CUDA support (for vLLM acceleration)
- **Fallback**: CPU-only deployment supported

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Easy way (with automatic LM Studio detection):
   ./run-local.sh
   
   # Manual way:
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Cloud Deployment (RunPod)

#### **🚀 Quick Setup (Recommended)**
```bash
curl -sSL https://raw.githubusercontent.com/aimerib/smollmfinetune/main/app/setup-runpod.sh | bash
```

This single command handles everything: system setup, dependencies, tmux session, and app launch.
See [RUNPOD-SETUP.md](RUNPOD-SETUP.md) for detailed instructions.

#### **🐳 Docker Deployment (Alternative)**
1. **Build Docker image**
   ```bash
   docker build -t character-ai-studio .
   ```

2. **Deploy to RunPod**
   - Upload the Docker image to your registry
   - Create a new RunPod instance with the image
   - Expose port 8888
   - The app will automatically use vLLM for high-performance inference
   - Access via the provided URL

## 🚀 **Inference Engines**

The app automatically selects the best inference engine for your environment:

### **vLLM** (Recommended for RunPod/Cloud GPU)
- **Best for**: High-performance GPU deployment  
- **Model**: PocketDoc/Dans-PersonalityEngine-V1.3.0-24b (24B parameters)
- **Performance**: Up to 20x faster than standard inference
- **Requirements**: CUDA GPU, Linux, ~48GB VRAM
- **Auto-detected**: When GPU is available

### **LM Studio** (Recommended for Local Mac Development)
- **Best for**: Local development on macOS
- **Performance**: Good interactive performance
- **Requirements**: LM Studio app running locally
- **Auto-detected**: When LM Studio is accessible


### **Manual Engine Selection**
Set the `INFERENCE_ENGINE` environment variable to force a specific engine:
```bash
export INFERENCE_ENGINE=vllm      # Force vLLM
export INFERENCE_ENGINE=lmstudio  # Force LM Studio
```

## 🧠 **Model Architecture**

The app uses different models optimized for different tasks:

### **Synthetic Data Generation**
- **Cloud (vLLM)**: `PocketDoc/Dans-PersonalityEngine-V1.3.0-24b`
  - 24B parameter model specialized for character personality
  - High-quality, diverse synthetic conversations
  - Requires substantial GPU memory (~48GB VRAM)

- **Local (LM Studio)**: User-configured model
  - Flexible model choice based on your preferences
  - Managed manually through LM Studio interface

### **Model Training & Inference**
- **Target Model**: `HuggingFaceTB/SmolLM2-135M-Instruct`
  - 135M parameter model for efficient fine-tuning
  - Fast training and inference
  - Suitable for character-specific adaptations

This **two-model approach** gives you the best of both worlds:
- 🎯 **Quality**: Large model generates diverse, high-quality training data
- ⚡ **Efficiency**: Small model trains quickly and runs efficiently
- 💰 **Cost-effective**: Only use expensive large model for data generation

### **Performance Summary**
- **Data Generation**: PersonalityEngine-24B → High-quality synthetic conversations
- **Training Target**: SmolLM2-135M → Fast, efficient fine-tuning
- **RunPod Speed**: Up to **60x faster** generation with optimized vLLM batching
- **Memory Usage**: Optimized for 48GB VRAM with 90% utilization
- **Batch Processing**: 8 samples simultaneously for maximum efficiency

> 📖 **See [VLLM-OPTIMIZATION.md](VLLM-OPTIMIZATION.md) for detailed performance improvements**

## 📖 Usage Guide

### 1. Character Upload
- Navigate to "📁 Character Upload"
- Upload your SillyTavern-compatible JSON character card
- Review character information and ensure it looks correct

### 2. Dataset Generation
- Go to "🔍 Dataset Preview"
- Configure generation settings (number of samples, temperature, etc.)
- Click "🚀 Generate Dataset" and wait for completion
- Review sample quality and statistics

### 3. Training Configuration
- Open "⚙️ Training Config"
- Adjust hyperparameters based on recommendations
- Pay attention to overfitting warnings
- Click "🚀 Start Training" to begin

### 4. Monitor Training
- Switch to "📊 Training Dashboard"
- Watch real-time loss curves and metrics
- Use pause/resume controls as needed
- Test checkpoints during training

### 5. Test Your Model
- Visit "🧪 Model Testing"
- Select your trained model
- Try different prompts and generation settings
- Compare multiple model versions

## ⚙️ Configuration

### Hyperparameter Guidelines

| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| **Epochs** | 1-5 | More epochs = longer training, risk of overfitting |
| **Learning Rate** | 1e-5 to 5e-5 | Higher = faster learning, risk of instability |
| **LoRA Rank** | 4-16 | Higher = more parameters, slower training |
| **Batch Size** | 1-4 | Limited by GPU memory |

### Quality Metrics

- **Uniqueness Ratio**: Percentage of unique responses (target: >80%)
- **Template Diversity**: Variety of prompt templates used (target: >0.7)
- **Quality Score**: Combined metric (target: >70%)

## 🎨 Customization

### Styling
The app uses custom CSS with CSS variables for easy theming. Key colors:
- Primary: `#6366f1` (Indigo)
- Secondary: `#8b5cf6` (Purple)
- Accent: `#06b6d4` (Cyan)
- Success: `#10b981` (Emerald)

### Adding New Templates
Edit `utils/dataset.py` to add new synthetic data templates:

```python
self.templates.append((
    "template_name",
    "Your template with {name} and {other_vars}"
))
```

## 🔧 Troubleshooting

### Common Issues

**LM Studio Connection**
- Ensure LM Studio is running and accessible
- Check API endpoints and credentials

**Memory Issues**
- Reduce batch size or model size
- Use gradient accumulation for larger effective batch sizes
- Clear model cache between training runs

**Training Failures**
- Check dataset quality and size
- Verify character card format
- Monitor system resources

**Performance Optimization**
- Use GPU acceleration when available
- Optimize batch size for your hardware
- Consider using FP16 training

## 📁 File Structure

```
app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── utils/
│   ├── character.py      # Character card management
│   ├── dataset.py        # Synthetic data generation
│   ├── training.py       # Training management
│   └── inference.py      # Model testing
└── training_output/      # Generated models and outputs
    ├── adapters/         # LoRA adapters
    └── prompts/          # Prompt embeddings
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Hugging Face** for transformers and datasets
- **Microsoft** for the LoRA implementation
- **LM Studio** for local LLM inference

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Join our Discord community
- Check the documentation wiki

---

**Happy training! 🎭✨** 
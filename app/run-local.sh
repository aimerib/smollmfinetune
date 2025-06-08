#!/bin/bash

# Character AI Training Studio - Local Development Setup
echo "🎭 Starting Character AI Training Studio (Local Development)"
echo "=================================================="

# Check if LM Studio is running
echo "🔍 Checking for LM Studio..."
if curl -s http://localhost:1234/health > /dev/null 2>&1; then
    echo "✅ LM Studio detected and running"
    export INFERENCE_ENGINE=lmstudio
else
    echo "⚠️  LM Studio not detected - will use fallback engine"
    echo "   Start LM Studio and load a model for optimal performance"
fi

# Set local development environment
export PYTHONUNBUFFERED=1

# Create local directories
mkdir -p training_output/adapters
mkdir -p training_output/prompts

echo ""
echo "🚀 Starting Streamlit app..."
echo "   Open your browser to: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

# Start the app
streamlit run app.py --server.address localhost --server.port 8501 
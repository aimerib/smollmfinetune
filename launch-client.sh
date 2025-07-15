#!/bin/bash

# Launch script for the Character Chat Platform
# Starts both the inference server and React client

echo "🚀 Starting Character Chat Platform..."

# Check if we're in the correct directory
if [ ! -f "app/inference_engine/core.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n🛑 Shutting down..."
    kill $SERVER_PID 2>/dev/null
    kill $CLIENT_PID 2>/dev/null
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

# Start the inference server
echo "📡 Starting inference server..."
python scripts/run_inference_server.py &
SERVER_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for server to be ready..."
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Error: Inference server failed to start"
    exit 1
fi

echo "✅ Inference server is running on http://localhost:8000"

# Start the React client
echo "🎨 Starting React client..."
cd client
npm start &
CLIENT_PID=$!
cd ..

echo "✅ React client is starting on http://localhost:3000"
echo ""
echo "🎉 Character Chat Platform is running!"
echo ""
echo "📱 Open http://localhost:3000 in your browser"
echo "📊 API docs available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait 
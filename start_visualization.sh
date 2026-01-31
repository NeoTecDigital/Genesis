#!/bin/bash
# Start Genesis Visualization System

echo "🚀 Starting Genesis Visualization..."
echo ""

# Check if already running
if pgrep -f "vite" > /dev/null; then
    echo "⚠️  Frontend already running"
else
    echo "Starting frontend (Vite)..."
    cd visualization
    npm run dev > /tmp/genesis_vite.log 2>&1 &
    FRONTEND_PID=$!
    echo "  Frontend PID: $FRONTEND_PID"
    cd ..
fi

sleep 2

# Check if backend already running
if pgrep -f "test_server.py" > /dev/null; then
    echo "⚠️  Backend already running"
else
    echo "Starting backend (WebSocket server)..."
    source venv/bin/activate 2>/dev/null || true
    python src/visualization/test_server.py > /tmp/genesis_server.log 2>&1 &
    BACKEND_PID=$!
    echo "  Backend PID: $BACKEND_PID"
fi

sleep 2

echo ""
echo "✅ Visualization system started!"
echo ""
echo "📊 Frontend: http://localhost:5173"
echo "🔌 WebSocket: ws://localhost:8000/ws"
echo ""
echo "To stop:"
echo "  pkill -f vite"
echo "  pkill -f test_server.py"
echo ""
echo "Logs:"
echo "  Frontend: tail -f /tmp/genesis_vite.log"
echo "  Backend:  tail -f /tmp/genesis_server.log"


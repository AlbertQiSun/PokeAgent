#!/bin/bash
set -e

echo "============================================"
echo "      PokeAgent Automated Setup script"
echo "============================================"

# 1. Update submodules
echo "[1/4] Updating Git Submodules (pokemon-showdown)..."
git submodule update --init --recursive

# 2. Setup Pokemon Showdown Node Environment
echo "[2/4] Setting up Pokemon Showdown Node environment..."
cd pokemon-showdown
npm install
cd ..

# 3. Setup Python Virtual Environment
echo "[3/4] Creating Python virtual environment in bot/venv..."
if [ ! -d "bot/venv" ]; then
    python3 -m venv bot/venv
fi

echo "[4/4] Installing Python dependencies..."
source bot/venv/bin/activate
pip install --upgrade pip
pip install -r bot/requirements.txt

# Create .env template if not exists
if [ ! -f "bot/.env" ]; then
    echo "GEMINI_API_KEY=your_api_key_here" > bot/.env
    echo ""
    echo "NOTE: Created bot/.env — add your Gemini API key there"
    echo "      (or export GEMINI_API_KEY in your shell profile)."
fi

echo ""
echo "============================================"
echo " Setup Complete!"
echo ""
echo " Make sure GEMINI_API_KEY is set (in bot/.env or shell profile)."
echo ""
echo " Usage:"
echo "   ./start.sh user gemini ppo ppo-team   # play vs bots"
echo "   ./start.sh arena gemini ppo-team 20   # bot vs bot arena"
echo "   ./start.sh train random 200000 --history      # train PPO"
echo "   ./start.sh train-team random 300000 --history  # train team PPO"
echo "   ./start.sh bc 50000 10                # BC pretrain from replays"
echo "============================================"

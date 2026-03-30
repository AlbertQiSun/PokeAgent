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
if [ ! -f ".env" ]; then
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    echo ""
    echo "⚠️ NOTE: We created a .env file for you."
    echo "Please open the .env file and paste your Gemini API Key there."
fi

echo ""
echo "============================================"
echo " Setup Complete! 🎉"
echo " "
echo " Remember to:"
echo " 1. Add your GEMINI_API_KEY to the .env file."
echo " 2. Run ./start_server.sh to start the local server"
echo " 3. Run ./start.sh to launch the bot!"
echo "============================================"

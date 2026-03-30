#!/bin/bash
set -e

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/pokemon-showdown"
BOT_DIR="$SCRIPT_DIR/bot"
PYTHON="$BOT_DIR/venv/bin/python"

# ── Environment ──────────────────────────────────────────────────────────────
# Ensure your GEMINI_API_KEY is set in the .env file instead of hardcoding it here.

# ── Mode selection ───────────────────────────────────────────────────────────
MODE="${1:-bot}"      # bot | user | eval
BATTLES="${2:-5}"    # number of battles per matchup (eval mode only)

echo "============================================"
echo "  Pokemon Showdown + Gemini Bot Launcher"
echo "  Mode: $MODE"
echo "============================================"

# ── Start pokemon-showdown server in background ───────────────────────────────
echo "[1/3] Starting Pokemon Showdown server..."
cd "$SERVER_DIR"
node pokemon-showdown start --no-security &> /tmp/ps-server.log &
SERVER_PID=$!
echo "      Server PID: $SERVER_PID"

# ── Wait until server is ready (port 8000) ───────────────────────────────────
echo "[2/3] Waiting for server to be ready on port 8000..."
for i in $(seq 1 30); do
    if nc -z localhost 8000 2>/dev/null; then
        echo "      Server is up!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Server did not start in time. Check /tmp/ps-server.log"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ── Run the bot ───────────────────────────────────────────────────────────────
echo "[3/3] Starting Gemini bot..."
cd "$BOT_DIR"

cleanup() {
    echo ""
    echo "Shutting down server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

if [ "$MODE" = "user" ]; then
    echo "      Open http://localhost:8000 and challenge 'Gemini Bot'"
    "$PYTHON" play_user.py
elif [ "$MODE" = "eval" ]; then
    echo "      Running evaluation ($BATTLES battles per matchup)..."
    "$PYTHON" evaluate.py --battles "$BATTLES"
elif [ "$MODE" = "bc" ]; then
    echo "      Running behavioral cloning from replay dataset..."
    "$PYTHON" bc_pretrain.py --samples "${2:-50000}" --epochs "${3:-10}"
elif [ "$MODE" = "train" ]; then
    OPPONENT="${2:-random}"
    STEPS="${3:-200000}"
    BC_WEIGHTS="${4:-}"
    echo "      Training PPO ($STEPS steps vs $OPPONENT)..."
    if [ -n "$BC_WEIGHTS" ]; then
        "$PYTHON" train_rl.py --steps "$STEPS" --opponent "$OPPONENT" --bc-weights "$BC_WEIGHTS"
    else
        "$PYTHON" train_rl.py --steps "$STEPS" --opponent "$OPPONENT"
    fi
else
    "$PYTHON" main.py
fi

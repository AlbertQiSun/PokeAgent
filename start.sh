#!/bin/bash
set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/pokemon-showdown"
BOT_DIR="$SCRIPT_DIR/bot"
PYTHON="$BOT_DIR/venv/bin/python"

MODE="${1:-bot}"   # bot | user | eval | train | train-team | bc

echo "============================================"
echo "  Pokemon Showdown Bot Launcher  |  $MODE"
echo "============================================"

# ── Start pokemon-showdown server ─────────────────────────────────────────────
echo "[1/3] Starting Pokemon Showdown server..."
cd "$SERVER_DIR"
node pokemon-showdown start --no-security &> /tmp/ps-server.log &
SERVER_PID=$!
echo "      Server PID: $SERVER_PID"

echo "[2/3] Waiting for server on port 8000..."
for i in $(seq 1 30); do
    nc -z localhost 8000 2>/dev/null && echo "      Server is up!" && break
    [ $i -eq 30 ] && echo "ERROR: Server timed out. Check /tmp/ps-server.log" && kill $SERVER_PID 2>/dev/null && exit 1
    sleep 1
done

echo "[3/3] Starting bots..."
cd "$BOT_DIR"

# ── user: start one or more bots for human challenges ─────────────────────────
if [ "$MODE" = "user" ]; then
    BOTS=("${@:2}")
    [ ${#BOTS[@]} -eq 0 ] && BOTS=("gemini")

    BOT_PIDS=()
    for BOT in "${BOTS[@]}"; do
        echo "      Launching $BOT..."
        "$PYTHON" play_user.py --player "$BOT" &
        BOT_PIDS+=($!)
    done

    echo "============================================"
    echo "  Open http://localhost:8000 and challenge:"
    for BOT in "${BOTS[@]}"; do
        case "$BOT" in
            gemini)   echo "    • Gemini Bot      (BSS Reg J)" ;;
            ppo)      echo "    • PPO Bot          (BSS Reg J)" ;;
            ppo-team) echo "    • PPO Team Bot     (BSS Reg J)" ;;
        esac
    done
    echo "  Uncheck Best-of-3 when challenging."
    echo "  Press Ctrl+C to stop all bots."
    echo "============================================"

    cleanup() {
        echo ""
        echo "Stopping bots and server..."
        for PID in "${BOT_PIDS[@]}"; do kill "$PID" 2>/dev/null; done
        kill $SERVER_PID 2>/dev/null
    }
    trap cleanup EXIT INT TERM
    wait "${BOT_PIDS[@]}"

# ── bot: bot vs bot ───────────────────────────────────────────────────────────
elif [ "$MODE" = "bot" ]; then
    PLAYER="${2:-gemini}"
    OPPONENT="${3:-random}"
    BATTLES="${4:-1}"
    cleanup() { kill $SERVER_PID 2>/dev/null; }
    trap cleanup EXIT
    echo "      $PLAYER vs $OPPONENT ($BATTLES battles)"
    "$PYTHON" main.py --player "$PLAYER" --opponent "$OPPONENT" --battles "$BATTLES"

# ── eval ──────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "eval" ]; then
    BATTLES="${2:-5}"
    cleanup() { kill $SERVER_PID 2>/dev/null; }
    trap cleanup EXIT
    echo "      Cross-evaluation ($BATTLES battles per matchup)"
    "$PYTHON" evaluate.py --battles "$BATTLES"

# ── train ─────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "train" ]; then
    OPPONENT="${2:-random}"
    STEPS="${3:-200000}"
    BC_WEIGHTS="${4:-}"
    cleanup() { kill $SERVER_PID 2>/dev/null; }
    trap cleanup EXIT
    echo "      Training PPO ($STEPS steps vs $OPPONENT)"
    if [ -n "$BC_WEIGHTS" ]; then
        "$PYTHON" train_rl.py --steps "$STEPS" --opponent "$OPPONENT" --bc-weights "$BC_WEIGHTS"
    else
        "$PYTHON" train_rl.py --steps "$STEPS" --opponent "$OPPONENT"
    fi

# ── train-team ────────────────────────────────────────────────────────────────
elif [ "$MODE" = "train-team" ]; then
    OPPONENT="${2:-random}"
    STEPS="${3:-300000}"
    cleanup() { kill $SERVER_PID 2>/dev/null; }
    trap cleanup EXIT
    echo "      Training PPO team-builder ($STEPS steps vs $OPPONENT)"
    "$PYTHON" train_rl_team.py --steps "$STEPS" --opponent "$OPPONENT" --format gen9anythinggoes

# ── bc ────────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "bc" ]; then
    cleanup() { kill $SERVER_PID 2>/dev/null; }
    trap cleanup EXIT
    echo "      Behavioral cloning from replay dataset"
    "$PYTHON" bc_pretrain.py --samples "${2:-50000}" --epochs "${3:-10}"

else
    cleanup() { kill $SERVER_PID 2>/dev/null; }
    trap cleanup EXIT
    "$PYTHON" main.py
fi

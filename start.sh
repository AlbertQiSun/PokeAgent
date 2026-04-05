#!/bin/bash
set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/pokemon-showdown"
BOT_DIR="$SCRIPT_DIR/bot"
PYTHON="$BOT_DIR/venv/bin/python"

MODE="${1:-bot}"   # bot | user | arena | eval | train | train-team | bc | online | vllm

SERVER_PID=""

echo "============================================"
echo "  Pokemon Showdown Bot Launcher  |  $MODE"
echo "============================================"

# ── Helper: start Pokemon Showdown server ────────────────────────────────────
start_showdown() {
    echo "[1/2] Starting Pokemon Showdown server..."
    cd "$SERVER_DIR"
    node pokemon-showdown start --no-security &> /tmp/ps-server.log &
    SERVER_PID=$!
    echo "      Server PID: $SERVER_PID"

    echo "[2/2] Waiting for server on port 8000..."
    for i in $(seq 1 30); do
        nc -z localhost 8000 2>/dev/null && echo "      Server is up!" && break
        [ $i -eq 30 ] && echo "ERROR: Server timed out. Check /tmp/ps-server.log" && kill $SERVER_PID 2>/dev/null && exit 1
        sleep 1
    done
    cd "$BOT_DIR"
}

# ── Cleanup helper ───────────────────────────────────────────────────────────
cleanup_all() {
    echo ""
    echo "Stopping services..."
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && echo "  Stopped Showdown ($SERVER_PID)"
    for pid in "$@"; do
        kill "$pid" 2>/dev/null
    done
}

# ── vllm: optionally start a vLLM server on CUDA machines ───────────────────
if [ "$MODE" = "vllm" ]; then
    MODEL="${2:-Qwen/Qwen3-8B}"
    VLLM_PORT="${3:-8001}"
    echo "      Starting vLLM server (model: $MODEL, port: $VLLM_PORT)"
    echo "      Set LOCAL_LLM_BASE_URL=http://localhost:$VLLM_PORT/v1 in bot/.env"
    echo ""
    exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$VLLM_PORT" \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --max-model-len 8192 \
        --dtype auto
fi

# ── Start Showdown server (all modes except online/vllm) ─────────────────────
if [ "$MODE" = "online" ]; then
    echo "[skip] No local Showdown server needed for online play."
else
    start_showdown
fi

echo "Starting bots..."
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
            gemini)   echo "    • Gemini Bot       (BSS Reg J)" ;;
            ppo)      echo "    • PPO Bot          (BSS Reg J)" ;;
            ppo-team) echo "    • PPO Team Bot     (BSS Reg J)" ;;
            qwen)     echo "    • Qwen Bot         (BSS Reg J)" ;;
        esac
    done
    echo "  Uncheck Best-of-3 when challenging."
    echo "  Press Ctrl+C to stop all bots."
    echo "============================================"

    trap "cleanup_all ${BOT_PIDS[*]}" EXIT INT TERM
    wait "${BOT_PIDS[@]}"

# ── bot: bot vs bot ───────────────────────────────────────────────────────────
elif [ "$MODE" = "bot" ]; then
    PLAYER="${2:-gemini}"
    OPPONENT="${3:-random}"
    BATTLES="${4:-1}"
    trap cleanup_all EXIT
    echo "      $PLAYER vs $OPPONENT ($BATTLES battles)"
    "$PYTHON" main.py --player "$PLAYER" --opponent "$OPPONENT" --battles "$BATTLES"

# ── arena: bot vs bot with detailed logging ───────────────────────────────────
elif [ "$MODE" = "arena" ]; then
    P1="${2:-gemini}"
    P2="${3:-ppo-team}"
    BATTLES="${4:-10}"
    FORMAT="${5:-gen9bssregj}"
    trap cleanup_all EXIT
    echo "      $P1 vs $P2  ($BATTLES battles, $FORMAT)"
    "$PYTHON" arena.py --p1 "$P1" --p2 "$P2" --battles "$BATTLES" --format "$FORMAT"

# ── eval ──────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "eval" ]; then
    BATTLES="${2:-5}"
    QWEN_FLAG=""
    for arg in "${@:2}"; do
        [ "$arg" = "qwen" ] || [ "$arg" = "--qwen" ] && QWEN_FLAG="--qwen"
    done
    trap cleanup_all EXIT
    echo "      Cross-evaluation ($BATTLES battles per matchup) $QWEN_FLAG"
    "$PYTHON" evaluate.py --battles "$BATTLES" $QWEN_FLAG

# ── train ─────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "train" ]; then
    OPPONENT="${2:-random}"
    STEPS="${3:-200000}"
    EXTRA_ARGS="${@:4}"
    trap cleanup_all EXIT
    echo "      Training PPO ($STEPS steps vs $OPPONENT) $EXTRA_ARGS"
    "$PYTHON" train_rl.py --steps "$STEPS" --opponent "$OPPONENT" $EXTRA_ARGS

# ── train-team ────────────────────────────────────────────────────────────────
elif [ "$MODE" = "train-team" ]; then
    OPPONENT="${2:-random}"
    STEPS="${3:-300000}"
    EXTRA_ARGS="${@:4}"
    trap cleanup_all EXIT
    echo "      Training PPO team-builder ($STEPS steps vs $OPPONENT) $EXTRA_ARGS"
    "$PYTHON" train_rl_team.py --steps "$STEPS" --opponent "$OPPONENT" --format gen9anythinggoes $EXTRA_ARGS

# ── bc ────────────────────────────────────────────────────────────────────────
elif [ "$MODE" = "bc" ]; then
    trap cleanup_all EXIT
    echo "      Behavioral cloning from replay dataset"
    "$PYTHON" bc_pretrain.py --samples "${2:-50000}" --epochs "${3:-10}"

# ── online: play on real Showdown ladder ─────────────────────────────────────
elif [ "$MODE" = "online" ]; then
    PLAYER="${2:-ppo-team}"
    USERNAME="${3:?Usage: start.sh online <player> <username> <password> [battles] [format]}"
    PASSWORD="${4:?Usage: start.sh online <player> <username> <password> [battles] [format]}"
    BATTLES="${5:-5}"
    FORMAT="${6:-gen9battlestadiumsingles}"
    trap cleanup_all EXIT
    echo "      Playing online as '$USERNAME' ($PLAYER, $BATTLES battles, $FORMAT)"
    "$PYTHON" play_online.py --player "$PLAYER" --username "$USERNAME" --password "$PASSWORD" --battles "$BATTLES" --format "$FORMAT"

else
    trap cleanup_all EXIT
    "$PYTHON" main.py
fi

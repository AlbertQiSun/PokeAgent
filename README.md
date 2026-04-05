# PokeAgent

A multi-agent Pokemon Showdown battle system for Gen 9 Battle Stadium Singles (Reg J).
Three distinct AI agents — Gemini LLM, local Qwen3-8B, and PPO reinforcement learning — compete
using a shared battle intelligence stack that provides pre-computed damage calculations,
metagame scouting, and in-battle opponent tracking.

## Architecture

```
                         +---------------------+
                         |   Metagame KB       |  30 Pokemon: common sets,
                         |   (metagame_kb.py)  |  teammates, usage rates
                         +----------+----------+
                                    |
                                    v
                         +----------+----------+
  Team Preview --------->| Battle Notebook     |  Per-battle state tracker:
  Observations --------->| (battle_notebook.py)|  priors -> observations -> surprises
                         +----------+----------+
                                    |
                                    v
                         +----------+----------+
  Battle state --------->| Battle Context      |  Pre-computed each turn:
  (poke-env)             | (battle_context.py) |  damage calcs, speed, matchups
                         +----------+----------+
                                    |
                                    v
                    +---------------+---------------+
                    |               |               |
              +-----+----+   +-----+----+   +------+-----+
              |  Gemini   |   |  Qwen3   |   |    PPO     |
              |  Player   |   |  Player  |   |   Player   |
              +----------+   +----------+   +------------+
```

### Battle Intelligence Stack

1. **Metagame Knowledge Base** (`metagame_kb.py`) — Static database of 30 competitive Pokemon with their common sets (item, ability, moves, EV spread, nature), usage percentages, and teammate co-occurrence rates. Used as Bayesian priors at team preview.

2. **Battle Notebook** (`battle_notebook.py`) — Per-battle state tracker. Initializes from KB priors at team preview, then updates with confirmed observations (moves, items, abilities, tera types). Detects surprises when reality doesn't match prediction (unexpected bulk, speed, or moves). Tracks confirmed info with checkmarks and estimates with tildes.

3. **Battle Context** (`battle_context.py`) — Pre-computes all numerical analysis each turn and formats it into a single string injected into the LLM prompt:
   - Damage calculations for every available move vs the opponent
   - Opponent's predicted moves vs your active Pokemon
   - Speed checks with priority move awareness
   - Switch analysis with type matchup scoring
   - Field conditions (weather, terrain, side conditions)
   - HP status for all Pokemon on both sides
   - Surprise alerts from the notebook

4. **Damage Calculator** (`damage_calc.py`) — Gen 9 damage formula implementation supporting STAB, type effectiveness, weather, terrain, items (Life Orb, Choice Band/Specs), abilities (Huge Power, Technician, Adaptability, Multiscale), stat boosts, burns, crits, and multi-hit moves.

5. **Pokemon Data** (`pokedata.py`) — Data layer using poke-env's GenData for the Gen 9 Pokedex and moves. Includes curated competitive descriptions for abilities and items, plus a type effectiveness chart.

### AI Agents

| Agent | File | Backend | Description |
|-------|------|---------|-------------|
| **Gemini** | `gemini_player.py` | Google Gemini 3.1 Flash Lite | Cloud LLM. Uses async API for team preview, sync for battle moves. Rebuilds team between battles. |
| **Qwen** | `local_llm_player.py` | Qwen3-8B (transformers or vLLM) | Local LLM. Runs on CUDA (fp16) or MPS (Apple Silicon, fp16). Optional vLLM server mode via `LOCAL_LLM_BASE_URL`. |
| **PPO** | `ppo_player.py` | Stable-Baselines3 MaskablePPO | RL agent trained via self-play. Fixed team. |
| **PPO-Team** | `ppo_team_player.py` | Stable-Baselines3 MaskablePPO | RL agent that also builds its own team of 6 from the pool and selects 3 at preview. |

Both LLM agents (Gemini and Qwen) share the same 3-phase battle loop:
1. **Team Building** — LLM picks 6 from the 20-Pokemon pool
2. **Team Preview** — LLM picks 3 to bring (with lead order) using KB scouting data
3. **Battle Moves** — LLM receives pre-computed context and outputs a JSON action (no tool calls)

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for the Pokemon Showdown server)
- A Google API Key (`GEMINI_API_KEY`) for the Gemini agent
- (Optional) CUDA or Apple Silicon GPU for the local Qwen agent

### Installation

```bash
git clone https://github.com/AlbertQiSun/PokeAgent.git
cd PokeAgent
chmod +x setup.sh
./setup.sh
```

The setup script:
- Initializes the Pokemon Showdown submodule and installs Node dependencies
- Creates a Python venv in `bot/venv` and installs all pip dependencies
- Creates a `bot/.env` template for your API key

Then set your API key:
```bash
# Either edit bot/.env:
echo "GEMINI_API_KEY=your_key_here" > bot/.env

# Or export in your shell:
export GEMINI_API_KEY=your_key_here
```

For the local Qwen agent, you also need `torch` with CUDA or MPS support. The model downloads automatically on first run (~16 GB for Qwen3-8B).

## Usage

All commands go through `start.sh`, which automatically starts/stops the local Showdown server.

### Play Against Bots (Human vs AI)

Open `http://localhost:8000` in your browser and challenge the bot.

```bash
# Play against Gemini
./start.sh user gemini

# Play against PPO team-builder
./start.sh user ppo-team

# Play against local Qwen
./start.sh user qwen

# Launch multiple bots at once
./start.sh user gemini ppo-team qwen
```

### Bot vs Bot (Quick Match)

```bash
# Gemini vs random baseline (1 battle)
./start.sh bot gemini random 1

# Qwen vs PPO (3 battles)
./start.sh bot qwen ppo-team 3
```

### Arena (Detailed Logging)

Runs battles with full statistics and saves results to `bot/arena_logs/`.

```bash
# Gemini vs PPO-Team (10 battles)
./start.sh arena gemini ppo-team 10

# Qwen vs Gemini (5 battles)
./start.sh arena qwen gemini 5

# PPO self-play (20 battles)
./start.sh arena ppo-team ppo-team 20

# Custom format
./start.sh arena gemini ppo-team 10 gen9bssregj
```

Arena output includes win rates, average turns, team usage stats, and per-battle CSV logs.

### Cross-Evaluation

Runs a round-robin between all agents.

```bash
# 5 battles per matchup
./start.sh eval 5

# Include Qwen in evaluation
./start.sh eval 5 --qwen
```

### Train PPO

```bash
# Train basic PPO (fixed team, 200k steps)
./start.sh train random 200000

# Train with history features
./start.sh train random 200000 --history

# Train team-builder PPO (300k steps)
./start.sh train-team random 300000

# Train team-builder with history
./start.sh train-team random 300000 --history
```

### Behavioral Cloning

Pre-train from replay datasets.

```bash
# 50k samples, 10 epochs
./start.sh bc 50000 10
```

### Play Online (Showdown Ladder)

Connect to the real Pokemon Showdown server.

```bash
./start.sh online ppo-team YourUsername YourPassword 5 gen9battlestadiumsingles
./start.sh online gemini YourUsername YourPassword 10
./start.sh online qwen YourUsername YourPassword 5
```

### vLLM Server (Optional, CUDA Only)

For faster Qwen inference on CUDA machines, run a dedicated vLLM server:

```bash
# Terminal 1: start vLLM server
./start.sh vllm Qwen/Qwen3-8B 8001

# Terminal 2: set env and run battles
echo "LOCAL_LLM_BASE_URL=http://localhost:8001/v1" >> bot/.env
./start.sh arena qwen gemini 10
```

Without `LOCAL_LLM_BASE_URL`, the Qwen agent loads the model in-process via transformers (no server needed).

## Environment Variables

Set these in `bot/.env` or export in your shell:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | For Gemini agent | Google AI API key |
| `LOCAL_LLM_MODEL` | No | Model name for Qwen agent (default: `Qwen/Qwen3-8B`) |
| `LOCAL_LLM_BASE_URL` | No | vLLM server URL (e.g., `http://localhost:8001/v1`). If unset, loads model in-process. |
| `LOCAL_LLM_API_KEY` | No | API key for vLLM server (default: `not-needed`) |

## Project Structure

```
pokemon/
  start.sh                  # Main launcher (all modes)
  setup.sh                  # One-time setup
  pokemon-showdown/         # Local Showdown server (git submodule)
  bot/
    main.py                 # Bot vs bot entry point
    play_user.py            # Human vs bot entry point
    play_online.py          # Online ladder play
    arena.py                # Arena with detailed logging
    evaluate.py             # Cross-evaluation round-robin
    gemini_player.py        # Gemini LLM agent
    local_llm_player.py     # Local Qwen3-8B agent
    ppo_player.py           # PPO RL agent (fixed team)
    ppo_team_player.py      # PPO RL agent (builds team)
    battle_context.py       # Pre-computed battle analysis
    battle_notebook.py      # Per-battle opponent tracker
    metagame_kb.py          # Metagame knowledge base (30 Pokemon)
    damage_calc.py          # Gen 9 damage calculator
    pokedata.py             # Pokemon/move data via poke-env GenData
    pokemon_pool.py         # 20-Pokemon competitive pool
    teams.py                # Static team definitions
    rl_env.py               # Gym environment for PPO training
    rl_env_team.py          # Gym environment for team-builder PPO
    train_rl.py             # PPO training script
    train_rl_team.py        # Team-builder PPO training script
    bc_pretrain.py          # Behavioral cloning from replays
    requirements.txt        # Python dependencies
    ppo_gen9random/         # Trained PPO model (basic)
    ppo_team_builder/       # Trained PPO model (team-builder)
    arena_logs/             # Battle logs from arena runs
```

## Available Player Types

Use these names with `start.sh`:

| Name | Agent | Notes |
|------|-------|-------|
| `gemini` | Gemini 3.1 Flash Lite | Requires `GEMINI_API_KEY` |
| `qwen` | Qwen3-8B local LLM | Requires GPU (CUDA/MPS) or patience (CPU) |
| `ppo` | PPO (fixed team) | Uses pre-trained model in `ppo_gen9random/` |
| `ppo-team` | PPO (team builder) | Uses pre-trained model in `ppo_team_builder/` |
| `random` | Random baseline | Available as opponent in `bot` mode |

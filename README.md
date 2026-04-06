# PokeAgent

A multi-agent Pokemon Showdown battle system for Gen 9 Battle Stadium Singles (Reg J).
Four distinct AI agents — Gemini LLM, local Qwen3-8B, PPO reinforcement learning, and a Hybrid dual-process agent — compete using a shared battle intelligence stack that provides pre-computed damage calculations, metagame scouting, and in-battle opponent tracking.

Includes a full training pipeline: collect human replay data, fine-tune the LLM (LoRA SFT), pre-train PPO via behavioral cloning, and train PPO via self-play RL.

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
              +----------+----------+----------+----------+
              |          |          |          |          |
        +-----+--+ +----+---+ +---+----+ +---+--------+ |
        | Hybrid  | | Gemini | | Qwen3  | |    PPO     | |
        | Player  | | Player | | Player | |   Player   | |
        +---------+ +--------+ +--------+ +------------+ |
          S1+S2       Cloud      Local       RL Agent     |
                                                          |
                                          +---------------+
                                          |
                                    +-----+------+
                                    | PPO-Team   |
                                    | Player     |
                                    +------------+
                                      RL + Team
```

### Battle Intelligence Stack

1. **Metagame Knowledge Base** (`metagame_kb.py`) -- Static database of 30 competitive Pokemon with their common sets (item, ability, moves, EV spread, nature), usage percentages, and teammate co-occurrence rates. Used as Bayesian priors at team preview.

2. **Battle Notebook** (`battle_notebook.py`) -- Per-battle state tracker. Initializes from KB priors at team preview, then updates with confirmed observations (moves, items, abilities, tera types). Detects surprises when reality doesn't match prediction (unexpected bulk, speed, or moves). Tracks confirmed info with checkmarks and estimates with tildes.

3. **Battle Context** (`battle_context.py`) -- Pre-computes all numerical analysis each turn and formats it into a single string injected into the LLM prompt:
   - Damage calculations for every available move vs the opponent
   - Opponent's predicted moves vs your active Pokemon
   - Speed checks with priority move awareness
   - Switch analysis with type matchup scoring
   - Field conditions (weather, terrain, side conditions)
   - HP status for all Pokemon on both sides
   - Surprise alerts from the notebook

4. **Damage Calculator** (`damage_calc.py`) -- Gen 9 damage formula implementation supporting STAB, type effectiveness, weather, terrain, items (Life Orb, Choice Band/Specs), abilities (Huge Power, Technician, Adaptability, Multiscale), stat boosts, burns, crits, and multi-hit moves.

5. **Pokemon Data** (`pokedata.py`) -- Data layer using poke-env's GenData for the Gen 9 Pokedex and moves. Includes curated competitive descriptions for abilities and items, plus a type effectiveness chart.

### AI Agents

| Agent | File | Backend | Description |
|-------|------|---------|-------------|
| **Hybrid** | `hybrid_player.py` | PPO + Gemini/Qwen | **Dual-process agent (System 1 + System 2).** PPO handles routine turns; LLM is invoked only on surprises or low PPO confidence. |
| **Gemini** | `gemini_player.py` | Google Gemini 2.5 Flash | Cloud LLM. Uses async API for team preview, sync for battle moves. Rebuilds team between battles. |
| **Qwen** | `local_llm_player.py` | Qwen3-8B (transformers or vLLM) | Local LLM. Runs on CUDA (fp16) or MPS (Apple Silicon). Optional vLLM server mode. |
| **PPO** | `ppo_player.py` | Stable-Baselines3 MaskablePPO | RL agent trained via self-play. Fixed team. |
| **PPO-Team** | `ppo_team_player.py` | Stable-Baselines3 MaskablePPO | RL agent that also builds its own team of 6 from the pool and selects 3 at preview. |

### Dual-Process Architecture (Hybrid Agent)

The hybrid agent implements a **System 1 / System 2** architecture inspired by Kahneman's *Thinking, Fast and Slow*:

```
Each Turn
    |
    v
PPO (System 1)          Fast, instinctive        ~0ms
    |
    v
Confidence Check        PPO softmax distribution
    |                   max_prob < 0.4 = uncertain
    v
Surprise Detector       Compare expected vs actual
(Notebook + Calculator) damage, speed, moves, items
    |
    |-- High confidence, no surprise --> Use PPO action (fast path)
    |
    +-- Low confidence OR surprise! --> LLM (System 2)     ~1-3s
                                        Deep strategic reasoning
                                        Adapts to new information
```

- **System 1 (PPO):** Learned pattern matching. Handles 80-90% of turns.
- **System 2 (LLM):** Deliberate reasoning. Called only when something unexpected happens -- an unknown item, a speed surprise, bulk that doesn't match standard spreads -- or when PPO's softmax distribution is flat (low confidence).
- **Confidence probing:** PPO's softmax output reveals uncertainty. A flat distribution (max prob < 0.4) means PPO is unsure, triggering LLM escalation. No retraining needed.
- **Surprise detection:** The battle notebook compares observations against metagame KB priors. Deviations trigger escalation.

Both standalone LLM agents (Gemini and Qwen) share the same 3-phase battle loop:
1. **Team Building** -- LLM picks 6 from the 20-Pokemon pool
2. **Team Preview** -- LLM picks 3 to bring (with lead order) using KB scouting data
3. **Battle Moves** -- LLM receives pre-computed context and outputs a JSON action (no tool calls)

### Training Pipeline

```
Human Replays (HuggingFace)
    |
    v
Replay Parser ---------> sft_data.jsonl    (prompt, response) pairs
(replay_parser.py)        from winning players
    |
    v
SFT Fine-tune ---------> LoRA adapter      Qwen3-8B learns battle decisions
(train_sft.py)            checkpoints/sft_qwen/
    |
    v
Behavioral Cloning -----> PPO pretrained    BC from replay observations
(bc_pretrain.py)           on human data
    |
    v
RL Self-Play -----------> PPO final model   Train vs random/self-play
(train_rl.py)              ppo_gen9random/
(train_rl_team.py)         ppo_team_builder/
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for the Pokemon Showdown server)
- A Google API Key (`GEMINI_API_KEY`) for the Gemini agent
- (Optional) CUDA GPU for the local Qwen agent and SFT training
- (Optional) Apple Silicon GPU (MPS) also works for Qwen inference

### Installation

```bash
git clone https://github.com/AlbertQiSun/PokeAgent.git
cd PokeAgent
chmod +x setup.sh start.sh
./setup.sh
```

The setup script:
- Initializes the Pokemon Showdown submodule and installs Node dependencies
- Creates a Python venv in `bot/venv` and installs all pip dependencies (including peft, trl for SFT)
- Creates a `bot/.env` template for your API keys

Then set your API key:
```bash
# Either edit bot/.env:
echo "GEMINI_API_KEY=your_key_here" > bot/.env

# Or export in your shell:
export GEMINI_API_KEY=your_key_here
```

For the local Qwen agent, you also need `torch` with CUDA or MPS support. The model downloads automatically on first run (~16 GB for Qwen3-8B).

## Usage

All commands go through `start.sh`, which automatically starts/stops the local Showdown server when needed.

### Play Against Bots (Human vs AI)

Open `http://localhost:8000` in your browser and challenge the bot.

```bash
# Play against Gemini
./start.sh user gemini

# Play against PPO team-builder
./start.sh user ppo-team

# Play against local Qwen
./start.sh user qwen

# Play against Hybrid (System 1+2)
./start.sh user hybrid

# Launch multiple bots at once
./start.sh user gemini ppo-team qwen hybrid
```

### Bot vs Bot (Quick Match)

```bash
# Gemini vs random baseline (1 battle)
./start.sh bot gemini random 1

# Qwen vs PPO (3 battles)
./start.sh bot qwen ppo-team 3

# Hybrid vs Gemini (1 battle)
./start.sh bot hybrid gemini 1
```

### Arena (Detailed Logging)

Runs battles with full statistics and saves results to `bot/arena_logs/`.

```bash
# Gemini vs PPO-Team (10 battles)
./start.sh arena gemini ppo-team 10

# Qwen vs Gemini (5 battles)
./start.sh arena qwen gemini 5

# Hybrid (System 1+2) vs PPO (5 battles)
./start.sh arena hybrid ppo-team 5

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

### Collect Replay Data

Parse human replays from HuggingFace into SFT training data. Only winning player's decisions are kept.

```bash
# Collect 2000 random battle replays (any generation)
./start.sh parse-replays 2000 any-random sft_data.jsonl

# Collect gen9 random battles specifically
./start.sh parse-replays 5000 gen9randombattle sft_gen9.jsonl

# Collect any singles format
./start.sh parse-replays 3000 any-singles sft_singles.jsonl
```

Format options: exact format ID (`gen9randombattle`, `gen9ou`), `any-random`, `any-singles`, or `any`.

### Fine-tune Qwen (SFT with LoRA)

Train Qwen3-8B on collected replay data using parameter-efficient LoRA fine-tuning. Requires a GPU (CUDA recommended, MPS supported).

```bash
# Fine-tune for 3 epochs on collected data
./start.sh sft sft_data.jsonl 3

# With custom learning rate and LoRA rank
./start.sh sft sft_data.jsonl 5 --lr 2e-4 --lora-rank 32

# Resume from checkpoint
./start.sh sft sft_data.jsonl 3 --resume checkpoints/sft_qwen/checkpoint-500

# Merge LoRA adapter into base model for inference
cd bot && python train_sft.py merge checkpoints/sft_qwen/final
```

After merging, set `LOCAL_LLM_MODEL` to the merged model path to use it.

### Behavioral Cloning

Pre-train PPO from replay observation vectors.

```bash
# 50k samples, 10 epochs
./start.sh bc 50000 10
```

### Train PPO (Reinforcement Learning)

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

### Play Online (Showdown Ladder)

Connect to the real Pokemon Showdown server.

```bash
./start.sh online ppo-team YourUsername YourPassword 5 gen9battlestadiumsingles
./start.sh online gemini YourUsername YourPassword 10
./start.sh online qwen YourUsername YourPassword 5
./start.sh online hybrid YourUsername YourPassword 5
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
| `GEMINI_API_KEY` | For Gemini/Hybrid agent | Google AI API key |
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
    # ── Entry points ──
    main.py                 # Bot vs bot entry point
    play_user.py            # Human vs bot entry point
    play_online.py          # Online ladder play
    arena.py                # Arena with detailed logging
    evaluate.py             # Cross-evaluation round-robin
    # ── AI Agents ──
    hybrid_player.py        # Dual-process agent (PPO + LLM)
    gemini_player.py        # Gemini LLM agent
    local_llm_player.py     # Local Qwen3-8B agent
    ppo_player.py           # PPO RL agent (fixed team)
    ppo_team_player.py      # PPO RL agent (builds team)
    # ── Battle Intelligence ──
    battle_context.py       # Pre-computed battle analysis
    battle_notebook.py      # Per-battle opponent tracker
    metagame_kb.py          # Metagame knowledge base (30 Pokemon)
    damage_calc.py          # Gen 9 damage calculator
    pokedata.py             # Pokemon/move data via poke-env GenData
    pokemon_pool.py         # 20-Pokemon competitive pool
    teams.py                # Static team definitions
    # ── Training ──
    train_sft.py            # SFT fine-tune Qwen (LoRA)
    train_rl.py             # PPO training script
    train_rl_team.py        # Team-builder PPO training script
    bc_pretrain.py          # Behavioral cloning from replays
    replay_parser.py        # Parse HuggingFace replays into SFT data
    rl_env.py               # Gym environment for PPO training
    rl_env_team.py          # Gym environment for team-builder PPO
    # ── Data & Models ──
    requirements.txt        # Python dependencies
    ppo_gen9random/         # Trained PPO model (basic)
    ppo_team_builder/       # Trained PPO model (team-builder)
    checkpoints/            # SFT training checkpoints
    sft_data.jsonl          # Collected SFT training data
    arena_logs/             # Battle logs from arena runs
    online_logs/            # Battle logs from online play
```

## Available Player Types

Use these names with `start.sh`:

| Name | Agent | Notes |
|------|-------|-------|
| `hybrid` | Dual-process (PPO + LLM) | System 1/System 2 agent. Requires `GEMINI_API_KEY` + PPO model. |
| `gemini` | Gemini 2.5 Flash | Requires `GEMINI_API_KEY` |
| `qwen` | Qwen3-8B local LLM | Requires GPU (CUDA/MPS) or patience (CPU) |
| `ppo` | PPO (fixed team) | Uses pre-trained model in `ppo_gen9random/` |
| `ppo-team` | PPO (team builder) | Uses pre-trained model in `ppo_team_builder/` |
| `random` | Random baseline | Available as opponent in `bot` mode |

## Full Training Recipe (WSL/CUDA)

End-to-end pipeline to train all models from scratch:

```bash
# 1. Setup
./setup.sh

# 2. Collect human replay data
./start.sh parse-replays 5000 any-random sft_data.jsonl

# 3. Fine-tune Qwen on human data
./start.sh sft sft_data.jsonl 3

# 4. Train PPO via RL self-play
./start.sh train random 200000 --history
./start.sh train-team random 300000 --history

# 5. Pre-train PPO from replays (optional, alternative to step 4)
./start.sh bc 50000 10

# 6. Run evaluation
./start.sh eval 5

# 7. Arena battles
./start.sh arena hybrid ppo-team 20
./start.sh arena gemini qwen 10
```

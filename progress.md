# PokeAgent — Weekly Progress Report

## Overview

PokeAgent is a multi-agent Pokemon Showdown battle system that combines reinforcement learning, large language models, and game-theoretic reasoning. The system plays Gen 9 Battle Stadium Singles (BSS Reg J) format.

---

## 1. Agent Architecture

### Five Agents Implemented

| Agent | Type | Description |
|-------|------|-------------|
| **Gemini** | LLM | Google Gemini API with structured battle context prompt |
| **Qwen** | LLM | Local Qwen3-8B (MPS/CUDA), same prompt framework as Gemini |
| **PPO** | RL | PPO policy trained via self-play, obs dim 421 (5-frame history) |
| **PPO-Team** | RL | PPO with team selection phase from 30-mon pool before battle |
| **Hybrid** | Dual-Process | System 1 (PPO) + System 2 (LLM), the core contribution |

### Hybrid Dual-Process Architecture (System 1 / System 2)

```
                    Battle Turn
                        |
                  [PPO Policy S1]
                   softmax(logits)
                        |
              confidence > 0.4?  ───── YES ──→  Execute PPO action
                   |                              (fast, ~85% of turns)
                   NO
                   |
           [Surprise Detection]
           Battle Notebook flags:
             • unexpected moves
             • item/ability reveals
             • bulk/speed deviations
                   |
              [LLM Agent S2]
           Full battle context +
           notebook surprises →
           strategic reasoning
                   |
              Execute LLM action
              (slow, ~15% of turns)
```

**Key Insight**: PPO handles routine turns efficiently. When its softmax distribution is flat (max probability < **0.4 threshold**), it signals uncertainty → escalate to LLM for deeper reasoning. The battle notebook also detects "surprises" (unexpected events) that trigger LLM escalation.

---

## 2. Battle Intelligence Stack

The intelligence stack transforms raw battle state into rich strategic context for agent decisions.

```
┌─────────────────────────────────────────────────┐
│  Metagame Knowledge Base (KB)                   │
│  30 Pokemon × multiple sets (item/moves/spread) │
│  Prior probabilities per set                    │
└──────────────────┬──────────────────────────────┘
                   │ priors
┌──────────────────▼──────────────────────────────┐
│  Battle Notebook (per opponent Pokemon)          │
│  Bayesian set inference from observations:       │
│  • Confirmed moves, item, ability               │
│  • Damage-based item inference (Life Orb, Band)  │
│  • Speed observations → Scarf detection          │
│  • Spread inference (offensive vs defensive)     │
│  Updates posterior beliefs each turn             │
└──────────────────┬──────────────────────────────┘
                   │ updated beliefs
┌──────────────────▼──────────────────────────────┐
│  Battle Context (pre-computed analysis)          │
│  • Damage calcs: our moves vs opp, opp vs us    │
│  • Speed check: who goes first                   │
│  • KO analysis: can we KO? can they KO us?      │
│  • Tera analysis: damage boost if we tera        │
│  • Opponent bench threats: best move vs each     │
│  • Opponent action prediction (MLP probabilities)│
│  • Switch options with matchup analysis          │
└──────────────────┬──────────────────────────────┘
                   │ formatted text
┌──────────────────▼──────────────────────────────┐
│  Agent Decision                                  │
│  LLM reads context → picks action + reasoning   │
│  PPO reads observation vector → picks action     │
└─────────────────────────────────────────────────┘
```

### Bayesian Set Inference (battle_notebook.py)

Each opponent Pokemon has multiple possible "sets" (item + moves + EV spread) in the KB. As the battle progresses, observations narrow down which set they're running.

**Evidence sources and log-probability weights:**

| Evidence | Match | Mismatch |
|----------|-------|----------|
| Confirmed item | +2.0 | -2.0 |
| Inferred item (from damage) | +1.0 | -0.5 |
| Confirmed ability | +1.5 | -1.5 |
| Each confirmed move in set | +0.5 | -0.3 per missing |
| Speed modifier (Scarf) | +2.0 | -1.0 |
| Spread type match | +0.5 to +0.8 | — |

**Item inference from damage observations:**

| Damage Ratio (actual/expected) | Inferred Item |
|-------------------------------|---------------|
| 1.25 – 1.40 | Life Orb |
| > 1.40 (physical move) | Choice Band |
| > 1.40 (special move) | Choice Specs |

### Opponent Action Prediction (opponent_model.py)

MLP classifier that predicts what the opponent will do next turn.

**Architecture**: Input(40) → Linear(128) → ReLU → Dropout(0.1) → Linear(6) → Softmax

**40 Input Features:**
1. HP fractions (ours + opponent) — 2 features
2. Base stats (ours + opponent: hp/atk/def/spa/spd/spe) — 12 features
3. Speed comparison (binary: are we faster?) — 1 feature
4. Type matchup effectiveness (our types vs opp, opp types vs us) — 4 features
5. Stat boosts (atk/def/spa/spd/spe for both sides) — 10 features
6. Game state (turn number, bench counts, weather, terrain) — 6 features
7. Opponent intel (# moves seen, has item?, has ability?, best move power) — 5 features

**6 Output Categories:**
1. `attack_physical` — physical damage move
2. `attack_special` — special damage move
3. `attack_status` — status move (toxic, thunder wave, etc.)
4. `switch` — voluntary switch to another Pokemon
5. `setup` — stat-boosting move (swords dance, calm mind, etc.)
6. `protect` — protective move (protect, detect, etc.)

**Integration**: Each turn, battle_context.py calls `predict_opponent_action()` and displays the probability distribution to the LLM agent.

### Tera Type Support (damage_calc.py + battle_context.py)

- **Offensive tera STAB**: If move type matches both original type AND tera type → 2.0x STAB (double STAB); if matches tera type only → 1.5x STAB
- **Defensive tera**: Defender uses single tera type for effectiveness instead of dual types
- **"IF YOU TERA" section**: battle_context.py shows damage boost when tera gain is meaningful (>2% difference)
- **Opponent bench threats**: Shows best move + effectiveness vs each known bench Pokemon

---

## 3. Training Data & Pipeline

### 3.1 SFT Training Data (sft_data.jsonl)

**Source**: HuggingFace `HolidayOugi/pokemon-showdown-replays` dataset (streaming)

**Collection**: `replay_parser.py` parses replay logs into (battle_state, action) pairs

**Format**: JSONL, each line is a chat-format training sample:
```json
{
  "messages": [
    {"role": "system", "content": "You are a Pokemon battle expert..."},
    {"role": "user", "content": "<full battle context: HP, moves, matchups, damage calcs>"},
    {"role": "assistant", "content": "ACTION: move Earthquake\nREASON: <strategic reasoning>"}
  ]
}
```

**Stats**: 48,969 samples from 2,000 replays (82% moves, 18% switches)

**Note**: Mostly gen1-gen4 replays. Gen9 random battles start at row ~105k in the HF dataset. Use `--scan-limit 5000000` for gen9 data.

**Training**: LoRA fine-tuning of Qwen3-8B
- LoRA rank=16, alpha=32, dropout=0.05
- Cosine learning rate schedule
- Command: `./start.sh sft sft_data.jsonl 3`

### 3.2 Opponent Prediction Training Data (opp_train.jsonl)

**Source**: Same HuggingFace replay dataset

**Collection**: `train_opponent.py collect` extracts (features, opponent_action) from both player perspectives per replay

**Format**: JSONL, each line:
```json
{
  "features": [0.85, 1.0, 100, 130, ...],  // 40 floats
  "category": 0,                             // 0-5 index into ACTION_CATEGORIES
  "action": "move Earthquake",               // human-readable label
  "turn": 5                                  // turn number
}
```

**Training**: PyTorch MLP with class-weighted cross-entropy loss
- Adam optimizer, lr=1e-3, weight_decay=1e-4
- Cosine annealing LR scheduler
- 90/10 train/val split
- Command: `./start.sh train-opp 3000 30`

### 3.3 PPO Training

- **PPO Random Battle**: 200k steps with 5-frame history stacking, obs dim=421, action space=26
  - Result: 84.1% win rate vs random player
  - Command: `./start.sh train random 200000 --history`

- **PPO Team Builder**: 300k steps with team selection + battle, obs dim includes team selection phase
  - Result: 97.1% win rate vs random player
  - Command: `./start.sh train-team random 300000 --history`

### Training Pipeline Vision

```
Human Replays (HF) ──→ SFT Data ──→ Fine-tune Qwen (LoRA)
                                          |
                                    Qwen labels battles
                                          |
                                    Distill into PPO (BC)
                                          |
                                    RL fine-tune PPO
                                          |
                                    GRPO improve Qwen
                                          ↺ repeat
```

---

## 4. Battle Logging System (battle_logger.py)

Every battle is logged turn-by-turn for analysis and debugging.

### Log Structure

```
battle_logs/
  <battle_tag>/
    turn_01.txt        # Full context + decision for turn 1
    turn_02.txt        # Full context + decision for turn 2
    ...
    summary.txt        # Battle result summary (human-readable)
    summary.json       # Battle result summary (machine-readable)
```

### Turn Log Contents

Each `turn_XX.txt` contains:

```
========== TURN 5 ==========
Agent: Gemini
System: —
Confidence: —

── BATTLE CONTEXT ──
<full battle context as seen by the agent>
  - Our Pokemon HP, moves, damage calcs
  - Opponent Pokemon HP, estimated set
  - Speed check, KO analysis
  - Tera analysis
  - Opponent bench threats
  - Opponent action prediction probabilities
  - Switch options

── ACTION ──
move Earthquake

── REASONING ──
Earthquake is a guaranteed 2HKO on the opponent's Heatran,
and we outspeed. The opponent is likely to stay in based on
the prediction model showing 45% physical attack probability.
```

### Summary File

`summary.json`:
```json
{
  "battle_tag": "battle-gen9bssregj-12345",
  "agent": "Gemini",
  "result": "win",
  "total_turns": 12,
  "our_team": ["Garchomp", "Heatran", "Rotom-Wash"],
  "opp_team": ["Dragapult", "Toxapex", "Corviknight"]
}
```

### Agent Integration

| Agent | System Field | Extra Info Logged |
|-------|-------------|-------------------|
| Gemini | — | reasoning from LLM response |
| Qwen | — | reasoning from LLM response |
| Hybrid (S1 path) | "S1" | PPO confidence value |
| Hybrid (S2 path) | "S2" | surprises list, LLM reasoning |

---

## 5. Complete File Inventory

### Files Created This Sprint

| File | Lines | Purpose |
|------|-------|---------|
| `bot/hybrid_player.py` | ~300 | Dual-process agent (PPO S1 + LLM S2) |
| `bot/battle_logger.py` | ~100 | Turn-by-turn battle logging |
| `bot/opponent_model.py` | ~200 | Opponent action prediction MLP (40→128→6) |
| `bot/train_opponent.py` | ~405 | Data collection + MLP training pipeline |
| `bot/replay_parser.py` | ~400 | Parse HF replays into SFT JSONL |
| `bot/train_sft.py` | ~250 | LoRA SFT fine-tuning for Qwen3-8B |
| `bot/sft_data.jsonl` | 48,969 | SFT training samples from 2,000 replays |

### Files Modified This Sprint

| File | Changes |
|------|---------|
| `bot/battle_context.py` | Tera analysis, opponent bench threats, opponent action prediction, KB fallback |
| `bot/battle_notebook.py` | Bayesian set inference (`_bayesian_update`), `observe_damage_taken()`, item/spread inference |
| `bot/damage_calc.py` | `atk_tera_type` and `def_tera_type` params for tera STAB/defensive calcs |
| `bot/gemini_player.py` | Decision priority prompt, battle logger integration |
| `bot/local_llm_player.py` | Decision priority prompt, battle logger integration |
| `bot/arena.py` | Hybrid player support |
| `bot/play_user.py` | Hybrid player support |
| `bot/play_online.py` | Hybrid player support |
| `bot/main.py` | Hybrid player support |
| `start.sh` | Added modes: train-opp, sft, parse-replays; server skip for offline modes |
| `setup.sh` | Updated .env template, training pipeline commands |
| `bot/requirements.txt` | Added peft, trl |
| `README.md` | Full rewrite with architecture diagrams and training recipes |

---

## 6. Key Design Decisions

1. **Prediction > Action Selection**: The battle intelligence stack already pre-computes optimal actions. The bigger win is predicting *what the opponent will do* — this lets the agent play proactively (e.g., predict a switch and use a coverage move).

2. **Bayesian inference over rule-based**: Instead of hard-coding "if they deal X damage, they have Choice Band", we use log-probability scoring across all evidence to rank sets. This handles ambiguous situations gracefully.

3. **Confidence probing for escalation**: Using PPO's own softmax distribution to detect uncertainty is elegant — no separate classifier needed. A flat distribution (entropy) naturally signals "I don't know what to do."

4. **Dual-perspective data collection**: Training the opponent model from both player perspectives per replay doubles the training data and prevents bias toward one side.

5. **Class-weighted loss**: Opponent actions are imbalanced (mostly attacks). Class-weighted cross-entropy prevents the model from always predicting "physical attack."

---

## 7. Commands Reference

```bash
# Battle modes
./start.sh bot gemini random 5          # Gemini vs random (5 battles)
./start.sh arena gemini ppo-team 10     # Gemini vs PPO-Team (10 battles, logged)
./start.sh user gemini hybrid           # Human can challenge both bots
./start.sh online ppo-team user pass 5  # Play on real Showdown ladder

# Training
./start.sh train random 200000 --history       # Train PPO (random battles)
./start.sh train-team random 300000 --history   # Train PPO team builder
./start.sh train-opp 3000 30                    # Train opponent predictor
./start.sh parse-replays 5000 gen9randombattle  # Collect SFT data
./start.sh sft sft_data.jsonl 3                 # Fine-tune Qwen (LoRA)
```

---

## 8. Results So Far

| Model | Metric | Value |
|-------|--------|-------|
| PPO (random battle) | Win rate vs random | 84.1% |
| PPO (team builder) | Win rate vs random | 97.1% |
| Opponent predictor | Validation accuracy | ~37.5% (gen1 only, 562 samples — needs more data) |

**Next steps on WSL/CUDA:**
1. Train opponent predictor with 3000+ replays for meaningful accuracy
2. Run arena battles to generate battle logs for analysis
3. Optional: SFT training for Qwen
4. Collect gen9-specific replay data (`--scan-limit 5000000`)

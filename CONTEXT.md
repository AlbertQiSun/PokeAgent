# Pokemon Showdown AI Bot — Project Context

## Overview

A Pokemon Showdown battle bot system with three distinct AI agents, a shared team pool, and an arena evaluation framework. All bots run against a local `pokemon-showdown` Node.js server.

---

## Stack

| Component | Version / Details |
|-----------|-------------------|
| Pokemon Showdown server | v25.8.0, `--no-security` flag |
| poke_env | 0.12.1 |
| google-genai | 1.69.0 |
| stable-baselines3 | PPO algorithm |
| Python | 3.12 (venv at `bot/venv/`) |
| Battle format (user/arena) | `gen9bssregj` (BSS Reg J) |
| Training format (RL) | `gen9anythinggoes` / `gen9randombattle` |

---

## Repository Layout

```
pokemon/
├── pokemon-showdown/          Node.js server
├── start.sh                   Unified launcher (all modes)
├── CONTEXT.md                 This file
└── bot/
    ├── .env                   GEMINI_API_KEY (empty — key lives in ~/.zshrc)
    ├── gemini_player.py        Gemini LLM agent
    ├── ppo_player.py           PPO battle-only agent
    ├── ppo_team_player.py      PPO team-building + battle agent
    ├── pokemon_pool.py         Pool of 20 BSS-legal Pokemon
    ├── rl_env.py               Base RL environment (80-dim obs, 26 actions)
    ├── rl_env_team.py          Team-building RL env (101-dim obs)
    ├── train_rl.py             PPO training (battle only)
    ├── train_rl_team.py        PPO training (team-build + battle)
    ├── bc_pretrain.py          Behavioral cloning from HuggingFace replays
    ├── eval_ppo.py             Calibration evaluation vs baselines
    ├── evaluate.py             Cross-evaluation (poke_env cross_evaluate)
    ├── arena.py                Bot-vs-bot arena with detailed logging
    ├── main.py                 Bot vs bot (CLI, non-arena)
    ├── play_user.py            Accept challenges from human player
    ├── teams.py                Fixed BSS teams (TEAM_1, TEAM_2)
    ├── ppo_gen9random.zip      Trained model: battle-only PPO (200k steps)
    ├── ppo_team_builder.zip    Trained model: team-building PPO (300k steps)
    ├── ppo_logs/               Tensorboard logs (battle-only training)
    ├── ppo_team_logs/          Tensorboard logs (team-building training)
    └── arena_logs/             Arena results (JSON + CSV per run)
```

---

## Agents

### 1. GeminiPlayer (`gemini_player.py`)

LLM-powered agent using `gemini-3.1-flash-lite-preview`.

**Three-phase flow:**

| Phase | Method | What happens |
|-------|--------|--------------|
| Team build | `next_team` (property) | Asks Gemini to pick 6 from the pool with synergy reasoning. Built **at startup** (sync) to avoid blocking the async event loop. Rebuilt in background after each battle via `_battle_finished_callback`. |
| Lead select | `teampreview()` (async) | Asks Gemini to pick best 3 from own 6 against the opponent's 6 shown at team preview. Uses `client.aio` (async Gemini client) so the WebSocket stays alive. |
| Battle | `choose_move()` | Formats battle state as text, calls Gemini, parses JSON `{"action_type", "target_name", "reasoning"}`. Falls back to random on parse error. |

**API key:** Set `GEMINI_API_KEY` in `~/.zshrc`. The `.env` file is intentionally empty.

---

### 2. PPOPlayer (`ppo_player.py`)

Loads `ppo_gen9random.zip` — trained 200k steps on `gen9randombattle` vs random opponent.

- Uses **fixed team** `TEAM_2` from `teams.py` for BSS battles.
- `choose_move()`: embeds battle state → `model.predict()` → `SinglesEnv.action_to_order()` with `strict=False` (invalid actions fall back to random).
- Win rate at calibration: ~66% vs random, ~30% vs MaxBasePower, ~4% vs SimpleHeuristics.

---

### 3. PPOTeamPlayer (`ppo_team_player.py`)

Loads `ppo_team_builder.zip` — trained 300k steps on `gen9anythinggoes` vs random opponent.

**Three-phase flow (same as Gemini but RL-driven):**

| Phase | Method | What happens |
|-------|--------|--------------|
| Team build | `next_team` (property) | Runs the PPO policy 6 times with `phase=0` obs to pick from the pool. `update_team()` converts Showdown format → packed format. |
| Lead select | `teampreview()` | Type-matchup scorer: for each of own 6, score offensive coverage + defensive resistance vs opponent's 6. Pick top 3. |
| Battle | `choose_move()` | Same as PPOPlayer: embed → predict → action_to_order. |

- Win rate: **93.3% vs random** (team-building env).
- Self-play via arena produces different teams each match (pool index diversity).

---

## Pokemon Pool (`pokemon_pool.py`)

20 competitive Gen 9 Pokemon, each with a **unique item** (Item Clause compliant). Any 6 selected automatically satisfy BSS Item Clause.

| # | Pokemon | Item |
|---|---------|------|
| 0 | Rillaboom | Choice Band |
| 1 | Tornadus-Therian | Assault Vest |
| 2 | Urshifu-Rapid-Strike | Choice Scarf |
| 3 | Incineroar | Sitrus Berry |
| 4 | Flutter Mane | Choice Specs |
| 5 | Iron Hands | Life Orb |
| 6 | Gholdengo | Expert Belt |
| 7 | Landorus-Therian | Rocky Helmet |
| 8 | Calyrex-Ice | Lum Berry |
| 9 | Amoonguss | Black Sludge |
| 10 | Primarina | Throat Spray |
| 11 | Kingambit | Black Glasses |
| 12 | Chi-Yu | Wise Glasses |
| 13 | Chien-Pao | Focus Sash |
| 14 | Skeledirge | Heavy-Duty Boots |
| 15 | Annihilape | Safety Goggles |
| 16 | Dragonite | Weakness Policy |
| 17 | Ting-Lu | Leftovers |
| 18 | Meowscarada | Scope Lens |
| 19 | Garganacl | Shed Shell |

---

## RL Environments

### `rl_env.py` — Base battle env

- `OBS_SIZE = 80`: HP, types, moves, bench, status/boosts, weather, team counts, turn
- `N_ACTIONS = 26`: 4 moves × 5 gimmick variants + 6 switches
- `PokemonSinglesEnv(SinglesEnv)`: implements `embed_battle()` and `calc_reward()` (fainted=2, hp=1, status=0.5, victory=30)
- `make_env(opponent_type, battle_format)`: wraps in `SingleAgentWrapper` for stable-baselines3

### `rl_env_team.py` — Team-building env

- `OBS_TEAM_SIZE = 101`: [phase(1)] + [pool_selected_flags(20)] + [battle_obs(80)]
- Same 26-action space — actions mean different things per phase:
  - **Phase 0** (team select): action `% 20` → pick pool slot. Runs 6 times.
  - **Phase 1** (battle): action `0–25` → standard battle order.
- `TeamBuildingEnv.step()`: manages phase transitions, applies team via `agent1.update_team()`.
- `make_team_env()`: uses `gen9anythinggoes`, gives both agents a default team so the server accepts the format.

---

## Training

### Battle-only PPO (`train_rl.py`)

```bash
./start.sh train random 200000
./start.sh train heuristic 500000   # harder curriculum
```

- MLP policy `[256, 256]`, lr=3e-4, n_steps=2048, batch=64
- Always runs on CPU (SB3 MLP is faster on CPU than MPS/CUDA)
- Optional BC warm-start: `--bc-weights bc_policy.pth`
- Saves to `ppo_pokemon.zip` (or `--model-path`)

### Team-building PPO (`train_rl_team.py`)

```bash
./start.sh train-team random 300000
./start.sh train-team heuristic 500000
```

- Same hyperparameters, 101-dim obs
- Format: `gen9anythinggoes` (accepts custom teams, no legality restrictions)
- Saves to `ppo_team_builder.zip`

### Behavioral Cloning (`bc_pretrain.py`)

```bash
./start.sh bc 50000 10
```

- Dataset: HuggingFace `HolidayOugi/pokemon-showdown-replays`, filtered to `gen9randombattle`
- `PolicyNet`: Linear(80→256)→ReLU→Linear(256→256)→ReLU→Linear(256→26)
- Weights saved CPU-portable for loading into PPO policy

---

## `start.sh` — All Modes

```bash
# Human vs bot(s) — start one or more bots simultaneously
./start.sh user gemini
./start.sh user ppo
./start.sh user ppo-team
./start.sh user gemini ppo ppo-team       # all three at once

# Bot vs bot arena with logging
./start.sh arena gemini ppo-team 20       # 20 battles, results in arena_logs/
./start.sh arena ppo ppo-team 10
./start.sh arena ppo-team ppo-team 10     # self-play
./start.sh arena gemini ppo 5 gen9bssregj # specify format

# RL training
./start.sh train random 200000
./start.sh train heuristic 500000
./start.sh train-team random 300000
./start.sh train-team heuristic 500000

# Evaluation
./start.sh eval 5                         # cross-evaluate all agents
./start.sh bc 50000 10                    # behavioral cloning
```

**Human play:** Open `http://localhost:8000`, pick a username, Find User → bot name, challenge with **BSS Reg J** format, uncheck Best-of-3.

| Bot name in browser | Player type |
|---------------------|-------------|
| Gemini Bot | `gemini` |
| PPO Bot | `ppo` |
| PPO Team Bot | `ppo-team` |

---

## Arena Logging (`arena.py`)

Each run saves two files in `bot/arena_logs/`:

- **`arena_P1_vs_P2_TIMESTAMP.json`** — full report:
  - Summary: win rates, avg/min/max turns, battles/minute
  - Team usage: how often each species was picked (team-building bots)
  - Survival rate per species
  - Per-battle details: tag, won, turns, teams, fainted, survived

- **`arena_P1_vs_P2_TIMESTAMP.csv`** — one row per battle:
  - `battle, won, turns, our_team, opp_team, our_fainted, our_survived`

---

## Known Issues & Notes

- **`strict=False`** throughout: invalid PPO actions (dynamax/zmove/mega in Gen 9) silently fall back to a random valid move. Warnings in logs are expected.
- **Calyrex-Ice** in pool: it's a restricted legendary. In `gen9bssregj` only 1 restricted mon per team is allowed. If training in BSS format, ensure team builder doesn't pick two restricted mons. (Currently not an issue since pool has only one restricted mon.)
- **Gemini `choose_move`** is synchronous (blocks event loop briefly per turn). This is acceptable for human play but may cause slowdowns in arena mode. `teampreview` is properly async.
- **PPO team-building model** was trained on `gen9anythinggoes` (no restrictions), not `gen9bssregj`. The battle policy transfers well but team selection wasn't trained against BSS-specific opponents.
- **Lead selection for PPO-Team** uses a type-matchup heuristic (not RL-trained). Training lead selection with RL would require integrating teampreview into the `step()` loop — a future improvement.

# PokeAgent
Great question. Let me show you what the raw data actually looks like and the full conversion pipeline.

---

## What Showdown replays actually look like

Raw replay logs are a line-by-line protocol format:

```
|move|p1a: Garchomp|Earthquake|p2a: Rotom-Wash
|split|p2
|-damage|p2a: Rotom-Wash|142/230
|-damage|p2a: Rotom-Wash|142/230|[from] item: Rocky Helmet
|move|p2a: Rotom-Wash|Will-O-Wisp|p1a: Garchomp
|-status|p1a: Garchomp|brn
|turn|8
```

Every line starts with a `|` tag. It's telling you what happened mechanically — not anything about *why*, not the full state, not what was known at that moment. So you have to **reconstruct** the full battle state turn by turn by reading forward through the log.

---

## The conversion pipeline — 4 stages

### Stage 1: Parse replay → turn-by-turn state dict

You simulate the battle by reading the log line by line and maintaining a running state object. After each `|turn|N` tag, you snapshot it:

```python
state = {
  "turn": 7,
  "our_active": {
    "species": "Garchomp",
    "hp": 284, "max_hp": 355,  # parsed from |p1a: Garchomp|284/355
    "status": None,
    "boosts": {"atk": 2, "spe": 0, ...},
    "revealed_moves": ["Earthquake", "Dragon Claw"]
    # only moves we've SEEN so far
  },
  "opp_active": {
    "species": "Rotom-Wash",
    "hp": "?",         # we haven't seen it yet
    "item": "?",       # unknown until revealed
    "revealed_moves": ["Hydro Pump"]
  },
  "weather": "none",
  "field": {},
  "opp_action_this_turn": "Will-O-Wisp"  # the label
}
```

The key constraint: **you only include information that was visible at that point in the game** — not what you know from hindsight. This is critical for training data integrity.

---

### Stage 2: State dict → numerical vector (for fast path / PPO)

A deterministic encoding function. Every field maps to a fixed position:

```python
def encode_numerical(state):
    vec = []

    # Our active mon (32 dims)
    vec += [state.our_hp / state.our_max_hp]          # 1
    vec += type_onehot(state.our_type1, state.our_type2)  # 18
    vec += status_onehot(state.our_status)              # 7
    vec += [b / 6 for b in state.our_boosts.values()]  # 7  (normalized -6 to +6)

    # Opponent active (32 dims, partial)
    vec += [state.opp_hp / state.opp_max_hp
            if state.opp_hp != "?" else -1.0]          # 1  (-1 = unknown)
    vec += type_onehot(state.opp_type1, state.opp_type2)
    vec += status_onehot(state.opp_status)
    vec += [1.0 if move in state.opp_revealed_moves
            else 0.0 for move in ALL_MOVES[:4]]        # 4 flags

    # Team slots (12 dims)
    vec += [s.hp_pct for s in state.our_team]          # 6
    vec += [1.0 if s.alive else 0.0
            for s in state.opp_team]                   # 6

    # Field (12 dims)
    vec += weather_onehot(state.weather)               # 5
    vec += terrain_onehot(state.terrain)               # 4
    vec += [state.turn / 50.0]                         # 1
    vec += hazard_flags(state.hazards)                 # 2

    # Slow path belief (12 dims, starts as zeros)
    vec += state.belief_vec                            # 12

    return np.array(vec, dtype=np.float32)  # → (128,)
```

---

### Stage 3: State dict → natural language string (for slow LLM)

A templated text generation function. This is what goes into the LLM prompt:

```python
def encode_natural_lang(state):
    return f"""
Turn {state.turn}. Your {state.our_active.species} 
({state.our_hp}/{state.our_max_hp} HP, 
{state.our_status or 'no status'}) 
faces the opponent's {state.opp_active.species} 
(HP unknown, no status observed).

Your team remaining: {team_summary(state.our_team)}
Opponent team: {opp_team_summary(state.opp_team)}

Moves you have seen from the opponent: 
{', '.join(state.opp_revealed_moves) or 'none yet'}

Weather: {state.weather or 'clear'}
Field: {field_summary(state.field)}

Your stat boosts: {boost_summary(state.our_boosts)}

Last turn: opponent used {state.last_opp_action}.
Observed damage: {state.last_damage_dealt} 
(expected range: {state.expected_dmg_low}–{state.expected_dmg_high})
""".strip()
```

Which produces something like:

```
Turn 7. Your Garchomp (284/355 HP, no status) faces
the opponent's Rotom-Wash (HP unknown, no status observed).

Your team remaining: Rotom-W (100%), Clefable (78%), 
Toxapex (fainted), Weavile (100%), Corviknight (55%)
Opponent team: 5 slots remaining, 1 revealed

Moves you have seen from the opponent: Hydro Pump
Weather: clear
Field: none

Your stat boosts: +2 Attack

Last turn: opponent used Will-O-Wisp.
Observed damage: 0 (expected range: 0–0)
[Surprise: Rocky Helmet recoil 71 HP — out of expected range]
```

---

### Stage 4: Tagging slow turns + generating reasoning chains

For turns where `actual_damage` falls outside `expected_dmg_low–high`, you tag `is_slow=True` and then **prompt a capable LLM** (GPT-4o or Claude) to retroactively generate the reasoning trace:

```python
ANNOTATION_PROMPT = """
You are annotating a Pokémon battle replay for training data.

Battle state:
{state_nl}

Anomaly: The opponent's Rotom-Wash dealt 71 HP recoil 
to your Garchomp, which is outside the expected range 
(0–0). The opponent had not revealed their item.

Looking at the full replay (which you have access to),
the item was: Rocky Helmet.

Generate a reasoning chain as if you discovered this 
mid-battle using tools:
1. What tool would you call first?
2. What would it return?
3. How does that change your opponent model?
4. What is your updated win condition?
5. What action would you take this turn?

Format as a JSON tool_call chain.
"""
```

This produces the synthetic `tool_calls` + `belief_update` + `reasoning_chain` fields that go in your slow path training examples. You do this for every `is_slow=True` turn across your 100k replays — probably ~1.5M slow turns total, though you'd sample a subset for annotation cost reasons.

---

## Summary of what you're actually building

| Stage | Input | Output | Tool |
|---|---|---|---|
| 1 | Raw `.log` replay text | State dict per turn | Custom Python parser |
| 2 | State dict | 128-dim float vector | `encode_numerical()` |
| 3 | State dict | NL string | `encode_natural_lang()` |
| 4 | NL string + anomaly + hindsight | Reasoning chain JSON | GPT-4o annotation |
| 5 | All of the above | HuggingFace Dataset | `datasets` library |

The hardest and most expensive part is Stage 4 — and it's also the most novel contribution of your data pipeline. It's worth calling that out explicitly in the presentation.

Want me to update the data slides in the deck to reflect this actual pipeline?

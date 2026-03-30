"""
RL Environment for Pokemon battles.

Wraps poke_env's SinglesEnv with:
- A concrete observation embedding (100-dim feature vector)
- A shaped reward function
- SingleAgentWrapper so stable-baselines3 can use it directly
"""

import numpy as np
from typing import Optional, Union
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status


# ── Observation dimensions ───────────────────────────────────────────────────
# Active pokemon (own):   HP + 18 types + 4 moves*(base_power+type+accuracy) = 1+18+4*3 = 31
# Active pokemon (opp):   HP + 18 types = 19
# Team bench (5 mons):    HP + fainted = 5*2 = 10
# Own status + boosts:    1 + 7 = 8
# Opp status:             1
# Weather (one-hot 8):    8
# Remaining team counts:  2
# Turn fraction:          1
# Total: 31 + 19 + 10 + 8 + 1 + 8 + 2 + 1 = 80
OBS_SIZE  = 80
N_ACTIONS = 26  # gen9: 6 switches + 4 moves * 5 gimmick variants
N_TYPES = len(PokemonType) + 1  # +1 for unknown


def _type_index(t) -> int:
    try:
        return t.value if t is not None else 0
    except Exception:
        return 0


def embed_battle(battle: AbstractBattle) -> np.ndarray:
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    # ── Own active pokemon ──────────────────────────────────────────────────
    active = battle.active_pokemon
    if active:
        obs[idx] = active.current_hp_fraction
        idx += 1
        # types (18 possible, one-hot — just set both)
        type_vec = np.zeros(18, dtype=np.float32)
        for t in [active.type_1, active.type_2]:
            if t is not None:
                ti = _type_index(t) % 18
                type_vec[ti] = 1.0
        obs[idx:idx+18] = type_vec
        idx += 18
        # moves: base_power (norm), type_index (norm), accuracy (norm)
        moves = list(active.moves.values())[:4]
        for i in range(4):
            if i < len(moves):
                m = moves[i]
                obs[idx]   = min(m.base_power, 250) / 250.0
                obs[idx+1] = _type_index(m.type) / 18.0
                obs[idx+2] = (m.accuracy if isinstance(m.accuracy, (int, float)) and m.accuracy > 1 else 100) / 100.0
            idx += 3
    else:
        idx += 1 + 18 + 12  # skip

    # ── Opponent active pokemon ─────────────────────────────────────────────
    opp = battle.opponent_active_pokemon
    if opp:
        obs[idx] = opp.current_hp_fraction
        idx += 1
        type_vec = np.zeros(18, dtype=np.float32)
        for t in [opp.type_1, opp.type_2]:
            if t is not None:
                ti = _type_index(t) % 18
                type_vec[ti] = 1.0
        obs[idx:idx+18] = type_vec
        idx += 18
    else:
        idx += 19

    # ── Bench (5 slots) ─────────────────────────────────────────────────────
    bench = [p for p in battle.team.values() if p != active][:5]
    for i in range(5):
        if i < len(bench):
            p = bench[i]
            obs[idx]   = p.current_hp_fraction
            obs[idx+1] = 1.0 if p.fainted else 0.0
        idx += 2

    # ── Status + boosts ─────────────────────────────────────────────────────
    if active:
        obs[idx] = 1.0 if active.status else 0.0
        idx += 1
        boosts = active.boosts
        for stat in ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]:
            obs[idx] = boosts.get(stat, 0) / 6.0
            idx += 1
    else:
        idx += 8

    # ── Opponent status ──────────────────────────────────────────────────────
    if opp:
        obs[idx] = 1.0 if opp.status else 0.0
    idx += 1

    # ── Weather (8 slots) ────────────────────────────────────────────────────
    weather_map = {
        "sunnyday": 0, "raindance": 1, "sandstorm": 2, "hail": 3,
        "snow": 4, "desolateland": 5, "primordialsea": 6, "deltastream": 7,
    }
    for w in battle.weather:
        wname = w.name.lower()
        wi = weather_map.get(wname, -1)
        if wi >= 0:
            obs[idx + wi] = 1.0
    idx += 8

    # ── Remaining pokemon counts ─────────────────────────────────────────────
    own_alive = sum(1 for p in battle.team.values() if not p.fainted)
    opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)
    obs[idx]   = own_alive / 6.0
    obs[idx+1] = opp_alive / 6.0
    idx += 2

    # ── Turn fraction ────────────────────────────────────────────────────────
    obs[idx] = min(battle.turn, 50) / 50.0

    return obs


class PokemonSinglesEnv(SinglesEnv):
    """SinglesEnv with concrete embed_battle and calc_reward."""

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return embed_battle(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    @property
    def observation_spaces(self):
        from gymnasium.spaces import Box
        space = Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        return {agent: space for agent in self.possible_agents}


def make_env(opponent_type: str = "random", battle_format: str = "gen9randombattle"):
    """Create a SingleAgentWrapper gym env for stable-baselines3."""
    env = PokemonSinglesEnv(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
        strict=False,  # invalid actions fall back to random valid move, no crash
    )

    if opponent_type == "random":
        opponent = RandomPlayer(
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
    elif opponent_type == "heuristic":
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    return SingleAgentWrapper(env, opponent)

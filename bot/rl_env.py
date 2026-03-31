"""
RL Environment for Pokemon battles.

Wraps poke_env's SinglesEnv with:
- A concrete observation embedding (80-dim base feature vector)
- Optional frame-stacking for battle history (N past observations)
- A shaped reward function
- SingleAgentWrapper so stable-baselines3 can use it directly
"""

import numpy as np
import gymnasium
from collections import deque
from typing import Optional, Union
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status


# ── Observation dimensions ───────────────────────────────────────────────────
# Base (single frame): 80
# With history (1 current + 4 past): 80 * 5 = 400
OBS_SIZE      = 80
N_ACTIONS     = 26   # gen9: 6 switches + 4 moves * 5 gimmick variants
HISTORY_LEN   = 4    # number of past frames to include
OBS_SIZE_HIST = OBS_SIZE * (1 + HISTORY_LEN)  # 400
N_TYPES       = len(PokemonType) + 1


def remap_action_gen9(action: int) -> int:
    """Remap mega/zmove/dynamax actions to normal moves for Gen 9.

    poke_env action layout:
      0-5:   switch
      6-9:   move (normal)
      10-13: move + mega     → never valid in Gen 9
      14-17: move + zmove    → never valid in Gen 9
      18-21: move + dynamax  → never valid in Gen 9
      22-25: move + tera

    We remap 10-21 to 6-9 so no training step is wasted.
    """
    if 10 <= action <= 21:
        return 6 + (action - 6) % 4
    return action


def _type_index(t) -> int:
    try:
        return t.value if t is not None else 0
    except Exception:
        return 0


def embed_battle(battle: AbstractBattle) -> np.ndarray:
    """80-dim observation of the current battle state (single frame)."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    # ── Own active pokemon ──────────────────────────────────────────────────
    active = battle.active_pokemon
    if active:
        obs[idx] = active.current_hp_fraction
        idx += 1
        type_vec = np.zeros(18, dtype=np.float32)
        for t in [active.type_1, active.type_2]:
            if t is not None:
                ti = _type_index(t) % 18
                type_vec[ti] = 1.0
        obs[idx:idx+18] = type_vec
        idx += 18
        moves = list(active.moves.values())[:4]
        for i in range(4):
            if i < len(moves):
                m = moves[i]
                obs[idx]   = min(m.base_power, 250) / 250.0
                obs[idx+1] = _type_index(m.type) / 18.0
                obs[idx+2] = (m.accuracy if isinstance(m.accuracy, (int, float)) and m.accuracy > 1 else 100) / 100.0
            idx += 3
    else:
        idx += 1 + 18 + 12

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


# ── Battle history tracker ───────────────────────────────────────────────────

class BattleHistory:
    """Tracks per-battle observation history for frame stacking."""

    def __init__(self, history_len: int = HISTORY_LEN):
        self.history_len = history_len
        self._buffers: dict[str, deque] = {}

    def embed_with_history(self, battle: AbstractBattle) -> np.ndarray:
        """Return (OBS_SIZE_HIST,) = [current_obs | past_1 | past_2 | ... | past_N]."""
        tag = getattr(battle, "battle_tag", id(battle))
        if tag not in self._buffers:
            self._buffers[tag] = deque(maxlen=self.history_len)

        current = embed_battle(battle)
        buf = self._buffers[tag]

        full = np.zeros(OBS_SIZE * (1 + self.history_len), dtype=np.float32)
        full[:OBS_SIZE] = current
        for i, prev in enumerate(buf):
            full[OBS_SIZE * (i + 1): OBS_SIZE * (i + 2)] = prev

        buf.append(current.copy())
        return full

    def reset(self, battle_tag: str = None):
        if battle_tag:
            self._buffers.pop(battle_tag, None)
        else:
            self._buffers.clear()


# ── Env classes ──────────────────────────────────────────────────────────────

class PokemonSinglesEnv(SinglesEnv):
    """SinglesEnv with concrete embed_battle and calc_reward."""

    def __init__(self, *args, use_history: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_history = use_history
        self._history = BattleHistory() if use_history else None

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        if self._history:
            return self._history.embed_with_history(battle)
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
        dim = OBS_SIZE_HIST if self.use_history else OBS_SIZE
        space = Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        return {agent: space for agent in self.possible_agents}


def _build_action_mask(battle: AbstractBattle) -> np.ndarray:
    """Build boolean mask of valid actions (0-25) for the current battle state."""
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    if battle is None or battle.finished:
        mask[0] = 1.0  # default fallback
        return mask

    valid_strs = {str(o) for o in battle.valid_orders}
    team = list(battle.team.values())

    # Switches (0-5)
    for i, mon in enumerate(team):
        order = Player.create_order(mon)
        if str(order) in valid_strs:
            mask[i] = 1.0

    # Moves (6-9) and tera (22-25)
    if battle.active_pokemon:
        mvs = (
            battle.available_moves
            if len(battle.available_moves) == 1
            and battle.available_moves[0].id in ["struggle", "recharge"]
            else list(battle.active_pokemon.moves.values())
        )
        for j, move in enumerate(mvs[:4]):
            order_normal = Player.create_order(move)
            if str(order_normal) in valid_strs:
                mask[6 + j] = 1.0
            order_tera = Player.create_order(move, terastallize=True)
            if str(order_tera) in valid_strs:
                mask[22 + j] = 1.0

    # Actions 10-21 (mega/zmove/dynamax) always masked out in Gen 9

    # Ensure at least one action is valid
    if mask.sum() == 0:
        mask[0] = 1.0
    return mask


def _get_current_battle(env):
    """Walk the wrapper chain to find the current battle object."""
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    # inner should now be the SinglesEnv (PettingZoo)
    if hasattr(inner, 'agent1'):
        battles = inner.agent1.battles
        if battles:
            return list(battles.values())[-1]
    if hasattr(inner, 'battles'):
        battles = inner.battles
        if battles:
            return list(battles.values())[-1]
    return None


class MaskableEnvWrapper(gymnasium.Wrapper):
    """Adds action_masks() for MaskablePPO and remaps dead Gen 9 actions."""

    def action_masks(self) -> np.ndarray:
        battle = _get_current_battle(self)
        if battle is None:
            mask = np.ones(N_ACTIONS, dtype=np.float32)
            mask[10:22] = 0.0
            return mask
        return _build_action_mask(battle)

    def step(self, action):
        action = np.int64(remap_action_gen9(int(action)))
        return self.env.step(action)


def make_env(opponent_type: str = "random", battle_format: str = "gen9randombattle",
             use_history: bool = False):
    """Create a SingleAgentWrapper gym env for stable-baselines3."""
    env = PokemonSinglesEnv(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
        strict=False,
        use_history=use_history,
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

    wrapped = SingleAgentWrapper(env, opponent)
    # MaskableEnvWrapper must be the outermost wrapper so MaskablePPO can see action_masks()
    return MaskableEnvWrapper(wrapped)


def make_masked_env(opponent_type: str = "random", battle_format: str = "gen9randombattle",
                    use_history: bool = False):
    """Create env with Monitor inside MaskableEnvWrapper (for MaskablePPO training)."""
    from stable_baselines3.common.monitor import Monitor
    env = PokemonSinglesEnv(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
        strict=False,
        use_history=use_history,
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
    wrapped = SingleAgentWrapper(env, opponent)
    monitored = Monitor(wrapped)
    return MaskableEnvWrapper(monitored)

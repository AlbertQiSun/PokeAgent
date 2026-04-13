"""
RL Environment for Pokemon battles.

Wraps poke_env's SinglesEnv with:
- A concrete observation embedding (86-dim base feature vector)
- Optional frame-stacking for battle history (N past observations)
- A shaped reward function
- SingleAgentWrapper so stable-baselines3 can use it directly
"""

import re

import numpy as np
import gymnasium
from collections import deque
from typing import Optional, Union
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from opponent_model import predict_opponent_action, N_CATEGORIES


# ── Observation dimensions ───────────────────────────────────────────────────
# Base (single frame): 80 original + 6 opponent prediction = 86
# With history (1 current + 4 past): 86 * 5 = 430
OBS_SIZE      = 86
N_ACTIONS     = 26   # gen9: 6 switches + 4 moves * 5 gimmick variants
HISTORY_LEN   = 4    # number of past frames to include
OBS_SIZE_HIST = OBS_SIZE * (1 + HISTORY_LEN)  # 430
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
    """86-dim observation of the current battle state (single frame)."""
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
    idx += 1

    # ── Opponent action prediction (6) ─────────────────────────────────────
    if active and opp:
        our_species = re.sub(r'[^a-zA-Z0-9]', '', active.species).lower()
        opp_species = re.sub(r'[^a-zA-Z0-9]', '', opp.species).lower()

        # Gather opponent intel from what poke-env exposes
        opp_moves_seen = list(opp.moves.keys()) if opp.moves else []
        our_boosts = active.boosts if active.boosts else {}
        opp_boosts = opp.boosts if opp.boosts else {}

        our_bench_alive = sum(1 for p in battle.team.values()
                              if p != active and not p.fainted)
        opp_bench_alive = sum(1 for p in battle.opponent_team.values()
                              if p != opp and not p.fainted)

        weather_str = ""
        if battle.weather:
            weather_str = ",".join(w.name for w in battle.weather)
        terrain_str = ""
        if battle.fields:
            terrain_str = ",".join(f.name for f in battle.fields)

        probs = predict_opponent_action(
            our_species, opp_species,
            active.current_hp_fraction, opp.current_hp_fraction,
            opp_intel=None,
            our_boosts=our_boosts, opp_boosts=opp_boosts,
            turn=battle.turn,
            our_bench_count=our_bench_alive,
            opp_bench_count=opp_bench_alive,
            weather=weather_str, terrain=terrain_str,
            opp_moves_list=opp_moves_seen,
        )
        if probs:
            from opponent_model import ACTION_CATEGORIES
            for cat in ACTION_CATEGORIES:
                obs[idx] = probs.get(cat, 0.0)
                idx += 1
        else:
            idx += N_CATEGORIES
    else:
        idx += N_CATEGORIES

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

    @staticmethod
    def action_to_order(
        action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        """Override to silently return DefaultBattleOrder when only /choose default is valid.

        This prevents warnings when the RL agent picks a switch/move action during
        forced-default turns (e.g. waiting for opponent, end-of-turn animations).
        """
        if action != -2 and action != -1 and not fake:
            valid_strs = {str(o) for o in battle.valid_orders}
            if not valid_strs or valid_strs == {"/choose default"}:
                return DefaultBattleOrder()
        return SinglesEnv.action_to_order(action, battle, fake, strict)

    @staticmethod
    def order_to_action(
        order: BattleOrder, battle: Battle, fake: bool = False, strict: bool = True
    ) -> np.int64:
        """Override to silently return default action when only /choose default is valid.

        The opponent's choose_move() can produce switch orders during transition turns
        where battle.valid_orders is only ['/choose default']. The parent order_to_action
        validates the switch against valid_orders and logs noisy warnings. We intercept
        this case and return the default action (-2) directly.
        """
        if not fake:
            valid_strs = {str(o) for o in battle.valid_orders}
            if not valid_strs or valid_strs == {"/choose default"}:
                return np.int64(-2)
        return SinglesEnv.order_to_action(order, battle, fake, strict)

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

    # If no valid orders or only default — battle in transition state
    valid_strs = {str(o) for o in battle.valid_orders}
    if not valid_strs or valid_strs == {"/choose default"}:
        mask[0] = 1.0
        return mask

    team = list(battle.team.values())

    # ── Switches (0-5) — use available_switches directly for reliability ──
    available_switches = battle.available_switches
    if available_switches:
        switch_species = {mon.species for mon in available_switches}
        for i, mon in enumerate(team):
            if mon.species in switch_species:
                mask[i] = 1.0

    # ── Moves (6-9) and tera (22-25) ──
    available_moves = battle.available_moves
    if available_moves and battle.active_pokemon:
        all_moves = list(battle.active_pokemon.moves.values())
        for j, move in enumerate(all_moves[:4]):
            if move in available_moves:
                mask[6 + j] = 1.0
            # Check tera variant
            order_tera = Player.create_order(move, terastallize=True)
            if str(order_tera) in valid_strs:
                mask[22 + j] = 1.0
    elif available_moves:
        # Struggle/recharge — only one move available
        for j, move in enumerate(available_moves[:4]):
            mask[6 + j] = 1.0

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

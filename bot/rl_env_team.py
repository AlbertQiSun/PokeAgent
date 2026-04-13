"""
Team-building RL environment.

Extends the standard battle env with a team-selection phase:
  Phase 0 (team_select): Agent picks 6 Pokemon from the pool of 20 sequentially.
  Phase 1 (battle): Normal battle with the selected team.

Unified spaces:
  observation_space: Box(shape=(OBS_TEAM_SIZE,))
    [0]       phase flag  (0.0 = selecting, 1.0 = battling)
    [1..20]   one-hot selected flags for pool positions
    [21..106] battle observation (86-dim from embed_battle)
    Total: 1 + 20 + 86 = 107

  action_space: Discrete(26)
    Phase 0: action mod POOL_SIZE (0-19) → pick that pool slot
    Phase 1: standard battle actions 0-25

The inner env must be a SingleAgentWrapper around a PokemonSinglesEnv.
Use make_team_env() to create the full env.
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer

from rl_env import PokemonSinglesEnv, OBS_SIZE, OBS_SIZE_HIST, N_ACTIONS, remap_action_gen9, _build_action_mask
from poke_env.environment import SingleAgentWrapper
from pokemon_pool import POOL, POOL_SIZE, POOL_NAMES, build_team_string


# ── Type coverage tables for team-selection reward shaping ───────────────────
# Maps pool index → set of types that Pokemon covers (STAB + move types)
_POOL_TYPES: dict[int, set[str]] = {
    0:  {"grass", "dark", "bug"},               # Rillaboom
    1:  {"flying", "fire", "dark", "bug"},       # Tornadus-T
    2:  {"water", "fighting", "bug"},            # Urshifu-RS
    3:  {"fire", "dark", "bug"},                 # Incineroar
    4:  {"fairy", "ghost", "fire", "psychic"},   # Flutter Mane
    5:  {"electric", "fighting", "steel"},       # Iron Hands
    6:  {"steel", "ghost", "electric", "fighting"}, # Gholdengo
    7:  {"ground", "flying", "bug", "rock"},     # Landorus-T
    8:  {"ice", "ground", "steel"},              # Calyrex-Ice
    9:  {"grass", "poison"},                     # Amoonguss
    10: {"water", "fairy", "grass", "psychic"},  # Primarina
    11: {"dark", "steel"},                       # Kingambit
    12: {"fire", "dark", "psychic"},             # Chi-Yu
    13: {"ice", "fighting", "dark"},             # Chien-Pao
    14: {"fire", "ghost"},                       # Skeledirge
    15: {"fighting", "ghost", "bug"},            # Annihilape
    16: {"normal", "dragon", "fire", "ice"},     # Dragonite
    17: {"ground", "dark", "rock"},              # Ting-Lu
    18: {"grass", "dark", "fighting", "bug"},    # Meowscarada
    19: {"rock", "fighting"},                    # Garganacl
}

# Roles for role diversity bonus (pick at least 1 from each role category)
_POOL_ROLES: dict[int, str] = {
    0: "phy_atk", 1: "spe_atk", 2: "phy_atk", 3: "support",
    4: "spe_atk", 5: "phy_atk", 6: "spe_atk", 7: "support",
    8: "phy_atk", 9: "support", 10: "spe_atk", 11: "phy_atk",
    12: "spe_atk", 13: "phy_atk", 14: "wall", 15: "phy_atk",
    16: "phy_atk", 17: "wall", 18: "phy_atk", 19: "wall",
}


def _team_coverage_reward(picks: list[int]) -> float:
    """Small shaped reward for team composition quality.

    Rewards:
      Per pick: +0.05 for each NEW type added to the team's coverage.
      Final (6th pick) bonus:
        +0.2 if team covers >= 10 types
        +0.1 if team has >= 3 distinct roles
        -0.1 if team covers < 6 types (too narrow)

    These are deliberately small relative to battle rewards (victory=30)
    to guide without overwhelming the battle learning.
    """
    if not picks:
        return 0.0

    latest = picks[-1]
    prev_types = set()
    for p in picks[:-1]:
        prev_types |= _POOL_TYPES.get(p, set())
    new_types = _POOL_TYPES.get(latest, set()) - prev_types
    reward = 0.05 * len(new_types)

    # Final pick bonus
    if len(picks) == 6:
        all_types = set()
        roles = set()
        for p in picks:
            all_types |= _POOL_TYPES.get(p, set())
            roles.add(_POOL_ROLES.get(p, "other"))
        if len(all_types) >= 10:
            reward += 0.2
        elif len(all_types) < 6:
            reward -= 0.1
        if len(roles) >= 3:
            reward += 0.1

    return reward


class SinglesAgentWrapper(SingleAgentWrapper):
    """SingleAgentWrapper patched to handle team preview in singles formats.

    poke-env's SingleAgentWrapper only supports team preview for VGC (doubles).
    This subclass handles singles team preview by having the opponent auto-pick
    via random_teampreview, then converting the team order to an action.
    """

    def step(self, action):
        assert self.env.battle2 is not None
        if not self.env.battle2.teampreview:
            # Normal battle turn — delegate to parent
            return super().step(action)

        # Singles team preview: opponent picks randomly, we pass through
        tp_order = self.opponent.teampreview(self.env.battle2)
        # Mark selected pokemon
        picks = [int(c) for c in tp_order.replace("/team ", "")]
        for i, mon in enumerate(self.env.battle2.team.values(), start=1):
            mon._selected_in_teampreview = i in picks

        # Convert opponent's team order to an action the env can process
        opp_action = self.env.order_to_action(
            Player.create_order(list(self.env.battle2.team.values())[picks[0] - 1]),
            self.env.battle2,
            fake=True,
            strict=False,
        )

        actions = {
            self.env.agent1.username: action,
            self.env.agent2.username: opp_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        return (
            obs[self.env.agent1.username],
            rewards[self.env.agent1.username],
            terms[self.env.agent1.username],
            truncs[self.env.agent1.username],
            infos[self.env.agent1.username],
        )

# Default opponent team: first 6 from the pool (used when format requires own team)
_DEFAULT_OPPONENT_TEAM = build_team_string(list(range(6)))

OBS_TEAM_SIZE = 1 + POOL_SIZE + OBS_SIZE  # 1 + 20 + 86 = 107
OBS_TEAM_SIZE_HIST = 1 + POOL_SIZE + OBS_SIZE_HIST  # 1 + 20 + 430 = 451
N_TEAM_ACTIONS = 26                        # same as battle action space


class TeamBuildingEnv(gymnasium.Env):
    """
    Gymnasium env that adds a team-selection phase before each battle.

    Parameters
    ----------
    battle_env : SingleAgentWrapper
        The underlying poke_env single-agent battle env.
    team_size : int
        How many Pokemon to pick (default 6).
    """

    metadata = {"render_modes": []}

    def __init__(self, battle_env: SingleAgentWrapper, team_size: int = 6,
                 use_history: bool = False):
        super().__init__()
        self.battle_env = battle_env
        self.team_size = team_size
        self.use_history = use_history

        self._battle_obs_size = OBS_SIZE_HIST if use_history else OBS_SIZE
        self._obs_size = 1 + POOL_SIZE + self._battle_obs_size

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_TEAM_ACTIONS)

        self._phase = 0          # 0 = selecting, 1 = battling
        self._picks: list[int] = []  # pool indices chosen so far

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _selection_obs(self) -> np.ndarray:
        obs = np.zeros(self._obs_size, dtype=np.float32)
        obs[0] = 0.0  # phase = selecting
        for idx in self._picks:
            obs[1 + idx] = 1.0
        # Encode pick progress in the first battle-obs slot so the agent
        # can distinguish pick 1 from pick 6
        if self._battle_obs_size > 0:
            obs[1 + POOL_SIZE] = len(self._picks) / self.team_size
        return obs

    def _wrap_battle_obs(self, battle_obs: np.ndarray) -> np.ndarray:
        obs = np.zeros(self._obs_size, dtype=np.float32)
        obs[0] = 1.0  # phase = battling
        # selection flags stay zero (team already chosen)
        end = min(1 + POOL_SIZE + len(battle_obs), self._obs_size)
        obs[1 + POOL_SIZE: end] = battle_obs[:end - 1 - POOL_SIZE]
        return obs

    def _apply_team(self):
        """Format the selected team and set it on the inner env's agent."""
        team_str = build_team_string(self._picks)
        # Update team on both env players so the server receives the right team
        self.battle_env.env.agent1.update_team(team_str)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._phase = 0
        self._picks = []
        return self._selection_obs(), {}

    def step(self, action: int):
        action = int(action)

        # ── Phase 0: team selection ──────────────────────────────────────────
        if self._phase == 0:
            pick = action % POOL_SIZE
            # Avoid duplicates: if already picked, take the next free slot
            if pick in self._picks:
                for i in range(POOL_SIZE):
                    if i not in self._picks:
                        pick = i
                        break

            self._picks.append(pick)

            # Small shaped reward for type coverage
            pick_reward = _team_coverage_reward(self._picks)

            if len(self._picks) < self.team_size:
                # Need more picks
                return self._selection_obs(), pick_reward, False, False, {}

            # All picks done — apply team and start battle
            self._apply_team()
            self._phase = 1
            battle_obs, info = self.battle_env.reset()
            return self._wrap_battle_obs(battle_obs), pick_reward, False, False, info

        # ── Phase 1: battle ──────────────────────────────────────────────────
        action = remap_action_gen9(action)
        battle_obs, reward, terminated, truncated, info = self.battle_env.step(np.int64(action))
        wrapped = self._wrap_battle_obs(battle_obs)

        if terminated or truncated:
            # Reset selection state for next episode
            self._phase = 0
            self._picks = []

        return wrapped, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO."""
        if self._phase == 0:
            # Team selection: mask out already-picked pool slots
            # Actions 0-19 map directly to pool indices; 20-25 wrap via % POOL_SIZE
            mask = np.zeros(N_TEAM_ACTIONS, dtype=np.float32)
            for a in range(N_TEAM_ACTIONS):
                pool_idx = a % POOL_SIZE
                if pool_idx not in self._picks:
                    mask[a] = 1.0
            if mask.sum() == 0:
                mask[:] = 1.0  # safety fallback
            return mask

        # Phase 1: battle — use the battle action mask
        poke_env = self.battle_env.env if hasattr(self.battle_env, 'env') else self.battle_env
        battle = None
        if hasattr(poke_env, 'agent1'):
            battles = poke_env.agent1.battles
            if battles:
                battle = list(battles.values())[-1]
        if battle is not None:
            return _build_action_mask(battle)

        # Fallback: allow switches and normal moves, block mega/zmove/dynamax
        mask = np.ones(N_TEAM_ACTIONS, dtype=np.float32)
        mask[10:22] = 0.0
        return mask

    def render(self):
        pass

    def close(self):
        self.battle_env.close()


def make_team_env(
    opponent_type: str = "random",
    battle_format: str = "gen9bssregj",
    team_size: int = 6,
    use_history: bool = False,
) -> TeamBuildingEnv:
    """Create a TeamBuildingEnv backed by a real poke_env battle env."""
    inner = PokemonSinglesEnv(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
        strict=False,
        team=_DEFAULT_OPPONENT_TEAM,
        use_history=use_history,
        choose_on_teampreview=False,
    )

    if opponent_type == "random":
        opponent = RandomPlayer(
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            team=_DEFAULT_OPPONENT_TEAM,
        )
    elif opponent_type == "heuristic":
        opponent = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            team=_DEFAULT_OPPONENT_TEAM,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    wrapped = SinglesAgentWrapper(inner, opponent)
    return TeamBuildingEnv(wrapped, team_size=team_size, use_history=use_history)


def make_masked_team_env(
    opponent_type: str = "random",
    battle_format: str = "gen9bssregj",
    team_size: int = 6,
    use_history: bool = False,
):
    """Create TeamBuildingEnv with Monitor, keeping action_masks() visible."""
    from stable_baselines3.common.monitor import Monitor

    class MonitoredMaskableTeamEnv(gymnasium.Wrapper):
        """Monitor wrapper that exposes action_masks() from inner TeamBuildingEnv."""
        def action_masks(self) -> np.ndarray:
            inner = self.env
            while hasattr(inner, 'env') and not hasattr(inner, 'action_masks'):
                inner = inner.env
            if hasattr(inner, 'action_masks'):
                return inner.action_masks()
            return np.ones(N_TEAM_ACTIONS, dtype=np.float32)

    team_env = make_team_env(opponent_type, battle_format, team_size, use_history)
    monitored = Monitor(team_env)
    return MonitoredMaskableTeamEnv(monitored)

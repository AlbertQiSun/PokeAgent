"""
Team-building RL environment.

Extends the standard battle env with a team-selection phase:
  Phase 0 (team_select): Agent picks 6 Pokemon from the pool of 20 sequentially.
  Phase 1 (battle): Normal battle with the selected team.

Unified spaces:
  observation_space: Box(shape=(OBS_TEAM_SIZE,))
    [0]       phase flag  (0.0 = selecting, 1.0 = battling)
    [1..20]   one-hot selected flags for pool positions
    [21..100] battle observation (80-dim from embed_battle)
    Total: 1 + 20 + 80 = 101

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
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

from rl_env import PokemonSinglesEnv, OBS_SIZE, OBS_SIZE_HIST, N_ACTIONS, remap_action_gen9, _build_action_mask
from poke_env.environment import SingleAgentWrapper
from pokemon_pool import POOL, POOL_SIZE, build_team_string

# Default opponent team: first 6 from the pool (used when format requires own team)
_DEFAULT_OPPONENT_TEAM = build_team_string(list(range(6)))

OBS_TEAM_SIZE = 1 + POOL_SIZE + OBS_SIZE  # 1 + 20 + 80 = 101
OBS_TEAM_SIZE_HIST = 1 + POOL_SIZE + OBS_SIZE_HIST  # 1 + 20 + 400 = 421
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
        # battle obs slots remain zero during selection
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

            if len(self._picks) < self.team_size:
                # Need more picks
                return self._selection_obs(), 0.0, False, False, {}

            # All picks done — apply team and start battle
            self._apply_team()
            self._phase = 1
            battle_obs, info = self.battle_env.reset()
            return self._wrap_battle_obs(battle_obs), 0.0, False, False, info

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
            # Team selection: all 26 actions valid (action % POOL_SIZE picks a slot)
            # But mask out already-picked slots mapped directly
            mask = np.ones(N_TEAM_ACTIONS, dtype=np.float32)
            # Slightly prefer unpicked slots by masking direct hits on picked indices
            # But since action % POOL_SIZE handles collisions, all actions are technically valid
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
    battle_format: str = "gen9anythinggoes",
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

    wrapped = SingleAgentWrapper(inner, opponent)
    return TeamBuildingEnv(wrapped, team_size=team_size, use_history=use_history)


def make_masked_team_env(
    opponent_type: str = "random",
    battle_format: str = "gen9anythinggoes",
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

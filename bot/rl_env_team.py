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

from rl_env import PokemonSinglesEnv, OBS_SIZE
from poke_env.environment import SingleAgentWrapper
from pokemon_pool import POOL, POOL_SIZE, build_team_string

# Default opponent team: first 6 from the pool (used when format requires own team)
_DEFAULT_OPPONENT_TEAM = build_team_string(list(range(6)))

OBS_TEAM_SIZE = 1 + POOL_SIZE + OBS_SIZE  # 1 + 20 + 80 = 101
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

    def __init__(self, battle_env: SingleAgentWrapper, team_size: int = 6):
        super().__init__()
        self.battle_env = battle_env
        self.team_size = team_size

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_TEAM_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_TEAM_ACTIONS)

        self._phase = 0          # 0 = selecting, 1 = battling
        self._picks: list[int] = []  # pool indices chosen so far

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _selection_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
        obs[0] = 0.0  # phase = selecting
        for idx in self._picks:
            obs[1 + idx] = 1.0
        # battle obs slots (21..100) remain zero during selection
        return obs

    def _wrap_battle_obs(self, battle_obs: np.ndarray) -> np.ndarray:
        obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
        obs[0] = 1.0  # phase = battling
        # selection flags stay zero (team already chosen)
        obs[1 + POOL_SIZE: 1 + POOL_SIZE + OBS_SIZE] = battle_obs
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
        battle_obs, reward, terminated, truncated, info = self.battle_env.step(np.int64(action))
        wrapped = self._wrap_battle_obs(battle_obs)

        if terminated or truncated:
            # Reset selection state for next episode
            self._phase = 0
            self._picks = []

        return wrapped, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.battle_env.close()


def make_team_env(
    opponent_type: str = "random",
    battle_format: str = "gen9anythinggoes",
    team_size: int = 6,
) -> TeamBuildingEnv:
    """Create a TeamBuildingEnv backed by a real poke_env battle env."""
    # Both env agents (agent1 = RL player, agent2 = internal bot) need a team
    # for bring-your-own-team formats.  We give agent2 the default opponent team;
    # agent1's team is overridden before each battle by TeamBuildingEnv._apply_team().
    inner = PokemonSinglesEnv(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        start_listening=True,
        strict=False,
        team=_DEFAULT_OPPONENT_TEAM,   # default for agent2; agent1 gets updated dynamically
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
    return TeamBuildingEnv(wrapped, team_size=team_size)

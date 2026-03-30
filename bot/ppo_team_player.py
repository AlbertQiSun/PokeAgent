"""
PPO player that uses the team-building model.

Full BSS flow:
  1. next_team  — model picks 6 Pokemon from the pool
  2. teampreview— type-matchup scorer picks the best 3 to bring
  3. choose_move— model's battle policy picks moves each turn
"""

import numpy as np
from stable_baselines3 import PPO
from poke_env.player import Player
from poke_env.environment import SinglesEnv
from poke_env.battle import AbstractBattle, Battle

from rl_env import embed_battle, OBS_SIZE
from rl_env_team import OBS_TEAM_SIZE
from pokemon_pool import POOL, POOL_SIZE, POOL_NAMES, build_team_string


def _selection_obs(picks: list[int]) -> np.ndarray:
    obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
    obs[0] = 0.0
    for idx in picks:
        obs[1 + idx] = 1.0
    return obs


def _battle_obs(battle: AbstractBattle) -> np.ndarray:
    obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
    obs[0] = 1.0
    obs[1 + POOL_SIZE: 1 + POOL_SIZE + OBS_SIZE] = embed_battle(battle)
    return obs


def _lead_score(our_mon, opp_mons) -> float:
    """
    Score our_mon against the opponent's team.
    Offensive: reward types that hit opponents super-effectively.
    Defensive: reward resisting common opponent types.
    """
    score = 0.0
    for opp in opp_mons:
        # Offensive score: best type effectiveness we can achieve
        best_eff = 1.0
        for move in our_mon.moves.values():
            if move.base_power > 0 and move.type is not None:
                try:
                    eff = opp.damage_multiplier(move)
                    best_eff = max(best_eff, eff)
                except Exception:
                    pass
        score += best_eff

        # Defensive score: how well do we resist their types?
        for opp_type in [opp.type_1, opp.type_2]:
            if opp_type is not None:
                try:
                    incoming = our_mon.damage_multiplier(opp_type)
                    if incoming < 1.0:
                        score += 0.5   # resist bonus
                    elif incoming > 1.0:
                        score -= 0.25  # weak penalty
                except Exception:
                    pass
    return score


class PPOTeamPlayer(Player):
    """
    Player that uses the team-building PPO model.

    Phase 1 — next_team: model picks 6 Pokemon from the pool.
    Phase 2 — teampreview: type-matchup scorer selects 3 to bring.
    Phase 3 — choose_move: model's battle policy picks moves.
    """

    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PPO.load(model_path, device="cpu")
        self._selected_team: list[int] = []

    # ── Phase 1: build team of 6 ──────────────────────────────────────────────

    def _select_team(self) -> list[int]:
        picks: list[int] = []
        while len(picks) < 6:
            obs = _selection_obs(picks).reshape(1, -1)
            action, _ = self.model.predict(obs, deterministic=True)
            pick = int(action[0]) % POOL_SIZE
            if pick in picks:
                for i in range(POOL_SIZE):
                    if i not in picks:
                        pick = i
                        break
            picks.append(pick)
        return picks

    @property
    def next_team(self) -> str:  # type: ignore[override]
        self._selected_team = self._select_team()
        names = [POOL_NAMES[i] for i in self._selected_team]
        print(f"[PPOTeamPlayer] Built team of 6: {', '.join(names)}")
        team_str = build_team_string(self._selected_team)
        self.update_team(team_str)
        return self._team.yield_team()

    # ── Phase 2: pick best 3 from our 6 based on opponent team preview ────────

    def teampreview(self, battle: AbstractBattle) -> str:
        our_mons  = list(battle.team.values())
        opp_mons  = list(battle.teampreview_opponent_team)

        scores = [_lead_score(mon, opp_mons) for mon in our_mons]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        picks  = ranked[:3]   # top-3 indices (0-based)

        chosen_names = [our_mons[i].species for i in picks]
        print(f"[PPOTeamPlayer] Bringing 3: {', '.join(chosen_names)}")

        # Showdown expects 1-indexed: /team 135  (lead is first)
        return "/team " + "".join(str(i + 1) for i in picks)

    # ── Phase 3: battle ───────────────────────────────────────────────────────

    def choose_move(self, battle: Battle):
        obs = _battle_obs(battle).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action[0])
        try:
            order = SinglesEnv.action_to_order(
                np.int64(action), battle, fake=False, strict=False
            )
            return order
        except Exception:
            return self.choose_random_move(battle)

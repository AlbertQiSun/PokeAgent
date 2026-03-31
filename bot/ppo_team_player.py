"""
PPO player that uses the team-building model.

Full BSS flow:
  1. next_team  — model picks a NEW team of 6 every battle (stochastic sampling)
  2. teampreview— type-matchup scorer picks the best 3 to bring
  3. choose_move— model's battle policy picks moves (with optional history)
"""

import numpy as np
import torch
from poke_env.player import Player
from poke_env.environment import SinglesEnv
from poke_env.battle import AbstractBattle, Battle

from rl_env import embed_battle, OBS_SIZE, OBS_SIZE_HIST, BattleHistory, N_ACTIONS


def _load_model(path: str):
    """Load either MaskablePPO or PPO model (backward compatible)."""
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    except Exception:
        from stable_baselines3 import PPO
        return PPO.load(path, device="cpu")
from rl_env_team import OBS_TEAM_SIZE, OBS_TEAM_SIZE_HIST
from pokemon_pool import POOL, POOL_SIZE, POOL_NAMES, build_team_string


def _selection_obs(picks: list[int]) -> np.ndarray:
    obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
    obs[0] = 0.0
    for idx in picks:
        obs[1 + idx] = 1.0
    return obs


def _battle_obs(battle: AbstractBattle, history: BattleHistory = None) -> np.ndarray:
    obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
    obs[0] = 1.0
    if history:
        hist_emb = history.embed_with_history(battle)
        end = min(1 + POOL_SIZE + len(hist_emb), OBS_TEAM_SIZE)
        obs[1 + POOL_SIZE: end] = hist_emb[:end - 1 - POOL_SIZE]
    else:
        obs[1 + POOL_SIZE: 1 + POOL_SIZE + OBS_SIZE] = embed_battle(battle)
    return obs


def _valid_action_mask(battle: Battle) -> np.ndarray:
    """Boolean mask of which action indices (0-25) produce valid battle orders."""
    valid_strs = {str(o) for o in battle.valid_orders}
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for a in range(N_ACTIONS):
        try:
            order = SinglesEnv.action_to_order(np.int64(a), battle, fake=True, strict=True)
            if str(order) in valid_strs:
                mask[a] = True
        except (ValueError, IndexError):
            pass
    return mask


def _lead_score(our_mon, opp_mons) -> float:
    score = 0.0
    for opp in opp_mons:
        best_eff = 1.0
        for move in our_mon.moves.values():
            if move.base_power > 0 and move.type is not None:
                try:
                    eff = opp.damage_multiplier(move)
                    best_eff = max(best_eff, eff)
                except Exception:
                    pass
        score += best_eff
        for opp_type in [opp.type_1, opp.type_2]:
            if opp_type is not None:
                try:
                    incoming = our_mon.damage_multiplier(opp_type)
                    if incoming < 1.0:
                        score += 0.5
                    elif incoming > 1.0:
                        score -= 0.25
                except Exception:
                    pass
    return score


class PPOTeamPlayer(Player):
    """
    PPO team-building + battle player.
    Builds a fresh team each battle (stochastic), picks 3 leads by matchup,
    battles with optional history context.
    """

    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = _load_model(model_path)
        self._selected_team: list[int] = []
        self._history = BattleHistory()   # for battle-phase history tracking

    # ── Phase 1: build a NEW team of 6 every battle ──────────────────────────

    def _select_team(self) -> list[int]:
        """Stochastic team selection — samples from policy distribution."""
        picks: list[int] = []
        while len(picks) < 6:
            obs = _selection_obs(picks).reshape(1, -1)
            # deterministic=False → sample from the action distribution
            # so the team varies between battles
            action, _ = self.model.predict(obs, deterministic=False)
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

    # ── Phase 2: pick best 3 from our 6 vs opponent ─────────────────────────

    def teampreview(self, battle: AbstractBattle) -> str:
        our_mons = list(battle.team.values())
        opp_mons = list(battle.teampreview_opponent_team)

        scores = [_lead_score(mon, opp_mons) for mon in our_mons]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        picks  = ranked[:3]

        chosen_names = [our_mons[i].species for i in picks]
        print(f"[PPOTeamPlayer] Bringing 3: {', '.join(chosen_names)}")
        return "/team " + "".join(str(i + 1) for i in picks)

    # ── Phase 3: battle with history context ─────────────────────────────────

    def choose_move(self, battle: Battle):
        obs = _battle_obs(battle, self._history).reshape(1, -1)

        # Get action probabilities, mask invalid actions, pick best valid one
        obs_tensor = torch.as_tensor(obs).float().to(self.model.device)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs[0].cpu().numpy()

        mask = _valid_action_mask(battle)
        masked_probs = probs * mask

        if masked_probs.sum() > 0:
            action = int(np.argmax(masked_probs))
            try:
                return SinglesEnv.action_to_order(
                    np.int64(action), battle, fake=False, strict=False
                )
            except Exception:
                pass

        return self.choose_random_move(battle)

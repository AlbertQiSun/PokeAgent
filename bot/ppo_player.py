"""
PPO-trained RL player for Pokemon Showdown.

Uses the trained stable-baselines3 PPO model to choose moves,
via the same observation embedding used during training (rl_env.py).

Supports both 86-dim (no history) and 430-dim (5-frame history) models.
"""

import numpy as np
import torch
from poke_env.player import Player
from poke_env.environment import SinglesEnv
from poke_env.battle import Battle

from rl_env import embed_battle, OBS_SIZE, OBS_SIZE_HIST, BattleHistory, N_ACTIONS


def _load_model(path: str):
    """Load either MaskablePPO or PPO model (backward compatible)."""
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, device="cpu")
    except Exception:
        from stable_baselines3 import PPO
        return PPO.load(path, device="cpu")


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


class PPOPlayer(Player):
    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = _load_model(model_path)

        # Detect whether model expects history obs
        expected_obs = self.model.observation_space.shape[0]
        self.use_history = (expected_obs == OBS_SIZE_HIST)
        self._history = BattleHistory() if self.use_history else None

        if self.use_history:
            print(f"[PPOPlayer] Using 5-frame history ({expected_obs}-dim obs)")

    def choose_move(self, battle: Battle):
        if self._history:
            obs = self._history.embed_with_history(battle)
        else:
            obs = embed_battle(battle)

        # Get action probabilities, mask invalid actions, pick best valid one
        obs_tensor = torch.as_tensor(obs.reshape(1, -1)).float().to(self.model.device)
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

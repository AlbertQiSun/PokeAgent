"""
PPO-trained RL player for Pokemon Showdown.

Uses the trained stable-baselines3 PPO model to choose moves,
via the same observation embedding used during training (rl_env.py).
"""

import numpy as np
from stable_baselines3 import PPO
from poke_env.player import Player
from poke_env.environment import SinglesEnv
from poke_env.battle import AbstractBattle, Battle

from rl_env import embed_battle, OBS_SIZE


class PPOPlayer(Player):
    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PPO.load(model_path, device="cpu")
        # Dummy env needed to resolve action → order mapping
        self._singles_env = None

    def choose_move(self, battle: Battle):
        obs = embed_battle(battle).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action[0])

        # Convert action index to BattleOrder using SinglesEnv's static method
        try:
            order = SinglesEnv.action_to_order(
                np.int64(action), battle, fake=False, strict=False
            )
            return order
        except Exception:
            return self.choose_random_move(battle)

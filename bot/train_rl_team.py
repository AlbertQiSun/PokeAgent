"""
PPO training with team-building phase.

The agent learns BOTH to select a team from the pool (6 picks from 20)
AND to battle with it.  Uses a unified 101-dim obs space and 26-action space.

Usage:
    # Train from scratch vs random opponent (BSS format)
    python train_rl_team.py

    # Train vs heuristic for 500k steps
    python train_rl_team.py --steps 500000 --opponent heuristic

    # Evaluate saved model
    python train_rl_team.py --eval --model-path ppo_team_builder
"""

import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_env_team import make_team_env, OBS_TEAM_SIZE
from pokemon_pool import POOL_SIZE


def get_device() -> str:
    if torch.cuda.is_available():
        import os
        return "cuda" if os.environ.get("FORCE_GPU") else "cpu"
    return "cpu"


class WinRateCallback(BaseCallback):
    def __init__(self, log_freq: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.total = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.total += 1
                if info["episode"]["r"] > 0:
                    self.wins += 1
                if self.total % self.log_freq == 0:
                    wr = self.wins / self.total * 100
                    print(
                        f"  [Step {self.num_timesteps:>8}] "
                        f"Episodes: {self.total}  Win rate: {wr:.1f}%"
                    )
        return True


def train(args):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Setting up team-building env (opponent={args.opponent}, format={args.format})...")

    env = Monitor(make_team_env(opponent_type=args.opponent, battle_format=args.format))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./ppo_team_logs/",
        device=device,
    )

    print(
        f"Obs size: {OBS_TEAM_SIZE}  "
        f"(1 phase + {POOL_SIZE} pool flags + 80 battle)\n"
        f"Action size: 26  (0-{POOL_SIZE-1} = pool picks, 0-25 = battle actions)\n"
        f"Training for {args.steps:,} steps..."
    )

    model.learn(
        total_timesteps=args.steps,
        callback=WinRateCallback(log_freq=50),
        progress_bar=True,
    )
    model.save(args.model_path)
    print(f"\nSaved model to {args.model_path}.zip")
    env.close()


def evaluate(args):
    device = get_device()
    print(f"Loading model from {args.model_path}.zip ...")
    env = make_team_env(opponent_type=args.opponent, battle_format=args.format)
    model = PPO.load(args.model_path, env=env, device=device)

    wins, n = 0, args.eval_episodes
    obs, _ = env.reset()
    episode_count = 0
    episode_reward = 0.0

    print(f"Evaluating {n} episodes vs {args.opponent}...")
    while episode_count < n:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            if episode_reward > 0:
                wins += 1
            episode_count += 1
            episode_reward = 0.0
            obs, _ = env.reset()

    print(f"Win rate: {wins}/{n} = {wins/n*100:.1f}%")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",         type=int, default=300_000)
    parser.add_argument("--opponent",      type=str, default="random",
                        choices=["random", "heuristic"])
    parser.add_argument("--format",        type=str, default="gen9anythinggoes",
                        help="Battle format with custom teams (gen9anythinggoes, gen9ou, gen9bssregj)")
    parser.add_argument("--model-path",    type=str, default="ppo_team_builder")
    parser.add_argument("--eval",          action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)

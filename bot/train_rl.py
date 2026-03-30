"""
PPO RL training for Pokemon battles using stable-baselines3.

Optionally loads a behavioral cloning checkpoint to warm-start the policy.

Usage:
    # Train from scratch vs random opponent
    python train_rl.py

    # Warm-start from BC checkpoint, train vs heuristic opponent
    python train_rl.py --bc-weights bc_policy.pth --opponent heuristic --steps 500000

    # Evaluate a saved model
    python train_rl.py --eval --model-path ppo_pokemon.zip
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_env import make_env, OBS_SIZE, N_ACTIONS


def get_device() -> str:
    """Return the best available device string for stable-baselines3."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


# ── Callback: print win rate every N episodes ────────────────────────────────
class WinRateCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.total = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.total += 1
                # reward > 0 at episode end = win
                if info["episode"]["r"] > 0:
                    self.wins += 1
                if self.total % self.log_freq == 0:
                    wr = self.wins / self.total * 100
                    print(f"  [Step {self.num_timesteps:>8}] Episodes: {self.total}  Win rate: {wr:.1f}%")
        return True


# ── Optional: load BC weights into PPO policy ────────────────────────────────
def load_bc_weights(model: PPO, bc_path: str):
    """Copy BC policy network weights into the PPO actor network."""
    from bc_pretrain import PolicyNet
    bc = PolicyNet(OBS_SIZE, N_ACTIONS)
    bc.load_state_dict(torch.load(bc_path, map_location="cpu"))
    bc = bc.to(next(model.policy.parameters()).device)

    # PPO's mlp_extractor + action_net map to our bc.net layers
    policy = model.policy
    with torch.no_grad():
        # Copy first two linear layers into mlp_extractor.policy_net
        sbl_layers = list(policy.mlp_extractor.policy_net.children())
        bc_layers  = [l for l in bc.net.children() if isinstance(l, nn.Linear)]

        for sbl_l, bc_l in zip(sbl_layers, bc_layers[:-1]):
            if isinstance(sbl_l, nn.Linear) and sbl_l.weight.shape == bc_l.weight.shape:
                sbl_l.weight.data.copy_(bc_l.weight.data)
                sbl_l.bias.data.copy_(bc_l.bias.data)

        # Copy final layer into action_net
        last_bc = bc_layers[-1]
        if (hasattr(policy, "action_net") and
                isinstance(policy.action_net, nn.Linear) and
                policy.action_net.weight.shape == last_bc.weight.shape):
            policy.action_net.weight.data.copy_(last_bc.weight.data)
            policy.action_net.bias.data.copy_(last_bc.bias.data)

    print(f"Loaded BC weights from {bc_path}")


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args):
    device = get_device()
    print(f"Setting up environment (opponent={args.opponent})...")
    env = Monitor(make_env(opponent_type=args.opponent))

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
        tensorboard_log="./ppo_logs/",
        device=device,
    )

    if args.bc_weights:
        load_bc_weights(model, args.bc_weights)

    callback = WinRateCallback(log_freq=500)

    print(f"Training PPO for {args.steps:,} steps...")
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=True)

    model.save(args.model_path)
    print(f"\nSaved model to {args.model_path}.zip")
    env.close()


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(args):
    device = get_device()
    print(f"Loading model from {args.model_path}.zip ...")
    env = make_env(opponent_type=args.opponent)
    model = PPO.load(args.model_path, env=env, device=device)

    wins = 0
    n = args.eval_episodes
    obs, _ = env.reset()
    episode_count = 0
    episode_reward = 0.0

    print(f"Evaluating over {n} episodes...")
    while episode_count < n:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            if episode_reward > 0:
                wins += 1
            episode_count += 1
            episode_reward = 0.0
            obs, _ = env.reset()

    print(f"\nWin rate vs {args.opponent}: {wins}/{n} = {wins/n*100:.1f}%")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",         type=int,   default=200_000)
    parser.add_argument("--opponent",      type=str,   default="random",
                        choices=["random", "heuristic"])
    parser.add_argument("--bc-weights",    type=str,   default=None,
                        help="Path to BC pretrained weights (.pth)")
    parser.add_argument("--model-path",    type=str,   default="ppo_pokemon")
    parser.add_argument("--eval",          action="store_true")
    parser.add_argument("--eval-episodes", type=int,   default=100)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)

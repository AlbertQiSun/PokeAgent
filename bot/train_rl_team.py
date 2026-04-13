"""
PPO training with team-building phase.

The agent learns BOTH to select a team from the pool (6 picks from 20)
AND to battle with it.  Uses a unified obs space and 26-action space.

Usage:
    # Train from scratch vs random opponent (BSS format)
    python train_rl_team.py

    # Train vs heuristic for 500k steps
    python train_rl_team.py --steps 500000 --opponent heuristic

    # Resume training from existing checkpoint
    python train_rl_team.py --steps 1000000 --opponent heuristic --resume

    # Evaluate saved model
    python train_rl_team.py --eval --model-path ppo_team_builder
"""

import argparse
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_env_team import make_team_env, make_masked_team_env, OBS_TEAM_SIZE, OBS_TEAM_SIZE_HIST
from pokemon_pool import POOL_SIZE


def get_device() -> str:
    import os
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if os.environ.get("FORCE_CPU"):
        device = "cpu"
    print(f"Using device: {device}  (set FORCE_CPU=1 to override)")
    return device


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """Linear schedule from initial_value to final_value."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class WinRateCallback(BaseCallback):
    """Tracks both cumulative and windowed (recent) win rates."""

    def __init__(self, log_freq: int = 50, window: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.window = window
        self.wins = 0
        self.total = 0
        self._recent: list[bool] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.total += 1
                won = info["episode"]["r"] > 0
                if won:
                    self.wins += 1
                self._recent.append(won)
                if len(self._recent) > self.window:
                    self._recent.pop(0)

                if self.total % self.log_freq == 0:
                    cum_wr = self.wins / self.total * 100
                    recent_wr = sum(self._recent) / len(self._recent) * 100
                    lr = self.model.lr_schedule(self.model._current_progress_remaining)
                    print(
                        f"  [Step {self.num_timesteps:>8}] "
                        f"Ep: {self.total}  "
                        f"WR: {cum_wr:.1f}%  "
                        f"Recent({len(self._recent)}): {recent_wr:.1f}%  "
                        f"LR: {lr:.2e}"
                    )
        return True


def train(args):
    device = get_device()
    use_hist = args.history
    obs_dim = OBS_TEAM_SIZE_HIST if use_hist else OBS_TEAM_SIZE
    print(f"Using device: {device}")
    print(f"Setting up team-building env (opponent={args.opponent}, format={args.format}, "
          f"history={use_hist}, obs={obs_dim})...")

    env = make_masked_team_env(
        opponent_type=args.opponent,
        battle_format=args.format,
        use_history=use_hist,
    )

    if args.resume:
        import os
        model_file = args.model_path + ".zip"
        if not os.path.exists(model_file):
            print(f"ERROR: Cannot resume — {model_file} not found.")
            return
        print(f"Resuming from {model_file}...")
        model = MaskablePPO.load(
            args.model_path,
            env=env,
            device=device,
        )
        # For resume, use a constant lower LR (schedule breaks with cumulative progress)
        resume_lr = args.lr * 0.5
        model.learning_rate = resume_lr
        model.ent_coef = args.ent_coef
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.tensorboard_log = "./ppo_team_logs/"
        print(f"  Loaded. Continuing for {args.steps:,} more steps.")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=linear_schedule(args.lr, args.lr * 0.1),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs=dict(net_arch=args.net_arch),
            tensorboard_log="./ppo_team_logs/",
            device=device,
        )

    battle_obs_str = f"400 history" if use_hist else "80 battle"
    print(
        f"Obs size: {obs_dim}  "
        f"(1 phase + {POOL_SIZE} pool flags + {battle_obs_str})\n"
        f"Action size: 26  (0-{POOL_SIZE-1} = pool picks, 0-25 = battle actions)\n"
        f"Net arch: {args.net_arch}  |  LR: {args.lr} → {args.lr*0.1:.1e}  "
        f"|  Entropy: {args.ent_coef}  |  Batch: {args.batch_size}  |  Rollout: {args.n_steps}\n"
        f"Training MaskablePPO for {args.steps:,} steps..."
    )

    model.learn(
        total_timesteps=args.steps,
        callback=WinRateCallback(log_freq=50, window=200),
        progress_bar=True,
        reset_num_timesteps=not args.resume,
    )
    model.save(args.model_path)
    print(f"\nSaved model to {args.model_path}.zip")
    env.close()


def evaluate(args):
    device = get_device()
    print(f"Loading model from {args.model_path}.zip ...")
    env = make_team_env(opponent_type=args.opponent, battle_format=args.format,
                        use_history=args.history)
    model = MaskablePPO.load(args.model_path, env=env, device=device)

    wins, n = 0, args.eval_episodes
    obs, _ = env.reset()
    episode_count = 0
    episode_reward = 0.0

    print(f"Evaluating {n} episodes vs {args.opponent}...")
    while episode_count < n:
        action_masks = env.action_masks() if hasattr(env, 'action_masks') else None
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
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
    parser.add_argument("--history",       action="store_true",
                        help="Use 5-frame history stacking (421-dim obs)")

    # Training hyperparameters
    parser.add_argument("--resume",        action="store_true",
                        help="Resume training from --model-path checkpoint")
    parser.add_argument("--lr",            type=float, default=3e-4,
                        help="Initial learning rate (decays to lr*0.1)")
    parser.add_argument("--ent-coef",      type=float, default=0.02,
                        help="Entropy coefficient (exploration vs exploitation)")
    parser.add_argument("--n-steps",       type=int, default=4096,
                        help="Rollout buffer size per update")
    parser.add_argument("--batch-size",    type=int, default=256,
                        help="Minibatch size for PPO updates")
    parser.add_argument("--net-arch",      type=int, nargs="+", default=[512, 256],
                        help="Hidden layer sizes for policy and value networks")

    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)

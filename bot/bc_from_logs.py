"""
Behavioral Cloning from Gemini battle logs.

Reads battle_logs/ directories, extracts (battle_context → action) pairs
from WINNING games, and trains a BC policy that can initialize PPO.

The key insight: Gemini's battle context already contains the full obs
description. We re-encode it into the same 86-dim obs space that PPO uses,
paired with the action Gemini chose.

Usage:
    # Build dataset from all winning battle logs
    python bc_from_logs.py --build

    # Train BC policy from the dataset
    python bc_from_logs.py --train --epochs 20

    # Both
    python bc_from_logs.py --build --train --epochs 20
"""

import argparse
import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Action space matches rl_env.py
N_ACTIONS = 26
# Obs matches rl_env_team.py: 1 phase + 20 pool + 86 battle = 107
OBS_BATTLE = 86
POOL_SIZE = 20
OBS_TEAM_SIZE = 1 + POOL_SIZE + OBS_BATTLE  # 107


def parse_hp(text: str) -> float:
    """Extract HP fraction from strings like '46%' or '100%'."""
    m = re.search(r'(\d+)%', text)
    return int(m.group(1)) / 100.0 if m else 1.0


def parse_move_power(text: str) -> float:
    """Extract base power from move line like 'moonblast  Fairy SPECIAL BP: 95'."""
    m = re.search(r'BP:\s*(\d+)', text)
    return int(m.group(1)) / 200.0 if m else 0.0


def parse_damage_range(text: str) -> tuple:
    """Extract damage range from '→  37.7- 44.4%'."""
    m = re.search(r'→\s+([\d.]+)-\s*([\d.]+)%', text)
    if m:
        return float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
    return 0.0, 0.0


def action_to_index(action_str: str, context: str) -> int | None:
    """Convert action string like 'move moonblast' or 'switch incineroar' to action index.

    For moves: try to find the move's position in the YOUR MOVES section (0-3) → action 6-9
    For switches: find the pokemon's position in SWITCH OPTIONS (0-5) → action 0-5
    """
    # Normalize: "/choose move hurricane" → "move hurricane"
    action_str = re.sub(r'^/choose\s+', '', action_str.strip())
    parts = action_str.strip().split(maxsplit=1)
    if len(parts) < 2:
        return None

    atype = parts[0].lower()
    target = re.sub(r'[^a-z0-9]', '', parts[1].lower())

    if atype == "move":
        # Find move position in context
        move_lines = re.findall(r'^\s+(\w+)\s+\w+\s+(PHYSICAL|SPECIAL|STATUS)', context, re.MULTILINE)
        for i, (move_id, _) in enumerate(move_lines):
            if i >= 4:
                break
            clean = re.sub(r'[^a-z0-9]', '', move_id.lower())
            if clean == target or target in clean or clean in target:
                return 6 + i  # move actions are 6-9
        # Fallback: action 6 (first move)
        return 6

    elif atype == "switch":
        # Find switch position in context (SWITCH OPTIONS section)
        switch_lines = re.findall(r'^\s+(\w+)\s+\(', context, re.MULTILINE)
        for i, mon in enumerate(switch_lines):
            if i >= 5:
                break
            clean = re.sub(r'[^a-z0-9]', '', mon.lower())
            if clean == target or target in clean or clean in target:
                return i  # switch actions are 0-5
        # Fallback: action 0 (first switch)
        return 0

    return None


def context_to_obs(context: str) -> np.ndarray:
    """Convert battle context text to 86-dim observation vector.

    We extract what we can from the text and map it to the same features
    the PPO agent sees. Not a perfect reconstruction, but captures the
    key signals: HP, type matchups, damage calcs, speed, bench status.
    """
    obs = np.zeros(OBS_BATTLE, dtype=np.float32)

    # Our active HP
    m = re.search(r'Your \w+:\s+(\d+)%', context)
    if m:
        obs[0] = int(m.group(1)) / 100.0

    # Opponent HP
    m = re.search(r'Opp \w+:\s+(\d+)%', context)
    if m:
        obs[37] = int(m.group(1)) / 100.0

    # Our moves damage (from YOUR MOVES section)
    move_section = re.findall(r'→\s+([\d.]+)-\s*([\d.]+)%', context)
    for i, (lo, hi) in enumerate(move_section[:4]):
        obs[1 + i*4] = float(lo) / 100.0  # damage low
        obs[2 + i*4] = float(hi) / 100.0  # damage high

    # Speed check
    if 'You outspeed' in context:
        obs[36] = 1.0
    elif 'Opponent outspeeds' in context:
        obs[36] = 0.0
    else:
        obs[36] = 0.5

    # Bench HP
    bench_matches = re.findall(r'(\w+)\((\d+)%\)', context)
    for i, (_, hp) in enumerate(bench_matches[:5]):
        obs[19 + i] = int(hp) / 100.0

    # Bench alive count
    bench_line = re.search(r'Your bench:(.+)', context)
    if bench_line:
        alive = len(re.findall(r'\d+%', bench_line.group(1)))
        obs[78] = alive / 5.0

    # Opponent bench
    opp_bench = re.search(r'Opp bench:.*?(\d+)\s+remaining', context)
    if opp_bench:
        obs[79] = int(opp_bench.group(1)) / 5.0

    # Turn number
    turn_m = re.search(r'TURN (\d+)', context)
    if turn_m:
        obs[80] = min(int(turn_m.group(1)), 50) / 50.0

    # OHKO indicator
    if 'OHKO' in context.split('YOUR MOVES')[0] if 'YOUR MOVES' in context else '':
        obs[81] = 1.0  # opponent can OHKO us

    our_moves = context.split('YOUR MOVES')[1] if 'YOUR MOVES' in context else ''
    if 'OHKO' in our_moves.split('OPPONENT')[0] if 'OPPONENT' in our_moves else our_moves:
        obs[82] = 1.0  # we can OHKO opponent

    # Opponent prediction probabilities (last 6 dims)
    pred_section = re.findall(r'(attack_physical|attack_special|attack_status|switch|setup|protect)\s+(\d+)%', context)
    cat_map = {'attack_physical': 0, 'attack_special': 1, 'attack_status': 2, 'switch': 3, 'setup': 4, 'protect': 5}
    for cat, pct in pred_section:
        if cat in cat_map:
            obs[80 + cat_map[cat]] = int(pct) / 100.0

    return obs


def build_dataset(log_dir: str = "battle_logs", output: str = "bc_gemini_data.npz",
                  wins_only: bool = True):
    """Build BC dataset from battle log directories."""
    obs_list = []
    act_list = []
    skipped = 0

    for battle_dir in sorted(os.listdir(log_dir)):
        summary_path = os.path.join(log_dir, battle_dir, "summary.json")
        if not os.path.isfile(summary_path):
            continue

        summary = json.load(open(summary_path))

        if wins_only and summary.get("result") != "WIN":
            continue

        # Process each decision
        for dec in summary.get("decisions", []):
            action_str = dec.get("action", "")

            # Read the corresponding turn file for context
            turn = dec.get("turn", 0)
            turn_file = os.path.join(log_dir, battle_dir, f"turn_{turn:02d}.txt")
            if not os.path.isfile(turn_file):
                skipped += 1
                continue

            context = open(turn_file).read()

            # Convert action to index
            action_idx = action_to_index(action_str, context)
            if action_idx is None:
                skipped += 1
                continue

            # Convert context to observation
            obs = context_to_obs(context)

            # Wrap in team obs format (phase=1.0 battling, pool flags=0)
            team_obs = np.zeros(OBS_TEAM_SIZE, dtype=np.float32)
            team_obs[0] = 1.0  # battle phase
            team_obs[1 + POOL_SIZE:] = obs  # battle obs at the end

            obs_list.append(team_obs)
            act_list.append(action_idx)

    if not obs_list:
        print("No valid samples found!")
        return

    obs_arr = np.stack(obs_list)
    act_arr = np.array(act_list, dtype=np.int64)

    np.savez(output, obs=obs_arr, actions=act_arr)
    print(f"Built BC dataset: {len(obs_arr)} samples from {output}")
    print(f"  Skipped: {skipped} decisions (no turn file or unparseable action)")
    print(f"  Action distribution:")
    for a in range(N_ACTIONS):
        count = (act_arr == a).sum()
        if count > 0:
            label = f"switch_{a}" if a < 6 else f"move_{a-6}" if a < 10 else f"tera_move_{a-18}"
            print(f"    {label:15s}: {count:4d} ({count/len(act_arr)*100:.1f}%)")


class BCDataset(Dataset):
    def __init__(self, path: str = "bc_gemini_data.npz"):
        data = np.load(path)
        self.obs = data["obs"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        print(f"Loaded BC dataset: {len(self.obs)} samples")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return self.obs[i], self.actions[i]


class PolicyNet(nn.Module):
    """Matches the architecture used by MaskablePPO's MlpPolicy [512, 256]."""
    def __init__(self, obs_size: int = OBS_TEAM_SIZE, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_bc(data_path: str = "bc_gemini_data.npz", epochs: int = 20,
             output: str = "bc_gemini_policy.pth", batch_size: int = 128):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = BCDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PolicyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            logits = model(obs_batch)
            loss = loss_fn(logits, act_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(obs_batch)
            correct += (logits.argmax(dim=1) == act_batch).sum().item()

        avg_loss = total_loss / len(dataset)
        acc = correct / len(dataset) * 100
        print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}  acc={acc:.1f}%")

    torch.save(model.cpu().state_dict(), output)
    print(f"\nSaved BC policy to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build dataset from battle logs")
    parser.add_argument("--train", action="store_true", help="Train BC policy")
    parser.add_argument("--log-dir", default="battle_logs")
    parser.add_argument("--data", default="bc_gemini_data.npz")
    parser.add_argument("--output", default="bc_gemini_policy.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--all-games", action="store_true", help="Include losses too (default: wins only)")
    args = parser.parse_args()

    if args.build:
        build_dataset(log_dir=args.log_dir, output=args.data, wins_only=not args.all_games)

    if args.train:
        train_bc(data_path=args.data, epochs=args.epochs, output=args.output)

    if not args.build and not args.train:
        print("Use --build to create dataset, --train to train, or both.")

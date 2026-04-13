"""
Behavioral Cloning pre-training from Pokemon Showdown replay dataset.

Downloads gen9randombattle replays from HuggingFace, parses the battle logs
to extract (observation, action) pairs, and trains a policy network via
supervised learning. The saved model can be used to initialize PPO training.

Usage:
    python bc_pretrain.py [--samples N] [--epochs E] [--output model.pth]
"""

import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ── Action mapping (matches poke_env SinglesEnv for gen9) ───────────────────
# 0-5:   switch to slot 0-5
# 6-9:   move 0-3 (normal)
# 10-13: move + mega   (dead in gen9 → remap to 6-9)
# 14-17: move + zmove  (dead in gen9 → remap to 6-9)
# 18-21: move + dynamax(dead in gen9 → remap to 6-9)
# 22-25: move 0-3 + terastallize
N_ACTIONS = 26
OBS_SIZE   = 86  # must match rl_env.py


# ── Log parser ───────────────────────────────────────────────────────────────

def parse_log(log: str):
    """
    Parse a PS replay log into a sequence of (simple_obs, action_index) pairs.
    Returns a list of (obs_array, action_int) tuples.
    Only extracts move/switch choices from player 1's perspective.
    """
    pairs = []
    lines = log.split("\n")

    # Track simple state: HP fractions (12 slots: 6 own + 6 opp), turn
    own_hp  = np.ones(6, dtype=np.float32)
    opp_hp  = np.ones(6, dtype=np.float32)
    own_fnt = np.zeros(6, dtype=np.float32)
    opp_fnt = np.zeros(6, dtype=np.float32)
    turn = 0
    own_active = 0
    opp_active = 0
    own_name_to_slot = {}
    opp_name_to_slot = {}

    def simple_obs():
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0] = own_hp[own_active]
        obs[19:25] = own_hp[1:]          # bench HP (rough)
        obs[29:35] = own_fnt[1:]         # bench fainted
        obs[37] = opp_hp[opp_active]     # opp HP
        obs[78] = sum(1-own_fnt) / 6.0
        obs[79] = min(turn, 50) / 50.0
        return obs.copy()

    def name_to_slot(name_map, name, max_slots=6):
        name = name.strip().lower()
        if name not in name_map and len(name_map) < max_slots:
            name_map[name] = len(name_map)
        return name_map.get(name, 0)

    move_names = []  # ordered list of own moves seen

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        if cmd == "turn":
            turn = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else turn + 1

        elif cmd == "move":
            # |move|p1a: Pokemon|Move Name|...
            if len(parts) >= 4 and parts[2].startswith("p1"):
                move = re.sub(r'[^a-z0-9]', '', parts[3].lower())
                if move not in move_names:
                    move_names.append(move)
                slot = move_names.index(move) if move in move_names else 0
                slot = min(slot, 3)  # clamp to 4 moves
                action = 6 + slot   # move action (no gimmick)
                pairs.append((simple_obs(), action))

        elif cmd == "switch":
            # |switch|p1a: Pokemon|...
            if len(parts) >= 4 and parts[2].startswith("p1"):
                mon_name = parts[2].split(":")[1].strip() if ":" in parts[2] else parts[2]
                slot = name_to_slot(own_name_to_slot, mon_name)
                if slot > 0:  # 0 = active, 1-5 = bench slots
                    action = slot - 1  # switch action 0-4
                    pairs.append((simple_obs(), action))
                own_active = slot

            elif len(parts) >= 4 and parts[2].startswith("p2"):
                mon_name = parts[2].split(":")[1].strip() if ":" in parts[2] else parts[2]
                slot = name_to_slot(opp_name_to_slot, mon_name)
                opp_active = slot

        elif cmd == "-damage" or cmd == "-heal":
            if len(parts) >= 4 and parts[2].startswith("p1"):
                hp_str = parts[3].split("/")
                if len(hp_str) == 2:
                    try:
                        cur = float(hp_str[0])
                        mx  = float(hp_str[1].split()[0])
                        own_hp[own_active] = cur / mx if mx > 0 else 0.0
                    except ValueError:
                        pass
            elif len(parts) >= 4 and parts[2].startswith("p2"):
                hp_str = parts[3].split("/")
                if len(hp_str) == 2:
                    try:
                        cur = float(hp_str[0])
                        mx  = float(hp_str[1].split()[0])
                        opp_hp[opp_active] = cur / mx if mx > 0 else 0.0
                    except ValueError:
                        pass

        elif cmd == "faint":
            if len(parts) >= 3 and parts[2].startswith("p1"):
                own_fnt[own_active] = 1.0
                own_hp[own_active]  = 0.0
            elif len(parts) >= 3 and parts[2].startswith("p2"):
                opp_fnt[opp_active] = 1.0
                opp_hp[opp_active]  = 0.0

    return pairs


# ── Dataset ──────────────────────────────────────────────────────────────────

class ReplayDataset(Dataset):
    def __init__(self, samples: int = 50_000, format_filter: str = "gen9randombattle",
                 scan_limit: int = 5_000_000):
        print(f"Loading replay dataset (target={samples} replays, format={format_filter})...")

        # Shuffle the stream so rows come from all shards (all gens) uniformly,
        # rather than scanning sequentially from Gen 1. Buffer of 10k rows.
        ds = load_dataset(
            "HolidayOugi/pokemon-showdown-replays",
            split="train",
            streaming=True,
        ).shuffle(seed=42, buffer_size=10_000)

        self.obs_list = []
        self.act_list = []
        matched = 0
        scanned = 0
        pbar = tqdm(total=samples, desc=f"Collecting {format_filter} replays")
        for row in ds:
            if matched >= samples or scanned >= scan_limit:
                break
            scanned += 1
            if row.get("formatid") != format_filter:
                continue
            try:
                pairs = parse_log(row["log"])
                for obs, act in pairs:
                    self.obs_list.append(obs)
                    self.act_list.append(act)
            except Exception:
                pass
            matched += 1
            pbar.update(1)
        pbar.close()
        print(f"  Scanned {scanned:,} rows → found {matched:,} '{format_filter}' replays")

        if not self.obs_list:
            raise RuntimeError(
                f"No replays found for format '{format_filter}' after scanning "
                f"{scanned:,} rows. Try --format gen9randombattle."
            )

        self.obs = np.stack(self.obs_list).astype(np.float32)
        self.act = np.array(self.act_list, dtype=np.int64)
        print(f"  Extracted {len(self.obs):,} state-action pairs from {len(ds):,} replays")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return self.obs[i], self.act[i]


# ── Policy network ────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train(samples: int, epochs: int, output: str, batch_size: int = 256,
          format_filter: str = "gen9randombattle", scan_limit: int = 5_000_000):
    device  = get_device()
    dataset = ReplayDataset(samples=samples, format_filter=format_filter,
                            scan_limit=scan_limit)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=(device.type == "cuda"))

    model   = PolicyNet().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\nTraining policy network for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for obs_batch, act_batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            logits = model(obs_batch)
            loss   = loss_fn(logits, act_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(obs_batch)
            correct    += (logits.argmax(dim=1) == act_batch).sum().item()

        avg_loss = total_loss / len(dataset)
        acc      = correct / len(dataset) * 100
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  acc={acc:.1f}%")

    # Always save weights on CPU so they load anywhere
    torch.save(model.cpu().state_dict(), output)
    print(f"\nSaved pretrained policy to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",    type=int, default=50_000, help="Target replays to collect")
    parser.add_argument("--scan-limit", type=int, default=5_000_000, help="Max rows to scan in dataset")
    parser.add_argument("--format",     type=str, default="gen9randombattle", help="Format ID to filter (e.g. gen9randombattle, gen9ou, gen9battlestadiumsingles)")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--output",     type=str, default="bc_policy.pth")
    args = parser.parse_args()
    train(args.samples, args.epochs, args.output, format_filter=args.format,
          scan_limit=args.scan_limit)

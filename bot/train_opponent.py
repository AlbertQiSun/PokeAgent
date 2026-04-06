"""
Train the opponent action prediction MLP from replay data.

Collects (battle_state, opponent_action) pairs from replays and trains
a classifier to predict what the opponent will do next.

Usage:
    # Collect training data from replays
    python train_opponent.py collect --samples 5000 --output opp_train.jsonl

    # Train the model
    python train_opponent.py train --data opp_train.jsonl --epochs 20

    # Both in one step
    python train_opponent.py all --samples 5000 --epochs 20
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset as hf_load
from pokedata import get_pokemon, get_move, type_effectiveness
from opponent_model import (
    OpponentPredictor, N_FEATURES, N_CATEGORIES, ACTION_CATEGORIES,
    extract_features, classify_action, MODEL_PATH,
)
from replay_parser import BattleState, format_battle_prompt


# ── Data collection from replays ─────────────────────────────────────────────

def collect_opponent_data(args):
    """Parse replays and extract (features, opponent_action) pairs."""
    print(f"Collecting opponent action data...")
    print(f"  Format: {args.format}")
    print(f"  Target replays: {args.samples}")

    ds = hf_load(
        "HolidayOugi/pokemon-showdown-replays",
        split="train",
        streaming=True,
    )

    format_filter = args.format

    def format_matches(fmt: str) -> bool:
        if format_filter == "any-random":
            return "randombattle" in fmt
        elif format_filter == "any-singles":
            return any(x in fmt for x in ("ou", "uu", "singles", "bss"))
        elif format_filter == "any":
            return True
        return fmt == format_filter

    samples = []
    matched = 0
    scanned = 0
    pbar = tqdm(total=args.samples, desc="Parsing replays")

    for row in ds:
        if matched >= args.samples or scanned >= args.scan_limit:
            break
        scanned += 1

        fmt = row.get("formatid", "")
        if not format_matches(fmt):
            continue

        try:
            replay_samples = _extract_opponent_actions(row["log"])
            samples.extend(replay_samples)
            matched += 1
            pbar.update(1)
        except Exception:
            pass

    pbar.close()

    print(f"\nResults:")
    print(f"  Scanned: {scanned:,} rows")
    print(f"  Matched: {matched:,} replays")
    print(f"  Samples: {len(samples):,} (state, opponent_action) pairs")

    if not samples:
        print("No samples collected!")
        return

    # Category distribution
    cats = [s["category"] for s in samples]
    for i, name in enumerate(ACTION_CATEGORIES):
        count = cats.count(i)
        if count > 0:
            print(f"  {name:20s}: {count:6d} ({count/len(cats)*100:.1f}%)")

    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved to {args.output}")


def _extract_opponent_actions(log: str) -> list[dict]:
    """From a replay, extract per-turn features + what the opponent did."""
    state = BattleState()
    lines = log.split("\n")

    # First pass: get player names and winner
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3 and parts[1].strip() in ("win", "player"):
            state.process_line(line)

    winner = state.winner
    if not winner:
        return []

    # We collect from BOTH perspectives — predict what each player does
    state = BattleState()
    samples = []

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1].strip()

        # Before processing: if this is an opponent action, capture features
        if cmd == "move" and len(parts) >= 4:
            actor, name = state._extract_name(parts[2])
            # For the OTHER player, this is what the opponent did
            perspective = "p2" if actor == "p1" else "p1"

            # Get battle state from perspective's view
            if perspective == "p1":
                our_active = state.p1_active
                opp_active = state.p2_active
                our_team = state.p1_team
                opp_team = state.p2_team
            else:
                our_active = state.p2_active
                opp_active = state.p1_active
                our_team = state.p2_team
                opp_team = state.p1_team

            if our_active and opp_active and state.turn > 0:
                our_mon = our_team.get(our_active)
                opp_mon = opp_team.get(opp_active)

                if our_mon and opp_mon:
                    move_name = parts[3].strip()
                    category = classify_action("move", move_name)

                    features = _state_to_features(our_mon, opp_mon, state, perspective)
                    if features is not None:
                        samples.append({
                            "features": features.tolist(),
                            "category": category,
                            "action": f"move {move_name}",
                            "turn": state.turn,
                        })

        elif cmd == "switch" and len(parts) >= 4:
            actor, name = state._extract_name(parts[2])
            perspective = "p2" if actor == "p1" else "p1"

            if perspective == "p1":
                our_active = state.p1_active
                opp_active = state.p2_active
                our_team = state.p1_team
                opp_team = state.p2_team
            else:
                our_active = state.p2_active
                opp_active = state.p1_active
                our_team = state.p2_team
                opp_team = state.p1_team

            if our_active and opp_active and state.turn > 0:
                our_mon = our_team.get(our_active)
                opp_mon = opp_team.get(opp_active)
                opp_active_mon = opp_team.get(opp_active)

                # Only voluntary switches (active not fainted)
                if our_mon and opp_mon and opp_active_mon and not opp_active_mon.fainted:
                    features = _state_to_features(our_mon, opp_mon, state, perspective)
                    if features is not None:
                        samples.append({
                            "features": features.tolist(),
                            "category": 3,  # switch
                            "action": f"switch {name}",
                            "turn": state.turn,
                        })

        state.process_line(line)

    return samples


def _state_to_features(our_mon, opp_mon, state: BattleState, perspective: str) -> np.ndarray | None:
    """Convert MonState objects to feature vector."""
    our_data = get_pokemon(re.sub(r'[^a-zA-Z0-9]', '', our_mon.species).lower())
    opp_data = get_pokemon(re.sub(r'[^a-zA-Z0-9]', '', opp_mon.species).lower())
    if not our_data or not opp_data:
        return None

    our_types = our_data.get("types", [])
    opp_types = opp_data.get("types", [])
    our_stats = our_data.get("baseStats", {})
    opp_stats = opp_data.get("baseStats", {})

    if perspective == "p1":
        our_bench = [m for n, m in state.p1_team.items()
                     if n != state.p1_active and not m.fainted]
        opp_bench = [m for n, m in state.p2_team.items()
                     if n != state.p2_active and not m.fainted]
    else:
        our_bench = [m for n, m in state.p2_team.items()
                     if n != state.p2_active and not m.fainted]
        opp_bench = [m for n, m in state.p1_team.items()
                     if n != state.p1_active and not m.fainted]

    return extract_features(
        our_mon.species, opp_mon.species,
        our_mon.hp_frac, opp_mon.hp_frac,
        our_types, opp_types,
        our_stats, opp_stats,
        opp_mon.moves_seen,
        opp_mon.item, opp_mon.ability,
        our_mon.boosts, opp_mon.boosts,
        state.turn, len(our_bench), len(opp_bench),
        state.weather, state.terrain,
    )


# ── Training ─────────────────────────────────────────────────────────────────

class OppDataset(Dataset):
    def __init__(self, path: str):
        self.features = []
        self.labels = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                self.features.append(row["features"])
                self.labels.append(row["category"])
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train(args):
    print(f"\n{'='*50}")
    print(f"  Training Opponent Action Predictor")
    print(f"{'='*50}")
    print(f"  Data:   {args.data}")
    print(f"  Epochs: {args.epochs}")

    ds = OppDataset(args.data)
    print(f"  Samples: {len(ds):,}")

    # Class distribution
    labels = ds.labels.numpy()
    for i, name in enumerate(ACTION_CATEGORIES):
        count = (labels == i).sum()
        if count > 0:
            print(f"    {name:20s}: {count:6d} ({count/len(labels)*100:.1f}%)")

    # Train/val split
    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    # Class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=N_CATEGORIES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * N_CATEGORIES
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = OpponentPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels_batch in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels_batch).sum().item()
            total += len(labels_batch)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels_batch in val_loader:
                logits = model(features)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += len(labels_batch)

        train_acc = correct / total * 100
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(MODEL_PATH))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/total:.4f} "
                  f"train_acc={train_acc:.1f}% val_acc={val_acc:.1f}%")

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to {MODEL_PATH}")

    # Per-class accuracy on validation
    model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True))
    model.eval()
    class_correct = np.zeros(N_CATEGORIES)
    class_total = np.zeros(N_CATEGORIES)
    with torch.no_grad():
        for features, labels_batch in val_loader:
            logits = model(features)
            preds = logits.argmax(dim=-1)
            for i in range(N_CATEGORIES):
                mask = labels_batch == i
                class_total[i] += mask.sum().item()
                class_correct[i] += (preds[mask] == i).sum().item()

    print("\nPer-class accuracy:")
    for i, name in enumerate(ACTION_CATEGORIES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            print(f"  {name:20s}: {acc:.1f}% ({int(class_total[i])} samples)")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opponent action prediction model")
    sub = parser.add_subparsers(dest="command")

    # Collect
    c = sub.add_parser("collect", help="Collect training data from replays")
    c.add_argument("--samples", type=int, default=5000)
    c.add_argument("--scan-limit", type=int, default=5000000)
    c.add_argument("--format", default="any-random")
    c.add_argument("--output", default="opp_train.jsonl")

    # Train
    t = sub.add_parser("train", help="Train the MLP")
    t.add_argument("--data", default="opp_train.jsonl")
    t.add_argument("--epochs", type=int, default=30)

    # All-in-one
    a = sub.add_parser("all", help="Collect + train")
    a.add_argument("--samples", type=int, default=3000)
    a.add_argument("--scan-limit", type=int, default=5000000)
    a.add_argument("--format", default="any-random")
    a.add_argument("--output", default="opp_train.jsonl")
    a.add_argument("--epochs", type=int, default=30)

    args = parser.parse_args()

    if args.command == "collect":
        collect_opponent_data(args)
    elif args.command == "train":
        train(args)
    elif args.command == "all":
        collect_opponent_data(args)
        args.data = args.output
        train(args)
    else:
        parser.print_help()

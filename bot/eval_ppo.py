"""
Comprehensive calibration check for the PPO model.

Tests:
1. Win rate vs Random (should be high ~80%+)
2. Win rate vs MaxBasePower (should be decent ~60%+)
3. Win rate vs SimpleHeuristics (harder benchmark)
4. Action distribution (is the model picking diverse actions or stuck?)
5. Episode length distribution (are games finishing normally?)
"""

import asyncio
import numpy as np
from collections import Counter
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from ppo_player import PPOPlayer
from rl_env import embed_battle

MODEL_PATH   = "ppo_gen9random"
FORMAT       = "gen9randombattle"
N_BATTLES    = 50   # per opponent


# ── Instrumented PPO player that logs actions ─────────────────────────────────
class InstrumentedPPOPlayer(PPOPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_log   = []
        self.episode_lens = []
        self._current_len = 0

    def choose_move(self, battle):
        self._current_len += 1
        obs = embed_battle(battle).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        self.action_log.append(int(action[0]))

        from poke_env.environment import SinglesEnv
        try:
            order = SinglesEnv.action_to_order(
                np.int64(int(action[0])), battle, fake=False, strict=False
            )
            return order
        except Exception:
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self.episode_lens.append(self._current_len)
        self._current_len = 0
        super()._battle_finished_callback(battle)


async def run_eval(ppo, opponent_cls, opponent_name, n):
    opponent = opponent_cls(
        account_configuration=AccountConfiguration(opponent_name, None),
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
    )
    ppo.action_log.clear()
    ppo.episode_lens.clear()
    ppo._current_len = 0
    ppo._battles = {}
    ppo._n_won_battles = 0
    ppo._n_finished_battles = 0

    await ppo.battle_against(opponent, n_battles=n)
    return ppo.n_won_battles


async def main():
    ppo = InstrumentedPPOPlayer(
        model_path=MODEL_PATH,
        account_configuration=AccountConfiguration("PPO Eval", None),
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
    )

    results = {}
    opponents = [
        (RandomPlayer,           "Random Bot"),
        (MaxBasePowerPlayer,     "MaxBP Bot"),
        (SimpleHeuristicsPlayer, "Heuristic Bot"),
    ]

    for cls, name in opponents:
        print(f"Evaluating vs {name} ({N_BATTLES} battles)...")
        wins = await run_eval(ppo, cls, name, N_BATTLES)
        wr = wins / N_BATTLES * 100
        results[name] = wr
        print(f"  → {wins}/{N_BATTLES} = {wr:.1f}%")

    print("\n========== CALIBRATION REPORT ==========")
    print(f"{'Opponent':<20} {'Win Rate':>10}  {'Assessment'}")
    print("-" * 50)
    thresholds = {
        "Random Bot":    (70, "Well calibrated", "Underfit"),
        "MaxBP Bot":     (50, "Good",             "Needs work"),
        "Heuristic Bot": (30, "Strong",           "Expected (needs more training)"),
    }
    for name, wr in results.items():
        lo, good_label, bad_label = thresholds[name]
        label = good_label if wr >= lo else bad_label
        print(f"  {name:<18} {wr:>8.1f}%   {label}")

    print("\n--- Action Distribution (last eval) ---")
    action_counts = Counter(ppo.action_log)
    action_labels = {
        **{i: f"switch_{i}" for i in range(6)},
        **{6+i: f"move_{i}" for i in range(4)},
        **{10+i: f"move_{i}_mega" for i in range(4)},
        **{14+i: f"move_{i}_zmove" for i in range(4)},
        **{18+i: f"move_{i}_dmax" for i in range(4)},
        **{22+i: f"move_{i}_tera" for i in range(4)},
    }
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items()):
        pct = count / total_actions * 100
        label = action_labels.get(action, f"action_{action}")
        bar = "█" * int(pct / 2)
        print(f"  {label:<20} {pct:5.1f}%  {bar}")

    print("\n--- Episode Length Distribution ---")
    if ppo.episode_lens:
        lens = np.array(ppo.episode_lens)
        print(f"  Mean turns/battle: {lens.mean():.1f}")
        print(f"  Min: {lens.min()}  Max: {lens.max()}  Std: {lens.std():.1f}")
        if lens.mean() < 3:
            print("  ⚠ Very short — model may be forfeiting or crashing")
        elif lens.mean() > 40:
            print("  ⚠ Very long — model may be stalling")
        else:
            print("  ✓ Episode length looks normal")

    print("\n========================================")


if __name__ == "__main__":
    asyncio.run(main())

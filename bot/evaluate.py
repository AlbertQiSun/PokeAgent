"""
Evaluate GeminiPlayer against three built-in baselines:
  - RandomPlayer         (makes random legal moves)
  - MaxBasePowerPlayer   (always picks the highest base-power move)
  - SimpleHeuristicsPlayer (type-effectiveness heuristics)

Usage:
    python evaluate.py [--battles N]

Results are printed as a win-rate table at the end.
"""

import asyncio
import argparse
from tabulate import tabulate
from dotenv import load_dotenv
load_dotenv()

from poke_env import AccountConfiguration, LocalhostServerConfiguration, cross_evaluate
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from gemini_player import GeminiPlayer

BATTLE_FORMAT = "gen9randombattle"
SERVER = LocalhostServerConfiguration


def make_player(cls, name):
    return cls(
        account_configuration=AccountConfiguration(name, None),
        battle_format=BATTLE_FORMAT,
        server_configuration=SERVER,
    )


async def main(n_battles: int, include_qwen: bool = False):
    print(f"Setting up players ({n_battles} battles each matchup)...\n")

    gemini    = make_player(GeminiPlayer,           "Gemini Bot")
    random_p  = make_player(RandomPlayer,           "Random")
    maxbp     = make_player(MaxBasePowerPlayer,     "MaxBasePower")
    heuristic = make_player(SimpleHeuristicsPlayer, "Heuristics")

    players = [gemini, random_p, maxbp, heuristic]

    if include_qwen:
        from local_llm_player import LocalLLMPlayer
        qwen = make_player(LocalLLMPlayer, "Qwen Bot")
        players.append(qwen)

    print("Running cross-evaluation...\n")
    results = await cross_evaluate(players, n_challenges=n_battles)

    # Build a table: rows = challenger, cols = opponent
    names = [p.username for p in players]
    rows = []
    for p_name in names:
        row = [p_name]
        for o_name in names:
            if p_name == o_name:
                row.append("—")
            else:
                wr = results[p_name][o_name]
                row.append(f"{wr*100:.1f}%" if wr is not None else "N/A")
        rows.append(row)

    print("\n========== EVALUATION RESULTS ==========")
    print(tabulate(rows, headers=["Player \\ Opponent"] + names, tablefmt="rounded_outline"))
    print()

    # Summary for Gemini
    g_name = gemini.username
    print("Gemini Bot win rates:")
    for o_name in names:
        if o_name == g_name:
            continue
        wr = results[g_name][o_name]
        print(f"  vs {o_name:20s}: {wr*100:.1f}%")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=5,
                        help="Number of battles per matchup (default: 5)")
    parser.add_argument("--qwen", action="store_true",
                        help="Include Qwen (local LLM) in evaluation")
    args = parser.parse_args()
    asyncio.run(main(args.battles, include_qwen=args.qwen))

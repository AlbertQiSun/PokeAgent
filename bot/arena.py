"""
Arena: run any two bots against each other with detailed logging.

Usage:
    python arena.py --p1 gemini --p2 ppo-team --battles 20
    python arena.py --p1 ppo --p2 ppo-team --battles 10
    python arena.py --p1 ppo-team --p2 ppo-team --battles 10   # self-play
    python arena.py --p1 gemini --p2 ppo --battles 5 --format gen9bssregj
"""

import asyncio
import argparse
import json
import csv
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from teams import TEAM_2

LOGS_DIR = Path("arena_logs")


# ── Player factory ─────────────────────────────────────────────────────────────

def make_player(player_type: str, slot: int, battle_format: str):
    """Create a named player instance. slot=1|2 ensures unique usernames."""
    name_map = {"gemini": "GeminiBot", "ppo": "PPOBot", "ppo-team": "PPOTeamBot", "qwen": "QwenBot"}
    username = f"{name_map.get(player_type, player_type)}{slot}"
    account = AccountConfiguration(username, None)
    base = dict(
        account_configuration=account,
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
    )

    if player_type == "gemini":
        from gemini_player import GeminiPlayer
        return GeminiPlayer(**base)

    elif player_type == "ppo":
        from ppo_player import PPOPlayer
        return PPOPlayer(model_path="ppo_gen9random", team=TEAM_2, **base)

    elif player_type == "ppo-team":
        from ppo_team_player import PPOTeamPlayer
        return PPOTeamPlayer(model_path="ppo_team_builder", **base)

    elif player_type == "qwen":
        from local_llm_player import LocalLLMPlayer
        return LocalLLMPlayer(**base)

    else:
        raise ValueError(f"Unknown player type: {player_type!r}")


# ── Per-battle stats extraction ────────────────────────────────────────────────

def _battle_stats(battle, player_username: str) -> dict:
    """Extract per-battle stats from a finished poke_env Battle object."""
    our_team  = list(battle.team.values())
    opp_team  = list(battle.opponent_team.values())

    our_species  = [m.species for m in our_team]
    opp_species  = [m.species for m in opp_team]
    our_fainted  = [m.species for m in our_team  if m.fainted]
    opp_fainted  = [m.species for m in opp_team  if m.fainted]
    our_survived = [m.species for m in our_team  if not m.fainted]

    return {
        "tag":          battle.battle_tag,
        "won":          battle.won,           # True / False / None (draw)
        "turns":        battle.turn,
        "our_team":     our_species,
        "opp_team":     opp_species,
        "our_fainted":  our_fainted,
        "opp_fainted":  opp_fainted,
        "our_survived": our_survived,
    }


# ── Aggregate statistics ───────────────────────────────────────────────────────

def compute_aggregate(battle_list: list[dict], p1_name: str, p2_name: str) -> dict:
    total  = len(battle_list)
    wins   = sum(1 for b in battle_list if b["won"] is True)
    losses = sum(1 for b in battle_list if b["won"] is False)
    draws  = total - wins - losses
    turns  = [b["turns"] for b in battle_list]

    # Team-usage: how often each species was selected (across all battles)
    team_usage:   dict[str, int] = defaultdict(int)
    opp_usage:    dict[str, int] = defaultdict(int)
    survival:     dict[str, int] = defaultdict(int)   # survived at least one battle

    for b in battle_list:
        for sp in b["our_team"]:
            team_usage[sp] += 1
        for sp in b["opp_team"]:
            opp_usage[sp] += 1
        for sp in b["our_survived"]:
            survival[sp] += 1

    # Sort by usage
    team_usage_sorted = dict(sorted(team_usage.items(), key=lambda x: x[1], reverse=True))
    opp_usage_sorted  = dict(sorted(opp_usage.items(),  key=lambda x: x[1], reverse=True))
    survival_rate     = {sp: round(survival[sp] / team_usage[sp], 3)
                         for sp in team_usage if team_usage[sp] > 0}
    survival_rate     = dict(sorted(survival_rate.items(), key=lambda x: x[1], reverse=True))

    return {
        "p1": p1_name, "p2": p2_name,
        "total_battles":  total,
        "p1_wins":        wins,
        "p2_wins":        losses,
        "draws":          draws,
        "p1_win_rate":    round(wins   / total, 4) if total else 0,
        "p2_win_rate":    round(losses / total, 4) if total else 0,
        "avg_turns":      round(sum(turns) / len(turns), 2) if turns else 0,
        "min_turns":      min(turns) if turns else 0,
        "max_turns":      max(turns) if turns else 0,
        "p1_team_usage":  team_usage_sorted,
        "p2_team_usage":  opp_usage_sorted,
        "p1_survival_rate": survival_rate,
    }


# ── Print a pretty summary ─────────────────────────────────────────────────────

def print_summary(agg: dict):
    p1, p2 = agg["p1"], agg["p2"]
    n = agg["total_battles"]
    bar = "=" * 52

    print(f"\n{bar}")
    print(f"  ARENA RESULTS: {p1}  vs  {p2}")
    print(bar)
    print(f"  Battles  : {n}")
    print(f"  {p1:<12}: {agg['p1_wins']:>3} wins  ({agg['p1_win_rate']*100:.1f}%)")
    print(f"  {p2:<12}: {agg['p2_wins']:>3} wins  ({agg['p2_win_rate']*100:.1f}%)")
    print(f"  Draws    : {agg['draws']}")
    print(f"  Avg turns: {agg['avg_turns']}  "
          f"(min {agg['min_turns']}, max {agg['max_turns']})")

    if agg["p1_team_usage"]:
        print(f"\n  {p1} team usage (out of {n} battles):")
        for sp, cnt in list(agg["p1_team_usage"].items())[:10]:
            sr = agg["p1_survival_rate"].get(sp, 0)
            print(f"    {sp:<25} {cnt:>3}×  survival {sr*100:.0f}%")

    if agg["p2_team_usage"]:
        print(f"\n  {p2} team usage:")
        for sp, cnt in list(agg["p2_team_usage"].items())[:10]:
            print(f"    {sp:<25} {cnt:>3}×")

    print(bar)


# ── Save logs ──────────────────────────────────────────────────────────────────

def _extract_battle_log(battle) -> str:
    """Extract the turn-by-turn battle log from a poke_env Battle object."""
    lines = []
    tag = battle.battle_tag
    lines.append(f"=== {tag} ===")
    lines.append(f"Players: {battle.player_username} vs {battle.opponent_username}")

    # poke_env stores raw messages in battle._observations or battle.observations
    # We reconstruct from available data
    team_mons = [m.species for m in battle.team.values()]
    opp_mons = [m.species for m in battle.opponent_team.values()]
    lines.append(f"P1 team: {', '.join(team_mons)}")
    lines.append(f"P2 team: {', '.join(opp_mons)}")
    lines.append("")

    # Extract turn-by-turn from observations (raw server messages)
    observations = getattr(battle, 'observations', {})
    if observations:
        for turn_key in sorted(observations.keys()):
            obs = observations[turn_key]
            events = obs.events if hasattr(obs, 'events') else []
            for event in events:
                event_str = "|".join(str(e) for e in event) if isinstance(event, (list, tuple)) else str(event)
                lines.append(event_str)
    else:
        lines.append("(no detailed observations available)")

    won = battle.won
    result = "WIN" if won is True else ("LOSS" if won is False else "DRAW")
    lines.append(f"\nResult: {result} in {battle.turn} turns")
    lines.append("")
    return "\n".join(lines)


def save_logs(agg: dict, battles: list[dict], p1_name: str, p2_name: str,
              raw_battles=None):
    LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"arena_{p1_name}_vs_{p2_name}_{ts}"

    # Full JSON report
    report = {"summary": agg, "battles": battles, "generated_at": datetime.now().isoformat()}
    json_path = LOGS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Full report : {json_path}")

    # Per-battle CSV for easy analysis
    csv_path = LOGS_DIR / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "battle", "won", "turns",
            "our_team", "opp_team", "our_fainted", "our_survived"
        ])
        writer.writeheader()
        for i, b in enumerate(battles, 1):
            writer.writerow({
                "battle":       i,
                "won":          b["won"],
                "turns":        b["turns"],
                "our_team":     "|".join(b["our_team"]),
                "opp_team":     "|".join(b["opp_team"]),
                "our_fainted":  "|".join(b["our_fainted"]),
                "our_survived": "|".join(b["our_survived"]),
            })
    print(f"  Per-battle CSV: {csv_path}")

    # Turn-by-turn battle logs
    if raw_battles:
        log_path = LOGS_DIR / f"{stem}.log"
        with log_path.open("w") as f:
            for battle in raw_battles:
                f.write(_extract_battle_log(battle))
                f.write("\n" + "=" * 60 + "\n\n")
        print(f"  Battle logs : {log_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def run_arena(p1_type: str, p2_type: str, n_battles: int, battle_format: str):
    print(f"\nSetting up {p1_type} (P1) and {p2_type} (P2)...")
    p1 = make_player(p1_type, 1, battle_format)
    p2 = make_player(p2_type, 2, battle_format)

    print(f"Starting {n_battles} battles ({battle_format})...")
    t0 = time.time()
    await p1.battle_against(p2, n_battles=n_battles)
    elapsed = time.time() - t0

    # Collect per-battle stats from p1's perspective
    battles = [_battle_stats(b, p1.username) for b in p1.battles.values()]
    agg = compute_aggregate(battles, p1_type, p2_type)
    agg["elapsed_seconds"]    = round(elapsed, 1)
    agg["battles_per_minute"] = round(n_battles / (elapsed / 60), 1) if elapsed > 0 else 0

    print_summary(agg)
    save_logs(agg, battles, p1_type, p2_type)
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1",      default="gemini",
                        choices=["gemini", "ppo", "ppo-team", "qwen"])
    parser.add_argument("--p2",      default="ppo-team",
                        choices=["gemini", "ppo", "ppo-team", "qwen"])
    parser.add_argument("--battles", type=int, default=10)
    parser.add_argument("--format",  default="gen9bssregj",
                        help="Battle format (default: gen9bssregj)")
    args = parser.parse_args()

    asyncio.run(run_arena(args.p1, args.p2, args.battles, args.format))

"""
Parse Pokemon Showdown replays into LLM SFT training pairs.

Downloads replays from HuggingFace, reconstructs per-turn battle state,
and formats each turn as (prompt, response) for supervised fine-tuning.

Output: JSONL file where each line is:
  {"prompt": "<battle context>", "response": "<action JSON>", "won": true/false}

Only winning player's turns are kept (learn from winners).

Usage:
    python replay_parser.py --samples 5000 --output sft_data.jsonl
    python replay_parser.py --samples 10000 --format gen9ou --output sft_ou.jsonl
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datasets import load_dataset
from tqdm import tqdm

from pokedata import get_pokemon, get_move, type_effectiveness


# ── Per-Pokemon state tracker ─────────────────────────────────────────────────

@dataclass
class MonState:
    species: str
    hp_frac: float = 1.0
    hp_cur: int = 0
    hp_max: int = 0
    fainted: bool = False
    status: str = ""
    item: str = ""
    ability: str = ""
    moves_seen: list = field(default_factory=list)
    types: list = field(default_factory=list)
    base_stats: dict = field(default_factory=dict)
    boosts: dict = field(default_factory=lambda: {
        "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0
    })
    tera_type: str = ""
    is_active: bool = False


# ── Battle state reconstructor ────────────────────────────────────────────────

class BattleState:
    """Reconstruct battle state from replay log lines."""

    def __init__(self):
        self.p1_team: dict[str, MonState] = {}  # name -> MonState
        self.p2_team: dict[str, MonState] = {}
        self.p1_active: str = ""
        self.p2_active: str = ""
        self.turn: int = 0
        self.weather: str = ""
        self.terrain: str = ""
        self.p1_side: list[str] = []  # reflect, lightscreen, etc.
        self.p2_side: list[str] = []
        self.winner: str = ""  # "p1" or "p2"
        self.p1_name: str = "Player 1"
        self.p2_name: str = "Player 2"

    def _get_or_create(self, team: dict, name: str) -> MonState:
        clean = name.strip()
        if clean not in team:
            # Look up base data
            species_key = re.sub(r'[^a-zA-Z0-9]', '', clean).lower()
            poke = get_pokemon(species_key)
            types = poke.get("types", []) if poke else []
            base_stats = poke.get("baseStats", {}) if poke else {}
            team[clean] = MonState(
                species=clean, types=types, base_stats=base_stats
            )
        return team[clean]

    def _parse_hp(self, hp_str: str) -> tuple[int, int, float]:
        """Parse '150/300' or '50/100 par' → (cur, max, frac)."""
        hp_part = hp_str.split()[0]  # strip status like 'par'
        if hp_part == "0":
            return 0, 100, 0.0
        parts = hp_part.split("/")
        if len(parts) == 2:
            try:
                cur = int(parts[0])
                mx = int(parts[1])
                return cur, mx, cur / mx if mx > 0 else 0.0
            except ValueError:
                pass
        return 100, 100, 1.0

    def _extract_name(self, player_mon: str) -> tuple[str, str]:
        """Extract player and mon name from 'p1a: Pikachu'."""
        if ":" in player_mon:
            player = player_mon.split(":")[0].strip()[:2]  # 'p1' or 'p2'
            name = player_mon.split(":")[1].strip()
            # Strip form details after comma: "Urshifu-Rapid-Strike, L50, M"
            name = name.split(",")[0].strip()
            return player, name
        return player_mon[:2], player_mon

    def process_line(self, line: str):
        parts = line.split("|")
        if len(parts) < 2:
            return
        cmd = parts[1].strip()

        if cmd == "turn":
            self.turn = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else self.turn + 1

        elif cmd == "player":
            if len(parts) >= 4:
                if parts[2].strip() == "p1":
                    self.p1_name = parts[3].strip()
                elif parts[2].strip() == "p2":
                    self.p2_name = parts[3].strip()

        elif cmd == "win":
            if len(parts) >= 3:
                winner_name = parts[2].strip()
                if winner_name == self.p1_name:
                    self.winner = "p1"
                elif winner_name == self.p2_name:
                    self.winner = "p2"

        elif cmd == "switch" or cmd == "drag":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team

                # Deactivate previous active
                if player == "p1" and self.p1_active in team:
                    team[self.p1_active].is_active = False
                    team[self.p1_active].boosts = {k: 0 for k in team[self.p1_active].boosts}
                elif player == "p2" and self.p2_active in team:
                    team[self.p2_active].is_active = False
                    team[self.p2_active].boosts = {k: 0 for k in team[self.p2_active].boosts}

                mon = self._get_or_create(team, name)
                mon.is_active = True

                # Parse HP from switch line: "Pikachu, L50, M|100/100"
                if len(parts) >= 5:
                    cur, mx, frac = self._parse_hp(parts[4])
                    mon.hp_cur = cur
                    mon.hp_max = mx
                    mon.hp_frac = frac

                if player == "p1":
                    self.p1_active = name
                else:
                    self.p2_active = name

        elif cmd == "move":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                move_name = parts[3].strip()
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                if move_name not in mon.moves_seen and len(mon.moves_seen) < 4:
                    mon.moves_seen.append(move_name)

        elif cmd in ("-damage", "-heal"):
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                cur, mx, frac = self._parse_hp(parts[3])
                mon.hp_cur = cur
                mon.hp_max = mx
                mon.hp_frac = frac
                if cur == 0:
                    mon.fainted = True

        elif cmd == "faint":
            if len(parts) >= 3:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                mon.fainted = True
                mon.hp_frac = 0.0
                mon.hp_cur = 0

        elif cmd == "-status":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                mon.status = parts[3].strip()

        elif cmd == "-curestatus":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                mon.status = ""

        elif cmd == "-ability":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                mon.ability = parts[3].strip()

        elif cmd == "-item" or cmd == "-enditem":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                if cmd == "-item":
                    mon.item = parts[3].strip()

        elif cmd == "-boost" or cmd == "-unboost":
            if len(parts) >= 5:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                stat = parts[3].strip()
                try:
                    amount = int(parts[4].strip())
                except ValueError:
                    amount = 1
                if stat in mon.boosts:
                    if cmd == "-boost":
                        mon.boosts[stat] = min(6, mon.boosts[stat] + amount)
                    else:
                        mon.boosts[stat] = max(-6, mon.boosts[stat] - amount)

        elif cmd == "-weather":
            if len(parts) >= 3:
                w = parts[2].strip()
                self.weather = "" if w == "none" else w

        elif cmd in ("-fieldstart", "-fieldend"):
            if len(parts) >= 3:
                field_name = parts[2].strip().lower()
                if "terrain" in field_name:
                    self.terrain = field_name if cmd == "-fieldstart" else ""

        elif cmd in ("-sidestart", "-sideend"):
            if len(parts) >= 4:
                player = parts[2].strip()[:2]
                condition = parts[3].strip()
                side = self.p1_side if player == "p1" else self.p2_side
                if cmd == "-sidestart":
                    if condition not in side:
                        side.append(condition)
                else:
                    if condition in side:
                        side.remove(condition)

        elif cmd == "-terastallize":
            if len(parts) >= 4:
                player, name = self._extract_name(parts[2])
                team = self.p1_team if player == "p1" else self.p2_team
                mon = self._get_or_create(team, name)
                mon.tera_type = parts[3].strip()


# ── Format battle state as LLM prompt ─────────────────────────────────────────

def format_battle_prompt(state: BattleState, perspective: str) -> str:
    """Format current battle state as a text prompt from perspective of p1 or p2."""
    if perspective == "p1":
        our_team = state.p1_team
        opp_team = state.p2_team
        our_active_name = state.p1_active
        opp_active_name = state.p2_active
        our_side = state.p1_side
        opp_side = state.p2_side
    else:
        our_team = state.p2_team
        opp_team = state.p1_team
        our_active_name = state.p2_active
        opp_active_name = state.p1_active
        our_side = state.p2_side
        opp_side = state.p1_side

    our_mon = our_team.get(our_active_name)
    opp_mon = opp_team.get(opp_active_name)

    if not our_mon or not opp_mon:
        return ""

    sections = []
    sections.append(f"=== TURN {state.turn} ===")
    sections.append(f"Your {our_mon.species} vs opponent's {opp_mon.species}")

    # Our active info
    our_types = "/".join(our_mon.types) if our_mon.types else "???"
    our_hp_str = f"{our_mon.hp_frac*100:.0f}%"
    sections.append(f"\nYour active: {our_mon.species} ({our_types}) HP: {our_hp_str}")
    if our_mon.status:
        sections.append(f"  Status: {our_mon.status}")
    if any(v != 0 for v in our_mon.boosts.values()):
        boosts = " ".join(f"{k}:{v:+d}" for k, v in our_mon.boosts.items() if v != 0)
        sections.append(f"  Boosts: {boosts}")
    if our_mon.moves_seen:
        sections.append(f"  Known moves: {', '.join(our_mon.moves_seen)}")
    if our_mon.item:
        sections.append(f"  Item: {our_mon.item}")
    if our_mon.ability:
        sections.append(f"  Ability: {our_mon.ability}")

    # Opponent active info
    opp_types = "/".join(opp_mon.types) if opp_mon.types else "???"
    opp_hp_str = f"{opp_mon.hp_frac*100:.0f}%"
    sections.append(f"\nOpponent active: {opp_mon.species} ({opp_types}) HP: {opp_hp_str}")
    if opp_mon.status:
        sections.append(f"  Status: {opp_mon.status}")
    if any(v != 0 for v in opp_mon.boosts.values()):
        boosts = " ".join(f"{k}:{v:+d}" for k, v in opp_mon.boosts.items() if v != 0)
        sections.append(f"  Boosts: {boosts}")
    if opp_mon.moves_seen:
        sections.append(f"  Known moves: {', '.join(opp_mon.moves_seen)}")
    if opp_mon.item:
        sections.append(f"  Item: {opp_mon.item}")
    if opp_mon.ability:
        sections.append(f"  Ability: {opp_mon.ability}")
    if opp_mon.tera_type:
        sections.append(f"  Tera: {opp_mon.tera_type}")

    # Type effectiveness
    if our_mon.types and opp_mon.types:
        for our_t in our_mon.types:
            for opp_t in opp_mon.types:
                eff = type_effectiveness(our_t, opp_t)
                if eff and eff != 1.0:
                    sections.append(f"  {our_t} → {opp_t}: {eff}x")

    # Our bench
    bench_lines = []
    for name, mon in our_team.items():
        if name == our_active_name or mon.fainted:
            continue
        types = "/".join(mon.types) if mon.types else "?"
        bench_lines.append(f"  {mon.species} ({types}) {mon.hp_frac*100:.0f}% HP")
    if bench_lines:
        sections.append("\nYour bench:")
        sections.extend(bench_lines)

    # Opponent bench (known info only)
    opp_bench = []
    for name, mon in opp_team.items():
        if name == opp_active_name:
            continue
        status = "fainted" if mon.fainted else f"{mon.hp_frac*100:.0f}% HP"
        opp_bench.append(f"  {mon.species} ({status})")
    if opp_bench:
        sections.append("\nOpponent bench:")
        sections.extend(opp_bench)

    # Field conditions
    field_lines = []
    if state.weather:
        field_lines.append(f"  Weather: {state.weather}")
    if state.terrain:
        field_lines.append(f"  Terrain: {state.terrain}")
    if our_side:
        field_lines.append(f"  Your side: {', '.join(our_side)}")
    if opp_side:
        field_lines.append(f"  Opponent side: {', '.join(opp_side)}")
    if field_lines:
        sections.append("\nField:")
        sections.extend(field_lines)

    return "\n".join(sections)


# ── Extract per-turn (prompt, action) from a replay ───────────────────────────

def parse_replay(log: str) -> list[dict]:
    """
    Parse a full replay log into per-turn SFT samples.

    Returns list of:
      {"prompt": str, "response": str, "won": bool, "turn": int, "perspective": str}
    """
    state = BattleState()
    samples = []
    lines = log.split("\n")

    # First pass: determine winner
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 3 and parts[1].strip() == "win":
            state.process_line(line)
        if len(parts) >= 4 and parts[1].strip() == "player":
            state.process_line(line)

    # Reset for second pass
    winner = state.winner
    if not winner:
        return []  # skip replays without a clear winner

    state = BattleState()

    # Process line by line, capture actions on move/switch events
    pending_action = None

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1].strip()

        # Capture action BEFORE processing (so prompt is from before the action)
        if cmd == "move" and len(parts) >= 4:
            player, name = state._extract_name(parts[2])
            if player == winner:  # only learn from winner
                prompt = format_battle_prompt(state, player)
                if prompt:
                    move_name = parts[3].strip()
                    response = json.dumps({
                        "action_type": "move",
                        "target_name": move_name,
                    })
                    samples.append({
                        "prompt": prompt,
                        "response": response,
                        "won": True,
                        "turn": state.turn,
                        "perspective": player,
                    })

        elif cmd == "switch" and len(parts) >= 4:
            player, name = state._extract_name(parts[2])
            if player == winner:
                # Only count voluntary switches (not forced after faint)
                # Heuristic: if our active is alive, this is voluntary
                team = state.p1_team if player == "p1" else state.p2_team
                active_name = state.p1_active if player == "p1" else state.p2_active
                active_mon = team.get(active_name)

                if active_mon and not active_mon.fainted and state.turn > 0:
                    prompt = format_battle_prompt(state, player)
                    if prompt:
                        response = json.dumps({
                            "action_type": "switch",
                            "target_name": name,
                        })
                        samples.append({
                            "prompt": prompt,
                            "response": response,
                            "won": True,
                            "turn": state.turn,
                            "perspective": player,
                        })

        # Now process the line to update state
        state.process_line(line)

    return samples


# ── Main: download, parse, save ───────────────────────────────────────────────

def collect(args):
    print(f"Collecting SFT data from replays...")
    print(f"  Format: {args.format}")
    print(f"  Target replays: {args.samples}")
    print(f"  Scan limit: {args.scan_limit}")

    ds = load_dataset(
        "HolidayOugi/pokemon-showdown-replays",
        split="train",
        streaming=True,
    )

    # If format starts with "gen" (e.g., "gen9randombattle"), match exactly.
    # If "any-random", match any genNrandombattle.
    # If "any-singles", match any singles/ou/uu format.
    format_filter = args.format

    def format_matches(fmt: str) -> bool:
        if format_filter == "any-random":
            return "randombattle" in fmt
        elif format_filter == "any-singles":
            return any(x in fmt for x in ("ou", "uu", "singles", "bss"))
        elif format_filter == "any":
            return True
        else:
            return fmt == format_filter

    all_samples = []
    matched = 0
    scanned = 0
    errors = 0
    pbar = tqdm(total=args.samples, desc="Parsing replays")

    for row in ds:
        if matched >= args.samples or scanned >= args.scan_limit:
            break
        scanned += 1

        fmt = row.get("formatid", "")
        if not format_matches(fmt):
            continue

        try:
            samples = parse_replay(row["log"])
            all_samples.extend(samples)
            matched += 1
            pbar.update(1)
        except Exception as e:
            errors += 1

    pbar.close()

    print(f"\nResults:")
    print(f"  Scanned: {scanned:,} rows")
    print(f"  Matched: {matched:,} replays ({args.format})")
    print(f"  Errors:  {errors:,}")
    print(f"  Samples: {len(all_samples):,} turn-level (prompt, response) pairs")

    if not all_samples:
        print("No samples collected! Try a different format.")
        return

    # Compute stats
    move_count = sum(1 for s in all_samples if '"move"' in s["response"])
    switch_count = sum(1 for s in all_samples if '"switch"' in s["response"])
    avg_prompt_len = sum(len(s["prompt"]) for s in all_samples) / len(all_samples)

    print(f"  Moves: {move_count:,} ({move_count/len(all_samples)*100:.0f}%)")
    print(f"  Switches: {switch_count:,} ({switch_count/len(all_samples)*100:.0f}%)")
    print(f"  Avg prompt length: {avg_prompt_len:.0f} chars")

    # Save as JSONL
    with open(args.output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\nSaved to {args.output}")

    # Also show a few examples
    print("\n── Sample entries ──")
    for i, s in enumerate(all_samples[:3]):
        print(f"\n[Sample {i+1}] Turn {s['turn']}")
        print(f"Prompt ({len(s['prompt'])} chars):")
        # Show first 500 chars
        preview = s["prompt"][:500]
        if len(s["prompt"]) > 500:
            preview += "\n..."
        print(preview)
        print(f"Response: {s['response']}")

    os._exit(0)  # Force exit — HuggingFace streaming threads don't terminate cleanly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Pokemon Showdown replays into LLM SFT training data."
    )
    parser.add_argument("--samples", type=int, default=5_000,
                        help="Target number of replays to parse")
    parser.add_argument("--scan-limit", type=int, default=5_000_000,
                        help="Max rows to scan in HuggingFace dataset")
    parser.add_argument("--format", type=str, default="gen9randombattle",
                        help="Format ID (gen9randombattle, gen9ou, gen9battlestadiumsingles)")
    parser.add_argument("--output", type=str, default="sft_data.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()
    collect(args)

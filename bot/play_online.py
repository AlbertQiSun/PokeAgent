"""
Play on the real Pokemon Showdown ladder against human players.

Usage:
    python play_online.py --player ppo-team --username YourAccount --password YourPass
    python play_online.py --player gemini --username YourAccount --password YourPass --battles 5
    python play_online.py --player ppo-team --username YourAccount --password YourPass --accept

Modes:
    --ladder N    : Queue for N ladder battles (default)
    --accept      : Sit in lobby and accept incoming challenges
"""

import asyncio
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

from poke_env import AccountConfiguration, ShowdownServerConfiguration

LOGS_DIR = Path("online_logs")

# ── Fixed competitive team for online play ────────────────────────────────────

ONLINE_TEAM = """
Miraidon @ Assault Vest
Ability: Hadron Engine
Level: 50
Shiny: Yes
Tera Type: Psychic
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Dazzling Gleam
- Electro Drift
- Draco Meteor
- U-turn

Ogerpon-Hearthflame @ Hearthflame Mask
Ability: Mold Breaker
Level: 50
Tera Type: Fire
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Ivy Cudgel
- Horn Leech
- Spiky Shield
- Swords Dance

Urshifu-Rapid-Strike (M) @ Silk Scarf
Ability: Unseen Fist
Tera Type: Stellar
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- U-turn

Skeledirge (F) @ Heavy-Duty Boots
Ability: Unaware
Level: 50
Shiny: Yes
Tera Type: Ghost
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 0 Atk
- Will-O-Wisp
- Scorching Sands
- Slack Off
- Torch Song

Garganacl @ Covert Cloak
Ability: Clear Body
Level: 50
Shiny: Yes
Tera Type: Water
EVs: 4 HP / 252 Def / 252 SpD
- Salt Cure
- Protect
- Stealth Rock
- Recover

Terapagos @ Leftovers
Ability: Tera Shift
Level: 50
Tera Type: Stellar
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Tera Starstorm
- Calm Mind
- Earth Power
- Dark Pulse
""".strip()


# ── Human-like timing wrapper ─────────────────────────────────────────────────

class HumanTimer:
    """Add realistic delays so moves aren't instant."""

    @staticmethod
    async def think(min_s: float = 2.0, max_s: float = 8.0):
        delay = random.uniform(min_s, max_s)
        await asyncio.sleep(delay)

    @staticmethod
    async def team_preview(min_s: float = 5.0, max_s: float = 15.0):
        delay = random.uniform(min_s, max_s)
        await asyncio.sleep(delay)


# ── Wrapped players with human-like delays ────────────────────────────────────

def make_online_bot(player_type: str, username: str, password: str, fmt: str):
    account = AccountConfiguration(username, password)
    base = dict(
        account_configuration=account,
        battle_format=fmt,
        server_configuration=ShowdownServerConfiguration,
    )

    # All online bots use the fixed competitive team
    base["team"] = ONLINE_TEAM

    if player_type == "ppo-team":
        from ppo_team_player import PPOTeamPlayer

        class OnlinePPOTeam(PPOTeamPlayer):
            def _select_team(self):
                # Skip random team building — use fixed team
                return []

            @property
            def next_team(self):
                # Use the fixed team set via constructor
                return self._team.yield_team()

            async def teampreview(self, battle):
                await HumanTimer.team_preview()
                result = super().teampreview(battle)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            def choose_move(self, battle):
                return super().choose_move(battle)

        bot = OnlinePPOTeam(model_path="ppo_team_builder", **base)
        return bot, "PPO-Team"

    elif player_type == "ppo":
        from ppo_player import PPOPlayer

        bot = PPOPlayer(model_path="ppo_gen9random", **base)
        return bot, "PPO"

    elif player_type == "gemini":
        from gemini_player import GeminiPlayer

        class OnlineGemini(GeminiPlayer):
            def _build_team_sync(self):
                # Skip Gemini team building — use fixed team
                print("[Gemini] Using fixed online team.")
                return self._team.yield_team()

            async def _build_team_async(self):
                # Don't rebuild team between battles
                pass

        bot = OnlineGemini(**base)
        return bot, "Gemini"

    elif player_type == "qwen":
        from local_llm_player import LocalLLMPlayer

        class OnlineQwen(LocalLLMPlayer):
            def _build_team_sync(self):
                print("[Qwen] Using fixed online team.")
                return self._team.yield_team()

            async def _build_team_async(self):
                pass

        bot = OnlineQwen(**base)
        return bot, "Qwen"

    else:
        raise ValueError(f"Unknown player type: {player_type}")


# ── Battle log saving ─────────────────────────────────────────────────────────

def save_battle_log(battle, bot_name: str, fmt: str):
    LOGS_DIR.mkdir(exist_ok=True)
    tag = battle.battle_tag
    won = battle.won
    result = "win" if won is True else ("loss" if won is False else "draw")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"online_{bot_name}_{ts}_{result}.log"

    lines = []
    lines.append(f"Battle: {tag}")
    lines.append(f"Player: {battle.player_username} vs {battle.opponent_username}")
    lines.append(f"Result: {result.upper()} in {battle.turn} turns")
    lines.append(f"Date: {datetime.now().isoformat()}")
    lines.append(f"Format: {fmt}")
    lines.append("")

    our_team = [m.species for m in battle.team.values()]
    opp_team = [m.species for m in battle.opponent_team.values()]
    our_fainted = [m.species for m in battle.team.values() if m.fainted]
    opp_fainted = [m.species for m in battle.opponent_team.values() if m.fainted]
    lines.append(f"Bot team: {', '.join(our_team)}")
    lines.append(f"Opponent team: {', '.join(opp_team)}")
    lines.append(f"Bot fainted: {', '.join(our_fainted) or 'none'}")
    lines.append(f"Opp fainted: {', '.join(opp_fainted) or 'none'}")
    lines.append("")

    observations = getattr(battle, 'observations', {})
    if observations:
        for turn_key in sorted(observations.keys()):
            obs = observations[turn_key]
            events = obs.events if hasattr(obs, 'events') else []
            for event in events:
                if isinstance(event, (list, tuple)):
                    event_str = "|".join(str(e) for e in event)
                else:
                    event_str = str(event)
                lines.append(event_str)

    log_file.write_text("\n".join(lines))
    print(f"  Log saved: {log_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_ladder(bot, bot_name: str, n_battles: int, fmt: str):
    """Queue for ladder battles."""
    print(f"\nSearching for {n_battles} ladder battles ({fmt})...")
    print("(This may take a moment to find opponents)\n")

    wins, losses, draws = 0, 0, 0
    prev_count = len(bot.battles)

    for i in range(n_battles):
        print(f"--- Battle {i+1}/{n_battles} ---")
        if i > 0:
            wait = random.uniform(5.0, 15.0)
            print(f"  Waiting {wait:.0f}s before next battle...")
            await asyncio.sleep(wait)

        await bot.ladder(1)

        new_battles = list(bot.battles.values())[prev_count:]
        prev_count = len(bot.battles)

        for battle in new_battles:
            won = battle.won
            if won is True:
                wins += 1
                print(f"  Result: WIN in {battle.turn} turns vs {battle.opponent_username}")
            elif won is False:
                losses += 1
                print(f"  Result: LOSS in {battle.turn} turns vs {battle.opponent_username}")
            else:
                draws += 1
                print(f"  Result: DRAW in {battle.turn} turns vs {battle.opponent_username}")
            save_battle_log(battle, bot_name, fmt)

    total = wins + losses + draws
    print(f"\n{'='*50}")
    print(f"  ONLINE RESULTS ({bot_name})")
    print(f"{'='*50}")
    if total:
        print(f"  Battles: {total}")
        print(f"  Wins:    {wins} ({wins/total*100:.1f}%)")
        print(f"  Losses:  {losses}")
        print(f"  Draws:   {draws}")
    print(f"  Logs in: {LOGS_DIR}/")
    print(f"{'='*50}")


async def run_accept(bot, bot_name: str, fmt: str):
    """Accept incoming challenges."""
    print(f"\n{bot_name} is online on Pokemon Showdown!")
    print("Waiting for challenges... (Ctrl+C to stop)\n")

    prev_count = len(bot.battles)
    battle_num = 0

    while True:
        await bot.accept_challenges(None, 1)
        battle_num += 1

        new_battles = list(bot.battles.values())[prev_count:]
        prev_count = len(bot.battles)

        for battle in new_battles:
            won = battle.won
            result = "WIN" if won is True else ("LOSS" if won is False else "DRAW")
            print(f"  Battle {battle_num}: {result} in {battle.turn} turns vs {battle.opponent_username}")
            save_battle_log(battle, bot_name, fmt)

        print("Waiting for next challenge...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player",   default="ppo-team", choices=["gemini", "ppo", "ppo-team", "qwen"])
    parser.add_argument("--username", required=True, help="Pokemon Showdown registered username")
    parser.add_argument("--password", required=True, help="Pokemon Showdown password")
    parser.add_argument("--battles",  type=int, default=5, help="Number of ladder battles")
    parser.add_argument("--format",   type=str, default="gen9battlestadiumsingles",
                        help="Battle format (default: gen9battlestadiumsingles)")
    parser.add_argument("--accept",   action="store_true", help="Accept challenges instead of ladder")
    args = parser.parse_args()

    bot, bot_name = make_online_bot(args.player, args.username, args.password, args.format)

    if args.accept:
        asyncio.run(run_accept(bot, bot_name, args.format))
    else:
        asyncio.run(run_ladder(bot, bot_name, args.battles, args.format))

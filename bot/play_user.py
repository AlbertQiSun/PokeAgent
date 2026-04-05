import asyncio
import argparse
import orjson
from datetime import datetime
from pathlib import Path
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from gemini_player import GeminiPlayer
from ppo_player import PPOPlayer
from ppo_team_player import PPOTeamPlayer
from teams import TEAM_1, TEAM_2
import os

BATTLE_FORMAT = "gen9bssregj"
LOGS_DIR = Path("battle_logs")


def make_bot(player_type: str):
    if player_type == "ppo":
        bot = PPOPlayer(
            model_path="ppo_gen9random",
            account_configuration=AccountConfiguration("PPO Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
            team=TEAM_2,
        )
        name = "PPO Bot"
    elif player_type == "ppo-team":
        model_path = "ppo_team_builder"
        # BSS Reg J: bot registers 6, teampreview picks best 3 vs opponent
        team_format = "gen9bssregj"
        if not os.path.exists(f"{model_path}.zip"):
            print(f"[WARNING] {model_path}.zip not found — falling back to PPO Bot with fixed team.")
            bot = PPOPlayer(
                model_path="ppo_gen9random",
                account_configuration=AccountConfiguration("PPO Team Bot", None),
                battle_format=team_format,
                server_configuration=LocalhostServerConfiguration,
                team=TEAM_2,
            )
        else:
            bot = PPOTeamPlayer(
                model_path=model_path,
                account_configuration=AccountConfiguration("PPO Team Bot", None),
                battle_format=team_format,
                server_configuration=LocalhostServerConfiguration,
            )
        name = "PPO Team Bot"
    elif player_type == "qwen":
        from local_llm_player import LocalLLMPlayer
        bot = LocalLLMPlayer(
            account_configuration=AccountConfiguration("Qwen Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
        name = "Qwen Bot"
    else:
        bot = GeminiPlayer(
            account_configuration=AccountConfiguration("Gemini Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
        name = "Gemini Bot"
    return bot, name


async def main(player_type: str = "gemini"):
    bot, bot_name = make_bot(player_type)
    print(f"Setting up {bot_name} to accept challenges...")

    # Patch challenge handlers to accept any format from the browser
    async def accept_any_update_challenges(split_message):
        challenges = orjson.loads(split_message[2]).get("challengesFrom", {})
        for user, fmt in challenges.items():
            print(f"[Bot] Incoming challenge from '{user}' format='{fmt}'")
            await bot._challenge_queue.put(user)

    async def accept_any_handle_challenge_request(split_message):
        challenging_player = split_message[2].strip()
        if challenging_player != bot.username:
            fmt = split_message[5] if len(split_message) >= 6 else "?"
            print(f"[Bot] Incoming challenge from '{challenging_player}' format='{fmt}'")
            await bot._challenge_queue.put(challenging_player)

    bot._update_challenges = accept_any_update_challenges
    bot._handle_challenge_request = accept_any_handle_challenge_request

    print("==================================================")
    print(f"{bot_name} is online and waiting for challenges!")
    print("1. Open your browser to http://localhost:8000")
    print("2. Choose any username for yourself.")
    print(f"3. Click 'Find User' and search for '{bot_name}'.")
    print("4. Challenge with 'BSS Reg J' format (uncheck Best-of-3).")
    print("==================================================")

    LOGS_DIR.mkdir(exist_ok=True)
    battle_count = 0
    prev_battle_count = len(bot.battles)

    while True:
        await bot.accept_challenges(None, 1)
        battle_count += 1

        # Save the latest battle log
        new_battles = list(bot.battles.values())[prev_battle_count:]
        prev_battle_count = len(bot.battles)

        for battle in new_battles:
            tag = battle.battle_tag
            won = battle.won
            result = "win" if won is True else ("loss" if won is False else "draw")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOGS_DIR / f"{bot_name.replace(' ','_')}_{ts}_{result}.log"

            lines = []
            lines.append(f"Battle: {tag}")
            lines.append(f"Player: {battle.player_username} vs {battle.opponent_username}")
            lines.append(f"Result: {result.upper()} in {battle.turn} turns")
            lines.append(f"Date: {datetime.now().isoformat()}")
            lines.append("")

            # Team info
            our_team = [m.species for m in battle.team.values()]
            opp_team = [m.species for m in battle.opponent_team.values()]
            our_fainted = [m.species for m in battle.team.values() if m.fainted]
            opp_fainted = [m.species for m in battle.opponent_team.values() if m.fainted]
            lines.append(f"Bot team: {', '.join(our_team)}")
            lines.append(f"Opponent team: {', '.join(opp_team)}")
            lines.append(f"Bot fainted: {', '.join(our_fainted) or 'none'}")
            lines.append(f"Opp fainted: {', '.join(opp_fainted) or 'none'}")
            lines.append("")

            # Turn-by-turn observations
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
            print(f"\n  Battle log saved: {log_file}")

        print(f"\nBattle {battle_count} finished! Waiting for next challenge...")
        print("(Press Ctrl+C to stop)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", default="gemini", choices=["gemini", "ppo", "ppo-team", "qwen"],
                        help="Which bot to use (default: gemini)")
    args = parser.parse_args()
    asyncio.run(main(args.player))

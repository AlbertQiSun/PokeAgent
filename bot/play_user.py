import asyncio
import argparse
import orjson
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from gemini_player import GeminiPlayer
from ppo_player import PPOPlayer
from ppo_team_player import PPOTeamPlayer
from teams import TEAM_1, TEAM_2
import os

BATTLE_FORMAT = "gen9bssregj"


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

    battle_count = 0
    while True:
        await bot.accept_challenges(None, 1)
        battle_count += 1
        print(f"\nBattle {battle_count} finished! Waiting for next challenge...")
        print("(Press Ctrl+C to stop)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", default="gemini", choices=["gemini", "ppo", "ppo-team"],
                        help="Which bot to use (default: gemini)")
    args = parser.parse_args()
    asyncio.run(main(args.player))

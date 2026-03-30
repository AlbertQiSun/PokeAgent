import asyncio
import orjson
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from gemini_player import GeminiPlayer
from teams import TEAM_2

async def main():
    print("Setting up Gemini Player to accept challenges...")
    # Initialize Gemini Player
    gemini_player = GeminiPlayer(
        account_configuration=AccountConfiguration("Gemini Bot", None),
        battle_format="gen9bssregj",
        server_configuration=LocalhostServerConfiguration,
        team=TEAM_2,
    )

    # Patch challenge handlers to accept ANY format (ignore format mismatch)
    async def accept_any_update_challenges(split_message):
        challenges = orjson.loads(split_message[2]).get("challengesFrom", {})
        for user, fmt in challenges.items():
            print(f"[Bot] Incoming challenge from '{user}' format='{fmt}'")
            await gemini_player._challenge_queue.put(user)

    async def accept_any_handle_challenge_request(split_message):
        challenging_player = split_message[2].strip()
        if challenging_player != gemini_player.username:
            fmt = split_message[5] if len(split_message) >= 6 else "?"
            print(f"[Bot] Incoming challenge from '{challenging_player}' format='{fmt}'")
            await gemini_player._challenge_queue.put(challenging_player)

    gemini_player._update_challenges = accept_any_update_challenges
    gemini_player._handle_challenge_request = accept_any_handle_challenge_request

    print("==================================================")
    print("Gemini Bot is online and waiting for challenges!")
    print("1. Open your browser to http://localhost:8000")
    print("2. Choose any username for yourself.")
    print("3. Click 'Find User' and search for 'Gemini Bot'.")
    print("4. Challenge 'Gemini Bot' to a 'BSS Reg J' match.")
    print("==================================================")
    
    # Keep accepting challenges indefinitely until Ctrl+C
    battle_count = 0
    while True:
        await gemini_player.accept_challenges(None, 1)
        battle_count += 1
        print(f"\nBattle {battle_count} finished! Waiting for next challenge...")
        print("(Press Ctrl+C to stop)")
    
if __name__ == "__main__":
    asyncio.run(main())

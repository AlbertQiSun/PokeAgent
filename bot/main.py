import asyncio
from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from gemini_player import GeminiPlayer

BATTLE_FORMAT = "gen9randombattle"

async def main():
    print("Setting up players...")
    # Initialize Random Player (Baseline)
    random_player = RandomPlayer(
        account_configuration=AccountConfiguration("Random Bot", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration,
    )

    # Initialize Gemini Player
    gemini_player = GeminiPlayer(
        account_configuration=AccountConfiguration("Gemini Bot", None),
        battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration,
    )

    print("Players initialized. Starting battle...")
    
    # We will run 1 battle
    await gemini_player.battle_against(random_player, n_battles=1)

    print(f"\nBattle finished!")
    print(f"Gemini won {gemini_player.n_won_battles} / 1 battles.")
    
if __name__ == "__main__":
    # check that we have an active event loop or run it
    asyncio.run(main())

import asyncio
import argparse
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from gemini_player import GeminiPlayer
from ppo_player import PPOPlayer

BATTLE_FORMAT = "gen9randombattle"


def make_opponent(name: str):
    if name == "random":
        return RandomPlayer(
            account_configuration=AccountConfiguration("Random Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
    elif name == "heuristic":
        return SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("Heuristic Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
    elif name == "gemini":
        return GeminiPlayer(
            account_configuration=AccountConfiguration("Gemini Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
    else:
        raise ValueError(f"Unknown opponent: {name}")


async def main(player_type: str, opponent_type: str, n_battles: int):
    print(f"Setting up {player_type} vs {opponent_type} ({n_battles} battles)...")

    opponent = make_opponent(opponent_type)

    if player_type == "ppo":
        player = PPOPlayer(
            model_path="ppo_gen9random",
            account_configuration=AccountConfiguration("PPO Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "PPO"
    elif player_type == "gemini":
        player = GeminiPlayer(
            account_configuration=AccountConfiguration("Gemini Bot", None),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "Gemini"
    else:
        raise ValueError(f"Unknown player type: {player_type}")

    print("Starting battles...")
    await player.battle_against(opponent, n_battles=n_battles)

    won = player.n_won_battles
    print(f"\nBattle finished!")
    print(f"{label} won {won} / {n_battles} battles  ({won/n_battles*100:.1f}%)")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player",   default="ppo",    choices=["ppo", "gemini"])
    parser.add_argument("--opponent", default="random", choices=["random", "heuristic", "gemini"])
    parser.add_argument("--battles",  default=1, type=int)
    args = parser.parse_args()
    asyncio.run(main(args.player, args.opponent, args.battles))

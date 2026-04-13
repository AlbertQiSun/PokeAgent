import asyncio
import argparse
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from gemini_player import GeminiPlayer
from ppo_player import PPOPlayer
from pokemon_pool import build_team_string

DEFAULT_FORMAT = "gen9randombattle"
TEAM_FORMAT = "gen9bssregj"
_DEFAULT_OPP_TEAM = build_team_string(list(range(6)))

# Surprise team: off-meta items/moves that KB won't predict
_SURPRISE_TEAM = """\
Flutter Mane @ Life Orb
Ability: Protosynthesis
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Moonblast
- Thunderbolt
- Mystical Fire
- Perish Song

Iron Hands @ Leftovers
Ability: Quark Drive
Tera Type: Grass
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Drain Punch
- Thunder Punch
- Swords Dance
- Protect

Incineroar @ Choice Band
Ability: Intimidate
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Flare Blitz
- Knock Off
- U-turn
- Earthquake

Rillaboom @ Assault Vest
Ability: Grassy Surge
Tera Type: Poison
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Wood Hammer
- Knock Off
- U-turn
- Drain Punch

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Water
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Nasty Plot
- Recover

Urshifu-Rapid-Strike @ Focus Sash
Ability: Unseen Fist
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- U-turn
"""


def make_opponent(name: str, battle_format: str, needs_team: bool = False):
    team = _DEFAULT_OPP_TEAM if needs_team else None
    if name == "random":
        return RandomPlayer(
            account_configuration=AccountConfiguration("Random Bot", None),
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            team=team,
        )
    elif name == "heuristic":
        return SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("Heuristic Bot", None),
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            team=team,
        )
    elif name == "gemini":
        return GeminiPlayer(
            account_configuration=AccountConfiguration("Gemini Bot", None),
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
    elif name == "qwen":
        from local_llm_player import LocalLLMPlayer
        return LocalLLMPlayer(
            account_configuration=AccountConfiguration("Qwen Bot", None),
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
    elif name == "hybrid":
        from hybrid_player import HybridPlayer
        return HybridPlayer(
            model_path="ppo_team_builder", llm_backend="gemini",
            account_configuration=AccountConfiguration("Hybrid Bot", None),
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
        )
    elif name == "surprise":
        return SimpleHeuristicsPlayer(
            account_configuration=AccountConfiguration("Surprise Bot", None),
            battle_format=battle_format,
            server_configuration=LocalhostServerConfiguration,
            team=_SURPRISE_TEAM,
        )
    else:
        raise ValueError(f"Unknown opponent: {name}")


async def main(player_type: str, opponent_type: str, n_battles: int):
    # Gemini, PPO-Team, and Hybrid build their own teams → need team-based format
    # PPO uses random battles (no team needed)
    needs_team = player_type in ("ppo-team", "hybrid", "gemini", "qwen")
    if needs_team:
        fmt = TEAM_FORMAT
    else:
        fmt = DEFAULT_FORMAT

    print(f"Setting up {player_type} vs {opponent_type} ({n_battles} battles, {fmt})...")

    opponent = make_opponent(opponent_type, fmt, needs_team=needs_team)

    if player_type == "ppo":
        player = PPOPlayer(
            model_path="ppo_gen9random",
            account_configuration=AccountConfiguration("PPO Bot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "PPO"
    elif player_type == "gemini":
        player = GeminiPlayer(
            account_configuration=AccountConfiguration("Gemini Bot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "Gemini"
    elif player_type == "qwen":
        from local_llm_player import LocalLLMPlayer
        player = LocalLLMPlayer(
            account_configuration=AccountConfiguration("Qwen Bot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "Qwen"
    elif player_type == "ppo-team":
        from ppo_team_player import PPOTeamPlayer
        player = PPOTeamPlayer(
            model_path="ppo_team_builder",
            account_configuration=AccountConfiguration("PPO Team Bot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "PPO-Team"
    elif player_type == "hybrid":
        from hybrid_player import HybridPlayer
        player = HybridPlayer(
            model_path="ppo_team_builder", llm_backend="gemini",
            account_configuration=AccountConfiguration("Hybrid Bot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
        )
        label = "Hybrid"
    else:
        raise ValueError(f"Unknown player type: {player_type}")

    print("Starting battles...")
    await player.battle_against(opponent, n_battles=n_battles)

    won = player.n_won_battles
    print(f"\nBattle finished!")
    print(f"{label} won {won} / {n_battles} battles  ({won/n_battles*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player",   default="ppo",    choices=["ppo", "ppo-team", "gemini", "qwen", "hybrid"])
    parser.add_argument("--opponent", default="random", choices=["random", "heuristic", "gemini", "qwen", "hybrid", "surprise"])
    parser.add_argument("--battles",  default=1, type=int)
    args = parser.parse_args()
    asyncio.run(main(args.player, args.opponent, args.battles))

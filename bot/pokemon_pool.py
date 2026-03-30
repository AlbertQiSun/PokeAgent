"""
Pool of 20 BSS Reg J / competitive Gen 9 Pokemon with standard movesets.
Used by the team-building RL environment.
"""

# Each entry is a Showdown-format team string for one Pokemon.
POOL = [
    # 0 – Rillaboom
    """\
Rillaboom @ Choice Band
Ability: Grassy Surge
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Drum Beating
- Wood Hammer
- U-turn
- Knock Off""",

    # 1 – Tornadus-Therian
    """\
Tornadus-Therian @ Assault Vest
Ability: Regenerator
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
- Hurricane
- Heat Wave
- Knock Off
- U-turn""",

    # 2 – Urshifu-Rapid-Strike
    """\
Urshifu-Rapid-Strike @ Choice Scarf
Ability: Unseen Fist
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- U-turn""",

    # 3 – Incineroar
    """\
Incineroar @ Sitrus Berry
Ability: Intimidate
EVs: 252 HP / 100 Def / 156 SpD
Careful Nature
- Fake Out
- Knock Off
- Flare Blitz
- Parting Shot""",

    # 4 – Flutter Mane
    """\
Flutter Mane @ Choice Specs
Ability: Protosynthesis
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Moonblast
- Shadow Ball
- Mystical Fire
- Psyshock""",

    # 5 – Iron Hands
    """\
Iron Hands @ Life Orb
Ability: Quark Drive
EVs: 252 HP / 252 Atk / 4 SpD
Brave Nature
IVs: 0 Spe
- Drain Punch
- Wild Charge
- Fake Out
- Heavy Slam""",

    # 6 – Gholdengo
    """\
Gholdengo @ Expert Belt
Ability: Good as Gold
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Thunderbolt
- Focus Blast""",

    # 7 – Landorus-Therian
    """\
Landorus-Therian @ Rocky Helmet
Ability: Intimidate
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Earthquake
- U-turn
- Stealth Rock
- Rock Slide""",

    # 8 – Calyrex-Ice
    """\
Calyrex-Ice @ Lum Berry
Ability: As One (Glastrier)
EVs: 252 HP / 252 Atk / 4 SpD
Brave Nature
IVs: 0 Spe
- Glacial Lance
- High Horsepower
- Icicle Crash
- Heavy Slam""",

    # 9 – Amoonguss
    """\
Amoonguss @ Black Sludge
Ability: Regenerator
EVs: 252 HP / 144 Def / 112 SpD
Calm Nature
- Spore
- Pollen Puff
- Rage Powder
- Protect""",

    # 10 – Primarina
    """\
Primarina @ Throat Spray
Ability: Liquid Voice
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Sparkling Aria
- Moonblast
- Energy Ball
- Psychic""",

    # 11 – Kingambit
    """\
Kingambit @ Black Glasses
Ability: Defiant
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Sucker Punch
- Kowtow Cleave
- Iron Head
- Swords Dance""",

    # 12 – Chi-Yu
    """\
Chi-Yu @ Wise Glasses
Ability: Beads of Ruin
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Overheat
- Dark Pulse
- Flamethrower
- Psychic""",

    # 13 – Chien-Pao
    """\
Chien-Pao @ Focus Sash
Ability: Sword of Ruin
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Ice Spinner
- Sacred Sword
- Crunch
- Sucker Punch""",

    # 14 – Skeledirge
    """\
Skeledirge @ Heavy-Duty Boots
Ability: Unaware
EVs: 252 HP / 4 SpA / 252 SpD
Calm Nature
- Torch Song
- Shadow Ball
- Slack Off
- Will-O-Wisp""",

    # 15 – Annihilape
    """\
Annihilape @ Safety Goggles
Ability: Vital Spirit
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Rage Fist
- Close Combat
- Shadow Claw
- U-turn""",

    # 16 – Dragonite
    """\
Dragonite @ Weakness Policy
Ability: Multiscale
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Extreme Speed
- Dragon Claw
- Fire Punch
- Ice Punch""",

    # 17 – Ting-Lu
    """\
Ting-Lu @ Leftovers
Ability: Vessel of Ruin
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Fissure
- Ruination
- Stealth Rock
- Whirlwind""",

    # 18 – Meowscarada
    """\
Meowscarada @ Scope Lens
Ability: Protean
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Flower Trick
- Night Slash
- Low Kick
- U-turn""",

    # 19 – Garganacl
    """\
Garganacl @ Shed Shell
Ability: Purifying Salt
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Salt Cure
- Body Press
- Recover
- Protect""",
]

# Short name for each pool entry (index-aligned with POOL)
POOL_NAMES = [
    "Rillaboom",
    "Tornadus-Therian",
    "Urshifu-Rapid-Strike",
    "Incineroar",
    "Flutter Mane",
    "Iron Hands",
    "Gholdengo",
    "Landorus-Therian",
    "Calyrex-Ice",
    "Amoonguss",
    "Primarina",
    "Kingambit",
    "Chi-Yu",
    "Chien-Pao",
    "Skeledirge",
    "Annihilape",
    "Dragonite",
    "Ting-Lu",
    "Meowscarada",
    "Garganacl",
]

POOL_SIZE = len(POOL)
assert len(POOL_NAMES) == POOL_SIZE == 20


def build_team_string(indices) -> str:
    """Return a Showdown-format team string from a list of pool indices."""
    return "\n\n".join(POOL[i] for i in indices)

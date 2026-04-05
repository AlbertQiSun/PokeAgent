"""
metagame_kb.py — Metagame knowledge base for BSS Reg J.

Stores common sets (items, abilities, moves, spreads) and teammate pairings
with usage percentages. Used as priors for the battle notebook.

Data sourced from Smogon usage stats and competitive analysis.
"""

from pokedata import get_pokemon


# ── Per-Pokemon set data ──────────────────────────────────────────────────────
# Format: {species_key: { sets: [...], teammates: [...] }}
# Each set: {item, ability, moves, spread, nature, usage %}
# spread is (HP, Atk, Def, SpA, SpD, Spe)

_KB: dict[str, dict] = {

    # ── Restricted / Box Legends ──────────────────────────────────────────────

    "miraidon": {
        "sets": [
            {"item": "Assault Vest", "ability": "Hadron Engine",
             "moves": ["Electro Drift", "Draco Meteor", "Dazzling Gleam", "U-turn"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.45},
            {"item": "Choice Specs", "ability": "Hadron Engine",
             "moves": ["Electro Drift", "Draco Meteor", "Volt Switch", "Dazzling Gleam"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.30},
            {"item": "Life Orb", "ability": "Hadron Engine",
             "moves": ["Electro Drift", "Draco Meteor", "Dazzling Gleam", "Calm Mind"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.15},
        ],
        "teammates": [
            ("Incineroar", 0.40), ("Rillaboom", 0.35), ("Urshifu-Rapid-Strike", 0.30),
            ("Flutter Mane", 0.28), ("Ogerpon-Hearthflame", 0.25), ("Landorus-Therian", 0.22),
        ],
    },

    "koraidon": {
        "sets": [
            {"item": "Assault Vest", "ability": "Orichalcum Pulse",
             "moves": ["Collision Course", "Flare Blitz", "U-turn", "Drain Punch"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.40},
            {"item": "Choice Band", "ability": "Orichalcum Pulse",
             "moves": ["Collision Course", "Flare Blitz", "U-turn", "Outrage"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.30},
        ],
        "teammates": [
            ("Incineroar", 0.38), ("Rillaboom", 0.35), ("Flutter Mane", 0.30),
            ("Landorus-Therian", 0.25), ("Amoonguss", 0.20),
        ],
    },

    "calyrexshadow": {
        "sets": [
            {"item": "Focus Sash", "ability": "As One",
             "moves": ["Astral Barrage", "Nasty Plot", "Draining Kiss", "Protect"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.50},
            {"item": "Choice Specs", "ability": "As One",
             "moves": ["Astral Barrage", "Psyshock", "Trick", "Energy Ball"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.25},
        ],
        "teammates": [
            ("Incineroar", 0.42), ("Kingambit", 0.30), ("Rillaboom", 0.28),
        ],
    },

    "calyrexice": {
        "sets": [
            {"item": "Assault Vest", "ability": "As One",
             "moves": ["Glacial Lance", "High Horsepower", "Trick Room", "Close Combat"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.45},
            {"item": "Clear Amulet", "ability": "As One",
             "moves": ["Glacial Lance", "High Horsepower", "Trick Room", "Protect"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Brave", "usage": 0.30},
        ],
        "teammates": [
            ("Incineroar", 0.45), ("Amoonguss", 0.35), ("Primarina", 0.28),
            ("Dondozo", 0.20),
        ],
    },

    "terapagos": {
        "sets": [
            {"item": "Leftovers", "ability": "Tera Shift",
             "moves": ["Tera Starstorm", "Calm Mind", "Earth Power", "Dark Pulse"],
             "spread": (252, 0, 0, 252, 4, 0), "nature": "Modest", "usage": 0.50},
            {"item": "Assault Vest", "ability": "Tera Shift",
             "moves": ["Tera Starstorm", "Earth Power", "Dark Pulse", "Dazzling Gleam"],
             "spread": (252, 0, 4, 252, 0, 0), "nature": "Modest", "usage": 0.25},
        ],
        "teammates": [
            ("Incineroar", 0.35), ("Ogerpon-Hearthflame", 0.30), ("Skeledirge", 0.25),
        ],
    },

    # ── Common support / core Pokemon ─────────────────────────────────────────

    "incineroar": {
        "sets": [
            {"item": "Safety Goggles", "ability": "Intimidate",
             "moves": ["Flare Blitz", "Knock Off", "U-turn", "Fake Out"],
             "spread": (252, 4, 0, 0, 252, 0), "nature": "Careful", "usage": 0.35},
            {"item": "Assault Vest", "ability": "Intimidate",
             "moves": ["Flare Blitz", "Knock Off", "U-turn", "Fake Out"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.30},
            {"item": "Sitrus Berry", "ability": "Intimidate",
             "moves": ["Flare Blitz", "Knock Off", "Parting Shot", "Fake Out"],
             "spread": (252, 4, 0, 0, 252, 0), "nature": "Careful", "usage": 0.20},
        ],
        "teammates": [
            ("Miraidon", 0.40), ("Calyrex-Ice", 0.35), ("Rillaboom", 0.30),
            ("Urshifu-Rapid-Strike", 0.28),
        ],
    },

    "rillaboom": {
        "sets": [
            {"item": "Choice Band", "ability": "Grassy Surge",
             "moves": ["Wood Hammer", "Drum Beating", "U-turn", "Knock Off"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.40},
            {"item": "Assault Vest", "ability": "Grassy Surge",
             "moves": ["Drum Beating", "Wood Hammer", "U-turn", "Knock Off"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.30},
            {"item": "Miracle Seed", "ability": "Grassy Surge",
             "moves": ["Drum Beating", "Wood Hammer", "U-turn", "Fake Out"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.15},
        ],
        "teammates": [
            ("Miraidon", 0.35), ("Incineroar", 0.30), ("Flutter Mane", 0.25),
        ],
    },

    "fluttermane": {
        "sets": [
            {"item": "Booster Energy", "ability": "Protosynthesis",
             "moves": ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.45},
            {"item": "Choice Specs", "ability": "Protosynthesis",
             "moves": ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Mystical Fire"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.25},
            {"item": "Focus Sash", "ability": "Protosynthesis",
             "moves": ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.15},
        ],
        "teammates": [
            ("Miraidon", 0.28), ("Incineroar", 0.35), ("Rillaboom", 0.25),
            ("Urshifu-Rapid-Strike", 0.22),
        ],
    },

    "urshifurapidstrike": {
        "sets": [
            {"item": "Choice Scarf", "ability": "Unseen Fist",
             "moves": ["Surging Strikes", "Close Combat", "U-turn", "Aqua Jet"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.35},
            {"item": "Focus Sash", "ability": "Unseen Fist",
             "moves": ["Surging Strikes", "Close Combat", "Aqua Jet", "Detect"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.25},
            {"item": "Silk Scarf", "ability": "Unseen Fist",
             "moves": ["Surging Strikes", "Close Combat", "U-turn", "Aqua Jet"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.20},
        ],
        "teammates": [
            ("Miraidon", 0.30), ("Incineroar", 0.28), ("Flutter Mane", 0.22),
        ],
    },

    "ogerponhearthflame": {
        "sets": [
            {"item": "Hearthflame Mask", "ability": "Mold Breaker",
             "moves": ["Ivy Cudgel", "Horn Leech", "Spiky Shield", "Swords Dance"],
             "spread": (252, 252, 0, 0, 0, 4), "nature": "Jolly", "usage": 0.50},
            {"item": "Hearthflame Mask", "ability": "Mold Breaker",
             "moves": ["Ivy Cudgel", "Horn Leech", "Spiky Shield", "Follow Me"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.30},
        ],
        "teammates": [
            ("Miraidon", 0.25), ("Terapagos", 0.30), ("Incineroar", 0.28),
        ],
    },

    "skeledirge": {
        "sets": [
            {"item": "Heavy-Duty Boots", "ability": "Unaware",
             "moves": ["Torch Song", "Slack Off", "Will-O-Wisp", "Scorching Sands"],
             "spread": (252, 0, 252, 0, 4, 0), "nature": "Bold", "usage": 0.55},
            {"item": "Leftovers", "ability": "Unaware",
             "moves": ["Torch Song", "Slack Off", "Will-O-Wisp", "Hex"],
             "spread": (252, 0, 252, 4, 0, 0), "nature": "Bold", "usage": 0.25},
        ],
        "teammates": [
            ("Garganacl", 0.40), ("Terapagos", 0.25), ("Dondozo", 0.22),
        ],
    },

    "garganacl": {
        "sets": [
            {"item": "Leftovers", "ability": "Purifying Salt",
             "moves": ["Salt Cure", "Recover", "Stealth Rock", "Protect"],
             "spread": (252, 0, 252, 0, 4, 0), "nature": "Impish", "usage": 0.40},
            {"item": "Covert Cloak", "ability": "Purifying Salt",
             "moves": ["Salt Cure", "Recover", "Body Press", "Iron Defense"],
             "spread": (252, 0, 252, 0, 4, 0), "nature": "Impish", "usage": 0.30},
            {"item": "Clear Amulet", "ability": "Clear Body",
             "moves": ["Salt Cure", "Recover", "Stealth Rock", "Protect"],
             "spread": (252, 0, 4, 0, 252, 0), "nature": "Careful", "usage": 0.15},
        ],
        "teammates": [
            ("Skeledirge", 0.40), ("Dondozo", 0.25), ("Ting-Lu", 0.20),
        ],
    },

    "kingambit": {
        "sets": [
            {"item": "Assault Vest", "ability": "Supreme Overlord",
             "moves": ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Low Kick"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.40},
            {"item": "Black Glasses", "ability": "Supreme Overlord",
             "moves": ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.30},
        ],
        "teammates": [
            ("Incineroar", 0.30), ("Calyrex-Shadow", 0.30), ("Rillaboom", 0.25),
        ],
    },

    "gholdengo": {
        "sets": [
            {"item": "Choice Scarf", "ability": "Good as Gold",
             "moves": ["Make It Rain", "Shadow Ball", "Trick", "Thunderbolt"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.35},
            {"item": "Assault Vest", "ability": "Good as Gold",
             "moves": ["Make It Rain", "Shadow Ball", "Thunderbolt", "Focus Blast"],
             "spread": (252, 0, 0, 252, 4, 0), "nature": "Modest", "usage": 0.30},
        ],
        "teammates": [
            ("Incineroar", 0.30), ("Rillaboom", 0.25), ("Landorus-Therian", 0.22),
        ],
    },

    "landorustherian": {
        "sets": [
            {"item": "Assault Vest", "ability": "Intimidate",
             "moves": ["Earthquake", "U-turn", "Rock Slide", "Stomping Tantrum"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.40},
            {"item": "Choice Scarf", "ability": "Intimidate",
             "moves": ["Earthquake", "U-turn", "Rock Slide", "Superpower"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.30},
        ],
        "teammates": [
            ("Miraidon", 0.22), ("Incineroar", 0.30), ("Flutter Mane", 0.25),
        ],
    },

    "ironhands": {
        "sets": [
            {"item": "Assault Vest", "ability": "Quark Drive",
             "moves": ["Drain Punch", "Wild Charge", "Ice Punch", "Fake Out"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.50},
            {"item": "Booster Energy", "ability": "Quark Drive",
             "moves": ["Drain Punch", "Wild Charge", "Ice Punch", "Swords Dance"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.25},
        ],
        "teammates": [
            ("Miraidon", 0.30), ("Rillaboom", 0.25), ("Incineroar", 0.22),
        ],
    },

    "chienpao": {
        "sets": [
            {"item": "Focus Sash", "ability": "Sword of Ruin",
             "moves": ["Ice Spinner", "Sucker Punch", "Sacred Sword", "Protect"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.45},
            {"item": "Life Orb", "ability": "Sword of Ruin",
             "moves": ["Ice Spinner", "Crunch", "Sacred Sword", "Ice Shard"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.25},
        ],
        "teammates": [
            ("Incineroar", 0.30), ("Flutter Mane", 0.28), ("Rillaboom", 0.22),
        ],
    },

    "chiyu": {
        "sets": [
            {"item": "Choice Scarf", "ability": "Beads of Ruin",
             "moves": ["Heat Wave", "Dark Pulse", "Overheat", "Psychic"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.40},
            {"item": "Choice Specs", "ability": "Beads of Ruin",
             "moves": ["Heat Wave", "Dark Pulse", "Overheat", "Flamethrower"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.30},
        ],
        "teammates": [
            ("Incineroar", 0.32), ("Rillaboom", 0.28), ("Flutter Mane", 0.20),
        ],
    },

    "dragonite": {
        "sets": [
            {"item": "Assault Vest", "ability": "Multiscale",
             "moves": ["Extreme Speed", "Ice Spinner", "Stomping Tantrum", "Dragon Claw"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.35},
            {"item": "Choice Band", "ability": "Multiscale",
             "moves": ["Extreme Speed", "Outrage", "Ice Spinner", "Stomping Tantrum"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Adamant", "usage": 0.30},
            {"item": "Lum Berry", "ability": "Multiscale",
             "moves": ["Dragon Dance", "Extreme Speed", "Outrage", "Earthquake"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.20},
        ],
        "teammates": [
            ("Incineroar", 0.30), ("Rillaboom", 0.28), ("Landorus-Therian", 0.22),
        ],
    },

    "annihilape": {
        "sets": [
            {"item": "Covert Cloak", "ability": "Defiant",
             "moves": ["Rage Fist", "Drain Punch", "Shadow Claw", "Bulk Up"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.40},
            {"item": "Safety Goggles", "ability": "Defiant",
             "moves": ["Rage Fist", "Close Combat", "Shadow Claw", "Protect"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.30},
        ],
        "teammates": [
            ("Incineroar", 0.30), ("Rillaboom", 0.25),
        ],
    },

    "tornadustherian": {
        "sets": [
            {"item": "Assault Vest", "ability": "Regenerator",
             "moves": ["Hurricane", "Heat Wave", "Knock Off", "U-turn"],
             "spread": (252, 0, 0, 4, 0, 252), "nature": "Timid", "usage": 0.40},
            {"item": "Focus Sash", "ability": "Regenerator",
             "moves": ["Hurricane", "Tailwind", "Rain Dance", "U-turn"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.30},
        ],
        "teammates": [
            ("Miraidon", 0.30), ("Urshifu-Rapid-Strike", 0.25),
        ],
    },

    "amoonguss": {
        "sets": [
            {"item": "Sitrus Berry", "ability": "Regenerator",
             "moves": ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
             "spread": (252, 0, 252, 0, 4, 0), "nature": "Bold", "usage": 0.50},
            {"item": "Rocky Helmet", "ability": "Regenerator",
             "moves": ["Spore", "Rage Powder", "Pollen Puff", "Clear Smog"],
             "spread": (252, 0, 252, 0, 4, 0), "nature": "Bold", "usage": 0.25},
        ],
        "teammates": [
            ("Calyrex-Ice", 0.35), ("Incineroar", 0.30), ("Koraidon", 0.20),
        ],
    },

    "tinglu": {
        "sets": [
            {"item": "Leftovers", "ability": "Vessel of Ruin",
             "moves": ["Earthquake", "Ruination", "Protect", "Stealth Rock"],
             "spread": (252, 0, 4, 0, 252, 0), "nature": "Careful", "usage": 0.45},
            {"item": "Assault Vest", "ability": "Vessel of Ruin",
             "moves": ["Earthquake", "Ruination", "Heavy Slam", "Stomping Tantrum"],
             "spread": (252, 0, 4, 0, 252, 0), "nature": "Careful", "usage": 0.30},
        ],
        "teammates": [
            ("Garganacl", 0.20), ("Skeledirge", 0.22), ("Incineroar", 0.28),
        ],
    },

    "primarina": {
        "sets": [
            {"item": "Assault Vest", "ability": "Liquid Voice",
             "moves": ["Hyper Voice", "Moonblast", "Flip Turn", "Icy Wind"],
             "spread": (252, 0, 0, 252, 4, 0), "nature": "Modest", "usage": 0.45},
            {"item": "Choice Specs", "ability": "Liquid Voice",
             "moves": ["Hyper Voice", "Moonblast", "Flip Turn", "Psychic"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Modest", "usage": 0.25},
        ],
        "teammates": [
            ("Calyrex-Ice", 0.28), ("Incineroar", 0.30),
        ],
    },

    "meowscarada": {
        "sets": [
            {"item": "Focus Sash", "ability": "Protean",
             "moves": ["Flower Trick", "Knock Off", "Sucker Punch", "Protect"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.50},
            {"item": "Choice Band", "ability": "Protean",
             "moves": ["Flower Trick", "Knock Off", "U-turn", "Sucker Punch"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.25},
        ],
        "teammates": [
            ("Incineroar", 0.28), ("Flutter Mane", 0.22),
        ],
    },

    "dondozo": {
        "sets": [
            {"item": "Leftovers", "ability": "Unaware",
             "moves": ["Wave Crash", "Earthquake", "Curse", "Rest"],
             "spread": (252, 4, 252, 0, 0, 0), "nature": "Impish", "usage": 0.45},
            {"item": "Sitrus Berry", "ability": "Unaware",
             "moves": ["Wave Crash", "Earthquake", "Yawn", "Protect"],
             "spread": (252, 0, 252, 0, 4, 0), "nature": "Impish", "usage": 0.25},
        ],
        "teammates": [
            ("Garganacl", 0.25), ("Skeledirge", 0.22),
        ],
    },

    "archaludon": {
        "sets": [
            {"item": "Assault Vest", "ability": "Stamina",
             "moves": ["Electro Shot", "Flash Cannon", "Body Press", "Draco Meteor"],
             "spread": (252, 0, 4, 252, 0, 0), "nature": "Modest", "usage": 0.45},
            {"item": "Leftovers", "ability": "Stamina",
             "moves": ["Electro Shot", "Flash Cannon", "Body Press", "Protect"],
             "spread": (252, 0, 252, 4, 0, 0), "nature": "Bold", "usage": 0.25},
        ],
        "teammates": [
            ("Pecharunt", 0.25), ("Incineroar", 0.28),
        ],
    },

    "pecharunt": {
        "sets": [
            {"item": "Focus Sash", "ability": "Poison Puppeteer",
             "moves": ["Malignant Chain", "Shadow Ball", "Nasty Plot", "Protect"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.50},
            {"item": "Life Orb", "ability": "Poison Puppeteer",
             "moves": ["Malignant Chain", "Shadow Ball", "Nasty Plot", "Hex"],
             "spread": (4, 0, 0, 252, 0, 252), "nature": "Timid", "usage": 0.25},
        ],
        "teammates": [
            ("Archaludon", 0.25), ("Incineroar", 0.30),
        ],
    },

    "gougingfire": {
        "sets": [
            {"item": "Booster Energy", "ability": "Protosynthesis",
             "moves": ["Flare Blitz", "Dragon Claw", "Breaking Swipe", "Howl"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.45},
            {"item": "Assault Vest", "ability": "Protosynthesis",
             "moves": ["Flare Blitz", "Dragon Claw", "Breaking Swipe", "Stomping Tantrum"],
             "spread": (252, 252, 0, 0, 4, 0), "nature": "Adamant", "usage": 0.25},
        ],
        "teammates": [
            ("Rillaboom", 0.25), ("Incineroar", 0.28),
        ],
    },

    "ironboulder": {
        "sets": [
            {"item": "Booster Energy", "ability": "Quark Drive",
             "moves": ["Mighty Cleave", "Close Combat", "Zen Headbutt", "Protect"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.50},
            {"item": "Focus Sash", "ability": "Quark Drive",
             "moves": ["Mighty Cleave", "Close Combat", "Zen Headbutt", "Swords Dance"],
             "spread": (4, 252, 0, 0, 0, 252), "nature": "Jolly", "usage": 0.25},
        ],
        "teammates": [
            ("Miraidon", 0.25), ("Incineroar", 0.28),
        ],
    },
}


# ── Lookup functions ──────────────────────────────────────────────────────────

def _normalize_key(name: str) -> str:
    """Normalize species name to KB key."""
    import re
    return re.sub(r'[^a-z0-9]', '', name.lower())


def get_sets(species: str) -> list[dict]:
    """Return common sets for a species, sorted by usage. Empty list if unknown."""
    key = _normalize_key(species)
    entry = _KB.get(key)
    if entry:
        return sorted(entry["sets"], key=lambda s: s["usage"], reverse=True)
    return []


def get_most_likely_set(species: str) -> dict | None:
    """Return the single most common set for a species."""
    sets = get_sets(species)
    return sets[0] if sets else None


def get_teammates(species: str) -> list[tuple[str, float]]:
    """Return common teammates as [(species, co_occurrence_rate)]."""
    key = _normalize_key(species)
    entry = _KB.get(key)
    if entry:
        return sorted(entry.get("teammates", []), key=lambda t: t[1], reverse=True)
    return []


def infer_team_from_preview(seen_species: list[str]) -> dict[str, dict]:
    """Given species seen at team preview, infer likely sets for each.

    Also uses teammate co-occurrence to boost/lower set probabilities.
    Returns {species: most_likely_set_dict}.
    """
    result = {}
    seen_keys = {_normalize_key(s) for s in seen_species}

    for species in seen_species:
        best_set = get_most_likely_set(species)
        if best_set:
            result[species] = dict(best_set)  # copy
        else:
            # Unknown Pokemon — fill in from pokedex basics
            poke = get_pokemon(species)
            if poke:
                result[species] = {
                    "item": "???", "ability": "???",
                    "moves": [], "spread": (0, 0, 0, 0, 0, 0),
                    "nature": "???", "usage": 0.0,
                }
            else:
                result[species] = {
                    "item": "???", "ability": "???",
                    "moves": [], "spread": (0, 0, 0, 0, 0, 0),
                    "nature": "???", "usage": 0.0,
                }
    return result


def predict_unseen(seen_species: list[str], team_size: int = 6) -> list[tuple[str, float]]:
    """Predict unseen team members based on teammate co-occurrence.

    Returns [(species, probability)] for the most likely unseen Pokemon.
    """
    seen_keys = {_normalize_key(s) for s in seen_species}
    candidate_scores: dict[str, float] = {}

    for species in seen_species:
        teammates = get_teammates(species)
        for mate_name, rate in teammates:
            mate_key = _normalize_key(mate_name)
            if mate_key not in seen_keys:
                candidate_scores[mate_name] = candidate_scores.get(mate_name, 0) + rate

    # Sort by score
    ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:team_size - len(seen_species)]


def format_kb_summary(species: str) -> str:
    """Human-readable KB summary for a species."""
    sets = get_sets(species)
    if not sets:
        return f"{species}: no KB data"

    lines = [f"{species} — common sets:"]
    for i, s in enumerate(sets[:3]):
        moves_str = ", ".join(s["moves"][:4])
        hp, atk, df, spa, spd, spe = s["spread"]
        spread_parts = []
        if hp: spread_parts.append(f"{hp} HP")
        if atk: spread_parts.append(f"{atk} Atk")
        if df: spread_parts.append(f"{df} Def")
        if spa: spread_parts.append(f"{spa} SpA")
        if spd: spread_parts.append(f"{spd} SpD")
        if spe: spread_parts.append(f"{spe} Spe")
        spread_str = " / ".join(spread_parts)
        lines.append(
            f"  {i+1}. {s['item']} | {s['ability']} ({s['usage']*100:.0f}%)\n"
            f"     Moves: {moves_str}\n"
            f"     Spread: {spread_str} {s['nature']}"
        )

    teammates = get_teammates(species)
    if teammates:
        mate_str = ", ".join(f"{n}({r*100:.0f}%)" for n, r in teammates[:4])
        lines.append(f"  Teammates: {mate_str}")

    return "\n".join(lines)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== metagame_kb self-test ===\n")
    print(format_kb_summary("Miraidon"))
    print()
    print(format_kb_summary("Skeledirge"))
    print()
    print(format_kb_summary("Garganacl"))
    print()

    # Predict unseen
    seen = ["Miraidon", "Incineroar", "Rillaboom"]
    print(f"Seen: {seen}")
    unseen = predict_unseen(seen)
    print(f"Predicted unseen: {unseen}")
    print()

    # Infer from preview
    preview = ["Skeledirge", "Garganacl", "Dondozo"]
    inferred = infer_team_from_preview(preview)
    for sp, s in inferred.items():
        print(f"{sp}: {s['item']} / {s['ability']} / {s['moves'][:4]}")
    print()
    print(f"KB entries: {len(_KB)}")
    print("=== Done ===")

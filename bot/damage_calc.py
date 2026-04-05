"""
damage_calc.py — Gen 9 damage calculator.

Implements the standard Pokemon damage formula with STAB, type effectiveness,
weather, terrain, and common item/ability modifiers.
"""

import math
from pokedata import get_pokemon, get_move, type_effectiveness


def calc_damage(
    atk_species: str,
    move_name: str,
    def_species: str,
    *,
    atk_level: int = 50,
    def_level: int = 50,
    atk_item: str = "",
    def_item: str = "",
    atk_ability: str = "",
    def_ability: str = "",
    atk_boost: int = 0,
    def_boost: int = 0,
    weather: str = "",
    terrain: str = "",
    is_stab: bool | None = None,
    atk_burned: bool = False,
    crit: bool = False,
) -> dict:
    """
    Calculate approximate damage range for a move.

    Returns dict with:
      min_damage, max_damage, min_pct, max_pct (% of target's HP),
      effectiveness (float multiplier), hits (for multi-hit moves),
      description (human-readable summary)
    """
    attacker = get_pokemon(atk_species)
    defender = get_pokemon(def_species)
    move = get_move(move_name)

    if not attacker:
        return {"error": f"Attacker '{atk_species}' not found"}
    if not defender:
        return {"error": f"Defender '{def_species}' not found"}
    if not move:
        return {"error": f"Move '{move_name}' not found"}

    bp = move.get("basePower", 0)
    category = move.get("category", "Status")

    if category == "Status" or bp == 0:
        return {
            "min_damage": 0, "max_damage": 0,
            "min_pct": 0, "max_pct": 0,
            "effectiveness": 1.0, "hits": 1,
            "description": f"{move.get('name', move_name)} is a status move (no direct damage)."
        }

    # Stats
    atk_stats = attacker.get("baseStats", {})
    def_stats = defender.get("baseStats", {})

    if category == "Physical":
        raw_atk = atk_stats.get("atk", 100)
        raw_def = def_stats.get("def", 100)
    else:
        raw_atk = atk_stats.get("spa", 100)
        raw_def = def_stats.get("spd", 100)

    # Stat stage multipliers
    def _stage_mult(stage: int) -> float:
        stage = max(-6, min(6, stage))
        if stage >= 0:
            return (2 + stage) / 2
        return 2 / (2 - stage)

    eff_atk = raw_atk * _stage_mult(atk_boost)
    eff_def = raw_def * _stage_mult(def_boost)

    # Type effectiveness
    def_types = defender.get("types", ["Normal"])
    def_t1 = def_types[0] if def_types else "Normal"
    def_t2 = def_types[1] if len(def_types) > 1 else None
    move_type = move.get("type", "Normal")
    eff = type_effectiveness(move_type, def_t1, def_t2)

    # STAB
    if is_stab is None:
        atk_types = attacker.get("types", [])
        is_stab = move_type in atk_types
    stab = 1.5 if is_stab else 1.0

    # Ability: Adaptability boosts STAB to 2x
    atk_ab = atk_ability.lower().replace(" ", "")
    if atk_ab == "adaptability" and is_stab:
        stab = 2.0

    # Item modifiers
    item_mod = 1.0
    atk_it = atk_item.lower().replace(" ", "").replace("-", "")
    if atk_it == "lifeorb":
        item_mod = 1.3
    elif atk_it == "choiceband" and category == "Physical":
        item_mod = 1.5
    elif atk_it == "choicespecs" and category == "Special":
        item_mod = 1.5

    # Ability modifiers on attack
    ability_atk_mod = 1.0
    if atk_ab in ("hugepower", "purepower") and category == "Physical":
        ability_atk_mod = 2.0
    elif atk_ab == "technician" and bp <= 60:
        bp = int(bp * 1.5)

    # Defender ability
    def_ab = def_ability.lower().replace(" ", "")
    ability_def_mod = 1.0
    if def_ab == "multiscale":
        ability_def_mod = 0.5  # assumes full HP
    if def_ab == "unaware":
        # Ignore attacker boosts
        eff_atk = raw_atk

    # Defender item
    def_it = def_item.lower().replace(" ", "").replace("-", "")
    if def_it == "assaultvest" and category == "Special":
        eff_def *= 1.5
    elif def_it == "eviolite":
        eff_def *= 1.5

    # Burn halves physical damage (unless Guts)
    burn_mod = 1.0
    if atk_burned and category == "Physical" and atk_ab != "guts":
        burn_mod = 0.5

    # Weather
    weather_mod = 1.0
    w = weather.lower()
    if w in ("sun", "sunnyday", "desolateland"):
        if move_type == "Fire":
            weather_mod = 1.5
        elif move_type == "Water":
            weather_mod = 0.5
    elif w in ("rain", "raindance", "primordialsea"):
        if move_type == "Water":
            weather_mod = 1.5
        elif move_type == "Fire":
            weather_mod = 0.5

    # Terrain
    terrain_mod = 1.0
    t = terrain.lower()
    if t == "electricterrain" and move_type == "Electric":
        terrain_mod = 1.3
    elif t == "grassyterrain" and move_type == "Grass":
        terrain_mod = 1.3
    elif t == "psychicterrain" and move_type == "Psychic":
        terrain_mod = 1.3

    # Critical hit
    crit_mod = 1.5 if crit else 1.0

    # Damage formula: ((2*level/5 + 2) * bp * atk/def) / 50 + 2
    eff_atk *= ability_atk_mod
    base = ((2 * atk_level / 5 + 2) * bp * eff_atk / eff_def) / 50 + 2

    # Apply modifiers
    final = base * stab * eff * item_mod * weather_mod * terrain_mod * burn_mod * crit_mod * ability_def_mod

    # Random roll: 85% to 100%
    min_dmg = int(final * 0.85)
    max_dmg = int(final)

    # Multi-hit
    hits = 1
    multihit = move.get("multihit")
    if multihit:
        if isinstance(multihit, int):
            hits = multihit
        elif isinstance(multihit, list) and len(multihit) == 2:
            hits = multihit[1]  # max hits
        min_dmg *= hits
        max_dmg *= hits

    # Target HP
    def_hp = def_stats.get("hp", 100)
    # Approximate actual HP at level 50: ((2*base + 31) * 50/100) + 50 + 10
    actual_hp = int(((2 * def_hp + 31) * atk_level / 100) + atk_level + 10)

    min_pct = round(min_dmg / actual_hp * 100, 1)
    max_pct = round(max_dmg / actual_hp * 100, 1)

    # Description
    eff_label = ""
    if eff == 0:
        eff_label = " (IMMUNE)"
    elif eff >= 4:
        eff_label = " (4x super effective)"
    elif eff >= 2:
        eff_label = " (super effective)"
    elif eff <= 0.25:
        eff_label = " (4x resisted)"
    elif eff <= 0.5:
        eff_label = " (resisted)"

    hit_str = f" x{hits}" if hits > 1 else ""
    desc = (
        f"{attacker.get('name', atk_species)} {move.get('name', move_name)} vs "
        f"{defender.get('name', def_species)}: "
        f"{min_dmg}-{max_dmg} ({min_pct}-{max_pct}%){hit_str}{eff_label}"
    )

    return {
        "min_damage": min_dmg,
        "max_damage": max_dmg,
        "min_pct": min_pct,
        "max_pct": max_pct,
        "effectiveness": eff,
        "hits": hits,
        "description": desc,
    }


def format_damage_calc(
    atk_species: str, move_name: str, def_species: str, **kwargs
) -> str:
    """Human-readable damage calculation result."""
    result = calc_damage(atk_species, move_name, def_species, **kwargs)
    if "error" in result:
        return result["error"]
    return result["description"]


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== damage_calc self-test ===\n")

    # Miraidon Electro Drift vs various targets
    print(format_damage_calc("Miraidon", "Electro Drift", "Skeledirge",
                              atk_ability="Hadron Engine", terrain="electricterrain"))
    print(format_damage_calc("Miraidon", "Draco Meteor", "Garganacl"))
    print(format_damage_calc("Urshifu-Rapid-Strike", "Surging Strikes", "Skeledirge",
                              atk_item="Silk Scarf"))
    print(format_damage_calc("Ogerpon-Hearthflame", "Ivy Cudgel", "Terapagos",
                              atk_boost=2))
    print(format_damage_calc("Garganacl", "Salt Cure", "Miraidon"))
    print()

    # Edge cases
    print(format_damage_calc("Miraidon", "Will-O-Wisp", "Garganacl"))  # status
    print(format_damage_calc("Pikachu", "Thunderbolt", "Landorus",
                              atk_item="Choice Specs"))
    print()
    print("=== Done ===")

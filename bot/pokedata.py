"""
pokedata.py — Pokemon data lookup using poke_env's built-in Gen 9 data.

Provides lookup functions for Pokedex, Moves, Items, Abilities, and type effectiveness.
All data comes from poke_env.data.GenData (gen 9), which is already parsed and reliable.
"""

import re
from poke_env.data import GenData

_GEN = GenData.from_gen(9)


def _normalize(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


# ── Lookup functions ──────────────────────────────────────────────────────────

def get_pokemon(name: str) -> dict | None:
    """Look up a Pokemon by name. Returns raw pokedex dict or None."""
    key = _normalize(name)
    entry = _GEN.pokedex.get(key)
    if entry:
        return entry
    for k, v in _GEN.pokedex.items():
        if key in k or k in key:
            return v
    return None


def get_move(name: str) -> dict | None:
    """Look up a move by name. Returns raw move dict or None."""
    key = _normalize(name)
    entry = _GEN.moves.get(key)
    if entry:
        return entry
    for k, v in _GEN.moves.items():
        if key in k or k in key:
            return v
    return None


def get_item(name: str) -> dict | None:
    """Look up an item by name. Returns description dict or None."""
    key = _normalize(name)
    if key in _ITEMS:
        return _ITEMS[key]
    for k, v in _ITEMS.items():
        if key in k or k in key:
            return v
    return None


def get_ability(name: str) -> dict | None:
    """Look up an ability by name. Returns description dict or None."""
    key = _normalize(name)
    if key in _ABILITIES:
        return _ABILITIES[key]
    for k, v in _ABILITIES.items():
        if key in k or k in key:
            return v
    # Also check pokedex — abilities are listed there by name
    return None


# ── Competitive items & abilities (descriptions for Gemini context) ────────────

_ABILITIES: dict[str, dict] = {
    "hadronengine":    {"name": "Hadron Engine", "desc": "On switch-in, sets Electric Terrain. While Electric Terrain is active, this Pokemon's Sp. Atk is 1.3333x."},
    "moldbreaker":     {"name": "Mold Breaker", "desc": "This Pokemon's moves ignore the target's Ability if it could modify the effectiveness or damage of the move."},
    "unseenfist":      {"name": "Unseen Fist", "desc": "This Pokemon's contact moves bypass the target's protection moves (Protect, Detect, etc.)."},
    "unaware":         {"name": "Unaware", "desc": "This Pokemon ignores the target's stat changes when calculating damage and accuracy."},
    "clearbody":       {"name": "Clear Body", "desc": "Prevents other Pokemon from lowering this Pokemon's stats."},
    "terashift":       {"name": "Tera Shift", "desc": "On switch-in, Terapagos transforms into its Terastal Form."},
    "protosynthesis":  {"name": "Protosynthesis", "desc": "While Sun is active or holding Booster Energy, this Pokemon's highest stat is boosted by 30% (50% for Speed)."},
    "quarkdrive":      {"name": "Quark Drive", "desc": "While Electric Terrain is active or holding Booster Energy, this Pokemon's highest stat is boosted by 30% (50% for Speed)."},
    "intimidate":      {"name": "Intimidate", "desc": "On switch-in, lowers adjacent foes' Attack by 1 stage."},
    "multiscale":      {"name": "Multiscale", "desc": "If this Pokemon is at full HP, damage from attacks is halved."},
    "drizzle":         {"name": "Drizzle", "desc": "On switch-in, sets Rain for 5 turns."},
    "drought":         {"name": "Drought", "desc": "On switch-in, sets Sun for 5 turns."},
    "sandstream":      {"name": "Sand Stream", "desc": "On switch-in, sets Sandstorm for 5 turns."},
    "snowwarning":     {"name": "Snow Warning", "desc": "On switch-in, sets Snow for 5 turns."},
    "levitate":        {"name": "Levitate", "desc": "This Pokemon is immune to Ground-type moves."},
    "sturdy":          {"name": "Sturdy", "desc": "If at full HP, survives any single hit with at least 1 HP. Immune to OHKO moves."},
    "magicguard":      {"name": "Magic Guard", "desc": "This Pokemon can only be damaged by direct attacks."},
    "regenerator":     {"name": "Regenerator", "desc": "This Pokemon restores 1/3 of its max HP when it switches out."},
    "swordofruin":     {"name": "Sword of Ruin", "desc": "Lowers the Defense of all active Pokemon except this one by 25%."},
    "tabletsofruinp":  {"name": "Tablets of Ruin", "desc": "Lowers the Attack of all active Pokemon except this one by 25%."},
    "vesselofruin":    {"name": "Vessel of Ruin", "desc": "Lowers the Sp. Atk of all active Pokemon except this one by 25%."},
    "beadsofruin":     {"name": "Beads of Ruin", "desc": "Lowers the Sp. Def of all active Pokemon except this one by 25%."},
    "poisonheal":      {"name": "Poison Heal", "desc": "If poisoned, this Pokemon heals 1/8 max HP each turn instead of taking damage."},
    "magicbounce":     {"name": "Magic Bounce", "desc": "Reflects most non-damaging status moves back at the user."},
    "prankster":       {"name": "Prankster", "desc": "This Pokemon's status moves have +1 priority. Dark types are immune."},
    "libero":          {"name": "Libero", "desc": "Once per switch-in, this Pokemon's type changes to match the move it's about to use."},
    "guts":            {"name": "Guts", "desc": "While statused, this Pokemon's Attack is 1.5x. Burn does not reduce physical damage."},
    "technician":      {"name": "Technician", "desc": "Moves with 60 or less base power have 1.5x power."},
    "adaptability":    {"name": "Adaptability", "desc": "This Pokemon's STAB bonus is 2x instead of 1.5x."},
    "hugepower":       {"name": "Huge Power", "desc": "This Pokemon's Attack is doubled."},
    "purepower":       {"name": "Pure Power", "desc": "This Pokemon's Attack is doubled."},
    "tintedlens":      {"name": "Tinted Lens", "desc": "Moves that are not very effective deal double their usual damage."},
    "serenegrace":     {"name": "Serene Grace", "desc": "Secondary effect chances are doubled."},
    "flamebody":       {"name": "Flame Body", "desc": "30% chance to burn on contact."},
    "roughskin":       {"name": "Rough Skin", "desc": "Contact moves deal 1/8 max HP damage to attacker."},
    "ironbarbs":       {"name": "Iron Barbs", "desc": "Contact moves deal 1/8 max HP damage to attacker."},
    "naturalcure":     {"name": "Natural Cure", "desc": "This Pokemon's status is cured when it switches out."},
}

_ITEMS: dict[str, dict] = {
    "assaultvest":     {"name": "Assault Vest", "desc": "Holder's Sp. Def is 1.5x, but it can only use attacking moves."},
    "choiceband":      {"name": "Choice Band", "desc": "Holder's Attack is 1.5x, but it can only use the first move selected."},
    "choicescarf":     {"name": "Choice Scarf", "desc": "Holder's Speed is 1.5x, but it can only use the first move selected."},
    "choicespecs":     {"name": "Choice Specs", "desc": "Holder's Sp. Atk is 1.5x, but it can only use the first move selected."},
    "leftovers":       {"name": "Leftovers", "desc": "Holder restores 1/16 max HP at the end of each turn."},
    "lifeorb":         {"name": "Life Orb", "desc": "Holder's attacks do 1.3x damage, but lose 1/10 max HP after each attack."},
    "focussash":       {"name": "Focus Sash", "desc": "If holder is at full HP and would be KO'd, survives with 1 HP. Single use."},
    "heavydutyboots":  {"name": "Heavy-Duty Boots", "desc": "Holder is immune to hazards (Stealth Rock, Spikes, etc.)."},
    "rockyhelmet":     {"name": "Rocky Helmet", "desc": "Contact moves deal 1/6 max HP damage to attacker."},
    "eviolite":        {"name": "Eviolite", "desc": "If holder can evolve, its Def and Sp. Def are 1.5x."},
    "blacksludge":     {"name": "Black Sludge", "desc": "Poison types restore 1/16 max HP per turn. Non-Poison types lose 1/8."},
    "silkscarf":       {"name": "Silk Scarf", "desc": "Holder's Normal-type moves have 1.2x power."},
    "covertcloak":     {"name": "Covert Cloak", "desc": "Holder is immune to secondary effects of moves."},
    "clearamulet":     {"name": "Clear Amulet", "desc": "Holder's stats cannot be lowered by other Pokemon."},
    "boosterenergy":   {"name": "Booster Energy", "desc": "Activates Protosynthesis/Quark Drive even without weather/terrain. Single use."},
    "sitrusberry":     {"name": "Sitrus Berry", "desc": "Holder restores 1/4 max HP when at 1/2 or less."},
    "lumberry":        {"name": "Lum Berry", "desc": "Holder cures any major status condition or confusion. Single use."},
    "mentalherb":      {"name": "Mental Herb", "desc": "Cures Taunt, Encore, Torment, Heal Block, Disable. Single use."},
    "whiteherb":       {"name": "White Herb", "desc": "Restores lowered stats. Single use."},
    "loadeddice":      {"name": "Loaded Dice", "desc": "Multi-hit moves always hit 4 or 5 times."},
    "widelens":        {"name": "Wide Lens", "desc": "Holder's move accuracy is 1.1x."},
    "scopelens":       {"name": "Scope Lens", "desc": "Holder's crit ratio is raised by 1."},
    "safetygoggles":   {"name": "Safety Goggles", "desc": "Holder is immune to weather damage and powder moves."},
    "shedshell":       {"name": "Shed Shell", "desc": "Holder can always switch out, even against trapping moves/abilities."},
    "airballoon":      {"name": "Air Balloon", "desc": "Holder is immune to Ground-type moves. Pops when hit."},
    "hearthflamemask": {"name": "Hearthflame Mask", "desc": "Ogerpon-Hearthflame's held item. Tera type becomes Fire."},
    "wellspringmask":  {"name": "Wellspring Mask", "desc": "Ogerpon-Wellspring's held item. Tera type becomes Water."},
    "cornerstonemask": {"name": "Cornerstone Mask", "desc": "Ogerpon-Cornerstone's held item. Tera type becomes Rock."},
}


# ── Type effectiveness ────────────────────────────────────────────────────────

_TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0, "Dark": 2, "Steel": 2, "Fairy": 0.5},
    "Poison":   {"Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0, "Fairy": 2},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2, "Steel": 0.5, "Fairy": 0.5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
    "Dragon":   {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
    "Dark":     {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Fairy": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5, "Fairy": 2},
    "Fairy":    {"Fire": 0.5, "Fighting": 2, "Poison": 0.5, "Dragon": 2, "Dark": 2, "Steel": 0.5},
}


def type_effectiveness(atk_type: str, def_type1: str, def_type2: str | None = None) -> float:
    """Calculate type effectiveness multiplier (0, 0.25, 0.5, 1, 2, or 4)."""
    atk = atk_type.strip().capitalize()
    d1 = def_type1.strip().capitalize()

    mult = _TYPE_CHART.get(atk, {}).get(d1, 1.0)
    if def_type2:
        d2 = def_type2.strip().capitalize()
        mult *= _TYPE_CHART.get(atk, {}).get(d2, 1.0)
    return mult


# ── Formatted lookup strings (for Gemini tool responses) ─────────────────────

def format_pokemon_info(name: str) -> str:
    """Return a human-readable summary of a Pokemon."""
    p = get_pokemon(name)
    if not p:
        return f"Pokemon '{name}' not found."
    stats = p.get('baseStats', {})
    types = p.get('types', [])
    abilities = p.get('abilities', {})
    ab_list = [v for v in abilities.values() if v]
    return (
        f"{p.get('name', name)} | Types: {'/'.join(types)}\n"
        f"  Stats: HP {stats.get('hp',0)} / Atk {stats.get('atk',0)} / Def {stats.get('def',0)} "
        f"/ SpA {stats.get('spa',0)} / SpD {stats.get('spd',0)} / Spe {stats.get('spe',0)}\n"
        f"  Abilities: {', '.join(ab_list)}\n"
        f"  Weight: {p.get('weightkg', '?')}kg"
    )


def format_move_info(name: str) -> str:
    """Return a human-readable summary of a move."""
    m = get_move(name)
    if not m:
        return f"Move '{name}' not found."
    secondary = m.get('secondary') or m.get('secondaries')
    sec_str = ""
    if secondary:
        if isinstance(secondary, dict):
            chance = secondary.get('chance', '')
            status = secondary.get('status', '')
            boosts = secondary.get('boosts', '')
            sec_str = f"\n  Secondary: {chance}% chance {status or boosts}"
        elif isinstance(secondary, list):
            parts = []
            for s in secondary:
                parts.append(f"{s.get('chance','')}% {s.get('status','') or s.get('boosts','')}")
            sec_str = f"\n  Secondaries: {'; '.join(parts)}"
    extra = ""
    if m.get('drain'):
        extra += f"\n  Drain: {m['drain'][0]}/{m['drain'][1]}"
    if m.get('recoil'):
        extra += f"\n  Recoil: {m['recoil'][0]}/{m['recoil'][1]}"
    if m.get('critRatio'):
        extra += f"\n  Crit ratio: +{m['critRatio']}"
    if m.get('priority', 0) != 0:
        extra += f"\n  Priority: {m['priority']}"
    if m.get('multihit'):
        extra += f"\n  Multi-hit: {m['multihit']}"
    return (
        f"{m.get('name', name)} | Type: {m.get('type','?')} | Category: {m.get('category','?')}\n"
        f"  Base Power: {m.get('basePower',0)} | Accuracy: {m.get('accuracy','—')} | PP: {m.get('pp',0)}"
        f"{sec_str}{extra}"
    )


def format_item_info(name: str) -> str:
    """Return a human-readable summary of an item."""
    it = get_item(name)
    if not it:
        return f"Item '{name}' not found."
    desc = it.get('shortDesc') or it.get('desc') or ''
    return f"{it.get('name', name)} | {desc}"


def format_ability_info(name: str) -> str:
    """Return a human-readable summary of an ability."""
    ab = get_ability(name)
    if not ab:
        return f"Ability '{name}' not found."
    desc = ab.get('shortDesc') or ab.get('desc') or ''
    return f"{ab.get('name', name)} | {desc}"


def format_type_effectiveness(atk_type: str, def_type1: str, def_type2: str | None = None) -> str:
    """Return a readable effectiveness string."""
    mult = type_effectiveness(atk_type, def_type1, def_type2)
    def_str = def_type1 + (f"/{def_type2}" if def_type2 else "")
    if mult == 0:
        label = "immune (0x)"
    elif mult < 1:
        label = f"resisted ({mult}x)"
    elif mult > 1:
        label = f"super effective ({mult}x)"
    else:
        label = "neutral (1x)"
    return f"{atk_type} vs {def_str}: {label}"


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== pokedata self-test ===\n")
    print(format_pokemon_info("Miraidon"))
    print()
    print(format_pokemon_info("Flutter Mane"))
    print()
    print(format_move_info("Electro Drift"))
    print()
    print(format_move_info("Surging Strikes"))
    print()
    print(format_ability_info("Hadron Engine"))
    print()
    print(format_type_effectiveness("Fire", "Grass", "Steel"))
    print(format_type_effectiveness("Normal", "Ghost"))
    print(format_type_effectiveness("Ground", "Flying"))
    print(format_type_effectiveness("Ice", "Dragon", "Flying"))
    print()
    print(f"Pokedex entries: {len(_GEN.pokedex)}")
    print(f"Move entries: {len(_GEN.moves)}")
    print("\n=== Done ===")

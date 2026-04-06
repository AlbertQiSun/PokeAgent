"""
battle_context.py — Pre-compute and format all battle data for LLM prompt injection.

Combines:
  - Notebook intel (opponent sets, confirmed/inferred info)
  - Damage calculations for every available move vs opponent
  - Speed checks
  - Switch analysis (type matchup scores)
  - Surprise flags

Produces a single formatted string that gets injected into the LLM prompt
so it never needs to call tools — all math is done instantly.
"""

from poke_env.battle import Battle
from pokedata import get_pokemon, type_effectiveness, get_move
from damage_calc import calc_damage
from battle_notebook import BattleNotebook, PokemonIntel


def _calc_stat(base: int, ev: int = 252, iv: int = 31, level: int = 50,
               nature: float = 1.0, is_hp: bool = False) -> int:
    """Calculate actual stat at level 50."""
    if is_hp:
        return int((2 * base + iv + ev // 4) * level / 100) + level + 10
    return int(((2 * base + iv + ev // 4) * level / 100 + 5) * nature)


def _speed_stat(base: int, ev: int = 252, nature: float = 1.0,
                scarf: bool = False, level: int = 50) -> int:
    """Calculate effective speed."""
    stat = _calc_stat(base, ev, nature=nature)
    if scarf:
        stat = int(stat * 1.5)
    return stat


def compute_battle_context(battle: Battle, notebook: BattleNotebook) -> str:
    """Compute the full battle context string for the current turn."""
    sections = []

    our = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    if not our or not opp:
        return notebook.format_full_notebook()

    our_species = our.species
    opp_species = opp.species
    opp_intel = notebook.get_intel(opp_species)

    # Get our Pokemon's info
    our_data = get_pokemon(our_species)
    our_stats = our_data.get("baseStats", {}) if our_data else {}
    our_types = our_data.get("types", []) if our_data else []

    opp_data = get_pokemon(opp_species)
    opp_stats = opp_data.get("baseStats", {}) if opp_data else {}
    opp_types = opp_data.get("types", []) if opp_data else []

    # ── Section 1: Our active vs their active ─────────────────────────────────
    sections.append(f"=== TURN {battle.turn} ANALYSIS ===")
    sections.append(f"Your {our_species} vs opponent's {opp_species}")

    # Our item/ability from battle state
    our_item = our.item if our.item else ""
    our_ability = our.ability if our.ability else ""

    # ── Section 2: Damage calcs for each available move ───────────────────────
    if battle.available_moves:
        sections.append("\n── YOUR MOVES vs " + opp_species.upper() + " ──")
        move_calcs = []
        for move in battle.available_moves:
            result = calc_damage(
                our_species, move.id, opp_species,
                atk_item=our_item, atk_ability=our_ability,
            )
            if "error" in result:
                move_calcs.append((move.id, "???", 0, 0, 1.0))
                continue

            min_pct = result["min_pct"]
            max_pct = result["max_pct"]
            eff = result["effectiveness"]
            hits = result["hits"]

            # Apply notebook bulk adjustment if we have observations
            if opp_intel and opp_intel.bulk_adjustment != 1.0:
                adj = opp_intel.bulk_adjustment
                min_pct = round(min_pct * adj, 1)
                max_pct = round(max_pct * adj, 1)

            # KO analysis
            mid = (min_pct + max_pct) / 2
            if mid >= 100:
                ko_label = "OHKO"
            elif mid >= 50:
                ko_label = "2HKO"
            elif mid >= 34:
                ko_label = "3HKO"
            else:
                ko_label = ""

            # Effectiveness label
            if eff == 0:
                eff_label = "IMMUNE"
            elif eff >= 4:
                eff_label = "4x SE"
            elif eff >= 2:
                eff_label = "2x SE"
            elif eff <= 0.25:
                eff_label = "4x resist"
            elif eff <= 0.5:
                eff_label = "resist"
            else:
                eff_label = "neutral"

            hit_str = f" x{hits}" if hits > 1 else ""
            ko_str = f" → {ko_label}" if ko_label else ""

            move_type = move.type.name.capitalize() if move.type else "???"
            bp = move.base_power
            cat = move.category.name if move.category else "???"
            pri = f" pri={move.priority}" if move.priority != 0 else ""

            line = (f"  {move.id:20s} {move_type:8s} {cat:8s} BP:{bp:3d}{pri} "
                    f"→ {min_pct:5.1f}-{max_pct:5.1f}%{hit_str} ({eff_label}){ko_str}")
            sections.append(line)
            move_calcs.append((move.id, ko_label, min_pct, max_pct, eff))

    # ── Section 3: Opponent's likely moves vs us ──────────────────────────────
    if opp_intel:
        sections.append(f"\n── OPPONENT'S LIKELY MOVES vs {our_species.upper()} ──")
        predicted = opp_intel.predicted_moves
        # Fallback: if notebook has no moves, try KB directly
        if not predicted:
            from metagame_kb import get_most_likely_set
            kb = get_most_likely_set(opp_species)
            if kb:
                predicted = kb.get("moves", [])
        opp_item_est = opp_intel.item
        opp_ability_est = opp_intel.ability

        for move_name in predicted:
            result = calc_damage(
                opp_species, move_name, our_species,
                atk_item=opp_item_est, atk_ability=opp_ability_est,
            )
            if "error" in result:
                sections.append(f"  {move_name:20s} → ???")
                continue

            min_pct = result["min_pct"]
            max_pct = result["max_pct"]
            eff = result["effectiveness"]
            hits = result["hits"]

            mid = (min_pct + max_pct) / 2
            if mid >= 100:
                ko_label = "OHKO!"
            elif mid >= 50:
                ko_label = "2HKO"
            elif mid >= 34:
                ko_label = "3HKO"
            else:
                ko_label = ""

            if eff == 0:
                eff_label = "IMMUNE"
            elif eff >= 2:
                eff_label = "SE"
            elif eff <= 0.5:
                eff_label = "resist"
            else:
                eff_label = ""

            confirmed = "✓" if move_name in opp_intel.confirmed_moves else "~"
            hit_str = f" x{hits}" if hits > 1 else ""
            ko_str = f" → {ko_label}" if ko_label else ""
            eff_str = f" ({eff_label})" if eff_label else ""

            sections.append(
                f"  {confirmed}{move_name:20s} → {min_pct:5.1f}-{max_pct:5.1f}%{hit_str}{eff_str}{ko_str}"
            )

    # ── Section 4: Speed check ────────────────────────────────────────────────
    our_base_spe = our_stats.get("spe", 0)
    opp_base_spe = opp_stats.get("spe", 0)

    sections.append(f"\n── SPEED CHECK ──")
    # Standard speed calc
    our_spe_stat = _speed_stat(our_base_spe)
    opp_spe_stat = _speed_stat(opp_base_spe)

    speed_note = ""
    if opp_intel and opp_intel.speed_modifier == "Choice Scarf":
        opp_spe_stat = int(opp_spe_stat * 1.5)
        speed_note = " [SCARF]"
    elif opp_intel and opp_intel.speed_modifier:
        speed_note = f" [{opp_intel.speed_modifier}]"

    if our_spe_stat > opp_spe_stat:
        sections.append(f"  You outspeed: {our_species}({our_base_spe} base) > {opp_species}({opp_base_spe} base){speed_note}")
    elif our_spe_stat < opp_spe_stat:
        sections.append(f"  Opponent outspeeds: {opp_species}({opp_base_spe} base){speed_note} > {our_species}({our_base_spe} base)")
    else:
        sections.append(f"  Speed tie: both {our_base_spe} base (50/50)")

    # Priority moves
    if battle.available_moves:
        prio_moves = [m for m in battle.available_moves if m.priority > 0]
        if prio_moves:
            sections.append(f"  Priority moves available: {', '.join(m.id for m in prio_moves)}")

    # ── Section 5: Switch analysis ────────────────────────────────────────────
    if battle.available_switches:
        sections.append(f"\n── SWITCH OPTIONS ──")
        for mon in battle.available_switches:
            mon_data = get_pokemon(mon.species)
            mon_types = mon_data.get("types", []) if mon_data else []
            hp_pct = mon.current_hp_fraction * 100

            # How well does this mon match up vs opponent?
            # Check opponent's STAB effectiveness against our switch
            defensive_score = 1.0
            for opp_t in opp_types:
                t2 = mon_types[1] if len(mon_types) > 1 else None
                eff = type_effectiveness(opp_t, mon_types[0], t2) if mon_types else 1.0
                if eff < defensive_score:
                    defensive_score = eff

            # Our switch's STAB vs opponent
            offensive_score = 1.0
            for our_t in mon_types:
                t2 = opp_types[1] if len(opp_types) > 1 else None
                eff = type_effectiveness(our_t, opp_types[0], t2) if opp_types else 1.0
                if eff > offensive_score:
                    offensive_score = eff

            def_label = ""
            if defensive_score == 0:
                def_label = "immune to STAB"
            elif defensive_score <= 0.25:
                def_label = "4x resists STAB"
            elif defensive_score <= 0.5:
                def_label = "resists STAB"

            off_label = ""
            if offensive_score >= 4:
                off_label = "4x SE STAB"
            elif offensive_score >= 2:
                off_label = "SE STAB"

            matchup_parts = [x for x in [def_label, off_label] if x]
            matchup = " | ".join(matchup_parts) if matchup_parts else "neutral"

            types_str = "/".join(mon_types)
            sections.append(f"  {mon.species:20s} ({types_str:15s} {hp_pct:5.1f}% HP) — {matchup}")

    # ── Section 6: Notebook summary ───────────────────────────────────────────
    sections.append(f"\n── OPPONENT SCOUTING NOTEBOOK ──")
    sections.append(notebook.format_full_notebook())

    # ── Section 7: Surprise alerts ────────────────────────────────────────────
    surprises = notebook.get_new_surprises()
    if surprises:
        sections.append(f"\n⚠ ACTIVE SURPRISES:")
        for s in surprises[-5:]:
            sections.append(f"  {s}")

    # ── Section 8: Field conditions ───────────────────────────────────────────
    if battle.weather or battle.fields:
        sections.append(f"\n── FIELD ──")
        if battle.weather:
            sections.append(f"  Weather: {', '.join(w.name for w in battle.weather)}")
        if battle.fields:
            sections.append(f"  Terrain: {', '.join(f.name for f in battle.fields)}")
        if battle.side_conditions:
            sections.append(f"  Our side: {', '.join(s.name for s in battle.side_conditions)}")
        if battle.opponent_side_conditions:
            sections.append(f"  Opp side: {', '.join(s.name for s in battle.opponent_side_conditions)}")

    # ── Section 9: HP summary ─────────────────────────────────────────────────
    sections.append(f"\n── HP STATUS ──")
    sections.append(f"  Your {our_species}: {our.current_hp_fraction*100:.0f}% "
                    f"{'[' + our.status.name + ']' if our.status else ''}")
    sections.append(f"  Opp {opp_species}: {opp.current_hp_fraction*100:.0f}% "
                    f"{'[' + opp.status.name + ']' if opp.status else ''}")

    # Bench HP
    our_bench = [m for m in battle.team.values() if m != our and not m.fainted]
    opp_bench_count = 6 - len([m for m in battle.opponent_team.values() if m.fainted]) - 1
    if our_bench:
        bench_str = ", ".join(f"{m.species}({m.current_hp_fraction*100:.0f}%)" for m in our_bench)
        sections.append(f"  Your bench: {bench_str}")
    sections.append(f"  Opp bench: ~{opp_bench_count} remaining")

    return "\n".join(sections)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== battle_context self-test ===")
    print("(This module requires a live Battle object to test fully.)")
    print("Integration tested via local_llm_player.py and gemini_player.py")

    # Test the helper functions
    print(f"\nSpeed calc: base 135, 252 EV = {_speed_stat(135)}")
    print(f"Speed calc: base 35, 252 EV, Scarf = {_speed_stat(35, scarf=True)}")
    print(f"HP calc: base 100, 252 EV = {_calc_stat(100, 252, is_hp=True)}")
    print("\n=== Done ===")

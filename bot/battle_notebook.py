"""
battle_notebook.py — Per-battle state tracker with surprise detection.

Tracks opponent Pokemon intel throughout a battle:
  - Confirmed moves, ability, item
  - Inferred EV spread / nature from damage and speed observations
  - Surprise flags when reality doesn't match prediction

Starts from metagame KB priors, updates with battle observations.
"""

import re
from dataclasses import dataclass, field
from pokedata import get_pokemon, type_effectiveness
from metagame_kb import get_most_likely_set, get_sets, predict_unseen, infer_team_from_preview
from damage_calc import calc_damage


@dataclass
class PokemonIntel:
    """What we know about one opponent Pokemon."""
    species: str
    types: list[str] = field(default_factory=list)
    base_stats: dict = field(default_factory=dict)
    base_speed: int = 0

    # Confirmed from observation
    confirmed_moves: list[str] = field(default_factory=list)
    confirmed_ability: str = ""
    confirmed_item: str = ""
    confirmed_tera_type: str = ""
    confirmed_status: str = ""

    # KB prior (most likely set)
    prior_item: str = "???"
    prior_ability: str = "???"
    prior_moves: list[str] = field(default_factory=list)
    prior_spread: tuple = (0, 0, 0, 0, 0, 0)
    prior_nature: str = "???"
    prior_usage: float = 0.0

    # Current best-guess (starts as prior, updated by observations)
    estimated_item: str = "???"
    estimated_ability: str = "???"
    estimated_spread: tuple = (0, 0, 0, 0, 0, 0)  # HP, Atk, Def, SpA, SpD, Spe
    estimated_nature: str = "???"

    # Speed observations
    outsped_us: list[tuple[int, bool]] = field(default_factory=list)
    # (our_base_speed, did_they_outspeed)
    speed_modifier: str = ""  # "Choice Scarf", "Tailwind", "Trick Room", etc.

    # Bulk observations: (expected_pct, actual_pct, move_category)
    damage_observations: list[dict] = field(default_factory=list)
    bulk_adjustment: float = 1.0  # multiplier for calcs if bulkier/frailer than expected

    # Surprises detected
    surprises: list[str] = field(default_factory=list)

    @property
    def item(self) -> str:
        return self.confirmed_item or self.estimated_item

    @property
    def ability(self) -> str:
        return self.confirmed_ability or self.estimated_ability

    @property
    def moves_known(self) -> int:
        return len(self.confirmed_moves)

    @property
    def predicted_moves(self) -> list[str]:
        """Confirmed moves + remaining slots filled from prior."""
        moves = list(self.confirmed_moves)
        for m in self.prior_moves:
            if m.lower() not in [x.lower() for x in moves] and len(moves) < 4:
                moves.append(m)
        return moves

    def effective_spread(self) -> tuple:
        """Return the best-guess spread for damage calcs."""
        if any(v > 0 for v in self.estimated_spread):
            return self.estimated_spread
        return self.prior_spread


class BattleNotebook:
    """Tracks all opponent intel for one battle."""

    def __init__(self):
        self.opponent_pokemon: dict[str, PokemonIntel] = {}
        self.unseen_predictions: list[tuple[str, float]] = []
        self.turn: int = 0
        self.our_pokemon_speeds: dict[str, int] = {}  # our species -> base speed

    def init_from_preview(self, opp_species: list[str], our_species: list[str] = None):
        """Initialize notebook from team preview using KB priors."""
        inferred = infer_team_from_preview(opp_species)

        for species in opp_species:
            poke = get_pokemon(species)
            base_stats = poke.get("baseStats", {}) if poke else {}
            types = poke.get("types", []) if poke else []

            kb_set = inferred.get(species, {})
            intel = PokemonIntel(
                species=species,
                types=types,
                base_stats=base_stats,
                base_speed=base_stats.get("spe", 0),
                prior_item=kb_set.get("item", "???"),
                prior_ability=kb_set.get("ability", "???"),
                prior_moves=kb_set.get("moves", []),
                prior_spread=kb_set.get("spread", (0, 0, 0, 0, 0, 0)),
                prior_nature=kb_set.get("nature", "???"),
                prior_usage=kb_set.get("usage", 0.0),
                estimated_item=kb_set.get("item", "???"),
                estimated_ability=kb_set.get("ability", "???"),
                estimated_spread=kb_set.get("spread", (0, 0, 0, 0, 0, 0)),
                estimated_nature=kb_set.get("nature", "???"),
            )
            self.opponent_pokemon[species.lower()] = intel

        # Predict unseen team members
        self.unseen_predictions = predict_unseen(opp_species)

        # Track our speeds
        if our_species:
            for sp in our_species:
                p = get_pokemon(sp)
                if p:
                    self.our_pokemon_speeds[sp.lower()] = p.get("baseStats", {}).get("spe", 0)

    def get_intel(self, species: str) -> PokemonIntel | None:
        """Get intel for an opponent Pokemon."""
        key = species.lower()
        # Try exact match first
        if key in self.opponent_pokemon:
            return self.opponent_pokemon[key]
        # Fuzzy match
        for k, v in self.opponent_pokemon.items():
            if key in k or k in key:
                return v
        return None

    def _get_or_create_intel(self, species: str) -> PokemonIntel:
        """Get or create intel entry."""
        intel = self.get_intel(species)
        if intel:
            return intel
        # New Pokemon not seen at preview
        poke = get_pokemon(species)
        base_stats = poke.get("baseStats", {}) if poke else {}
        types = poke.get("types", []) if poke else []
        kb_set = get_most_likely_set(species) or {}

        intel = PokemonIntel(
            species=species,
            types=types,
            base_stats=base_stats,
            base_speed=base_stats.get("spe", 0),
            prior_item=kb_set.get("item", "???"),
            prior_ability=kb_set.get("ability", "???"),
            prior_moves=kb_set.get("moves", []),
            prior_spread=kb_set.get("spread", (0, 0, 0, 0, 0, 0)),
            prior_nature=kb_set.get("nature", "???"),
            prior_usage=kb_set.get("usage", 0.0),
            estimated_item=kb_set.get("item", "???"),
            estimated_ability=kb_set.get("ability", "???"),
            estimated_spread=kb_set.get("spread", (0, 0, 0, 0, 0, 0)),
            estimated_nature=kb_set.get("nature", "???"),
        )
        self.opponent_pokemon[species.lower()] = intel
        return intel

    # ── Observation updates ───────────────────────────────────────────────────

    def observe_move(self, species: str, move_name: str):
        """Record that opponent used a move."""
        intel = self._get_or_create_intel(species)
        clean = move_name.lower().replace(" ", "").replace("-", "")
        for m in intel.confirmed_moves:
            if m.lower().replace(" ", "").replace("-", "") == clean:
                return  # already recorded
        intel.confirmed_moves.append(move_name)

        # If move isn't in any known set, flag surprise
        all_kb_moves = set()
        for s in get_sets(species):
            all_kb_moves.update(m.lower() for m in s.get("moves", []))
        if clean not in {m.lower().replace(" ", "").replace("-", "") for m in all_kb_moves} and all_kb_moves:
            intel.surprises.append(f"Unexpected move: {move_name}")

    def observe_ability(self, species: str, ability_name: str):
        """Record opponent's ability (revealed by game messages)."""
        intel = self._get_or_create_intel(species)
        intel.confirmed_ability = ability_name

        if intel.prior_ability != "???" and ability_name.lower() != intel.prior_ability.lower():
            intel.surprises.append(f"Unexpected ability: {ability_name} (expected {intel.prior_ability})")
            # Re-evaluate which set they might be running
            self._update_set_estimate(intel)

    def observe_item(self, species: str, item_name: str):
        """Record opponent's item (revealed by Knock Off, Trick, consumption, etc.)."""
        intel = self._get_or_create_intel(species)
        intel.confirmed_item = item_name

        if intel.prior_item != "???" and item_name.lower() != intel.prior_item.lower():
            intel.surprises.append(f"Unexpected item: {item_name} (expected {intel.prior_item})")
            self._update_set_estimate(intel)

    def observe_tera(self, species: str, tera_type: str):
        """Record opponent's Tera type."""
        intel = self._get_or_create_intel(species)
        intel.confirmed_tera_type = tera_type

    def observe_status(self, species: str, status: str):
        """Record status condition."""
        intel = self._get_or_create_intel(species)
        intel.confirmed_status = status

    def observe_speed(self, species: str, our_species: str, they_moved_first: bool):
        """Record speed observation from turn order."""
        intel = self._get_or_create_intel(species)
        our_base = self.our_pokemon_speeds.get(our_species.lower(), 0)
        if our_base == 0:
            return

        intel.outsped_us.append((our_base, they_moved_first))

        # Detect speed surprise
        their_base = intel.base_speed
        # Standard assumption: both 252 Spe with neutral nature
        # At level 50: stat = (2*base + 31 + 252/4) * 50/100 + 5
        their_est_speed = int((2 * their_base + 31 + 63) * 50 / 100 + 5)
        our_est_speed = int((2 * our_base + 31 + 63) * 50 / 100 + 5)

        if they_moved_first and their_base < our_base - 5:
            # They're slower base but moved first — something is boosting them
            if intel.speed_modifier == "":
                # Check if Choice Scarf explains it
                scarf_speed = int(their_est_speed * 1.5)
                if scarf_speed > our_est_speed:
                    intel.speed_modifier = "Choice Scarf"
                    intel.estimated_item = "Choice Scarf"
                    intel.surprises.append(
                        f"Speed surprise: {species} outsped {our_species} "
                        f"(base {their_base} vs {our_base}) — likely Choice Scarf"
                    )
                else:
                    intel.speed_modifier = "unknown boost"
                    intel.surprises.append(
                        f"Speed surprise: {species} outsped {our_species} "
                        f"(base {their_base} vs {our_base}) — unknown speed boost"
                    )

    def observe_damage(self, species: str, our_species: str, our_move: str,
                       actual_pct: float, our_item: str = "", our_ability: str = ""):
        """Record damage dealt to opponent and infer their set via Bayesian update.

        actual_pct: how much % HP the opponent lost.
        """
        intel = self._get_or_create_intel(species)

        # Calculate expected damage with standard spread
        result = calc_damage(
            our_species, our_move, species,
            atk_item=our_item, atk_ability=our_ability,
        )
        if "error" in result or result["max_pct"] == 0:
            return

        expected_mid = (result["min_pct"] + result["max_pct"]) / 2

        obs = {
            "move": our_move,
            "expected_mid": expected_mid,
            "expected_range": (result["min_pct"], result["max_pct"]),
            "actual_pct": actual_pct,
        }
        intel.damage_observations.append(obs)

        if actual_pct < 0.01:
            return  # immune, don't analyze

        ratio = actual_pct / expected_mid if expected_mid > 0 else 1.0

        # ── Bayesian item/spread inference from damage ──
        from pokedata import get_move
        move_data = get_move(our_move)
        cat = move_data.get("category", "Physical") if move_data else "Physical"

        inferred_item = None
        inferred_spread_type = None  # "defensive" or "offensive"

        if ratio < 0.65 and not intel.confirmed_item:
            # Much less damage → tankier than standard
            if cat == "Special":
                # Could be Assault Vest (+50% SpD) or SpD investment
                # AV would make ratio ~0.67, heavy SpD invest ~0.55-0.75
                if ratio < 0.55:
                    inferred_spread_type = "specially_defensive"
                    intel.surprises.append(
                        f"Bulk surprise: {our_move} did {actual_pct:.0f}% "
                        f"(expected {expected_mid:.0f}%) — heavy SpD investment"
                    )
                else:
                    inferred_item = "Assault Vest"
                    intel.estimated_item = "Assault Vest"
                    intel.surprises.append(
                        f"Bulk surprise: {our_move} did {actual_pct:.0f}% "
                        f"(expected {expected_mid:.0f}%) — likely Assault Vest"
                    )
            elif cat == "Physical":
                if ratio < 0.55:
                    inferred_spread_type = "physically_defensive"
                    intel.surprises.append(
                        f"Bulk surprise: {our_move} did {actual_pct:.0f}% "
                        f"(expected {expected_mid:.0f}%) — heavy Def investment"
                    )
                else:
                    intel.surprises.append(
                        f"Bulk surprise: {our_move} did {actual_pct:.0f}% "
                        f"(expected {expected_mid:.0f}%) — likely defensive item or Def invest"
                    )
            intel.bulk_adjustment = ratio

        elif ratio > 1.4:
            inferred_spread_type = "offensive"
            intel.surprises.append(
                f"Frailty note: {our_move} did {actual_pct:.0f}% "
                f"(expected {expected_mid:.0f}%) — likely offensive spread"
            )
            intel.bulk_adjustment = ratio

        # ── Bayesian set re-ranking with damage evidence ──
        self._bayesian_update(intel, damage_obs=obs, inferred_item=inferred_item,
                              inferred_spread_type=inferred_spread_type)

    def observe_damage_taken(self, species: str, our_species: str, opp_move: str,
                             actual_pct: float):
        """Record damage WE took from opponent's move → infer their item/boosts.

        actual_pct: how much % HP we lost.
        """
        intel = self._get_or_create_intel(species)

        # Calc expected damage from their move with their estimated set
        result = calc_damage(
            species, opp_move, our_species,
            atk_item=intel.item, atk_ability=intel.ability,
        )
        if "error" in result or result["max_pct"] == 0:
            return

        expected_mid = (result["min_pct"] + result["max_pct"]) / 2
        if expected_mid < 1:
            return

        ratio = actual_pct / expected_mid

        inferred_item = None
        from pokedata import get_move
        move_data = get_move(opp_move)
        cat = move_data.get("category", "Physical") if move_data else "Physical"

        if ratio > 1.25 and not intel.confirmed_item:
            # They did more damage than expected → boosting item
            if 1.25 <= ratio <= 1.40:
                inferred_item = "Life Orb"
                intel.estimated_item = "Life Orb"
                intel.surprises.append(
                    f"Damage inference: {opp_move} did {actual_pct:.0f}% to us "
                    f"(expected {expected_mid:.0f}%) — likely Life Orb"
                )
            elif ratio > 1.40 and cat == "Physical":
                inferred_item = "Choice Band"
                intel.estimated_item = "Choice Band"
                intel.surprises.append(
                    f"Damage inference: {opp_move} did {actual_pct:.0f}% to us "
                    f"(expected {expected_mid:.0f}%) — likely Choice Band"
                )
            elif ratio > 1.40 and cat == "Special":
                inferred_item = "Choice Specs"
                intel.estimated_item = "Choice Specs"
                intel.surprises.append(
                    f"Damage inference: {opp_move} did {actual_pct:.0f}% to us "
                    f"(expected {expected_mid:.0f}%) — likely Choice Specs"
                )

        self._bayesian_update(intel, inferred_item=inferred_item)

    def _bayesian_update(self, intel: PokemonIntel, damage_obs: dict = None,
                         inferred_item: str = None, inferred_spread_type: str = None):
        """Bayesian re-ranking of sets given all evidence."""
        sets = get_sets(intel.species)
        if not sets:
            return

        # Compute posterior probability for each set
        posteriors = []
        for s in sets:
            # Start with prior (usage %)
            log_prob = s["usage"]

            # Evidence: confirmed item
            if intel.confirmed_item:
                if s["item"].lower() == intel.confirmed_item.lower():
                    log_prob += 2.0
                else:
                    log_prob -= 2.0  # strong negative evidence

            # Evidence: inferred item from damage
            if inferred_item and not intel.confirmed_item:
                if s["item"].lower() == inferred_item.lower():
                    log_prob += 1.0
                elif inferred_item.lower() in ("choice band", "choice specs", "choice scarf"):
                    # If we inferred a choice item but this set has a different one
                    if "choice" in s["item"].lower():
                        log_prob += 0.5  # at least it's a choice item
                    else:
                        log_prob -= 0.5

            # Evidence: confirmed ability
            if intel.confirmed_ability:
                if s["ability"].lower() == intel.confirmed_ability.lower():
                    log_prob += 1.5
                else:
                    log_prob -= 1.5

            # Evidence: confirmed moves
            set_moves_lower = {m.lower() for m in s.get("moves", [])}
            for cm in intel.confirmed_moves:
                if cm.lower() in set_moves_lower:
                    log_prob += 0.5
                else:
                    log_prob -= 0.3  # move not in this set

            # Evidence: speed modifier
            if intel.speed_modifier == "Choice Scarf":
                if "choice scarf" in s["item"].lower():
                    log_prob += 2.0
                else:
                    log_prob -= 1.0

            # Evidence: spread type inference
            if inferred_spread_type:
                spread = s.get("spread", (0, 0, 0, 0, 0, 0))
                hp_ev, atk_ev, def_ev, spa_ev, spd_ev, spe_ev = spread
                if inferred_spread_type == "specially_defensive" and spd_ev >= 200:
                    log_prob += 0.8
                elif inferred_spread_type == "physically_defensive" and def_ev >= 200:
                    log_prob += 0.8
                elif inferred_spread_type == "offensive" and (atk_ev >= 200 or spa_ev >= 200):
                    log_prob += 0.5

            posteriors.append((s, log_prob))

        # Normalize and pick best
        if not posteriors:
            return

        posteriors.sort(key=lambda x: x[1], reverse=True)
        best_set = posteriors[0][0]

        # Update estimates from best matching set
        intel.estimated_item = intel.confirmed_item or best_set["item"]
        intel.estimated_ability = intel.confirmed_ability or best_set["ability"]
        intel.estimated_spread = best_set["spread"]
        intel.estimated_nature = best_set["nature"]

        # Update predicted moves: keep confirmed, fill from best set
        # (already handled by predicted_moves property using prior_moves)
        intel.prior_moves = best_set.get("moves", intel.prior_moves)

    def _update_set_estimate(self, intel: PokemonIntel):
        """Re-evaluate which set the opponent is running based on confirmed info."""
        self._bayesian_update(intel)

    # ── Formatting for LLM prompt ─────────────────────────────────────────────

    def format_pokemon_intel(self, species: str) -> str:
        """Format what we know about one opponent Pokemon."""
        intel = self.get_intel(species)
        if not intel:
            return f"{species}: no intel"

        lines = [f"┌─ {intel.species} ({'/'.join(intel.types)}) ─"]

        # Ability
        if intel.confirmed_ability:
            lines.append(f"│ Ability:  {intel.confirmed_ability} ✓")
        elif intel.estimated_ability != "???":
            lines.append(f"│ Ability:  ~{intel.estimated_ability} (KB estimate, {intel.prior_usage*100:.0f}%)")
        else:
            lines.append(f"│ Ability:  ???")

        # Item
        if intel.confirmed_item:
            lines.append(f"│ Item:    {intel.confirmed_item} ✓")
        elif intel.estimated_item != "???":
            lines.append(f"│ Item:    ~{intel.estimated_item} (KB estimate)")
        else:
            lines.append(f"│ Item:    ???")

        # Tera
        if intel.confirmed_tera_type:
            lines.append(f"│ Tera:    {intel.confirmed_tera_type} ✓")
        else:
            lines.append(f"│ Tera:    ???")

        # Moves
        moves_line = "│ Moves:  "
        for i in range(4):
            if i < len(intel.confirmed_moves):
                moves_line += f" {intel.confirmed_moves[i]} ✓,"
            elif i < len(intel.predicted_moves):
                moves_line += f" ~{intel.predicted_moves[i]},"
            else:
                moves_line += " ???,"
        lines.append(moves_line.rstrip(","))

        # Speed
        speed_info = f"│ Speed:   base {intel.base_speed}"
        if intel.speed_modifier:
            speed_info += f" [{intel.speed_modifier}]"
        lines.append(speed_info)

        # Bulk
        if intel.bulk_adjustment != 1.0:
            if intel.bulk_adjustment < 1.0:
                lines.append(f"│ Bulk:    tankier than standard (×{1/intel.bulk_adjustment:.1f} effective)")
            else:
                lines.append(f"│ Bulk:    frailer than standard")

        # Surprises
        if intel.surprises:
            lines.append(f"│ ⚠ Surprises:")
            for s in intel.surprises[-3:]:  # last 3
                lines.append(f"│   - {s}")

        lines.append(f"└─")
        return "\n".join(lines)

    def format_full_notebook(self) -> str:
        """Format the entire notebook for LLM prompt injection."""
        sections = ["OPPONENT NOTEBOOK"]

        for key in sorted(self.opponent_pokemon.keys()):
            sections.append(self.format_pokemon_intel(self.opponent_pokemon[key].species))

        if self.unseen_predictions:
            preds = ", ".join(f"{n}({p*100:.0f}%)" for n, p in self.unseen_predictions[:3])
            sections.append(f"Predicted unseen teammates: {preds}")

        return "\n".join(sections)

    def get_new_surprises(self) -> list[str]:
        """Return all surprise flags across all Pokemon (for LLM attention)."""
        all_surprises = []
        for intel in self.opponent_pokemon.values():
            for s in intel.surprises:
                all_surprises.append(f"[{intel.species}] {s}")
        return all_surprises


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== battle_notebook self-test ===\n")

    nb = BattleNotebook()
    nb.init_from_preview(
        opp_species=["Miraidon", "Skeledirge", "Garganacl"],
        our_species=["Urshifu-Rapid-Strike", "Ogerpon-Hearthflame", "Terapagos"],
    )

    print(nb.format_full_notebook())
    print()

    # Simulate battle observations
    print("--- Turn 1: Miraidon uses Electro Drift ---")
    nb.observe_move("Miraidon", "Electro Drift")
    nb.turn = 1

    print("--- Turn 2: We hit Miraidon with Surging Strikes, does 45% (expected ~55%) ---")
    nb.observe_damage("Miraidon", "Urshifu-Rapid-Strike", "Surging Strikes",
                       actual_pct=45.0, our_item="Choice Scarf")
    nb.turn = 2

    print("--- Turn 2: Miraidon ability revealed as Hadron Engine ---")
    nb.observe_ability("Miraidon", "Hadron Engine")

    print("--- Turn 3: Garganacl outspeeds our Urshifu (base 97 vs 35) ---")
    nb.observe_speed("Garganacl", "Urshifu-Rapid-Strike", they_moved_first=True)
    nb.turn = 3

    print()
    print(nb.format_full_notebook())
    print()

    surprises = nb.get_new_surprises()
    print(f"Surprises detected: {len(surprises)}")
    for s in surprises:
        print(f"  {s}")
    print()
    print("=== Done ===")

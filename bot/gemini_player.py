import asyncio
import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from poke_env.player import Player
from poke_env.battle import Battle, AbstractBattle

from pokemon_pool import POOL, POOL_NAMES, POOL_SIZE, build_team_string


def _pool_summary() -> str:
    lines = []
    for i, (name, entry) in enumerate(zip(POOL_NAMES, POOL)):
        item_m = re.search(r'@ (.+)', entry)
        item = item_m.group(1) if item_m else "?"
        moves = re.findall(r'- (.+)', entry)
        lines.append(f"  {i:2}: {name} @ {item} | Moves: {', '.join(moves)}")
    return "\n".join(lines)


def _opp_preview_summary(opp_team) -> str:
    lines = []
    for mon in opp_team:
        types_str = "/".join(
            t.name.capitalize() for t in [mon.type_1, mon.type_2] if t is not None
        )
        lines.append(f"  - {mon.species} ({types_str})")
    return "\n".join(lines) if lines else "  (unknown)"


TEAM_SYSTEM = (
    "You are a Pokemon BSS Reg J team-building expert."
    " Respond ONLY with a raw JSON object — no markdown, no extra text."
)

BATTLE_SYSTEM = (
    "You are a Pokemon Showdown expert playing Gen 9 Battle Stadium Singles."
    " Every turn, you will be given the battle state. Your task is to select the optimal move or switch."
    ' You must ONLY output a valid JSON with exactly these keys: "action_type" ("move" or "switch"),'
    ' "target_name" (the exact name of the move or Pokemon), and "reasoning" (a short explanation).'
    ' Example move: {"action_type": "move", "target_name": "moonblast", "reasoning": "Super effective."}'
    ' Example switch: {"action_type": "switch", "target_name": "dragonite", "reasoning": "Resists earthquake."}'
    " Do NOT output any markdown, just the raw JSON object."
)


class GeminiPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable.")

        self.client = genai.Client()
        self.model  = "gemini-3.1-flash-lite-preview"

        # Battle history: track last N turns per battle for context
        self._turn_history: dict[str, list[dict]] = {}
        self.history_len = 5

        # Pre-build the first team synchronously so it's ready before the event loop starts.
        print("[Gemini] Building first team (this may take a few seconds)...")
        self._packed_team: str = self._build_team_sync()
        self._building_next: bool = False  # guard against concurrent rebuilds
        self._battle_count: int = 0  # for team variety prompts

    # ── Gemini helpers ────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        return text.strip()

    def _sync_call(self, prompt: str, system: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.3),
        )
        return self._clean(resp.text)

    async def _async_call(self, prompt: str, system: str, temperature: float = 0.3) -> str:
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
        return self._clean(resp.text)

    # ── Phase 1: build team of 6 ──────────────────────────────────────────────

    def _team_prompt(self) -> str:
        variety = ""
        if self._battle_count > 0:
            variety = (
                f"\nThis is battle #{self._battle_count + 1}. "
                "Try a DIFFERENT team composition than before — explore new synergies, "
                "different cores, and alternative win conditions.\n"
            )
        return (
            "From the pool below, pick exactly 6 Pokemon to form the strongest BSS Reg J team.\n"
            "Consider: type coverage, speed control, win conditions, and team synergy.\n"
            f"{variety}\n"
            f"Pool:\n{_pool_summary()}\n\n"
            'Return JSON: {"picks": [i, i, i, i, i, i], "reasoning": "..."}\n'
            "picks must be exactly 6 unique integers from 0 to 19."
        )

    def _parse_team_picks(self, text: str) -> list[int]:
        data  = json.loads(text)
        raw   = [int(p) for p in data["picks"]]
        seen, unique = set(), []
        for p in raw:
            if p not in seen and 0 <= p < POOL_SIZE:
                seen.add(p); unique.append(p)
        for i in range(POOL_SIZE):
            if len(unique) >= 6: break
            if i not in seen: seen.add(i); unique.append(i)
        return unique[:6]

    def _apply_picks(self, picks: list[int]) -> str:
        names = [POOL_NAMES[i] for i in picks]
        print(f"[Gemini] Team of 6: {', '.join(names)}")
        team_str = build_team_string(picks)
        self.update_team(team_str)
        return self._team.yield_team()

    def _build_team_sync(self) -> str:
        """Blocking call — only used in __init__ before the event loop starts."""
        try:
            text  = self._sync_call(self._team_prompt(), TEAM_SYSTEM)
            picks = self._parse_team_picks(text)
            packed = self._apply_picks(picks)
            print("[Gemini] Team ready.")
            return packed
        except Exception as e:
            print(f"[Gemini] Team build failed ({e}), using default pool order.")
            return self._apply_picks(list(range(6)))

    async def _build_team_async(self) -> None:
        """Non-blocking rebuild called after each battle with higher temperature for variety."""
        if self._building_next:
            return
        self._building_next = True
        try:
            self._battle_count += 1
            print("[Gemini] Rebuilding team for next battle...")
            text  = await self._async_call(self._team_prompt(), TEAM_SYSTEM, temperature=0.7)
            picks = self._parse_team_picks(text)
            self._packed_team = self._apply_picks(picks)
            print("[Gemini] Next team ready.")
        except Exception as e:
            print(f"[Gemini] Async team build failed ({e}), keeping previous team.")
        finally:
            self._building_next = False

    @property
    def next_team(self) -> str:  # type: ignore[override]
        """Return the pre-built packed team (already ready, no blocking call)."""
        return self._packed_team

    # ── Phase 2: pick best 3 at team preview (async) ─────────────────────────

    async def teampreview(self, battle: AbstractBattle) -> str:  # type: ignore[override]
        our_mons = list(battle.team.values())
        opp_mons = list(battle.teampreview_opponent_team)

        our_summary = "\n".join(
            f"  {i+1}: {m.species} "
            f"({'/'.join(t.name.capitalize() for t in [m.type_1, m.type_2] if t)})"
            for i, m in enumerate(our_mons)
        )

        prompt = (
            "You are in BSS Reg J team preview. Choose 3 Pokemon to bring.\n\n"
            f"Your registered team (1-indexed):\n{our_summary}\n\n"
            f"Opponent's team:\n{_opp_preview_summary(opp_mons)}\n\n"
            "Pick 3 from YOUR team that best counter the opponent. Order matters — index 1 is your lead.\n"
            'Return JSON: {"picks": [a, b, c], "reasoning": "..."}\n'
            "picks must be exactly 3 unique integers from 1 to 6."
        )
        try:
            print("[Gemini] Thinking about lead selection...")
            text = await self._async_call(prompt, TEAM_SYSTEM)
            data = json.loads(text)
            raw  = [int(p) for p in data["picks"]]
            seen, valid = set(), []
            for p in raw:
                if 1 <= p <= len(our_mons) and p not in seen:
                    seen.add(p); valid.append(p)
            for i in range(1, len(our_mons) + 1):
                if len(valid) >= 3: break
                if i not in seen: seen.add(i); valid.append(i)
            picks = valid[:3]
            chosen = [our_mons[p - 1].species for p in picks]
            print(f"[Gemini] Bringing 3: {', '.join(chosen)}")
            print(f"[Gemini] Lead reasoning: {data.get('reasoning', '')}")
            return "/team " + "".join(str(p) for p in picks)
        except Exception as e:
            print(f"[Gemini] Teampreview failed ({e}), using default order.")
            return "/team 123"

    # ── Battle history tracking ─────────────────────────────────────────────────

    def _record_turn(self, battle: Battle) -> None:
        """Record what happened this turn for history context."""
        tag = battle.battle_tag
        if tag not in self._turn_history:
            self._turn_history[tag] = []

        entry = {"turn": battle.turn}
        if battle.active_pokemon:
            entry["our_active"] = battle.active_pokemon.species
            entry["our_hp"] = f"{battle.active_pokemon.current_hp_fraction*100:.0f}%"
        if battle.opponent_active_pokemon:
            entry["opp_active"] = battle.opponent_active_pokemon.species
            entry["opp_hp"] = f"{battle.opponent_active_pokemon.current_hp_fraction*100:.0f}%"
        self._turn_history[tag].append(entry)

    def _format_history(self, battle: Battle) -> str:
        """Format recent turn history for the prompt."""
        tag = battle.battle_tag
        history = self._turn_history.get(tag, [])
        if not history:
            return ""
        recent = history[-self.history_len:]
        lines = ["\n-- Recent Battle History (last few turns) --"]
        for h in recent:
            parts = [f"Turn {h.get('turn', '?')}:"]
            if "our_active" in h:
                parts.append(f"Our {h['our_active']} ({h['our_hp']})")
            if "opp_active" in h:
                parts.append(f"vs {h['opp_active']} ({h['opp_hp']})")
            lines.append("  " + " | ".join(parts))
        return "\n".join(lines)

    # ── After each battle: rebuild team in background ─────────────────────────

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        # Clean up history for finished battle
        tag = getattr(battle, "battle_tag", None)
        if tag and tag in self._turn_history:
            del self._turn_history[tag]
        asyncio.create_task(self._build_team_async())

    # ── Phase 3: battle moves ─────────────────────────────────────────────────

    def choose_move(self, battle: Battle):
        self._record_turn(battle)
        prompt = self._format_battle_state(battle)
        available_moves   = battle.available_moves
        available_switches = battle.available_switches

        try:
            text = self._sync_call(prompt, BATTLE_SYSTEM)
            data = json.loads(text)
            action_type = data.get("action_type", "").lower()
            target_name = data.get("target_name", "").lower()
            reasoning   = data.get("reasoning", "")

            print(f"\n[Gemini] Reasoning: {reasoning}")
            print(f"[Gemini] Action: {action_type} -> {target_name}")

            if action_type == "move" and available_moves:
                for move in available_moves:
                    clean_id     = move.id.lower()
                    clean_target = re.sub(r'[^a-z0-9]', '', target_name)
                    if clean_target in clean_id or clean_id in clean_target:
                        return self.create_order(move)
                return self.create_order(available_moves[0])

            elif action_type == "switch" and available_switches:
                for mon in available_switches:
                    clean_sp     = mon.species.lower()
                    clean_target = re.sub(r'[^a-z0-9]', '', target_name)
                    if clean_target in clean_sp or clean_sp in clean_target:
                        return self.create_order(mon)
                return self.create_order(available_switches[0])

        except Exception as e:
            print(f"\n[Gemini] Error: {e} — falling back to random.")

        return self.choose_random_move(battle)

    def _format_battle_state(self, battle: Battle) -> str:
        state = ["=== CURRENT BATTLE STATE ===", f"Turn: {battle.turn}"]
        if battle.active_pokemon:
            a = battle.active_pokemon
            state += [
                "\n-- Your Active Pokemon --",
                f"Species: {a.species}",
                f"HP: {a.current_hp_fraction*100:.1f}%",
                f"Status: {a.status.name if a.status else 'None'}",
                f"Stat Boosts: {a.boosts}",
            ]
        if battle.opponent_active_pokemon:
            o = battle.opponent_active_pokemon
            state += [
                "\n-- Opponent Active Pokemon --",
                f"Species: {o.species}",
                f"HP: {o.current_hp_fraction*100:.1f}%",
                f"Status: {o.status.name if o.status else 'None'}",
            ]
        if battle.weather or battle.fields:
            state += [
                "\n-- Field Conditions --",
                f"Weather: {[w.name for w in battle.weather]}",
                f"Fields: {[f.name for f in battle.fields]}",
            ]
        state.append("\n-- Your Available Actions --")
        if battle.available_moves:
            state.append("Moves: " + ", ".join(
                f"{m.id} (BP:{m.base_power}, Type:{m.type.name})"
                for m in battle.available_moves
            ))
        else:
            state.append("Moves: None")
        if battle.available_switches:
            state.append("Switches: " + ", ".join(
                f"{m.species} ({m.current_hp_fraction*100:.1f}% HP)"
                for m in battle.available_switches
            ))
        else:
            state.append("Switches: None")

        # Add battle history context
        history_text = self._format_history(battle)
        if history_text:
            state.append(history_text)

        state.append("\nBased on this game state and battle history, determine the single best action.")
        return "\n".join(state)

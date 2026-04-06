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
from battle_notebook import BattleNotebook
from battle_context import compute_battle_context
from battle_logger import BattleLogger


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
    "You are an expert Pokemon Showdown player (Gen 9 Battle Stadium Singles, Reg J).\n"
    "Each turn you receive a PRE-CALCULATED ANALYSIS with damage numbers, speed checks,\n"
    "type matchups, and an opponent scouting notebook. ALL MATH IS ALREADY DONE.\n\n"
    "Your job is STRATEGY — predict the opponent's play and choose the best action.\n\n"
    "DECISION PRIORITIES (follow in order):\n"
    "1. KO THREAT: If you can KO the opponent this turn, ATTACK. Do not switch.\n"
    "2. SURVIVE: If the opponent can KO you and you cannot KO them, switch to a resist.\n"
    "3. ATTACK WHEN AHEAD: If you can 2HKO and they cannot OHKO you, ATTACK. Do not pivot.\n"
    "4. MOMENTUM: If no clear KO path exists, use the best damaging move or a pivot move.\n"
    "5. SETUP: Only use setup moves (Swords Dance, etc.) if the opponent cannot threaten you.\n\n"
    "ANTI-PATTERNS TO AVOID:\n"
    "- DO NOT switch back and forth between the same Pokemon. This wastes turns.\n"
    "- DO NOT switch when you have a strong attack that does 30%+ damage.\n"
    "- DO NOT 'preserve' a Pokemon by switching it out repeatedly. Dead weight is worse.\n"
    "- DO NOT switch into a Pokemon that was just forced out — the threat is still there.\n\n"
    "Think about what the opponent will do THIS turn:\n"
    "- Will they stay in or switch? What is their best move?\n"
    "- Pay attention to SURPRISES — unexpected items, abilities, or EV spreads.\n\n"
    "Output ONLY a JSON object:\n"
    '{"action_type": "move" or "switch", "target_name": "exact name", '
    '"reasoning": "1-2 sentences explaining your prediction and decision"}\n'
    "No markdown. Just the raw JSON."
)


class GeminiPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable.")

        self.client = genai.Client()
        self.model  = "gemini-3.1-flash-lite-preview"

        # Per-battle notebooks and loggers
        self._notebooks: dict[str, BattleNotebook] = {}
        self._last_opp_hp: dict[str, float] = {}  # battle_tag -> last opp HP fraction
        self._loggers: dict[str, BattleLogger] = {}

        self._battle_count: int = 0
        self._building_next: bool = False

        print("[Gemini] Building first team (this may take a few seconds)...")
        self._packed_team: str = self._build_team_sync()

    # ── Notebook management ───────────────────────────────────────────────────

    def _get_notebook(self, battle: Battle) -> BattleNotebook:
        tag = battle.battle_tag
        if tag not in self._notebooks:
            self._notebooks[tag] = BattleNotebook()
        return self._notebooks[tag]

    def _update_notebook(self, battle: Battle):
        """Update notebook with observations from the current battle state."""
        nb = self._get_notebook(battle)
        tag = battle.battle_tag

        # Update opponent observations
        opp = battle.opponent_active_pokemon
        if opp:
            species = opp.species
            if opp.ability:
                nb.observe_ability(species, opp.ability)
            if opp.item:
                nb.observe_item(species, opp.item)
            if opp.status:
                nb.observe_status(species, opp.status.name)

            # Track opponent moves from battle log
            for move_id in opp.moves:
                nb.observe_move(species, move_id)

            # Damage tracking: compare HP change
            prev_hp = self._last_opp_hp.get(tag, 1.0)
            curr_hp = opp.current_hp_fraction
            if prev_hp > curr_hp and curr_hp > 0:
                dmg_pct = (prev_hp - curr_hp) * 100
                our = battle.active_pokemon
                if our and our.moves:
                    # We can't know exactly which move hit, but track anyway
                    pass  # damage observation requires knowing our move
            self._last_opp_hp[tag] = curr_hp

        # Speed tracking
        if opp and battle.active_pokemon:
            # poke_env doesn't directly expose turn order, but we track what we can
            pass

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
        return self._packed_team

    # ── Phase 2: pick best 3 at team preview ──────────────────────────────────

    async def teampreview(self, battle: AbstractBattle) -> str:  # type: ignore[override]
        our_mons = list(battle.team.values())
        opp_mons = list(battle.teampreview_opponent_team)

        # Initialize notebook from preview
        nb = self._get_notebook(battle)
        opp_species = [m.species for m in opp_mons]
        our_species = [m.species for m in our_mons]
        nb.init_from_preview(opp_species, our_species)

        our_summary = "\n".join(
            f"  {i+1}: {m.species} "
            f"({'/'.join(t.name.capitalize() for t in [m.type_1, m.type_2] if t)})"
            for i, m in enumerate(our_mons)
        )

        # Include KB scouting data in preview prompt
        kb_summary = nb.format_full_notebook()

        prompt = (
            "You are in BSS Reg J team preview. Choose 3 Pokemon to bring.\n\n"
            f"Your registered team (1-indexed):\n{our_summary}\n\n"
            f"Opponent's team:\n{_opp_preview_summary(opp_mons)}\n\n"
            f"Scouting data (common sets from metagame KB):\n{kb_summary}\n\n"
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

    # ── After each battle: cleanup + rebuild ──────────────────────────────────

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        tag = getattr(battle, "battle_tag", None)
        if tag:
            logger = self._loggers.pop(tag, None)
            if logger:
                our_team = [m.species for m in battle.team.values()]
                opp_team = [m.species for m in battle.opponent_team.values()]
                logger.log_summary(battle.won, battle.turn, our_team, opp_team)
            self._notebooks.pop(tag, None)
            self._last_opp_hp.pop(tag, None)
        asyncio.create_task(self._build_team_async())

    # ── Phase 3: battle moves (context injection, no tool calls) ──────────────

    def _get_logger(self, battle: Battle) -> BattleLogger:
        tag = battle.battle_tag
        if tag not in self._loggers:
            self._loggers[tag] = BattleLogger(tag, "Gemini")
        return self._loggers[tag]

    def choose_move(self, battle: Battle):
        self._update_notebook(battle)
        nb = self._get_notebook(battle)
        logger = self._get_logger(battle)

        # Pre-compute all battle context
        context = compute_battle_context(battle, nb)
        print(f"\n{context}\n")

        available_moves   = battle.available_moves
        available_switches = battle.available_switches

        prompt = context + (
            "\n\nBased on the analysis above, choose the best action. "
            "Think about what the opponent will do THIS turn."
        )

        try:
            text = self._sync_call(prompt, BATTLE_SYSTEM)
            data = json.loads(text)
            action_type = data.get("action_type", "").lower()
            target_name = data.get("target_name", "").lower()
            reasoning   = data.get("reasoning", "")

            print(f"[Gemini] Reasoning: {reasoning}")
            print(f"[Gemini] Action: {action_type} -> {target_name}")

            logger.log_turn(
                turn=battle.turn, context=context,
                action=f"{action_type} {target_name}",
                reasoning=reasoning, system="LLM",
            )

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
            logger.log_turn(
                turn=battle.turn, context=context,
                action="random (error)", system="LLM",
            )

        return self.choose_random_move(battle)

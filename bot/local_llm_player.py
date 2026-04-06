"""
Local LLM player using Qwen3-8B loaded directly via transformers.

Two modes (auto-detected):
  - LOCAL mode (default): loads model in-process via transformers. Works on CUDA + MPS.
  - SERVER mode: connects to a vLLM/OpenAI-compatible server if LOCAL_LLM_BASE_URL is set.

Uses pre-computed battle context injection (no tool calls during battle).
All damage calcs, type matchups, and opponent scouting are pre-computed
and injected into the prompt each turn.

Config via env vars (or bot/.env):
    LOCAL_LLM_MODEL     (default: Qwen/Qwen3-8B)
    LOCAL_LLM_BASE_URL  (set to use server mode, e.g. http://localhost:8001/v1)
"""

import asyncio
import json
import os
import re
import torch
from dotenv import load_dotenv
load_dotenv()

from poke_env.player import Player
from poke_env.battle import Battle, AbstractBattle

from pokemon_pool import POOL, POOL_NAMES, POOL_SIZE, build_team_string
from battle_notebook import BattleNotebook
from battle_context import compute_battle_context

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-8B"


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


# ── System prompts ────────────────────────────────────────────────────────────

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


# ── Local model backend (transformers) ────────────────────────────────────────

class _LocalBackend:
    """Load and generate with a local HuggingFace model. Works on CUDA + MPS."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _get_device()
        print(f"[LocalLLM] Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "mps":
            self.model = self.model.to("mps")

        print(f"[LocalLLM] Model loaded on {self.device}")

    def generate(self, messages: list[dict], temperature: float = 0.3,
                 max_new_tokens: int = 1024) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Server backend (OpenAI-compatible API) ────────────────────────────────────

class _ServerBackend:
    """Connect to a vLLM or other OpenAI-compatible server."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        print(f"[LocalLLM] Connecting to server at {base_url} (model: {model_name})")

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


# ── Player class ──────────────────────────────────────────────────────────────

class LocalLLMPlayer(Player):
    """
    Pokemon battle agent powered by a local LLM (Qwen3-8B).

    - Default: loads model in-process via transformers (CUDA / MPS / CPU).
    - If LOCAL_LLM_BASE_URL is set: connects to a vLLM server instead.

    Uses pre-computed battle context injection — no tool calls during battle.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_name = os.environ.get("LOCAL_LLM_MODEL", DEFAULT_MODEL)
        base_url   = os.environ.get("LOCAL_LLM_BASE_URL", "")

        if base_url:
            api_key = os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
            self._backend = _ServerBackend(base_url, api_key, model_name)
            self._use_server = True
        else:
            self._backend = _LocalBackend(model_name)
            self._use_server = False

        # Per-battle notebooks
        self._notebooks: dict[str, BattleNotebook] = {}
        self._last_opp_hp: dict[str, float] = {}

        self._battle_count: int = 0
        self._building_next: bool = False

        print("[LocalLLM] Building first team...")
        self._packed_team: str = self._build_team_sync()

    # ── Notebook management ───────────────────────────────────────────────────

    def _get_notebook(self, battle: Battle) -> BattleNotebook:
        tag = battle.battle_tag
        if tag not in self._notebooks:
            self._notebooks[tag] = BattleNotebook()
        return self._notebooks[tag]

    def _update_notebook(self, battle: Battle):
        nb = self._get_notebook(battle)
        tag = battle.battle_tag

        opp = battle.opponent_active_pokemon
        if opp:
            species = opp.species
            if opp.ability:
                nb.observe_ability(species, opp.ability)
            if opp.item:
                nb.observe_item(species, opp.item)
            if opp.status:
                nb.observe_status(species, opp.status.name)
            for move_id in opp.moves:
                nb.observe_move(species, move_id)

            prev_hp = self._last_opp_hp.get(tag, 1.0)
            self._last_opp_hp[tag] = opp.current_hp_fraction

    # ── LLM call helpers ──────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _call(self, prompt: str, system: str, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if self._use_server:
            raw = self._backend.chat(messages, temperature=temperature)
        else:
            raw = self._backend.generate(messages, temperature=temperature)
        return self._clean(raw)

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
        match = re.search(r'\{[^{}]*"picks"\s*:\s*\[[^\]]+\][^{}]*\}', text)
        if match:
            text = match.group(0)
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
        print(f"[LocalLLM] Team of 6: {', '.join(names)}")
        team_str = build_team_string(picks)
        self.update_team(team_str)
        return self._team.yield_team()

    def _build_team_sync(self) -> str:
        try:
            text  = self._call(self._team_prompt(), TEAM_SYSTEM)
            picks = self._parse_team_picks(text)
            packed = self._apply_picks(picks)
            print("[LocalLLM] Team ready.")
            return packed
        except Exception as e:
            print(f"[LocalLLM] Team build failed ({e}), using default pool order.")
            return self._apply_picks(list(range(6)))

    async def _build_team_async(self) -> None:
        if self._building_next:
            return
        self._building_next = True
        try:
            self._battle_count += 1
            print("[LocalLLM] Rebuilding team for next battle...")
            text  = self._call(self._team_prompt(), TEAM_SYSTEM, temperature=0.7)
            picks = self._parse_team_picks(text)
            self._packed_team = self._apply_picks(picks)
            print("[LocalLLM] Next team ready.")
        except Exception as e:
            print(f"[LocalLLM] Async team build failed ({e}), keeping previous team.")
        finally:
            self._building_next = False

    @property
    def next_team(self) -> str:  # type: ignore[override]
        return self._packed_team

    # ── Phase 2: pick best 3 at team preview ──────────────────────────────────

    def teampreview(self, battle: AbstractBattle) -> str:
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
            print("[LocalLLM] Thinking about lead selection...")
            text = self._call(prompt, TEAM_SYSTEM)
            match = re.search(r'\{[^{}]*"picks"\s*:\s*\[[^\]]+\][^{}]*\}', text)
            if match:
                text = match.group(0)
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
            print(f"[LocalLLM] Bringing 3: {', '.join(chosen)}")
            print(f"[LocalLLM] Lead reasoning: {data.get('reasoning', '')}")
            return "/team " + "".join(str(p) for p in picks)
        except Exception as e:
            print(f"[LocalLLM] Teampreview failed ({e}), using default order.")
            return "/team 123"

    # ── After each battle: cleanup + rebuild ──────────────────────────────────

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        tag = getattr(battle, "battle_tag", None)
        if tag:
            self._notebooks.pop(tag, None)
            self._last_opp_hp.pop(tag, None)
        asyncio.create_task(self._build_team_async())

    # ── Phase 3: battle moves (context injection, no tool calls) ──────────────

    def choose_move(self, battle: Battle):
        self._update_notebook(battle)
        nb = self._get_notebook(battle)

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
            text = self._call(prompt, BATTLE_SYSTEM)

            match = re.search(r'\{[^{}]*"action_type"\s*:.*?\}', text, re.DOTALL)
            if match:
                text = match.group(0)

            data = json.loads(text)
            action_type = data.get("action_type", "").lower()
            target_name = data.get("target_name", "").lower()
            reasoning   = data.get("reasoning", "")

            print(f"[LocalLLM] Reasoning: {reasoning}")
            print(f"[LocalLLM] Action: {action_type} -> {target_name}")

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
            print(f"\n[LocalLLM] Error: {e} — falling back to random.")

        return self.choose_random_move(battle)

"""
Hybrid dual-process player — System 1 (PPO) + System 2 (LLM).

On routine turns, the PPO policy acts instantly (System 1).
When the battle notebook detects surprises — unexpected moves, items,
abilities, bulk/speed deviations — the LLM is invoked for deeper
strategic reasoning (System 2).

Usage:
    HybridPlayer(model_path="ppo_team_builder", llm_backend="gemini")
    HybridPlayer(model_path="ppo_team_builder", llm_backend="qwen")

Config via env vars (or bot/.env):
    GEMINI_API_KEY          (required for llm_backend="gemini")
    LOCAL_LLM_MODEL         (default: Qwen/Qwen3-8B, for llm_backend="qwen")
    LOCAL_LLM_BASE_URL      (optional vLLM server URL for qwen backend)
"""

import json
import os
import re

import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()

from poke_env.player import Player
from poke_env.environment import SinglesEnv
from poke_env.battle import Battle, AbstractBattle

from pokemon_pool import POOL, POOL_NAMES, POOL_SIZE, build_team_string
from battle_notebook import BattleNotebook
from battle_context import compute_battle_context
from battle_logger import BattleLogger

# Reuse PPO infrastructure
from ppo_team_player import (
    _load_model, _selection_obs, _battle_obs, _valid_action_mask,
)
from rl_env import BattleHistory, OBS_SIZE
from rl_env_team import OBS_TEAM_SIZE, OBS_TEAM_SIZE_HIST


# ── Shared helpers ────────────────────────────────────────────────────────────

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


# ── Prompts ───────────────────────────────────────────────────────────────────

PREVIEW_SYSTEM = (
    "You are a Pokemon BSS Reg J team-building expert."
    " Respond ONLY with a raw JSON object — no markdown, no extra text."
)

BATTLE_SYSTEM = (
    "You are an expert Pokemon Showdown player (Gen 9 Battle Stadium Singles, Reg J).\n"
    "You have been called because something UNEXPECTED happened in the battle.\n"
    "A fast reactive model normally handles routine turns, but it escalated to you\n"
    "because surprises were detected that require deeper reasoning.\n\n"
    "Each turn you receive a PRE-CALCULATED ANALYSIS with damage numbers, speed checks,\n"
    "type matchups, and an opponent scouting notebook. ALL MATH IS ALREADY DONE.\n\n"
    "Your job is STRATEGY — reason about the surprise and adapt:\n\n"
    "DECISION PRIORITIES (follow in order):\n"
    "1. KO THREAT: If you can KO the opponent this turn, ATTACK. Do not switch.\n"
    "2. SURVIVE: If the opponent can KO you and you cannot KO them, switch to a resist.\n"
    "3. ATTACK WHEN AHEAD: If you can 2HKO and they cannot OHKO you, ATTACK. Do not pivot.\n"
    "4. MOMENTUM: If no clear KO path exists, use the best damaging move or a pivot move.\n"
    "5. SETUP: Only use setup moves (Swords Dance, etc.) if the opponent cannot threaten you.\n\n"
    "ANTI-PATTERNS TO AVOID:\n"
    "- DO NOT switch back and forth between the same Pokemon. This wastes turns.\n"
    "- DO NOT switch when you have a strong attack that does 30%+ damage.\n"
    "- DO NOT 'preserve' a Pokemon by switching it out repeatedly.\n"
    "- DO NOT switch into a Pokemon that was just forced out.\n\n"
    "Output ONLY a JSON object:\n"
    '{"action_type": "move" or "switch", "target_name": "exact name", '
    '"reasoning": "1-2 sentences explaining how you adapted to the surprise"}\n'
    "No markdown. Just the raw JSON."
)


# ── LLM backends ─────────────────────────────────────────────────────────────

class _GeminiBackend:
    def __init__(self):
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY required for hybrid player with gemini backend")
        self.client = genai.Client()
        self.model = "gemini-3.1-flash-lite-preview"
        self._types = types

    def call(self, prompt: str, system: str, temperature: float = 0.3) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                system_instruction=system, temperature=temperature
            ),
        )
        return resp.text or ""


class _QwenBackend:
    def __init__(self):
        model_name = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-8B")
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "")

        if base_url:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=base_url,
                api_key=os.environ.get("LOCAL_LLM_API_KEY", "not-needed"),
            )
            self._model_name = model_name
            self._use_server = True
            print(f"[Hybrid/Qwen] Server mode: {base_url}")
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"[Hybrid/Qwen] Loading {model_name} on {device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "mps":
                self._model = self._model.to("mps")
            self._use_server = False
            print(f"[Hybrid/Qwen] Model loaded on {device}")

    def call(self, prompt: str, system: str, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if self._use_server:
            resp = self._client.chat.completions.create(
                model=self._model_name, messages=messages,
                temperature=temperature, max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        else:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs, max_new_tokens=1024, do_sample=temperature > 0,
                    temperature=max(temperature, 1e-4), top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Hybrid Player ─────────────────────────────────────────────────────────────

class HybridPlayer(Player):
    """
    Dual-process Pokemon battle agent.

    System 1 (PPO): handles routine turns instantly via learned policy.
    System 2 (LLM): invoked when battle notebook detects surprises.

    Team building: PPO (stochastic selection from pool).
    Team preview: LLM (strategic pick with KB scouting).
    Battle moves: PPO by default, escalates to LLM on surprise.
    """

    def __init__(self, model_path: str = "ppo_team_builder",
                 llm_backend: str = "gemini", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ── System 1: PPO ────────────────────────────────────────────────────
        self.ppo = _load_model(model_path)
        expected_obs = self.ppo.observation_space.shape[0]
        self._use_history = (expected_obs == OBS_TEAM_SIZE_HIST)
        self._obs_size = expected_obs
        self._history = BattleHistory() if self._use_history else None
        print(f"[Hybrid/S1] PPO loaded (obs={expected_obs}, "
              f"history={'yes' if self._use_history else 'no'})")

        # ── System 2: LLM ────────────────────────────────────────────────────
        self._llm_backend_name = llm_backend
        if llm_backend == "gemini":
            self._llm = _GeminiBackend()
        elif llm_backend == "qwen":
            self._llm = _QwenBackend()
        else:
            raise ValueError(f"Unknown llm_backend: {llm_backend!r}")
        print(f"[Hybrid/S2] LLM backend: {llm_backend}")

        # ── Battle state ─────────────────────────────────────────────────────
        self._notebooks: dict[str, BattleNotebook] = {}
        self._last_opp_hp: dict[str, float] = {}
        self._seen_surprises: dict[str, set[str]] = {}
        self._loggers: dict[str, BattleLogger] = {}
        self._s1_count = 0
        self._s2_count = 0

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
            self._last_opp_hp[tag] = opp.current_hp_fraction

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _llm_call(self, prompt: str, system: str, temperature: float = 0.3) -> str:
        raw = self._llm.call(prompt, system, temperature)
        return self._clean(raw)

    # ── Phase 1: Team building (PPO) ──────────────────────────────────────────

    def _select_team(self) -> list[int]:
        picks: list[int] = []
        while len(picks) < 6:
            obs = _selection_obs(picks, self._obs_size).reshape(1, -1)
            action, _ = self.ppo.predict(obs, deterministic=False)
            pick = int(action[0]) % POOL_SIZE
            if pick in picks:
                for i in range(POOL_SIZE):
                    if i not in picks:
                        pick = i
                        break
            picks.append(pick)
        return picks

    @property
    def next_team(self) -> str:  # type: ignore[override]
        picks = self._select_team()
        names = [POOL_NAMES[i] for i in picks]
        print(f"[Hybrid/S1] Team of 6: {', '.join(names)}")
        team_str = build_team_string(picks)
        self.update_team(team_str)
        return self._team.yield_team()

    # ── Phase 2: Team preview (LLM — strategic pick with KB scouting) ─────────

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
            "Pick 3 from YOUR team that best counter the opponent. "
            "Order matters — index 1 is your lead.\n"
            'Return JSON: {"picks": [a, b, c], "reasoning": "..."}\n'
            "picks must be exactly 3 unique integers from 1 to 6."
        )

        try:
            print("[Hybrid/S2] Thinking about lead selection...")
            text = self._llm_call(prompt, PREVIEW_SYSTEM)

            match = re.search(r'\{[^{}]*"picks"\s*:\s*\[[^\]]+\][^{}]*\}', text)
            if match:
                text = match.group(0)
            data = json.loads(text)
            raw = [int(p) for p in data["picks"]]

            seen, valid = set(), []
            for p in raw:
                if 1 <= p <= len(our_mons) and p not in seen:
                    seen.add(p); valid.append(p)
            for i in range(1, len(our_mons) + 1):
                if len(valid) >= 3:
                    break
                if i not in seen:
                    seen.add(i); valid.append(i)
            picks = valid[:3]

            chosen = [our_mons[p - 1].species for p in picks]
            reasoning = data.get("reasoning", "")
            print(f"[Hybrid/S2] Bringing 3: {', '.join(chosen)}")
            print(f"[Hybrid/S2] Reasoning: {reasoning}")
            return "/team " + "".join(str(p) for p in picks)

        except Exception as e:
            print(f"[Hybrid] Preview failed ({e}), using PPO fallback.")
            # Fall back to PPO-style lead scoring
            from ppo_team_player import _lead_score
            scores = [_lead_score(mon, opp_mons) for mon in our_mons]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            picks = ranked[:3]
            return "/team " + "".join(str(i + 1) for i in picks)

    # ── Phase 3: Battle moves — dual-process routing ──────────────────────────

    # Confidence threshold: if PPO's best action probability (after masking)
    # is below this, the model is uncertain and we escalate to the LLM.
    # Typical confident PPO output: 60-90% on best action.
    # Uncertain PPO output: 25-35% spread across multiple actions.
    CONFIDENCE_THRESHOLD = 0.4

    def choose_move(self, battle: Battle):
        self._update_notebook(battle)
        nb = self._get_notebook(battle)
        tag = battle.battle_tag

        # Check for NEW surprises since last check
        all_surprises = nb.get_new_surprises()
        seen = self._seen_surprises.setdefault(tag, set())
        new_surprises = [s for s in all_surprises if s not in seen]
        seen.update(all_surprises)

        # Get PPO's action distribution (needed for both routing and S1 action)
        obs = _battle_obs(battle, self._history, self._obs_size).reshape(1, -1)
        obs_tensor = torch.as_tensor(obs).float().to(self.ppo.device)

        with torch.no_grad():
            dist = self.ppo.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs[0].cpu().numpy()

        mask = _valid_action_mask(battle)
        masked_probs = probs * mask
        total = masked_probs.sum()

        if total > 0:
            normalized = masked_probs / total
            best_prob = float(normalized.max())
            entropy = -float(np.sum(normalized[normalized > 0]
                                    * np.log(normalized[normalized > 0])))
        else:
            best_prob = 0.0
            entropy = 999.0

        # Routing decision: three triggers for System 2
        escalate = False
        reasons = list(new_surprises)

        if new_surprises:
            escalate = True

        if best_prob < self.CONFIDENCE_THRESHOLD:
            escalate = True
            reasons.append(
                f"PPO uncertainty: best action only {best_prob:.0%} confidence "
                f"(threshold {self.CONFIDENCE_THRESHOLD:.0%}, entropy={entropy:.2f})"
            )

        if escalate:
            return self._s2_choose(battle, nb, reasons)
        else:
            return self._s1_choose(battle, masked_probs, best_prob)

    def _get_logger(self, battle: Battle) -> BattleLogger:
        tag = battle.battle_tag
        if tag not in self._loggers:
            self._loggers[tag] = BattleLogger(tag, "Hybrid")
        return self._loggers[tag]

    def _s1_choose(self, battle: Battle, masked_probs: np.ndarray,
                   confidence: float):
        """System 1: PPO fast path — confident routine turn."""
        self._s1_count += 1
        logger = self._get_logger(battle)
        nb = self._get_notebook(battle)
        context = compute_battle_context(battle, nb)

        action = int(np.argmax(masked_probs))
        try:
            order = SinglesEnv.action_to_order(
                np.int64(action), battle, fake=False, strict=False
            )
            print(f"[Hybrid/S1] Turn {battle.turn}: {order} "
                  f"(confidence={confidence:.0%})")
            logger.log_turn(
                turn=battle.turn, context=context,
                action=str(order), system="S1",
                confidence=confidence,
            )
            return order
        except Exception:
            pass

        print(f"[Hybrid/S1] Turn {battle.turn}: fallback to random")
        logger.log_turn(
            turn=battle.turn, context=context,
            action="random (S1 fallback)", system="S1",
            confidence=confidence,
        )
        return self.choose_random_move(battle)

    def _s2_choose(self, battle: Battle, nb: BattleNotebook,
                   new_surprises: list[str]):
        """System 2: LLM slow path, triggered by surprise."""
        self._s2_count += 1
        logger = self._get_logger(battle)

        context = compute_battle_context(battle, nb)

        surprise_block = "SURPRISES DETECTED (reason you were called):\n"
        for s in new_surprises:
            surprise_block += f"  - {s}\n"

        prompt = (
            surprise_block + "\n"
            + context
            + "\n\nAdapt your strategy to the surprises above. "
            "What does this new information change about the best play?"
        )

        available_moves = battle.available_moves
        available_switches = battle.available_switches

        try:
            text = self._llm_call(prompt, BATTLE_SYSTEM)

            match = re.search(r'\{[^{}]*"action_type"\s*:.*?\}', text, re.DOTALL)
            if match:
                text = match.group(0)

            data = json.loads(text)
            action_type = data.get("action_type", "").lower()
            target_name = data.get("target_name", "").lower()
            reasoning = data.get("reasoning", "")

            print(f"[Hybrid/S2] Turn {battle.turn}: surprises={new_surprises}")
            print(f"[Hybrid/S2] Reasoning: {reasoning}")
            print(f"[Hybrid/S2] Action: {action_type} -> {target_name}")

            logger.log_turn(
                turn=battle.turn, context=context,
                action=f"{action_type} {target_name}",
                reasoning=reasoning, system="S2",
                surprises=new_surprises,
            )

            if action_type == "move" and available_moves:
                for move in available_moves:
                    clean_id = move.id.lower()
                    clean_target = re.sub(r'[^a-z0-9]', '', target_name)
                    if clean_target in clean_id or clean_id in clean_target:
                        return self.create_order(move)
                return self.create_order(available_moves[0])

            elif action_type == "switch" and available_switches:
                for mon in available_switches:
                    clean_sp = mon.species.lower()
                    clean_target = re.sub(r'[^a-z0-9]', '', target_name)
                    if clean_target in clean_sp or clean_sp in clean_target:
                        return self.create_order(mon)
                return self.create_order(available_switches[0])

        except Exception as e:
            print(f"[Hybrid/S2] LLM failed ({e}), falling back to S1.")
            logger.log_turn(
                turn=battle.turn, context=context,
                action="S1 fallback (S2 error)", system="S2",
                surprises=new_surprises,
            )
            return self._s1_choose(battle)

        return self.choose_random_move(battle)

    # ── Battle finished ───────────────────────────────────────────────────────

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        tag = getattr(battle, "battle_tag", None)
        total = self._s1_count + self._s2_count
        if total > 0:
            print(f"[Hybrid] Battle stats: S1(PPO)={self._s1_count} "
                  f"S2(LLM)={self._s2_count} "
                  f"({self._s2_count/total*100:.0f}% escalated)")
        # Log summary
        if tag:
            logger = self._loggers.pop(tag, None)
            if logger:
                our_team = [m.species for m in battle.team.values()]
                opp_team = [m.species for m in battle.opponent_team.values()]
                logger.log_summary(battle.won, battle.turn, our_team, opp_team)
            self._notebooks.pop(tag, None)
            self._last_opp_hp.pop(tag, None)
            self._seen_surprises.pop(tag, None)
        self._s1_count = 0
        self._s2_count = 0

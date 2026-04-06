"""
battle_logger.py — Turn-by-turn battle logging for all agents.

Saves the full battle context (damage calcs, predictions, speed checks)
and agent decisions to per-battle log files for analysis.

Logs are saved to battle_logs/<battle_tag>/
  turn_01.txt, turn_02.txt, ...
  summary.txt (after battle ends)
"""

import json
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("battle_logs")


class BattleLogger:
    """Per-battle logger. Create one per battle, call log_turn() each turn."""

    def __init__(self, battle_tag: str, agent_name: str):
        self.battle_tag = battle_tag.replace("/", "_")
        self.agent_name = agent_name
        self.battle_dir = LOGS_DIR / self.battle_tag
        self.battle_dir.mkdir(parents=True, exist_ok=True)
        self.turns: list[dict] = []

    def log_turn(
        self,
        turn: int,
        context: str,
        action: str,
        *,
        reasoning: str = "",
        system: str = "",
        confidence: float = -1,
        surprises: list[str] | None = None,
        extra: dict | None = None,
    ):
        """Log a single turn's context and decision."""
        entry = {
            "turn": turn,
            "system": system,
            "action": action,
            "confidence": confidence,
            "surprises": surprises or [],
            "reasoning": reasoning,
        }
        self.turns.append(entry)

        # Write turn file
        turn_file = self.battle_dir / f"turn_{turn:02d}.txt"
        lines = []
        lines.append(f"Battle: {self.battle_tag}")
        lines.append(f"Agent:  {self.agent_name}")
        lines.append(f"Turn:   {turn}")
        lines.append(f"System: {system or 'N/A'}")
        if confidence >= 0:
            lines.append(f"Confidence: {confidence:.0%}")
        if surprises:
            lines.append(f"Surprises: {', '.join(surprises)}")
        lines.append(f"Action: {action}")
        if reasoning:
            lines.append(f"\n── REASONING ──")
            lines.append(reasoning)
        lines.append(f"\n── BATTLE CONTEXT ──")
        lines.append(context)
        if extra:
            lines.append(f"\n── EXTRA ──")
            for k, v in extra.items():
                lines.append(f"  {k}: {v}")

        turn_file.write_text("\n".join(lines))

    def log_summary(
        self,
        won: bool | None,
        total_turns: int,
        our_team: list[str] = None,
        opp_team: list[str] = None,
    ):
        """Write battle summary after battle ends."""
        s1_count = sum(1 for t in self.turns if t["system"] == "S1")
        s2_count = sum(1 for t in self.turns if t["system"] == "S2")
        llm_count = sum(1 for t in self.turns if t["system"] in ("LLM", "S2"))

        result = "WIN" if won is True else ("LOSS" if won is False else "DRAW")

        lines = []
        lines.append(f"{'='*50}")
        lines.append(f"  BATTLE SUMMARY: {self.battle_tag}")
        lines.append(f"{'='*50}")
        lines.append(f"Agent:   {self.agent_name}")
        lines.append(f"Result:  {result}")
        lines.append(f"Turns:   {total_turns}")
        lines.append(f"Logged:  {len(self.turns)} decisions")
        if our_team:
            lines.append(f"Our team: {', '.join(our_team)}")
        if opp_team:
            lines.append(f"Opp team: {', '.join(opp_team)}")

        if s1_count or s2_count:
            total = s1_count + s2_count
            lines.append(f"\nRouting: S1={s1_count} ({s1_count/total*100:.0f}%) "
                          f"S2={s2_count} ({s2_count/total*100:.0f}%)")

        # Surprise summary
        all_surprises = []
        for t in self.turns:
            all_surprises.extend(t["surprises"])
        if all_surprises:
            lines.append(f"\nSurprises ({len(all_surprises)} total):")
            for s in all_surprises:
                lines.append(f"  - {s}")

        # Turn-by-turn action summary
        lines.append(f"\nTurn-by-turn:")
        for t in self.turns:
            sys_tag = f"[{t['system']}]" if t["system"] else ""
            conf_tag = f" ({t['confidence']:.0%})" if t["confidence"] >= 0 else ""
            surp_tag = " !" if t["surprises"] else ""
            lines.append(f"  T{t['turn']:2d}: {t['action']:30s} {sys_tag}{conf_tag}{surp_tag}")

        lines.append(f"\nTimestamp: {datetime.now().isoformat()}")

        summary_file = self.battle_dir / "summary.txt"
        summary_file.write_text("\n".join(lines))

        # Also save structured JSON
        json_file = self.battle_dir / "summary.json"
        json_file.write_text(json.dumps({
            "battle_tag": self.battle_tag,
            "agent": self.agent_name,
            "result": result,
            "turns": total_turns,
            "s1_count": s1_count,
            "s2_count": s2_count,
            "llm_count": llm_count,
            "our_team": our_team or [],
            "opp_team": opp_team or [],
            "decisions": self.turns,
        }, indent=2))

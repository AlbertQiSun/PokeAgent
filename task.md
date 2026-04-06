# PokeAgent — Remaining Tasks

## 3. GRPO Training for Qwen3-8B

Train the local Qwen3-8B model via Group Relative Policy Optimization to improve battle performance.

### Subtasks

- [ ] **Design reward function**
  - Primary: +1 win, -1 loss
  - Heuristic bonuses: KO opponent Pokemon (+0.2), preserve own Pokemon HP, use super-effective moves (+0.05), avoid ineffective moves (-0.05)
  - Penalty for excessive switching (-0.1 per switch after 2 in a row)

- [ ] **Build rollout pipeline**
  - Play N games with current Qwen policy, collecting (prompt, response, reward) tuples
  - Each game produces a trajectory of turn-level decisions
  - Store as HuggingFace Dataset format for training

- [ ] **Implement GRPO training loop**
  - Use `trl` library (GRPOTrainer) or custom implementation
  - Group size: 4-8 responses per prompt (play same position multiple times)
  - Relative ranking within group as advantage signal
  - KL penalty against reference model to prevent collapse

- [ ] **Training infrastructure**
  - Must run on CUDA (8B model GRPO needs ~40GB VRAM, or use LoRA/QLoRA for ~16GB)
  - Checkpoint saving every N episodes
  - Eval against PPO and Gemini every M episodes to track progress
  - Tensorboard logging for reward curves

- [ ] **LoRA/QLoRA option**
  - Full fine-tune of 8B model is expensive; LoRA rank 16-64 on attention layers is practical
  - Use `peft` library for LoRA integration with transformers
  - Merge LoRA weights back for inference

### Notes
- The current architecture (context injection, no tool calls) makes GRPO straightforward — the model just outputs JSON actions
- Reward can be computed entirely from game outcome + per-turn heuristics
- MPS (Apple Silicon) is fine for inference but too slow for training — use CUDA machine

---

## 4. Expand Metagame Knowledge Base

Current KB has 30 Pokemon. The competitive meta has 50+ viable picks.

- [ ] **Add missing common Pokemon** — Especially: Iron Bundle, Heatran, Zamazenta, Palafin, Dondozo-Tatsugiri, Sneasler, Arcanine, Porygon2, Maushold, Indeedee-F
- [ ] **Add multiple sets per Pokemon** — Many Pokemon run 2-3 viable sets (e.g., Kingambit can be Swords Dance or Assault Vest). Store all with usage percentages.
- [ ] **Auto-update from Smogon usage stats** — Parse pikalytics.com or Smogon usage stats JSON to keep the KB current with the metagame
- [ ] **Add move coverage data** — For each Pokemon, store what types they can hit so the notebook can predict coverage threats

---

## 5. Improve Battle Notebook Observations

- [ ] **Infer EV spreads from damage** — When we deal X% to an opponent, compare against expected damage for standard spreads (252/252/4, defensive, etc.) to narrow down their investment
- [ ] **Detect Tera type** — When opponent uses a move that changes type effectiveness (e.g., suddenly resists what should be SE), infer Tera
- [ ] **Track switching patterns** — Record which Pokemon the opponent switches to in each matchup to predict future switches
- [ ] **Track move order for speed inference** — If opponent outspeeds when they shouldn't (or vice versa), infer Choice Scarf, Tailwind, Trick Room, or speed-boosting nature/EVs
- [ ] **Confidence scoring** — Assign confidence levels to each inference (e.g., "90% Choice Scarf" vs "50% Tailwind") based on number of observations

---

## 6. Prompt Engineering Refinements

- [ ] **Add matchup-specific heuristics** — "Against setup sweepers, don't let them get free turns" / "Against stall, be aggressive"
- [ ] **Endgame awareness** — When both sides are low on Pokemon, switch from midgame strategy to endgame calculation (speed ties, priority moves, last-mon scenarios)
- [ ] **Tera timing guidance** — "Only Tera when it enables a KO, saves you from a KO, or removes a critical weakness. Don't waste Tera early."
- [ ] **History context** — Include last 2-3 turns of actions in the prompt so the LLM can detect opponent patterns (e.g., "they switched Skeledirge in the last 2 times you sent Kingambit")

---

## 7. Self-Play and Curriculum Training (PPO)

- [ ] **Self-play training** — Train PPO against itself or against a frozen copy for harder opponents
- [ ] **Curriculum learning** — Start vs random, then vs heuristic, then vs frozen-PPO, then vs Gemini
- [ ] **Reward shaping** — Add intermediate rewards for type-effective moves, KOs, HP preservation (not just win/loss)
- [ ] **Larger network** — Current 256x256 MLP may be too small; try 512x512 or add LSTM for temporal reasoning

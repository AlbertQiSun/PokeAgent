import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from poke_env.player import Player
from poke_env.battle import Battle

class GeminiPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Attempt to get the API key, throw a descriptive error if not found
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY environment variable to your Gemini API Key before running.")
        
        self.client = genai.Client()
        self.model = "gemini-3.1-pro-preview"
        
        # System instructions to ensure JSON output
        self.system_instruction = (
            "You are a Pokemon Showdown expert playing Gen 9 Battle Stadium Singles."
            " Every turn, you will be given the battle state. Your task is to select the optimal move or switch."
            ' You must ONLY output a valid JSON with exactly these keys: "action_type" ("move" or "switch"),'
            ' "target_name" (the exact name of the move or Pokemon), and "reasoning" (a short explanation).'
            ' Example move: {"action_type": "move", "target_name": "moonblast", "reasoning": "Super effective."}'
            ' Example switch: {"action_type": "switch", "target_name": "dragonite", "reasoning": "Resists earthquake."}'
            " Do NOT output any markdown, just the raw JSON object."
        )
        
    def choose_move(self, battle: Battle):
        # Format the battle state for the LLM prompt
        prompt = self._format_battle_state(battle)
        
        # We need available moves and switches to match the LLM's choice
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        
        # Call Gemini API
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.2, # Low temperature for more deterministic, optimal play
                )
            )
            
            # Basic parsing of the response
            llm_text = response.text.strip()
            
            # Clean up potential markdown formatting around JSON
            if llm_text.startswith("```json"):
                llm_text = llm_text[7:]
            if llm_text.startswith("```"):
                llm_text = llm_text[3:]
            if llm_text.endswith("```"):
                llm_text = llm_text[:-3]
            llm_text = llm_text.strip()

            data = json.loads(llm_text)
            action_type = data.get("action_type", "").lower()
            target_name = data.get("target_name", "").lower()
            reasoning = data.get("reasoning", "")
            
            print(f"\n[Gemini] Reasoning: {reasoning}")
            print(f"[Gemini] Intended Action: {action_type} -> {target_name}")

            # Try to match the action with available options
            if action_type == "move" and available_moves:
                for move in available_moves:
                    # Replace spaces and hyphens to match exact internal names if needed
                    # but typically target_name should be recognizable
                    clean_move_id = move.id.lower()
                    clean_target = re.sub(r'[^a-z0-9]', '', target_name)
                    if clean_target in clean_move_id or clean_move_id in clean_target:
                        return self.create_order(move)
                
                # Fallback to first available move if not found
                return self.create_order(available_moves[0])
                
            elif action_type == "switch" and available_switches:
                for mon in available_switches:
                    clean_mon_sp = mon.species.lower()
                    clean_target = re.sub(r'[^a-z0-9]', '', target_name)
                    if clean_target in clean_mon_sp or clean_mon_sp in clean_target:
                        return self.create_order(mon)
                
                # Fallback to first available switch if not found
                return self.create_order(available_switches[0])
                
        except Exception as e:
            print(f"\n[Gemini] Error calling API or parsing response: {e}")
            print("[Gemini] Falling back to a random action!")
        
        # Fallback to random move if LLM fails
        return self.choose_random_move(battle)
        
    def _format_battle_state(self, battle: Battle) -> str:
        # Construct a detailed prompt describing the game state
        state = []
        state.append("=== CURRENT BATTLE STATE ===")
        state.append(f"Turn: {battle.turn}")
        
        # Our active pokemon
        if battle.active_pokemon:
            active = battle.active_pokemon
            hp_perc = active.current_hp_fraction * 100
            state.append(f"\\n-- Your Active Pokemon --")
            state.append(f"Species: {active.species}")
            state.append(f"HP: {hp_perc:.1f}%")
            state.append(f"Status: {active.status.name if active.status else 'None'}")
            state.append(f"Stat Boosts: {active.boosts}")
            
        # Opponent active pokemon
        if battle.opponent_active_pokemon:
            opp = battle.opponent_active_pokemon
            hp_perc = opp.current_hp_fraction * 100
            state.append(f"\\n-- Opponent Active Pokemon --")
            state.append(f"Species: {opp.species}")
            state.append(f"HP: {hp_perc:.1f}%")
            state.append(f"Status: {opp.status.name if opp.status else 'None'}")
            
        # Field conditions
        if battle.weather or battle.fields:
            state.append(f"\\n-- Field Conditions --")
            state.append(f"Weather: {[w.name for w in battle.weather.keys()]}")
            state.append(f"Fields: {[f.name for f in battle.fields.keys()]}")

        # Available options
        state.append(f"\\n-- Your Available Actions --")
        if battle.available_moves:
            moves_str = ", ".join([f"{m.id} (Base Power: {m.base_power}, Type: {m.type.name})" for m in battle.available_moves])
            state.append(f"Moves: {moves_str}")
        else:
            state.append("Moves: None (Must switch or recharge)")
            
        if battle.available_switches:
            switches_str = ", ".join([f"{mon.species} ({mon.current_hp_fraction*100:.1f}% HP)" for mon in battle.available_switches])
            state.append(f"Switches: {switches_str}")
        else:
            state.append("Switches: None")
            
        state.append("\\nBased on this game state, determine the single best action to take.")
        
        return "\\n".join(state)

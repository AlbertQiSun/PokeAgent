"""
opponent_model.py — Predict what the opponent will do next.

A lightweight MLP that takes battle state features and predicts:
  - Probability of each opponent action category (attack, switch, tera, status)
  - Which move type they're most likely to use

Used by battle_context.py to augment damage calcs with action predictions.

Training data comes from replay_parser.py with --opponent-actions flag.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pokedata import get_pokemon, get_move, type_effectiveness

MODEL_PATH = Path("opponent_model.pt")

# ── Action categories ────────────────────────────────────────────────────────

ACTION_CATEGORIES = [
    "attack_physical",   # 0: physical attacking move
    "attack_special",    # 1: special attacking move
    "attack_status",     # 2: status move (will-o-wisp, toxic, etc.)
    "switch",            # 3: switch out
    "setup",             # 4: stat boost move (swords dance, calm mind, etc.)
    "protect",           # 5: protect / detect / spiky shield
]
N_CATEGORIES = len(ACTION_CATEGORIES)

# Known setup moves
SETUP_MOVES = {
    "swordsdance", "calmmind", "nastyplot", "dragondance", "irondefense",
    "bulkup", "shellsmash", "quiverdance", "coil", "bellydrum",
    "tailwind", "trickroom", "auroraveil", "reflect", "lightscreen",
}

PROTECT_MOVES = {
    "protect", "detect", "spikyshield", "banefulbunker", "kingsshield",
    "silktrap", "burningbulwark", "obstruct",
}


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(
    our_species: str,
    opp_species: str,
    our_hp_frac: float,
    opp_hp_frac: float,
    our_types: list[str],
    opp_types: list[str],
    our_base_stats: dict,
    opp_base_stats: dict,
    opp_moves_seen: list[str],
    opp_item: str = "",
    opp_ability: str = "",
    our_boosts: dict = None,
    opp_boosts: dict = None,
    turn: int = 1,
    our_bench_count: int = 2,
    opp_bench_count: int = 2,
    weather: str = "",
    terrain: str = "",
) -> np.ndarray:
    """Extract numeric feature vector from battle state for opponent prediction.

    Returns: numpy array of shape (N_FEATURES,)
    """
    features = []

    # ── HP (2) ──
    features.append(our_hp_frac)
    features.append(opp_hp_frac)

    # ── Base stats normalized to 0-1 range (12) ──
    for stat_key in ["hp", "atk", "def", "spa", "spd", "spe"]:
        features.append(our_base_stats.get(stat_key, 80) / 255.0)
    for stat_key in ["hp", "atk", "def", "spa", "spd", "spe"]:
        features.append(opp_base_stats.get(stat_key, 80) / 255.0)

    # ── Speed comparison (3) ──
    our_spe = our_base_stats.get("spe", 80)
    opp_spe = opp_base_stats.get("spe", 80)
    features.append(1.0 if our_spe > opp_spe else 0.0)  # we outspeed
    features.append(1.0 if opp_spe > our_spe else 0.0)  # they outspeed
    features.append(abs(our_spe - opp_spe) / 200.0)       # speed gap

    # ── Type matchup (4) ──
    # Our STAB effectiveness vs opponent
    our_best_eff = 1.0
    for ot in our_types:
        opp_t1 = opp_types[0] if opp_types else "Normal"
        opp_t2 = opp_types[1] if len(opp_types) > 1 else None
        eff = type_effectiveness(ot, opp_t1, opp_t2)
        if eff > our_best_eff:
            our_best_eff = eff

    # Opponent STAB effectiveness vs us
    opp_best_eff = 1.0
    for ot in opp_types:
        our_t1 = our_types[0] if our_types else "Normal"
        our_t2 = our_types[1] if len(our_types) > 1 else None
        eff = type_effectiveness(ot, our_t1, our_t2)
        if eff > opp_best_eff:
            opp_best_eff = eff

    features.append(our_best_eff / 4.0)   # our advantage (normalized)
    features.append(opp_best_eff / 4.0)   # their advantage
    features.append(1.0 if our_best_eff >= 2.0 else 0.0)  # we have SE
    features.append(1.0 if opp_best_eff >= 2.0 else 0.0)  # they have SE

    # ── Boosts (10) ──
    our_b = our_boosts or {}
    opp_b = opp_boosts or {}
    for stat in ["atk", "def", "spa", "spd", "spe"]:
        features.append(our_b.get(stat, 0) / 6.0)
    for stat in ["atk", "def", "spa", "spd", "spe"]:
        features.append(opp_b.get(stat, 0) / 6.0)

    # ── Game state (5) ──
    features.append(min(turn, 30) / 30.0)        # turn number
    features.append(our_bench_count / 5.0)         # our bench
    features.append(opp_bench_count / 5.0)         # their bench
    features.append(1.0 if weather else 0.0)
    features.append(1.0 if terrain else 0.0)

    # ── Opponent intel (4) ──
    features.append(len(opp_moves_seen) / 4.0)   # move reveal count
    features.append(1.0 if opp_item else 0.0)      # item known
    features.append(1.0 if opp_ability else 0.0)   # ability known
    is_choice = opp_item.lower().replace(" ", "") in ("choiceband", "choicespecs", "choicescarf") if opp_item else False
    features.append(1.0 if is_choice else 0.0)     # locked into one move

    # ── Opponent known moves vs us (3) ──
    # Best effectiveness and power of opponent's revealed moves against us
    opp_move_best_eff = 0.0
    opp_move_best_power = 0.0
    opp_move_has_se = 0.0
    our_t1 = our_types[0] if our_types else "Normal"
    our_t2 = our_types[1] if len(our_types) > 1 else None
    for mname in opp_moves_seen:
        mdata = get_move(mname.lower().replace(" ", "").replace("-", ""))
        if mdata:
            mtype = mdata.get("type", "Normal")
            eff = type_effectiveness(mtype, our_t1, our_t2)
            power = mdata.get("basePower", 0) / 150.0  # normalize
            eff_power = eff * power
            if eff_power > opp_move_best_eff * opp_move_best_power:
                opp_move_best_eff = eff / 4.0
                opp_move_best_power = power
            if eff >= 2.0:
                opp_move_has_se = 1.0
    features.append(opp_move_best_eff)
    features.append(opp_move_best_power)
    features.append(opp_move_has_se)

    # ── Opponent known move category distribution (3) ──
    n_phys_seen = 0
    n_spec_seen = 0
    n_status_seen = 0
    for mname in opp_moves_seen:
        mdata = get_move(mname.lower().replace(" ", "").replace("-", ""))
        if mdata:
            cat = mdata.get("category", "Status")
            if cat == "Physical":
                n_phys_seen += 1
            elif cat == "Special":
                n_spec_seen += 1
            else:
                n_status_seen += 1
    features.append(n_phys_seen / 4.0)
    features.append(n_spec_seen / 4.0)
    features.append(n_status_seen / 4.0)

    # ── Our threat to opponent (2) ──
    # Rough damage proxy: best STAB eff × our attacking stat / their defensive stat
    our_atk = our_base_stats.get("atk", 80)
    our_spa = our_base_stats.get("spa", 80)
    opp_def = opp_base_stats.get("def", 80)
    opp_spd = opp_base_stats.get("spd", 80)
    phys_threat = (our_atk / opp_def) * our_best_eff
    spec_threat = (our_spa / opp_spd) * our_best_eff
    our_threat = max(phys_threat, spec_threat)
    features.append(min(our_threat / 3.0, 1.0))  # our threat level (capped)
    # Opponent's threat to us (same idea)
    opp_atk = opp_base_stats.get("atk", 80)
    opp_spa = opp_base_stats.get("spa", 80)
    our_def = our_base_stats.get("def", 80)
    our_spd_stat = our_base_stats.get("spd", 80)
    opp_phys_threat = (opp_atk / our_def) * opp_best_eff
    opp_spec_threat = (opp_spa / our_spd_stat) * opp_best_eff
    opp_threat = max(opp_phys_threat, opp_spec_threat)
    features.append(min(opp_threat / 3.0, 1.0))  # their threat to us

    # ── Opponent physical vs special lean (1) ──
    # > 0.5 means physical leaning, < 0.5 means special leaning
    features.append(opp_atk / (opp_atk + opp_spa + 1e-6))

    # ── HP danger zones (2) ──
    features.append(1.0 if our_hp_frac < 0.3 else 0.0)   # we're low
    features.append(1.0 if opp_hp_frac < 0.3 else 0.0)   # they're low

    # ── Switch pressure (1) ──
    # If opponent has no bench, they can't switch
    features.append(0.0 if opp_bench_count == 0 else 1.0)

    return np.array(features, dtype=np.float32)


N_FEATURES = 52  # total features from extract_features


# ── MLP Model ────────────────────────────────────────────────────────────────

class OpponentPredictor(nn.Module):
    """3-layer MLP for opponent action prediction."""

    def __init__(self, n_features: int = N_FEATURES, n_categories: int = N_CATEGORIES,
                 hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_categories),
        )

    def forward(self, x):
        return self.net(x)

    def predict_probs(self, features: np.ndarray) -> dict[str, float]:
        """Predict action category probabilities from features."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self(x)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        return {ACTION_CATEGORIES[i]: float(probs[i]) for i in range(N_CATEGORIES)}


# ── Classify a move into action category ─────────────────────────────────────

def classify_action(action_type: str, move_name: str = "") -> int:
    """Classify a (action_type, move_name) into category index."""
    if action_type == "switch":
        return 3  # switch

    move_key = move_name.lower().replace(" ", "").replace("-", "")

    if move_key in PROTECT_MOVES:
        return 5  # protect
    if move_key in SETUP_MOVES:
        return 4  # setup

    move_data = get_move(move_key)
    if move_data:
        cat = move_data.get("category", "Status")
        if cat == "Physical":
            return 0
        elif cat == "Special":
            return 1
        elif cat == "Status":
            return 2

    return 2  # default to status


# ── Load/save ────────────────────────────────────────────────────────────────

def load_model(path: str = None) -> OpponentPredictor | None:
    """Load trained model if it exists."""
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        return None
    model = OpponentPredictor()
    model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ── Predict from battle objects ──────────────────────────────────────────────

_cached_model: OpponentPredictor | None = None
_cache_checked = False


def predict_opponent_action(
    our_species: str,
    opp_species: str,
    our_hp_frac: float,
    opp_hp_frac: float,
    opp_intel=None,
    our_boosts: dict = None,
    opp_boosts: dict = None,
    turn: int = 1,
    our_bench_count: int = 2,
    opp_bench_count: int = 2,
    weather: str = "",
    terrain: str = "",
    opp_moves_list: list[str] | None = None,
) -> dict[str, float] | None:
    """Predict opponent action probabilities. Returns None if model not available."""
    global _cached_model, _cache_checked

    if not _cache_checked:
        _cached_model = load_model()
        _cache_checked = True

    if _cached_model is None:
        return None

    our_data = get_pokemon(our_species)
    opp_data = get_pokemon(opp_species)
    if not our_data or not opp_data:
        return None

    our_types = our_data.get("types", [])
    opp_types = opp_data.get("types", [])
    our_stats = our_data.get("baseStats", {})
    opp_stats = opp_data.get("baseStats", {})

    if opp_intel:
        opp_moves_seen = opp_intel.confirmed_moves
    elif opp_moves_list:
        opp_moves_seen = opp_moves_list
    else:
        opp_moves_seen = []
    opp_item = (opp_intel.item if opp_intel else "") or ""
    opp_ability = (opp_intel.ability if opp_intel else "") or ""

    features = extract_features(
        our_species, opp_species,
        our_hp_frac, opp_hp_frac,
        our_types, opp_types,
        our_stats, opp_stats,
        opp_moves_seen, opp_item, opp_ability,
        our_boosts, opp_boosts,
        turn, our_bench_count, opp_bench_count,
        weather, terrain,
    )

    return _cached_model.predict_probs(features)

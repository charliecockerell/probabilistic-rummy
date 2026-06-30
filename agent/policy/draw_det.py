"""
Determinized-lookahead draw valuation.

The one-ply rule in `draw.py` scores each draw option by next-step deadwood.
This values them by *lookahead* instead: sample worlds consistent with the
belief (Madow), then roll each option forward to a terminal score with the
engine-faithful base playout. Take the discard iff its rollout EV beats the
stock's. No alpha/gamma knob -- the lookahead prices the choice directly.

Reuses the validated determinization + rollout pieces (the same ones the knock
optimal-stopping work uses); this module only adds the draw-time branch. Kept
out of the package __init__ so it imports cleanly without circularity.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from agent.cards import Card, make_deck
from agent.policy.knock import sample_opp_hands
from agent.policy.search import _rollout, _fast_risk_discard

_DECK = make_deck()


def determinize(bs, n_worlds: int, rng: np.random.Generator):
    """Madow-sample opponent hands consistent with the belief; the leftover
    uncertain cards form a shuffled stock. Returns a list of (opp_hand, stock)."""
    p = np.array([bs.prob(c) for c in _DECK])
    uncertain = [_DECK[i] for i in np.where((p > 1e-12) & (p < 1 - 1e-9))[0]]
    worlds = []
    for opp in sample_opp_hands(bs, n_worlds, rng):
        held = set(opp)
        stock = [c for c in uncertain if c not in held]
        rng.shuffle(stock)
        worlds.append((opp, stock))
    return worlds


def draw_values(hand: List[Card], top: Card, bs, n_det: int,
                rng: np.random.Generator) -> Tuple[float, float]:
    """(V_take, V_stock): signed terminal-score EVs of the two draw options,
    scored on one shared set of determinized worlds (common random numbers, so
    their difference is paired). Higher is better.

    Take: add the face-up `top`, discard best back to 10 (gin forbids
    re-discarding the card just taken), then the opponent moves.
    Stock: in each world the shuffled leftover stock fixes the card we draw,
    and we move."""
    worlds = determinize(bs, n_det, rng)
    me11 = list(hand) + [top]
    d = _fast_risk_discard(me11, bs, alpha=0.0, forbidden=top)   # naive min-deadwood drop
    kept = [c for c in me11 if c is not d]
    v_take = np.mean([_rollout(kept, opp, list(stock), me_to_move=False)
                      for opp, stock in worlds])
    v_stock = np.mean([_rollout(list(hand), opp, list(stock), me_to_move=True)
                       for opp, stock in worlds])
    return float(v_take), float(v_stock)
